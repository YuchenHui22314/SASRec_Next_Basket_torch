"""
SASRec
@author: Tianyu Zhu
@created: 2023/3/1
@modified: 2023/3/29
"""

import time
import argparse
import numpy as np
# import scipy.sparse as sp
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from evaluate import evaluate, save_result


class SASRec(object):
    def __init__(self, num_user, num_item, args):
        self.num_user = num_user
        self.num_item = num_item

        self.num_factor = args.num_factor
        self.l2_reg = args.l2_reg
        self.lr = args.lr
        self.max_seq_len = args.max_seq_len
        self.max_basket_len = args.max_basket_len
        self.num_block = args.num_block
        self.num_head = args.num_head

        self.dropout_rate = tf.placeholder(tf.float32)
        self.input_seq = tf.placeholder(tf.int32, [None, self.max_seq_len, self.max_basket_len], name="input_seq")
        self.label = tf.placeholder(tf.float32, [None, self.num_item], name="label")

        self.mask = tf.cast(tf.not_equal(self.input_seq, self.num_item), tf.float32) # [batch_size, max_seq_len, max_basket_len]

        with tf.name_scope("item_embedding"):
            item_embedding_ = tf.Variable(tf.random_normal([self.num_item, self.num_factor], stddev=0.01), name="item_embedding")
            item_embedding = tf.concat([item_embedding_, tf.zeros([1, self.num_factor], dtype=tf.float32)], 0)

        with tf.name_scope("positional_embedding"):
            position = tf.tile(tf.expand_dims(tf.range(self.max_seq_len), 0), [tf.shape(self.input_seq)[0], 1]) # [batch_size, max_seq_len]
            position_embedding = tf.Variable(tf.random_normal([self.max_seq_len, self.num_factor], stddev=0.01), name="position_embedding")
            p_emb = tf.nn.embedding_lookup(position_embedding, position) # [batch_size, max_seq_len, num_factor]
            seq_emb = tf.reduce_sum(tf.nn.embedding_lookup(item_embedding, self.input_seq), 2) / (tf.reduce_sum(self.mask, 2, True) + 1e-24) # [batch_size, max_seq_len, num_factor] / [batch_size, max_seq_len, 1]
            seq_emb = tf.nn.dropout(seq_emb + p_emb, keep_prob=1-self.dropout_rate) * (tf.reduce_sum(self.mask, 2, True) + 1e-24) # [batch_size, max_seq_len, num_factor] * [batch_size, max_seq_len, 1]

        with tf.name_scope("block"):
            for _ in range(self.num_block):
                # Self-attention
                # Linear projections
                seq = seq_emb
                seq_norm = self.layer_normalize(seq)
                Q = tf.layers.dense(seq_norm, self.num_factor, activation=None)
                K = tf.layers.dense(seq, self.num_factor, activation=None)
                V = tf.layers.dense(seq, self.num_factor, activation=None)

                # Split and concat
                Q_ = tf.concat(tf.split(Q, self.num_head, axis=2), axis=0)
                K_ = tf.concat(tf.split(K, self.num_head, axis=2), axis=0)
                V_ = tf.concat(tf.split(V, self.num_head, axis=2), axis=0)

                # Multiplication
                outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

                # Scale
                outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

                # Key Masking
                key_masks = tf.sign(tf.reduce_sum(tf.abs(seq), axis=-1))
                key_masks = tf.tile(key_masks, [self.num_head, 1])
                key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(seq_norm)[1], 1])

                paddings = tf.ones_like(outputs)*(-2**32+1)
                outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

                # Causality (Future blinding)
                diag_vals = tf.ones_like(outputs[0, :, :])
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

                paddings = tf.ones_like(masks)*(-2**32+1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

                # Activation
                outputs = tf.nn.softmax(outputs)

                # Query Masking
                query_masks = tf.sign(tf.reduce_sum(tf.abs(seq_norm), axis=-1))
                query_masks = tf.tile(query_masks, [self.num_head, 1])
                query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(seq)[1]])
                outputs *= query_masks

                # Dropouts
                outputs = tf.nn.dropout(outputs, keep_prob=1-self.dropout_rate)

                # Weighted sum
                outputs = tf.matmul(outputs, V_)

                # Restore shape
                outputs = tf.concat(tf.split(outputs, self.num_head, axis=0), axis=2)

                # Residual connection
                outputs += seq_norm

                # Layer normalization
                outputs = self.layer_normalize(outputs)

                # Feed forward
                # Layer 1
                outputs_ = tf.layers.dense(outputs, self.num_factor, activation=tf.nn.relu, use_bias=True)
                outputs_ = tf.nn.dropout(outputs_, keep_prob=1-self.dropout_rate)

                # Layer 2
                outputs_ = tf.layers.dense(outputs_, self.num_factor, activation=None, use_bias=True)
                outputs_ = tf.nn.dropout(outputs_, keep_prob=1-self.dropout_rate)

                # Residual connection
                outputs += outputs_

                seq_emb = outputs * (tf.reduce_sum(self.mask, 2, True) + 1e-24)

            seq_emb = self.layer_normalize(seq_emb)  # [batch_size, max_seq_len, num_factor]

        with tf.name_scope("train"):
            input_seq_emb = tf.reshape(seq_emb, [tf.shape(self.input_seq)[0] * self.max_seq_len, self.num_factor])  # [batch_size * max_seq_len, num_factor]
            logits = tf.matmul(input_seq_emb, tf.transpose(item_embedding_))  # [batch_size * max_seq_len, num_item]
            print(logits.shape)
            target = tf.reshape(tf.reduce_sum(tf.cast(tf.not_equal(self.input_seq, self.num_item), tf.float32), 2), [tf.shape(self.input_seq)[0] * self.max_seq_len])  # [batch_size * max_seq_len]
            target = target / (target + 1e-24)
            if args.loss == "sigmoid":
                logits_sigmoid = tf.nn.sigmoid(logits)  # [batch_size, num_item]
                loss = -tf.reduce_sum((tf.reduce_sum(self.label * tf.log(logits_sigmoid + 1e-24) + (1.0 - self.label) * tf.log(1.0 - logits_sigmoid + 1e-24), 1)) * target) / tf.reduce_sum(target)
            else:
                logits_log_softmax = tf.nn.log_softmax(logits)  # [batch_size * max_seq_len, num_item]
                loss = -tf.reduce_sum(tf.reduce_sum(self.label * logits_log_softmax, 1) * target) / tf.reduce_sum(target)

            self.loss = loss + self.l2_reg * tf.reduce_sum([tf.nn.l2_loss(va) for va in tf.trainable_variables()])
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.name_scope("test"):
            self.test_logits = tf.matmul(seq_emb[:, -1, :], tf.transpose(item_embedding_))

    def layer_normalize(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(self.num_factor))
        gamma = tf.Variable(tf.ones(self.num_factor))
        normalized = (inputs - mean) / ((variance + 1e-24) ** 0.5)
        outputs = gamma * normalized + beta
        return outputs


def get_sequences(train_dict, validate_dict, num_item, max_seq_len, max_basket_len):
    sequences = list()
    for user in train_dict:
        sequences_user = list()
        # 把每一个basket里面都补齐到max_basket_len
        # seuqences_user = [[1, 2, 3, 0, 0], [4, 5, 6, 7, 0], [8, 9, 10, 11, 12]]
        # 包含了training和validation的basket
        for basket in train_dict[user] + [validate_dict[user]]:
            sequences_user.append(basket + [num_item] * (max_basket_len - len(basket)))
        # sequences 把basket的个数补齐到max_seq_len，不够的话向左加。
        sequences.append([[num_item] * max_basket_len] * (max_seq_len + 2 - len(sequences_user)) + sequences_user)
    sequences = np.array(sequences, dtype=np.int32)
    return sequences


def get_batches(sequences, batch_size):
    batches = list()
    idx = list(range(len(sequences)))
    np.random.shuffle(idx)
    i = 0
    while i < len(sequences):
        batches.append(sequences[idx[i:i+batch_size]])
        i += batch_size
    return batches


def get_feed_dict_train(model, num_item, batch, dropout_rate):
    feed_dict = dict()
    feed_dict[model.dropout_rate] = dropout_rate
    # 取到倒数第二个train basket做input，因为要predict最后一个basket
    feed_dict[model.input_seq] = batch[:, :-2, :]  # batch: [batch_size, train[0]...train[-2] train[-1] validate[], max_basket_len]
    # 需要预测的是这个。和train seq比向右移动了一位
    pred_seq = batch[:, 1:-1, :]
    labels = list()
    for row in np.reshape(pred_seq, [-1, pred_seq.shape[-1]]):
        # 拆成一个个basket，然后为每一个basket生成一个label 向量，
        # 其中basket里面的item对应的位置为1，其他为0
        label_ = np.zeros(shape=num_item+1, dtype=np.float32)
        label_[row] = 1.0
        labels.append(label_[:-1])
    feed_dict[model.label] = np.array(labels)
    return feed_dict


def get_feed_dict_validate(model, batch):
    feed_dict = dict()
    feed_dict[model.dropout_rate] = 0.0
    feed_dict[model.input_seq] = batch[:, 1:-1, :]
    return feed_dict


def get_feed_dict_test(model, batch):
    feed_dict = dict()
    feed_dict[model.dropout_rate] = 0.0
    feed_dict[model.input_seq] = batch[:, 2:, :]
    return feed_dict


def get_top_K_index(pred_scores, K):
    ind = np.argpartition(pred_scores, -K)[:, -K:]
    arr_ind = pred_scores[np.arange(len(pred_scores))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_scores)), ::-1]
    batch_pred_list = ind[np.arange(len(pred_scores))[:, None], arr_ind_argsort]
    return batch_pred_list.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="SASRec")
    parser.add_argument("--dataset", type=str, default="metro_01") # metro_yogurt_02
    # common hyperparameters
    parser.add_argument("--num_factor", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2_reg", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--random_seed", type=int, default=2023)
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--K", type=int, default=20)
    # model-specific hyperparameters
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--num_block", type=int, default=1)
    parser.add_argument("--num_head", type=int, default=1)
    parser.add_argument("--loss", type=str, default="softmax")  # {sigmoid, softmax}

    args = parser.parse_args()
    for arg, arg_value in vars(args).items():
        print(arg, ":", arg_value)

    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    [train_dict, validate_dict, test_dict, num_user, num_item] = np.load("metro_02.npy", allow_pickle=True)
    print("num_user: %d, num_item: %d" % (num_user, num_item))
    seq_len = [len(train_dict[u] + [validate_dict[u]]) for u in train_dict]
    # basket的个数
    print("max seq len: %d, min seq len: %d, avg seq len: %.4f, med seq len: %.4f" % (np.max(seq_len), np.min(seq_len), np.mean(seq_len), np.median(seq_len)))
    basket_len = [len(b) for u in train_dict for b in train_dict[u] + [validate_dict[u]]]
    # 一个basket里面的item个数
    print("max basket len: %d, min basket len: %d, avg basket len: %.4f, med basket len: %.4f" % (np.max(basket_len), np.min(basket_len), np.mean(basket_len), np.median(basket_len)))
    args.max_seq_len, args.max_basket_len = np.max(seq_len), np.max(basket_len)
    sequences = get_sequences(train_dict, validate_dict, num_item, args.max_seq_len, args.max_basket_len)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        print("Model preparing...")
        model = SASRec(num_user, num_item, args)
        sess.run(tf.global_variables_initializer())

        print("Model training...")
        result_validate = list()
        result_test = list()
        for epoch in range(1, args.num_epoch+1):
            t1 = time.time()
            train_loss = list()
            batches = get_batches(sequences, args.batch_size)
            for batch in batches:
                loss, _ = sess.run([model.loss, model.train_op], feed_dict=get_feed_dict_train(model, num_item, batch, args.dropout_rate))
                train_loss.append(loss)
            train_loss = np.mean(train_loss)
            print("epoch: %d, %.2fs" % (epoch, time.time() - t1))
            print("training loss: %.4f" % train_loss)

            if epoch == 1 or epoch % args.N == 0:
                batch_size_test = args.batch_size
                rank_list = list()
                for start in range(0, num_user, batch_size_test):
                    test_logits = sess.run(model.test_logits, feed_dict=get_feed_dict_validate(model, sequences[start:start+batch_size_test]))
                    rank_list += get_top_K_index(test_logits, args.K)
                precision_k_validate, recall_k_validate, ndcg_k_validate = evaluate(rank_list, validate_dict, args.K)
                print('validate precision@{K}: %.4f, recall@{K}: %.4f, ndcg@{K}: %.4f'.format(K=args.K) % (precision_k_validate, recall_k_validate, ndcg_k_validate))
                result_validate.append([epoch] + evaluate(rank_list, validate_dict, 10) + evaluate(rank_list, validate_dict, 20))

                #print("Model testing...")
                rank_list = list()
                for start in range(0, num_user, batch_size_test):
                    test_logits = sess.run(model.test_logits, feed_dict=get_feed_dict_test(model, sequences[start:start+batch_size_test]))
                    rank_list += get_top_K_index(test_logits, args.K)
                precision_k_test, recall_k_test, ndcg_k_test = evaluate(rank_list, test_dict, args.K)
                print('test precision@{K}: %.4f, recall@{K}: %.4f, ndcg@{K}: %.4f'.format(K=args.K) % (precision_k_test, recall_k_test, ndcg_k_test))
                result_test.append([epoch] + evaluate(rank_list, test_dict, 10) + evaluate(rank_list, test_dict, 20))

        save_result(args, result_validate, result_test)
