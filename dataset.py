import numpy as np

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