import numpy as np
from utils import *
import csv
import torch

def get_sequences(train_dict, validate_dict, num_item, max_seq_len, max_basket_len):
    sequences = list()
    for user in train_dict:
        sequences_user = list()
        # 把每一个basket里面都补齐到max_basket_len
        # seuqences_user = [[1, 2, 3, 0, 0], [4, 5, 6, 7, 0], [8, 9, 10, 11, 12]]
        # 包含了training和validation的basket
        for basket in train_dict[user] + [validate_dict[user]]:
            sequences_user.append(basket + [num_item] * (max_basket_len - len(basket)))
        # sequences 把basket的个数补齐到 max_seq_len，不够的话向左加。
        # why + 2?
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


def get_inputs_train(num_item, batch):
    # 取到倒数第二个train basket做input，因为要predict最后一个basket
    input_seq = batch[:, :-2, :]  # batch: [batch_size, train[0]...train[-2] train[-1] validate[], max_basket_len]
    # 需要预测的是这个。和train seq比向右移动了一位
    pred_seq = batch[:, 1:-1, :]
    # Create a mask array to ignore padding item (num_item + 1)
    mask = (pred_seq != num_item )

    # Convert array a to one-hot encoding
    one_hot_a = np.eye(num_item + 1, dtype= np.int8)[pred_seq]

    # Apply mask to one-hot array
    one_hot_a = one_hot_a * mask[..., None]

    # Sum one-hot array along the second axis to get multi-hot representation
    multi_hot_a = one_hot_a.sum(axis=2)
    # pred_seq = torch.tensor(pred_seq).long()
    # mask = (pred_seq != 5).long()
    # # Convert tensor a to one-hot encoding
    # one_hot_pred = torch.nn.functional.one_hot(pred_seq, num_classes = num_item + 1)
    # # Apply mask to one-hot tensor
    # one_hot_pred = one_hot_pred * mask.unsqueeze(-1)
    # # Sum one-hot tensor along the second axis to get multi-hot representation
    # multi_hot_pred = one_hot_pred.sum(dim=2)


    # labels = list()
    # for row in np.reshape(pred_seq, [-1, pred_seq.shape[-1]]):
    #     # 拆成一个个basket，然后为每一个basket生成一个label 向量，
    #     # 其中basket里面的item对应的位置为1，其他为0
    #     label_ = np.zeros(shape=num_item+1, dtype=np.float32)
    #     label_[row] = 1.0
    #     labels.append(label_[:-1])
    # labels = np.array(labels)
    return input_seq, multi_hot_a, pred_seq 

def load_dataset_batches(args):
    #### load dataset 
    # validate_dict and test_dict :
    # one user one basket
    # if a user has less than 3 basket, then no validate_dict nor test_dict
    [train_dict, validate_dict, test_dict, num_user, num_item] = np.load("metro_02.npy", allow_pickle=True)

    # print number of users and items
    print("num_user: %d, num_item: %d" % (num_user, num_item))

    # how many basket does each user have, whey + validate dict?
    seq_len = [len(train_dict[u] + [validate_dict[u]]) for u in train_dict]
    print("max seq len: %d, min seq len: %d, avg seq len: %.4f, med seq len: %.4f" % (np.max(seq_len), np.min(seq_len), np.mean(seq_len), np.median(seq_len)))



    # how may items does each basket have 
    basket_len = [len(b) for u in train_dict for b in train_dict[u] + [validate_dict[u]]]
    print("max basket len: %d, min basket len: %d, avg basket len: %.4f, med basket len: %.4f" % (np.max(basket_len), np.min(basket_len), np.mean(basket_len), np.median(basket_len)))
    args.max_seq_len, args.max_basket_len = np.max(seq_len), np.max(basket_len)

    # sequences = [num_user, max_seq_len (max number of baskets for a user among all users), max_basket_len (max number of items for a basket)]. elements are item_id, if item_i = num_item, then it is padding

    '''
     max_seq_len = 3, max_basket_len = 3, num_item = 5,
    sequences = [
        [ [5,5,5], [5,5,5], [3,2,5] ], // the fist two baskets are padding, the last basket has 2 items, 3 and 2, 5 is padding
        [ [5,5,5], [1,2,3], [4,5,6] ],
        .....
    ]
    '''
    sequences = get_sequences(train_dict, validate_dict, num_item, args.max_seq_len, args.max_basket_len)

    # split to batches
    batches = get_batches(sequences, args.batch_size) #(random_shuffle)

    return batches, num_user, num_item, train_dict, validate_dict, test_dict, sequences

def get_feed_dict_validate( batch):
    input_seq = batch[:, 1:-1, :]
    return input_seq 


def get_feed_dict_test( batch):
    input_seq = batch[:, 2:, :]
    return input_seq


def get_top_K_index(pred_scores, K):
    ind = np.argpartition(pred_scores, -K)[:, -K:]
    arr_ind = pred_scores[np.arange(len(pred_scores))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_scores)), ::-1]
    batch_pred_list = ind[np.arange(len(pred_scores))[:, None], arr_ind_argsort]
    return batch_pred_list.tolist()


def save_result(args, result_valid, result_test):
    ndcg_10 = list(np.array(result_valid)[:, 6])
    ndcg_10_max = max(ndcg_10)
    result_report = result_test[ndcg_10.index(ndcg_10_max)]

    result_test_array = np.array(result_test)
    result_max = ["max", max(result_test_array[:, 1]), max(result_test_array[:, 2]), max(result_test_array[:, 3]), max(result_test_array[:, 4]), max(result_test_array[:, 5]), max(result_test_array[:, 6])]

    args_dict = vars(args)
    filename = ""
    for arg in args_dict:
        filename += str(args_dict[arg]) + "_"
    with open(filename + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "Precision@1", "Recall@1", "NDCG@1", "Precision@10", "Recall@10", "NDCG@10"])
        for line in result_test:
            writer.writerow(line)
        writer.writerow(result_report)
        writer.writerow(result_max)
        for arg in args_dict:
            writer.writerow(["", arg, args_dict[arg]] + [""] * (len(line) - 3))