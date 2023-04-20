import numpy as np
import csv


def evaluate(rank_list, test_dict, K):
    rank_list = np.array(rank_list)[:, :K]
    precision_list = list()
    recall_list = list()
    ndcg_list = list()
    for user in range(rank_list.shape[0]):
        if user in test_dict and len(test_dict[user]) > 0:
            hit = len(set(rank_list[user].tolist()) & set(test_dict[user]))
            precision_list.append(hit / K)
            recall_list.append(hit / len(test_dict[user]))
            index = np.arange(len(rank_list[user].tolist()))
            k = min(len(rank_list[user].tolist()), len(test_dict[user]))
            idcg = (1 / np.log2(2 + np.arange(k))).sum()
            dcg = (1 / np.log2(2 + index[np.isin(rank_list[user].tolist(), test_dict[user])])).sum()
            ndcg_list.append(dcg/idcg)
    precision_avg = np.mean(precision_list)
    recall_avg = np.mean(recall_list)
    ndcg_avg = np.mean(ndcg_list)
    return [precision_avg, recall_avg, ndcg_avg]


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