import os
import sys
import time
import torch
import argparse
import pickle
from datetime import datetime
from data import *
from model import SASRec
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()

# setting 
parser.add_argument('--dataset', default="metro_02")
parser.add_argument('--train_dir', default = "lg")

parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)

# common hyperparameters
parser.add_argument('--hidden_units', default=5, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_epoch', default=2, type=int)
parser.add_argument("--random_seed", type=int, default=2023)


# model specific hyperparameters
#parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument("--loss", type=str, default="softmax")  # {sigmoid, softmax}

args = parser.parse_args()
# save model hyperparameters and settings --------------------------------------
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)

with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()
#-------------------------------------

if __name__ == '__main__':

    batches, num_user, num_item, train_dict, validate_dict, test_dict, sequences= load_dataset_batches(args)  

    # train
    model = SASRec(num_user, num_item, args).to(args.device) # no ReLU activation in original SASRec implementation?

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    model.train() # enable model training

    result_train = list()
    result_validate = list()
    result_test = list()

    for epoch in range(1, args.num_epoch + 1):
        
        t1 = time.time()
        train_loss = list()
        for batch in batches:
            # batch = [num_user, max_seq_len, max_basket_len]
            # how we adopt sasREC to NBR? we take the average of a basket as the basket embedding, which is counterpart to a "item" in SASREC
            ## TODO: concat?
            # labels are used to calculate the modified softmax loss
            input_seqs, labels, _ = get_inputs_train(num_item, batch)
            loss_type = args.loss.lower() 
            loss, logits = model(input_seqs, labels, loss_type)
            train_loss.append(loss.item())

        mean_epoch_loss = np.mean(train_loss)
        result_train.append(mean_epoch_loss)
        print("epoch: %d, %.2fs" % (epoch, time.time() - t1))
        print("training loss: %.4f" % mean_epoch_loss)

        # validation and testing

        if epoch == 1 or epoch % args.N == 0:
            # for validation
            #如果num_user大于batch_size就相当于取前batch size
            #个user，如果num_user < batch_size, 就是取所有的
            # num_user,是这样吗？是出于num_user太大的情况下一次
            # evaluation的时间太长是不？
            # answer: 是的。也是出于显存只能放一个batch 

            batch_size_test = args.batch_size
            rank_list = list()

            for start in range(0, num_user, batch_size_test):
                input_seqs = get_feed_dict_validate(sequences[start:start+batch_size_test]) 
                valid_logits = model.predict(input_seqs) 
                rank_list += get_top_K_index(valid_logits, args.K)
            precision_k_validate, recall_k_validate, ndcg_k_validate = evaluate(rank_list, validate_dict, args.K)
            print('validate precision@{K}: %.4f, recall@{K}: %.4f, ndcg@{K}: %.4f'.format(K=args.K) % (precision_k_validate, recall_k_validate, ndcg_k_validate))
            result_validate.append([epoch] + evaluate(rank_list, validate_dict, 10) + evaluate(rank_list, validate_dict, 20))

            #print("Model testing...")
            rank_list = list()
            for start in range(0, num_user, batch_size_test):
                input_seqs = get_feed_dict_test(sequences[start:start+batch_size_test]) 
                test_logits = model.predict(input_seqs) 
                rank_list += get_top_K_index(test_logits, args.K)
            precision_k_test, recall_k_test, ndcg_k_test = evaluate(rank_list, test_dict, args.K)
            print('test precision@{K}: %.4f, recall@{K}: %.4f, ndcg@{K}: %.4f'.format(K=args.K) % (precision_k_test, recall_k_test, ndcg_k_test))
            result_test.append([epoch] + evaluate(rank_list, test_dict, 10) + evaluate(rank_list, test_dict, 20))

    save_result(args, result_validate, result_test)

    # create a folder named log if not exist
    if not os.path.exists('log'):
        os.makedirs('log')
    # save the training , validation and testing results as pickle file

    # result_train = list()
    # result_validate = list()
    # result_test = list()

    # get day and time
    now = datetime.now()
    dt_string = now.strftime("_%Y_%d_%m_%H_%M_%S")
    with open('log/' + args.dataset + '_train' + dt_string +'.pkl', 'wb') as f:
        pickle.dump(result_train, f)
    with open('log/' + args.dataset + '_validate' + dt_string +'.pkl', 'wb') as f:
        pickle.dump(result_validate, f)
    with open('log/' + args.dataset + '_test' + dt_string +'.pkl', 'wb') as f:
        pickle.dump(result_test, f)