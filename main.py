import os
import sys
import time
import torch
import argparse
import pickle
from datetime import datetime
from data import *
from model import SASRec
from evaluate import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()

# setting 
parser.add_argument('--dataset', default="metro_01")
parser.add_argument('--train_dir', default = "lg")

parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--debug', default= False, type=str2bool, help='if true, use 10% dataset')

# common hyperparameters
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--lr', default = 1e-3, type=float)
parser.add_argument('--l2_emb', default=1e-4, type=float)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_epoch', default=200, type=int)
parser.add_argument("--random_seed", type=int, default=2023)
parser.add_argument("--N", type=int, default=1)
parser.add_argument("--K", type=int, default=20)
parser.add_argument("--lr_sched", action="store_true", default=False)
parser.add_argument("--sig_loss_average", action="store_true", default=False)
parser.add_argument("--sig_loss_average6", action="store_true", default=False)
parser.add_argument('--adam_beta1', default = 0.9, type=float)
parser.add_argument('--adam_beta2', default=0.999, type=float)





# model specific hyperparameters
#parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.1, type=float)
parser.add_argument("--loss", type=str, default="softmax")  # {sigmoid, softmax}


args = parser.parse_args()
# save model hyperparameters and settings --------------------------------------
if not os.path.isdir("/content/assignment/" + args.dataset + '_' + args.train_dir):
    os.makedirs("/content/assignment/" +args.dataset + '_' + args.train_dir)

with open(os.path.join("/content/assignment/" +args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()
#-------------------------------------
# one can also remove this if __name__ == '__main__': 
if __name__ == '__main__':

    # set the random seed manually for reproducibility.
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)    
        torch.cuda.manual_seed(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

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

    # set the padding index (model.num_item) to zero, so that the padding index will not affect the loss
    model.item_emb.weight.data[model.item_num].fill_(0)
    model.train() # enable model training
    betas = (args.adam_beta1, args.adam_beta2)
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas = betas)

    result_train = list()
    result_validate = list()
    result_test = list()

    for epoch in range(1, args.num_epoch + 1):
        if epoch >= 60 and epoch % 20 == 0 and args.lr_sched:
            # set the learning rate to the half of the current learning rate
            for param_group in adam_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5

        model.train() 
        t1 = time.time()
        train_loss = list()
        for batch in batches:
            # !!!!!!!!!!!!!! --------------------
            adam_optimizer.zero_grad()
            # !!!!!!!!!!!!!! --------------------

            # batch = [num_user, max_seq_len, max_basket_len]
            # how we adopt sasREC to NBR? we take the average of a basket as the basket embedding, which is counterpart to a "item" in SASREC
            ## TODO: concat?
            # labels are used to calculate the modified softmax loss
            input_seqs, labels, _ = get_inputs_train(num_item, batch)
            loss_type = args.loss.lower() 
            loss, logits = model(input_seqs, labels, args)
            # regularization
            for param in model.parameters():
                loss += args.l2_emb * torch.norm(param) 

            # !!!!!!!!!!!!!! --------------------
            loss.backward()
            adam_optimizer.step()
            # !!!!!!!!!!!!!--------------------
            train_loss.append(loss.item())

        mean_epoch_loss = np.mean(train_loss)
        result_train.append(mean_epoch_loss)
        print("epoch: %d, %.2fs" % (epoch, time.time() - t1))
        print("training loss: %.4f" % mean_epoch_loss)

        # validation and testing

        if epoch == 1 or epoch % args.N == 0:
            model.eval()
            # If num_user is greater than batch_size, it is taking the first batch_size
            # users. If num_user < batch_size, it's taking all the num_user, right?
            # Is it because evaluating too many users at once takes too long?
            # answer: Yes. It's also because the GPU memory can only hold one batch at a time.

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

    now = datetime.now()
    dt_string = now.strftime("_%Y_%d_%m_%H_%M_%S")
    with open('log/' + args.dataset + '_train' + dt_string +'.pkl', 'wb') as f:
        pickle.dump(result_train, f)
    with open('log/' + args.dataset + '_validate' + dt_string +'.pkl', 'wb') as f:
        pickle.dump(result_validate, f)
    with open('log/' + args.dataset + '_test' + dt_string +'.pkl', 'wb') as f:
        pickle.dump(result_test, f)

    # colab
    if os.path.exists("/content/assignment"):
        with open('/content/assignment/' + args.dataset + '_train' + dt_string +'.pkl', 'wb') as f:
            pickle.dump(result_train, f)
        with open('/content/assignment/' + args.dataset + '_validate' + dt_string +'.pkl', 'wb') as f:
            pickle.dump(result_validate, f)
        with open('/content/assignment/' + args.dataset + '_test' + dt_string +'.pkl', 'wb') as f:
            pickle.dump(result_test, f)