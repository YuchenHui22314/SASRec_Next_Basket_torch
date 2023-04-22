import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1) # equivallent to nn.linear layer
        self.dropout1 = torch.nn.Dropout(p=dropout_rate) # dropout的位置已经成谜
        self.relu = torch.nn.ReLU() #不是gelu吗？
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length) Note: len = 1, C = hidden_units
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=item_num)  # attention! padding_idx is item_num, not 0 in metro project
        self.pos_emb = torch.nn.Embedding(args.max_seq_len, args.hidden_units) # TO IMPROVE how?
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(
                args.hidden_units,
                args.num_heads,
                args.dropout_rate)

            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def seq2embed(self, input_seqs):
        '''
        log_seqs: (U, T) where U is user_num, T is maxlen. so this is purchase history of users
        (item Recommendation)
        log_seqs: (user_num, Basket_num, item_num) (next basket recommendation) 
        '''
        # generate mask for log_seqs: cretiria: 1.size is (user_num, basket_num) 2.2. mask if the first item index is item_num. (this means the basket is a padded one)
        input_seqs = torch.LongTensor(input_seqs).to(self.dev)
        # timeline_mask的complement是要把padding的item/basket 全都置零。而其本身会喂给attention的key_padding_mask
        #timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        timeline_mask_bool = torch.where(input_seqs[:, :, 0] == self.item_num, True, False).to(self.dev)  
        seqs = self.item_emb(input_seqs)

        # take average of item embeddings as basket embedding
        seqs = torch.sum(seqs, dim = -2) / input_seqs.shape[2] 

        seqs *= self.item_emb.embedding_dim ** 0.5  # necessity?   introduced by transformer paper, but not understood yet 

        # positions is for positional enbedding index.
        positions = np.tile(np.array(range(input_seqs.shape[1])), [input_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))

        seqs = self.emb_dropout(seqs)
        seqs *= ~timeline_mask_bool.unsqueeze(-1) 
        # broadcast in last dim 有必要？(有。因为变成embedding了。)

        number_baskets = seqs.shape[1] 
        # causal mask
        attention_mask = ~torch.tril(torch.ones((number_baskets, number_baskets), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, 
                attn_mask=attention_mask,
                key_padding_mask=timeline_mask_bool
                )
                # key_padding_mask=timeline_mask
                # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

            # mask necessary? (yes...linear includes bias, lol, why am i so stupid)
            # but still, if we offer attn_mask, this line seems to be redundant.
            seqs *=  ~timeline_mask_bool.unsqueeze(-1)

        output_embedding = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return output_embedding, ~timeline_mask_bool 


    def forward(self, input_seqs, labels, loss_type):
        '''
        labels = [
        [1, 0,1,0], 
        [0, 1,1,1]
        ], dtype=torch.float32
         
        loss_type: "sigmoid" or "softmax"
        '''
        #output: [batch_size, seq_len, emb_dim] 
        #loss mask: True if not padding, False if padding
        ''' example:
            mask = torch.tensor(
                [ [False, True, True, True], [True, False, False, True] ], dtype=torch.bool)
        '''
        output, loss_mask = self.seq2embed(input_seqs)
        logits = torch.matmul(output, self.item_emb.weight.transpose(0, 1))
        loss_mask_logits = loss_mask.unsqueeze(-1).repeat(1, 1, logits.shape[-1])

        if loss_type == "sigmoid":
            criterion = torch.nn.BCEWithLogitsLoss(reduce=False)
            loss= criterion(logits, labels)
            loss = torch.sum(loss * loss_mask_logits)/ torch.sum(loss_mask_logits)
            return loss, logits
        
        elif loss_type == "softmax":
            criterion = torch.nn.CrossEntropyLoss(reduce=False)
            assert len(logits.shape) == 3, "logits should be 3D"
            labels = torch.tensor(labels,dtype=torch.float32).to(self.dev)
            # label should be floating point, not long
            loss= criterion(logits.transpose(1, 2), labels.transpose(1, 2))
            loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask)
            return loss, logits



    def predict(self, input_seqs): # for inference
        self.eval()
        output_embedding , _ = self.seq2embed(input_seqs)
        logits = torch.matmul(output_embedding, self.item_emb.weight.transpose(0, 1))
        # take the last embedding as the prediction
        logits = logits[:, -1, :] # (U, num_items)

        return logits # preds # (U, num_items)
