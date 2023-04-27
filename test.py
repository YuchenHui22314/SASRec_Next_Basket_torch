import torch
import numpy as np
from datetime import datetime

embedding = torch.nn.Embedding(5 + 1, 2, padding_idx=5)
print("embedding.weight", embedding.weight)
# now we have a tensor of shape(batch_size, seq_len, basket_size) a
# where each element is the item id, ranging from 0 to num_class (inclusive), here we have 0 ,1 ,2, 3, 4, 5 as possible item id 
a = [
    [[5,5,5],[3,2,3],[1,4,5]],
    [[5,5,5],[5,5,5],[1,4,5]]
]
# we want to get a tensor of shape(batch_size, seq_len, class_num)
# where class_num is 6 here. so for each basket, we want to get a multi-hot vector of length 6, where the element at position indicated by the last dimension of a is 1, and the rest are 0. However we will ignore the padding item 5, so the element at position 5 will be 0. for example, for a[0][0], we want to get [0,0,0,0,0,0], for a[0][1], we want to get [0,0,1,1,0,0], for a[0][2], we want to get [0,1,0,0,1,0]

# please generate code for this.

a = torch.LongTensor(a) 
mask = torch.where(a[:, :, 0] == 5, False, True).to("cpu")
print("the mask is", mask)

a_emb = embedding(a)
print(a_emb.shape)
print(a_emb)
# average
a_emb = torch.sum(a_emb, dim= -2) / a_emb.shape[2] 
print("average:\n",a_emb)

tl = a_emb.shape[1] # time dim len for enforce causality
attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool))
print(mask)
print(attention_mask)
print(a_emb.shape)


num_classes = 3
num_seq = 3
batch_size = 2 

# Randomly generate some data
# set random seed to 0
torch.manual_seed(0)
input_tensor = torch.randn(batch_size, num_seq, num_classes)
target_tensor = torch.tensor(
    [
        [[1, 0,1], [0, 1,0], [1,1,1]],
        [[1, 0,1], [0, 1,0], [1,1,1]],
        ], dtype=torch.float32
)
mask = torch.tensor(
    [ [False, True, True], [True, False, False] ], dtype=torch.bool)
print("the sum of mask is ", torch.sum(mask))

# Define loss function
#criterion = torch.nn.BCEWithLogitsLoss(pos_weight= torch.ones(num_classes))
criterion = torch.nn.BCEWithLogitsLoss()

# Calculate loss
loss1 = criterion(input_tensor, target_tensor)
print("loss1 unreduced", loss1)
# 2eme loss
logits_sigmoid = torch.nn.Sigmoid()(input_tensor) 
loss2 = -torch.sum(target_tensor * torch.log(logits_sigmoid) + (1 - target_tensor) * torch.log(1 - logits_sigmoid)) / (batch_size* num_classes * num_seq) 

# torch multilabel crossEntropyLoss
# cross Entropy Loss 是默认每个target里面只有一个维度可以是1的。但是计算的方法是一样的，都是 target element wise 去乘 tensor。 所以最后只要对齐一下。
# indices = torch.tensor(
#     [[0, 1], [0, 1]], dtype= torch.long 
# ) 
# input_tensor = input_tensor[indices]
#print(input_tensor.shape)
# input tensor 的维度应该是 batch_size, num_classes,num_seq, 666
criteria = torch.nn.CrossEntropyLoss()
loss3 = criteria(input_tensor.transpose(1,2), target_tensor.transpose(1,2))*batch_size*num_seq/ torch.sum(target_tensor) 

# manual multilable crossEntropyLoss
logits_log_softmax = torch.nn.LogSoftmax(dim=-1)(input_tensor)
loss4 = - torch.sum(target_tensor * logits_log_softmax) / torch.sum(target_tensor) 

# timelin mask
criteria = torch.nn.CrossEntropyLoss(reduce = False)
criterion = torch.nn.BCEWithLogitsLoss(reduce = False)
print(loss1)
print(loss2)
print(loss3)
print(loss4)

# timelin mask sigmoid
criterion = torch.nn.BCEWithLogitsLoss(reduce = False)
loss5 = criterion(input_tensor, target_tensor)
print("the shape of loss5 is", loss5.shape)
mask_sigmoid = mask.unsqueeze(-1).repeat(1,1,loss5.shape[-1])
loss5 = torch.sum(loss5 * mask_sigmoid)/ torch.sum(mask_sigmoid)

logits_sigmoid = torch.nn.Sigmoid()(input_tensor) 
loss6 = -torch.sum((target_tensor * torch.log(logits_sigmoid) + (1 - target_tensor) * torch.log(1 - logits_sigmoid)) * mask_sigmoid) / torch.sum(mask_sigmoid)

print("loss_sigmoid_mask_torch", loss5)
print("loss_sigmoid_mask_manual", loss6)

# timelin mask softmax
criteria = torch.nn.CrossEntropyLoss(reduce = False)
loss7 = criteria(input_tensor.transpose(1,2), target_tensor.transpose(1,2))
loss7 = torch.sum(loss7 * mask)/ torch.sum(mask)

logits_log_softmax = torch.nn.LogSoftmax(dim=-1)(input_tensor)
loss8 = - torch.sum((target_tensor * logits_log_softmax) * mask_sigmoid) / torch.sum(mask)

print("loss_softmax_mask_torch", loss7)
print("loss_softmax_mask_manual", loss8)



######## test matrix multiplication
m1 = torch.nn.Embedding(3, 2)
print("the shape of m1.weight", m1.weight.shape)
m2 = torch.randn(2, 2)

product = torch.matmul(m1.weight, m2)
print(product)

now = datetime.now()
dt_string = now.strftime("%Y_%d_%m_%H_%M_%S")
print("date and time =", dt_string)

###################3 test embedding ###############
print("shuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
emb = torch.nn.Embedding(3, 2)
print(emb.weight)
emb.weight.data[0].fill_(0)
print(emb.weight)

softmax = torch.nn.Softmax(dim=-1)
a = softmax(torch.tensor([-torch.inf,  -2e31]))
print(a)
attention_mask = torch.tril(torch.zeros((3, 3) ).fill_(-2e15), diagonal=0)# (T, T)
print("attention_mask\n", attention_mask)


previous = torch.tensor([[[1,2,3], [4,5,6], [7,8,9]], [[10,11,12], [13,14,15], [16,17,18]]])
print(previous)
medium = previous.reshape(-1, 3)
print(medium)
after = medium.reshape(2, -1, 3)
print(after)

# what if previous medium after are all numpy array?

previous = np.array([[[1,2,3], [4,5,6], [7,8,9]], [[10,11,12], [13,14,15], [16,17,18]]])
print(previous)
medium = previous.reshape(-1, 3)
print(medium)
after = medium.reshape(2, -1, 3)
print(after)
print(-2e15)