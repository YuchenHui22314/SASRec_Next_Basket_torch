import torch
import numpy as np

embedding = torch.nn.Embedding(5 + 1, 2, padding_idx=5)
print("embedding.weight", embedding.weight)
a = [
    [[5,5,5],[3,2,3],[1,4,5]],
    [[5,5,5],[5,5,5],[1,4,5]]
]

a = torch.LongTensor(a) 
mask = torch.where(a[:, :, 0] == 5, False, True)

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