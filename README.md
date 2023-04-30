# SASRec for next basket recommendation
This is a pytorch implementation of SASRec for next basket recommendation, as a part of my computer science project at the Université de Montréal.

It is modified based on [paper author's tensorflow implementation](https://github.com/kang205/SASRec) and another [pytorch implementation for next item recommendation](https://github.com/pmixer/SASRec.pytorch). The code is well commented and tested on a internal dataset to validate the correctness, using Pytorch 2.0.0. [Here](https://drive.google.com/file/d/1xNkVtjHt4Oha7gUqAIu_uuPqEGVrlpG6/view?usp=sharing) is an exemple Colab notebook for execution.


In next basket recommendation regime, we adapt the model by regarding a basket as an item in the original SASRec model. That is, each basket produces an embedding, and then this embedding is used as the Transformer input, and the transformer output at the same position is used to predict the next basket. A little bit like what GPT does.

We get the basket embedding by taking the average of the item embeddings in the basket. Item embeddings are the same as in the original SASRec model, i.e. a torch.nn.Embedding layer.

Please check paper author's [repo](https://github.com/kang205/SASRec) and for detailed intro and more complete README, and here's paper bib FYI :)

```
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```
