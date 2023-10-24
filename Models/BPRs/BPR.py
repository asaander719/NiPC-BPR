import torch
from torch import load, sigmoid, cat, rand, bmm, mean, matmul
from torch.nn.functional import logsigmoid
from torch.nn.init import uniform_
from torch.nn import *
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.nn.functional as F

class BPR(Module):
    def __init__(self, user_num, item_num, hidden_dim=512):
        super(BPR, self).__init__()
        self.hidden_dim = hidden_dim
       
        self.user_gama = Embedding(user_num, self.hidden_dim)
        self.item_gama = Embedding(item_num, self.hidden_dim)
        self.user_beta = Embedding(user_num, 1)
        self.item_beta = Embedding(item_num, 1)
        
        init.uniform_(self.user_gama.weight, 0, 0.01)
        init.uniform_(self.user_beta.weight, 0, 0.01)
        init.uniform_(self.item_gama.weight, 0, 0.01)
        init.uniform_(self.item_beta.weight, 0, 0.01)


    def forward(self, users, items): #for user_item latent
        batchsize = len(users)
        user_gama = self.user_gama(users)
        user_beta = self.user_beta(users)
        item_gama = self.item_gama(items)
        item_beta = self.item_beta(items)
        if conf["with_Nor"]:
            user_gama = F.normalize(user_gama, dim=0)
            user_beta = F.normalize(user_beta, dim=0)
            item_gama = F.normalize(item_gama, dim=0)
            item_beta = F.normalize(item_beta, dim=0)
        if conf["cos"]:
            pred = item_beta.view(batchsize) + user_beta.view(batchsize) + F.cosine_similarity(user_gama, item_gama, dim=-1)
        else:    
            pred = item_beta.view(batchsize) + user_beta.view(batchsize) + torch.sum(user_gama * item_gama, dim=-1)
        return pred

    def fit(self, tops, bottoms): #for top_bpttom latent
        batchsize = len(tops)      
        tops_gama = self.item_gama(tops)
        tops_beta = self.item_beta(tops)
        bottoms_gama = self.item_gama(bottoms)
        bottoms_beta = self.item_beta(bottoms)
        
        if conf["with_Nor"]:
            tops_gama = F.normalize(tops_gama, dim=0)
            tops_beta = F.normalize(tops_beta, dim=0)
            bottoms_gama = F.normalize(bottoms_gama, dim=0)
            bottoms_beta = F.normalize(bottoms_beta, dim=0)

        if conf["cos"]:
            pred =  tops_beta.view(batchsize) + bottoms_beta.view(batchsize) + F.cosine_similarity(tops_gama, bottoms_gama, dim=-1)
        else:
            pred =  tops_beta.view(batchsize) + bottoms_beta.view(batchsize) + torch.sum(tops_gama * bottoms_gama, dim=-1)
        return pred