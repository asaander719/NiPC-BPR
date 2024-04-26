import torch
from torch.nn.init import uniform_
from torch.nn import *
import torch.nn as nn
import torch.nn.functional as F

class BPR(Module):
    def __init__(self, user_num, item_num, hidden_dim=512, with_Nor=True, cos=True):
        super(BPR, self).__init__()
        self.hidden_dim = hidden_dim
       
        self.user_gama = nn.Embedding(user_num, self.hidden_dim)
        self.item_gama = nn.Embedding(item_num, self.hidden_dim)
        self.user_beta = nn.Embedding(user_num, 1)
        self.item_beta = nn.Embedding(item_num, 1)
        self.with_Nor = with_Nor
        self.cos = cos
        
        nn.init.uniform_(self.user_gama.weight, 0, 0.01)
        nn.init.uniform_(self.user_beta.weight, 0, 0.01)
        nn.init.uniform_(self.item_gama.weight, 0, 0.01)
        nn.init.uniform_(self.item_beta.weight, 0, 0.01)


    def forward(self, users, items): #for user_item latent
        batchsize = len(users)
        user_gama = self.user_gama(users)
        user_beta = self.user_beta(users)
        item_gama = self.item_gama(items)
        item_beta = self.item_beta(items)
        if self.with_Nor:
            user_gama = F.normalize(user_gama, dim=0)
            user_beta = F.normalize(user_beta, dim=0)
            item_gama = F.normalize(item_gama, dim=0)
            item_beta = F.normalize(item_beta, dim=0)
        if self.cos:
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
        
        if self.with_Nor:
            tops_gama = F.normalize(tops_gama, dim=0)
            tops_beta = F.normalize(tops_beta, dim=0)
            bottoms_gama = F.normalize(bottoms_gama, dim=0)
            bottoms_beta = F.normalize(bottoms_beta, dim=0)

        if self.cos:
            pred =  tops_beta.view(batchsize) + bottoms_beta.view(batchsize) + F.cosine_similarity(tops_gama, bottoms_gama, dim=-1)
        else:
            pred =  tops_beta.view(batchsize) + bottoms_beta.view(batchsize) + torch.sum(tops_gama * bottoms_gama, dim=-1)
        return pred
    
    def infer(self, bs, candi_num, Us, Js, Ks):
        user_gama = self.user_gama(Us) #bs, hd
        user_beta = self.user_beta(Us) #bs, 1
        item_gama_J = self.item_gama(Js)
        item_beta_J = self.item_beta(Js)
        item_gama_K = self.item_gama(Ks)
        item_beta_K = self.item_beta(Ks)
        if self.with_Nor:
            user_gama = F.normalize(user_gama, dim=0)
            user_beta = F.normalize(user_beta, dim=0)
            item_gama_J = F.normalize(item_gama_J, dim=0)
            item_beta_J = F.normalize(item_beta_J, dim=0)
            item_gama_K = F.normalize(item_gama_K, dim=0)
            item_beta_K = F.normalize(item_beta_K, dim=0)

        user_gama, item_gama = self.wide_infer(bs, candi_num, item_gama_J, item_gama_K, user_gama) # bs, candi_num, hd
        user_beta, item_beta = self.wide_infer(bs, candi_num, item_beta_J, item_beta_K, user_beta) # bs, candi_num, 1
        if self.cos:
            score = item_beta.view(bs, candi_num) + user_beta.view(bs, candi_num) + F.cosine_similarity(user_gama, item_gama, dim=-1) #bs, candi_num
        else:    
            score = item_beta.view(bs, candi_num) + user_beta.view(bs, candi_num) + torch.sum(user_gama * item_gama, dim=-1)
        return score
    
    def wide_infer(self, bs, candi_num, J, K, I): # replace I by U for personalization learning
        J = J.unsqueeze(1) #256,1,512
        K = K.unsqueeze(0).expand(bs, -1, -1) #1,256,512->256,256,512
        Jks = torch.cat([J, K], dim=1) #256,257,512 # dim=1 里面第一个为postive target(1+256)
        I= I.unsqueeze(1).expand(-1, candi_num, -1) # 256,257,512
        return I, Jks