import torch
from torch import load, sigmoid, cat, rand, bmm, mean, matmul
from torch.nn.functional import logsigmoid
from torch.nn.init import uniform_
from torch.nn import *
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
from utils.utils import get_parser
from Models.BPRs.BPR import BPR
from Models.BPRs.VTBPR import VTBPR
from Models.BPRs.TextCNN import TextCNN

class GPBPR(Module):
    def __init__(self, arg, embedding_weight, visual_features, text_features):        
        super(GPBPR, self) .__init__()
        self.arg = arg
        self.weight_P = arg.weight_P
        self.hidden_dim = arg.hidden_dim
        self.user_num = arg.user_num
        self.item_num = arg.item_num
        self.with_visual = arg.with_visual
        self.with_text = arg.with_text
        self.with_Nor = arg.with_Nor
        self.cos = arg.cos
        self.UC = arg.UC
        self.IC = arg.IC
        #for compatibility space
        self.visual_nn = Sequential(
            Linear(arg.visual_feature_dim, self.hidden_dim),
            nn.Sigmoid())
        self.visual_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.visual_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

        self.text_nn = Sequential(
            Linear(100 * arg.textcnn_layer, self.hidden_dim),
            nn.Sigmoid()) 
        self.text_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.text_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

        #for personalization space
        self.p_visual_nn = Sequential(
            Linear(arg.visual_feature_dim, self.hidden_dim),
            nn.Sigmoid())
        self.p_visual_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.p_visual_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

        self.p_text_nn = Sequential(
            Linear(100 * arg.textcnn_layer, self.hidden_dim),
            nn.Sigmoid())
        self.p_text_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.p_text_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

        #for UC space
        self.s_visual_nn = Sequential(
            Linear(arg.visual_feature_dim, self.hidden_dim),
            nn.Sigmoid())
        self.s_visual_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.s_visual_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

        self.s_text_nn = Sequential(
            Linear(100 * arg.textcnn_layer, self.hidden_dim),
            nn.Sigmoid())
        self.s_text_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.s_text_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

         #for IC space
        self.s3_visual_nn = Sequential(
            Linear(arg.visual_feature_dim, self.hidden_dim),
            nn.Sigmoid())
        self.s3_visual_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.s3_visual_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

        self.s3_text_nn = Sequential(
            Linear(100 * arg.textcnn_layer, self.hidden_dim),
            nn.Sigmoid())
        self.s3_text_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.s3_text_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

        self.sigmoid = nn.Sigmoid()
        
        if self.with_visual:
            self.visual_features = visual_features.to(arg.device)
        if self.with_text:
            self.max_sentense_length = arg.max_sentence
            self.text_features = text_features.to(arg.device)
            self.text_embedding = Embedding.from_pretrained(embedding_weight, freeze=False)
            self.textcnn = TextCNN(arg.textcnn_layer, sentence_size=(arg.max_sentence, arg.text_feature_dim), output_size=self.hidden_dim)

        self.vtbpr = VTBPR(self.user_num, self.item_num, hidden_dim=self.hidden_dim, 
            theta_text=self.with_text, theta_visual=self.with_visual, with_Nor=True, cos=True)
        print('Module already prepared, {} users, {} items'.format(self.user_num, self.item_num))
        self.bpr = BPR(self.user_num, self.item_num)
         
    def forward(self, batch, train, **args):
        Us = batch[0] #bs
        Is = batch[1]
        Js = batch[2]
        Ks = batch[3]
        bhis = batch[4]
        this = batch[5]
        tbhis = batch[6]
        bs = len(Us)

        if self.with_visual:
            vis_I = self.visual_features[Is] #bs,visual_feature_dim = 2048 = torch.Size([256, 2048])
            vis_J = self.visual_features[Js]
            vis_K = self.visual_features[Ks]
  
            I_visual_latent = self.visual_nn(vis_I) #bs, hidden_dim =torch.Size([256, 512])
            J_visual_latent = self.visual_nn(vis_J)
            K_visual_latent = self.visual_nn(vis_K)

            J_visual_latent_p = self.p_visual_nn(vis_J)
            K_visual_latent_p = self.p_visual_nn(vis_K)

            if self.with_Nor:
                I_visual_latent = F.normalize(I_visual_latent,dim=0)
                J_visual_latent = F.normalize(J_visual_latent,dim=0)
                K_visual_latent = F.normalize(K_visual_latent,dim=0)   

                J_visual_latent_p = F.normalize(J_visual_latent_p,dim=0)
                K_visual_latent_p = F.normalize(K_visual_latent_p,dim=0)

            if self.cos:
                visual_ij = F.cosine_similarity(I_visual_latent, J_visual_latent, dim=-1)
                visual_ik = F.cosine_similarity(I_visual_latent, K_visual_latent, dim=-1)
            else:
                visual_ij = torch.sum(I_visual_latent * J_visual_latent, dim=-1)
                visual_ik = torch.sum(I_visual_latent * K_visual_latent, dim=-1)
            #add similarity 
            if self.UC:    
                vis_bhis = self.visual_features[bhis]#bs,3,visual_feature_dim = 2048 torch.Size([256, 3, 2048])
                vis_bhis = self.s_visual_nn(vis_bhis) #bs,3,512
                # print(vis_bhis.size()) torch.Size([64, 3, 512])
                vis_J_p = self.s_visual_nn(vis_J)
                vis_K_p = self.s_visual_nn(vis_K)
                b_his_visual = torch.mean(vis_bhis, dim=-2)  #bs, visual_feature_dim = 2048 #torch.Size([256, 512])

                if self.with_Nor:
                    b_his_visual = F.normalize(b_his_visual,dim=0)
                    vis_J_p = F.normalize(vis_J_p,dim=0)
                    vis_K_p = F.normalize(vis_K_p,dim=0)
                if self.cos:
                    Visual_BuJ = F.cosine_similarity(b_his_visual, vis_J_p, dim=-1)
                    Visual_BuK = F.cosine_similarity(b_his_visual, vis_K_p, dim=-1)
                else:
                    Visual_BuJ = torch.sum(b_his_visual * vis_J_p, dim=-1)
                    Visual_BuK = torch.sum(b_his_visual * vis_K_p, dim=-1)

            if self.IC:    
                vis_this = self.visual_features[this]#bs,3,visual_feature_dim = 2048 torch.Size([256, 3, 2048])
                vis_this = self.s3_visual_nn(vis_this) #bs,3,512
                vis_J_c= self.s3_visual_nn(vis_J)
                vis_K_c = self.s3_visual_nn(vis_K)
                t_his_visual = torch.mean(vis_this, dim=-2)  #bs, visual_feature_dim = 2048 #torch.Size([256, 512])

                if self.with_Nor:
                    t_his_visual = F.normalize(t_his_visual,dim=0)
                    vis_J_c = F.normalize(vis_J_c,dim=0)
                    vis_K_c = F.normalize(vis_K_c,dim=0)

                if self.cos:
                    Visual_TuJ = F.cosine_similarity(t_his_visual, vis_J_c, dim=-1)
                    Visual_TuK = F.cosine_similarity(t_his_visual, vis_K_c, dim=-1)
                else:
                    Visual_TuJ = torch.sum(t_his_visual * vis_J_c, dim=-1)
                    Visual_TuK = torch.sum(t_his_visual * vis_K_c, dim=-1)

        if self.with_text:
            text_I = self.text_embedding(self.text_features[Is]) #256,83,300
            text_J = self.text_embedding(self.text_features[Js])
            text_K = self.text_embedding(self.text_features[Ks])

            I_text_fea = self.textcnn(text_I.unsqueeze(1))  #256,400
            J_text_fea = self.textcnn(text_J.unsqueeze(1))
            K_text_fea = self.textcnn(text_K.unsqueeze(1))

            I_text_latent = self.text_nn(I_text_fea) #256,512
            J_text_latent = self.text_nn(J_text_fea)
            K_text_latent = self.text_nn(K_text_fea)

            J_text_latent_p = self.p_text_nn(J_text_fea)
            K_text_latent_p = self.p_text_nn(K_text_fea)

            if self.with_Nor:
                I_text_latent = F.normalize(I_text_latent,dim=0)
                J_text_latent = F.normalize(J_text_latent,dim=0)
                K_text_latent = F.normalize(K_text_latent,dim=0)

                J_text_latent_p = F.normalize(J_text_latent_p,dim=0)
                K_text_latent_p = F.normalize(K_text_latent_p,dim=0)

            if self.cos:
                text_ij = F.cosine_similarity(I_text_latent, J_text_latent, dim=-1)
                text_ik = F.cosine_similarity(I_text_latent, K_text_latent, dim=-1)
            else:
                text_ij = torch.sum(I_text_latent * J_text_latent, dim=-1)
                text_ik = torch.sum(I_text_latent * K_text_latent, dim=-1)
            #add similarity 
            if self.UC:
                text_bhis = self.text_embedding(self.text_features[bhis]) #torch.Size([64, 3, 83, 300])

                bhis_text_fea = self.textcnn(text_bhis.reshape(bs * arg.num_his, arg.max_sentence, arg.text_feature_dim).unsqueeze(1))  #bs, 400(100*layers)
                bhis_text_fea = self.s_text_nn(bhis_text_fea) #torch.Size([192, 512])
                bhis_text_fea = bhis_text_fea.reshape(bs, arg.num_his, self.hidden_dim) #64, 3, 512
                bhis_text_fea_mean = torch.mean(bhis_text_fea, dim=-2) #torch.Size([bs, 512])

                text_J_p = self.s_text_nn(J_text_fea)
                text_K_p = self.s_text_nn(K_text_fea)
                if self.with_Nor:
                    bhis_text_fea_mean = F.normalize(bhis_text_fea_mean,dim=0)
                    text_J_p = F.normalize(text_J_p,dim=0)
                    text_K_p = F.normalize(text_K_p,dim=0)
                if self.cos:
                    text_BuJ = F.cosine_similarity(bhis_text_fea_mean, text_J_p, dim=-1)
                    text_BuK = F.cosine_similarity(bhis_text_fea_mean, text_K_p, dim=-1)
                else:
                    text_BuJ = torch.sum(bhis_text_fea_mean * text_J_p, dim=-1)
                    text_BuK = torch.sum(bhis_text_fea_mean * text_K_p, dim=-1)

            if self.IC:
                text_this = self.text_embedding(self.text_features[this]) #torch.Size([64, 3, 83, 300])

                this_text_fea = self.textcnn(text_this.reshape(bs * arg.num_his, arg.max_sentence, arg.text_feature_dim).unsqueeze(1))  #bs, 400(100*layers)
                this_text_fea = self.s3_text_nn(this_text_fea) #torch.Size([192, 512])
                this_text_fea = this_text_fea.reshape(bs, arg.num_his, self.hidden_dim) #64, 3, 512
                this_text_fea_mean = torch.mean(this_text_fea, dim=-2) #torch.Size([bs, 512])

                text_J_c = self.s3_text_nn(J_text_fea)
                text_K_c = self.s3_text_nn(K_text_fea)
                if self.with_Nor:
                    this_text_fea_mean = F.normalize(this_text_fea_mean,dim=0)
                    text_J_c = F.normalize(text_J_c,dim=0)
                    text_K_c = F.normalize(text_K_c,dim=0)
                if self.cos:
                    text_TuJ = F.cosine_similarity(this_text_fea_mean, text_J_c, dim=-1)
                    text_TuK = F.cosine_similarity(this_text_fea_mean, text_K_c, dim=-1)
                else:
                    text_TuJ = torch.sum(this_text_fea_mean * text_J_c, dim=-1)
                    text_TuK = torch.sum(this_text_fea_mean * text_K_c, dim=-1) 

        if self.with_visual and self.with_text:
            if self.arg.b_PC:
                cuj = self.vtbpr(Us, Js, J_visual_latent_p, J_text_latent_p) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, K_visual_latent_p, K_text_latent_p)
            else:
                cuj = self.vtbpr(Us, Js, J_visual_latent, J_text_latent) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, K_visual_latent, K_text_latent)

            p_ij = 0.5 * (visual_ij + text_ij)
            p_ik = 0.5 * (visual_ik + text_ik)

            pred = self.weight_P * p_ij + (1 - self.weight_P) * cuj - (self.weight_P * p_ik + (1 - self.weight_P) * cuk)

            if self.UC:
                C_BuJ = self.arg.UC_w  * (self.arg.UC_v_w  * Visual_BuJ + (1-self.arg.UC_v_w) * text_BuJ)
                C_BuK = self.arg.UC_w  * (self.arg.UC_v_w  * Visual_BuK + (1-self.arg.UC_v_w) * text_BuK)

                pred = pred + C_BuJ - C_BuK

            if self.IC:
                C_TuJ = self.arg.IC_w * (self.arg.IC_v_w * Visual_TuJ + (1-self.arg.IC_v_w) * text_TuJ)
                C_TuK = self.arg.IC_w * (self.arg.IC_v_w * Visual_TuK + (1-self.arg.IC_v_w) * text_TuK)

                pred = pred + C_TuJ - C_TuK   

        if self.with_visual and not self.with_text:
            if self.arg.b_PC:
                cuj = self.vtbpr(Us, Js, J_visual_latent_p, None) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, K_visual_latent_p, None)
            else:
                cuj = self.vtbpr(Us, Js, J_visual_latent, None) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, K_visual_latent, None)

            p_ij = visual_ij 
            p_ik = visual_ik 

            pred = self.weight_P * p_ij + (1 - self.weight_P) * cuj - (self.weight_P * p_ik + (1 - self.weight_P) * cuk)

            if self.UC:
                C_BuJ = self.arg.UC_w * Visual_BuJ 
                C_BuK = self.arg.UC_w * Visual_BuK 

                pred = pred + C_BuJ - C_BuK

            if self.IC:
                C_TuJ = self.arg.IC_w * Visual_TuJ 
                C_TuK = self.arg.IC_w * Visual_TuK 

                pred = pred + C_TuJ - C_TuK   
        
        if not self.with_visual and self.with_text:
            if self.arg.b_PC:
                cuj = self.vtbpr(Us, Js, None, J_text_latent_p) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, None, K_text_latent_p)
            else:
                cuj = self.vtbpr(Us, Js, None, J_text_latent) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, None, K_text_latent)

            p_ij = 0.5 * text_ij
            p_ik = 0.5 * text_ik

            pred = self.weight_P * p_ij + (1 - self.weight_P) * cuj - (self.weight_P * p_ik + (1 - self.weight_P) * cuk)

            if self.UC:
                C_BuJ = self.arg.UC_w * text_BuJ
                C_BuK = self.arg.UC_w * text_BuK

                pred = pred + C_BuJ - C_BuK

            if self.IC:
                C_TuJ = self.arg.IC_w * text_TuJ
                C_TuK = self.arg.IC_w * text_TuK

                pred = pred + C_TuJ - C_TuK   
   
        return pred        