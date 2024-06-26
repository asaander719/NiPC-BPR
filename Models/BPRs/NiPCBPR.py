import torch
from torch import load, sigmoid, cat, rand, bmm, mean, matmul
from torch.nn.functional import logsigmoid
from torch.nn.init import uniform_
from torch.nn import *
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
from util.utils import get_parser
from Models.BPRs.BPR import BPR
from Models.BPRs.VTBPR import VTBPR
from Models.BPRs.TextCNN import TextCNN

class NiPCBPR(Module):
    def __init__(self, args, embedding_weight, visual_features, text_features):        
        super(NiPCBPR, self) .__init__()
        self.args = args
        self.weight_P = args.weight_P
        self.hidden_dim = args.hidden_dim
        self.user_num = args.user_num
        self.item_num = args.item_num
        self.with_visual = args.with_visual
        self.with_text = args.with_text
        self.with_Nor = args.with_Nor
        self.cos = args.cos
        self.iPC = args.iPC
        #for compatibility space
        self.visual_nn = Sequential(
            Linear(args.visual_feature_dim, self.hidden_dim),
            nn.Sigmoid())
        self.visual_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.visual_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

        #for personalization space
        self.p_visual_nn = Sequential(
            Linear(args.visual_feature_dim, self.hidden_dim),
            nn.Sigmoid())
        self.p_visual_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.p_visual_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

        #for iPC space
        self.iPC_visual_nn = Sequential(
            Linear(args.visual_feature_dim, self.hidden_dim),
            nn.Sigmoid())
        self.iPC_visual_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.iPC_visual_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

        if self.args.dataset == 'IQON3000':
            self.text_nn = Sequential(
                Linear(100 * args.textcnn_layer, self.hidden_dim),
                nn.Sigmoid()) 

            self.p_text_nn = Sequential(
                Linear(100 * args.textcnn_layer, self.hidden_dim),
                nn.Sigmoid())

            self.iPC_text_nn = Sequential(
                Linear(100 * args.textcnn_layer, self.hidden_dim),
                nn.Sigmoid())

        elif self.args.dataset == 'Polyvore':
            self.text_nn = Sequential(
                Linear(args.text_feature_dim, self.hidden_dim),
                nn.Sigmoid()) 

            self.p_text_nn = Sequential(
                Linear(args.text_feature_dim, self.hidden_dim),
                nn.Sigmoid())

            self.iPC_text_nn = Sequential(
                Linear(args.text_feature_dim, self.hidden_dim),
                nn.Sigmoid())

        self.text_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.text_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))       
        self.p_text_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.p_text_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))     
        self.iPC_text_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.iPC_text_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))
        
        if self.with_visual:
            self.visual_features = visual_features.cuda()
        if self.with_text:
            # self.max_sentense_length = args.max_sentence
            self.text_features = text_features.cuda()
            self.text_embedding = Embedding.from_pretrained(embedding_weight, freeze=False)
            self.textcnn = TextCNN(args.textcnn_layer, sentence_size=(args.max_sentence, args.text_feature_dim), output_size=self.hidden_dim)

        self.vtbpr = VTBPR(self.user_num, self.item_num, hidden_dim=self.hidden_dim, 
            theta_text=self.with_text, theta_visual=self.with_visual, with_Nor=True, cos=True)
        print('Module already prepared, {} users, {} items'.format(self.user_num, self.item_num))
        self.bpr = BPR(self.user_num, self.item_num)
         
    def forward(self, batch, train, **args): #**args,
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

            if self.iPC:    
                vis_tbhis = self.visual_features[tbhis]#bs,3,visual_feature_dim = 2048 torch.Size([256, 3, 2048])
                vis_tbhis = self.iPC_visual_nn(vis_tbhis) #bs,3,512
                # print(vis_bhis.size()) torch.Size([64, 3, 512])
                vis_J_c= self.iPC_visual_nn(vis_J)
                vis_K_c = self.iPC_visual_nn(vis_K)
                t_his_visual = torch.mean(vis_tbhis, dim=-2)  #bs, visual_feature_dim = 2048 #torch.Size([256, 512])

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
            if self.args.dataset == 'IQON3000':
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

            elif self.args.dataset == 'Polyvore':
                text_I = self.text_features[Is] #256,83,300
                text_J = self.text_features[Js]
                text_K = self.text_features[Ks]

                I_text_latent = self.text_nn(text_I) #256,512
                J_text_latent = self.text_nn(text_J)
                K_text_latent = self.text_nn(text_K)

                J_text_latent_p = self.p_text_nn(text_J)
                K_text_latent_p = self.p_text_nn(text_K)
            
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
                
            if self.iPC:
                if self.args.dataset == 'IQON3000':
                    text_tbhis = self.text_embedding(self.text_features[tbhis]) #torch.Size([64, 3, 83, 300])
                    tbhis_text_fea = self.textcnn(text_tbhis.reshape(bs * self.args.num_his, self.args.max_sentence, self.args.text_feature_dim).unsqueeze(1))  #bs, 400(100*layers)
                    tbhis_text_fea = self.iPC_text_nn(tbhis_text_fea) #torch.Size([192, 512])
                    tbhis_text_fea = tbhis_text_fea.reshape(bs, self.args.num_his, self.hidden_dim) #64, 3, 512
                    tbhis_text_fea_mean = torch.mean(tbhis_text_fea, dim=-2) #torch.Size([bs, 512])

                    text_J_c = self.iPC_text_nn(J_text_fea)
                    text_K_c = self.iPC_text_nn(K_text_fea)
                    if self.with_Nor:
                        tbhis_text_fea_mean = F.normalize(tbhis_text_fea_mean,dim=0)
                        text_J_c = F.normalize(text_J_c,dim=0)
                        text_K_c = F.normalize(text_K_c,dim=0)
                    if self.cos:
                        text_TuJ = F.cosine_similarity(tbhis_text_fea_mean, text_J_c, dim=-1)
                        text_TuK = F.cosine_similarity(tbhis_text_fea_mean, text_K_c, dim=-1)
                    else:
                        text_TuJ = torch.sum(tbhis_text_fea_mean * text_J_c, dim=-1)
                        text_TuK = torch.sum(tbhis_text_fea_mean * text_K_c, dim=-1)

                elif self.args.dataset == 'Polyvore':
                    text_tbhis = self.text_features[tbhis]
                    text_tbhis = self.iPC_text_nn(text_tbhis)
                    text_J_c = self.iPC_text_nn(text_J)
                    text_K_c = self.iPC_text_nn(text_K)
                    t_his_text = torch.mean(text_tbhis, dim=-2)  #bs, visual_feature_dim = 2048 #torch.Size([256, 2048])
                    if self.with_Nor:
                        t_his_text = F.normalize(t_his_text,dim=0)
                        text_J_c = F.normalize(text_J_c,dim=0)
                        text_K_c = F.normalize(text_K_c,dim=0)

                    if self.cos:
                        text_TuJ = F.cosine_similarity(t_his_text, text_J_c, dim=-1)
                        text_TuK = F.cosine_similarity(t_his_text, text_K_c, dim=-1)
                    else:
                        text_TuJ = torch.sum(t_his_text * text_J_c, dim=-1)
                        text_TuK = torch.sum(t_his_text * text_K_c, dim=-1) 

        if self.with_visual and self.with_text:
            if self.args.b_PC:
                cuj = self.vtbpr(Us, Js, J_visual_latent_p, J_text_latent_p) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, K_visual_latent_p, K_text_latent_p)
            else:
                cuj = self.vtbpr(Us, Js, J_visual_latent, J_text_latent) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, K_visual_latent, K_text_latent)

            p_ij = 0.5 * (visual_ij + text_ij)
            p_ik = 0.5 * (visual_ik + text_ik)

            pred = self.weight_P * p_ij + (1 - self.weight_P) * cuj - (self.weight_P * p_ik + (1 - self.weight_P) * cuk) 

            if self.iPC:
                C_TuJ = self.args.iPC_w * (self.args.iPC_v_w * Visual_TuJ + (1-self.args.iPC_v_w) * text_TuJ)# + bhis_j_latent_mean)
                C_TuK = self.args.iPC_w * (self.args.iPC_v_w * Visual_TuK + (1-self.args.iPC_v_w) * text_TuK)

                pred = pred + C_TuJ - C_TuK  

        if self.with_visual and not self.with_text:
            if self.args.b_PC:
                cuj = self.vtbpr(Us, Js, J_visual_latent_p, None) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, K_visual_latent_p, None)
            else:
                cuj = self.vtbpr(Us, Js, J_visual_latent, None) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, K_visual_latent, None)

            p_ij = visual_ij 
            p_ik = visual_ik 

            pred = self.weight_P * p_ij + (1 - self.weight_P) * cuj - (self.weight_P * p_ik + (1 - self.weight_P) * cuk)

            if self.iPC:
                C_TuJ = self.args.iPC_w * Visual_TuJ 
                C_TuK = self.args.iPC_w * Visual_TuK 

                pred = pred + C_TuJ - C_TuK 
        
        if not self.with_visual and self.with_text:
            if self.args.b_PC:
                cuj = self.vtbpr(Us, Js, None, J_text_latent_p) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, None, K_text_latent_p)
            else:
                cuj = self.vtbpr(Us, Js, None, J_text_latent) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, None, K_text_latent)

            p_ij = 0.5 * text_ij
            p_ik = 0.5 * text_ik

            pred = self.weight_P * p_ij + (1 - self.weight_P) * cuj - (self.weight_P * p_ik + (1 - self.weight_P) * cuk)

            if self.iPC:
                C_TuJ = self.args.iPC_w * text_TuJ
                C_TuK = self.args.iPC_w * text_TuK

                pred = pred + C_TuJ - C_TuK 
   
        return pred        