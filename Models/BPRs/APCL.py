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

class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)

    def forward(self, x, mask = None):
        batch_size, seq_length, input_d = x.size()

        q = self.query(x) #(batch_size, seq_length, output_d)
        k = self.key(x)
        v = self.value(x)

        attention_score = torch.matmul(q, k.transpose(1,2)) / (self.output_dim ** 0.5) #(batch_size, seq_length, seq_length)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask ==0, float('-inf'))
        
        attention_weight = torch.softmax(attention_score, dim=-1) #(batch_size, seq_length, seq_length)
        weighted_values = torch.matmul(attention_weight, v) ##(batch_size, seq_length, output_d)

        output = torch.mean(weighted_values, dim=1) #(bs, out_dim) #average pooling
        return output

class APCL(Module):
    def __init__(self, args, embedding_weight, visual_features, text_features):        
        super(NattBPR, self) .__init__()
        self.args = args
        self.weight_P = args.weight_P
        self.hidden_dim = args.hidden_dim
        self.user_num = args.user_num
        self.item_num = args.item_num
        self.with_visual = args.with_visual
        self.with_text = args.with_text
        self.with_Nor = args.with_Nor
        self.cos = args.cos
        self.att = args.att
        self.use_weighted_loss = args.use_weighted_loss
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

        #for att space
        self.att_visual_nn = Sequential(
            Linear(args.visual_feature_dim, self.hidden_dim),
            nn.Sigmoid())
        self.att_visual_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.att_visual_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

        if self.args.dataset == 'IQON3000':
            self.text_nn = Sequential(
                Linear(100 * args.textcnn_layer, self.hidden_dim),
                nn.Sigmoid()) 

            self.p_text_nn = Sequential(
                Linear(100 * args.textcnn_layer, self.hidden_dim),
                nn.Sigmoid())

            self.att_text_nn = Sequential(
                Linear(100 * args.textcnn_layer, self.hidden_dim),
                nn.Sigmoid())

        elif self.args.dataset == 'Polyvore':
            self.text_nn = Sequential(
                Linear(args.text_feature_dim, self.hidden_dim),
                nn.Sigmoid()) 

            self.p_text_nn = Sequential(
                Linear(args.text_feature_dim, self.hidden_dim),
                nn.Sigmoid())

            self.att_text_nn = Sequential(
                Linear(args.text_feature_dim, self.hidden_dim),
                nn.Sigmoid())

        self.text_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.text_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))       
        self.p_text_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.p_text_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))     
        self.att_text_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.att_text_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))
        
        if self.with_visual:
            self.visual_features = visual_features.cuda()
        if self.with_text:
            self.max_sentense_length = args.max_sentence
            self.text_features = text_features.cuda()
            self.text_embedding = Embedding.from_pretrained(embedding_weight, freeze=False)
            self.textcnn = TextCNN(args.textcnn_layer, sentence_size=(args.max_sentence, args.text_feature_dim), output_size=self.hidden_dim)

        self.vtbpr = VTBPR(self.user_num, self.item_num, hidden_dim=self.hidden_dim, 
            theta_text=self.with_text, theta_visual=self.with_visual, with_Nor=True, cos=True)
        print('Module already prepared, {} users, {} items'.format(self.user_num, self.item_num))
        self.bpr = BPR(self.user_num, self.item_num)
         
        self.V_attention = SelfAttention(input_dim= args.visual_feature_dim, output_dim = args.visual_feature_dim)
        self.T_attention = SelfAttention(input_dim= args.text_feature_dim, output_dim = args.text_feature_dim)
  
        # self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.w1.data.fill_(1.0)
        # self.w2.data.fill_(1.0)
        # self.w1.data.clamp_(0,1)
        # self.w2.data.clamp_(0,1)
         
    def forward(self, batch, train, **args):
        Us = batch[0] #bs
        Is = batch[1]
        Js = batch[2]
        Ks = batch[3]
        all_u_pb = batch[4]
        ub_inter_weight = batch[5]
        tb_inter_weight = batch[6]
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

            if self.att:
                vis_all_u_pb = self.visual_features[all_u_pb]#bs,3,visual_feature_dim = 2048 torch.Size([256, 3, 2048])
                v_sim_bs_out = self.V_attention.forward(vis_all_u_pb) #(bs, 512)
                all_u_pb_visual = self.att_visual_nn(v_sim_bs_out) #bs,512
                vis_J_u = self.att_visual_nn(vis_J)
                vis_K_u = self.att_visual_nn(vis_K)

                if self.with_Nor:
                    all_u_pb_visual = F.normalize(all_u_pb_visual,dim=0)
                    vis_J_u = F.normalize(vis_J_u,dim=0)
                    vis_K_u = F.normalize(vis_K_u,dim=0)

                if self.cos:
                    Visual_UuJ = F.cosine_similarity(all_u_pb_visual, vis_J_u, dim=-1)
                    Visual_UuK = F.cosine_similarity(all_u_pb_visual, vis_K_u, dim=-1)
                else:
                    Visual_UuJ = torch.sum(all_u_pb_visual * vis_J_u, dim=-1)
                    Visual_UuK = torch.sum(all_u_pb_visual * vis_K_u, dim=-1)
      
        if self.with_text:
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

            # sim att
            if self.att:
                text_all_u_pb = self.text_features[all_u_pb]#bs,3,visual_feature_dim = 2048 torch.Size([256, 3, 2048])
                t_sim_bs_out = self.T_attention.forward(text_all_u_pb) #(bs, 512)
                all_u_pb_text = self.att_text_nn(t_sim_bs_out) #bs,512
                text_J_u = self.att_text_nn(text_J)
                text_K_u = self.att_text_nn(text_K)

                if self.with_Nor:
                    all_u_pb_text = F.normalize(all_u_pb_text,dim=0)
                    text_J_u = F.normalize(text_J_u,dim=0)
                    text_K_u = F.normalize(text_K_u,dim=0)

                if self.cos:
                    Text_UuJ = F.cosine_similarity(all_u_pb_text, text_J_u, dim=-1)
                    Text_UuK = F.cosine_similarity(all_u_pb_text, text_K_u, dim=-1)
                else:
                    Text_UuJ = torch.sum(all_u_pb_text * text_J_u, dim=-1)
                    Text_UuK = torch.sum(all_u_pb_text * text_K_u, dim=-1)

        if self.with_visual and self.with_text:
            if self.args.b_PC:
                cuj = self.vtbpr(Us, Js, J_visual_latent_p, J_text_latent_p) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, K_visual_latent_p, K_text_latent_p)

                #infoNCE loss: Bottom in p space and c space
                v_logits = torch.sum(J_visual_latent_p[:, None, :] * J_visual_latent[:, :, None], dim=-1) / conf['temperature']
                # 计算v和负样本的点积，然后除以温度
                v_neg_logits = torch.sum(J_visual_latent_p[:, None, :] * K_visual_latent_p[:, :, None], dim=-1) / conf['temperature']

                # 计算InfoNCE损失
                infoNCE_v_loss_pc = -v_logits + torch.logsumexp(torch.cat([v_logits.unsqueeze(2), v_neg_logits.unsqueeze(2)], dim=2), dim=2)
                infoNCE_v_loss_pc = infoNCE_v_loss_pc.mean()


                t_logits = torch.sum(J_text_latent_p[:, None, :] * J_text_latent[:, :, None], dim=-1) / conf['temperature']
                # 计算v和负样本的点积，然后除以温度
                t_neg_logits = torch.sum(J_text_latent_p[:, None, :] * K_text_latent_p[:, :, None], dim=-1) / conf['temperature']

                # 计算InfoNCE损失
                infoNCE_t_loss_pc = -t_logits + torch.logsumexp(torch.cat([t_logits.unsqueeze(2), t_neg_logits.unsqueeze(2)], dim=2), dim=2)
                infoNCE_t_loss_pc = infoNCE_t_loss_pc.mean()

            else:
                cuj = self.vtbpr(Us, Js, J_visual_latent, J_text_latent) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, K_visual_latent, K_text_latent)
                infoNCE_v_loss_pc == 0
                infoNCE_t_loss_pc == 0

            # p_ij = 0.5 * (visual_ij + text_ij)
            # p_ik = 0.5 * (visual_ik + text_ik)
            p_ij = (visual_ij + text_ij)
            p_ik = (visual_ik + text_ik)

            # pred = self.uniform_value * p_ij + (1 - self.uniform_value) * cuj - (self.uniform_value * p_ik + (1 - self.uniform_value) * cuk)
            if self.use_weighted_loss:
                pred = ub_inter_weight * (p_ij - p_ik) + tb_inter_weight * (cuj - cuk)

            else:
                pred = self.uniform_value * (p_ij - p_ik) + (1 - self.uniform_value) * (cuj - cuk)

            if self.att:
                U_BuJ = self.args.uu_w * (conf["uu_v_w"] * Visual_UuJ + (1-conf["uu_v_w"]) * Text_UuJ)
                U_BuK = self.args.uu_w * (conf["uu_v_w"] * Visual_UuK + (1-conf["uu_v_w"]) * Text_UuK)
                # U_BuJ = self.w1 * (conf["uu_v_w"] * Visual_UuJ + (1-conf["uu_v_w"]) * Text_UuJ)
                # U_BuK = self.w1 * (conf["uu_v_w"] * Visual_UuK + (1-conf["uu_v_w"]) * Text_UuK)

                pred = pred + U_BuJ - U_BuK

                #infoNCE loss: Bottom in p space and ps space
                v2_logits = torch.sum(J_visual_latent_p[:, None, :] * vis_J_u[:, :, None], dim=-1) / conf['temperature']
                # [:, None, :] 将 J 的形状从 [batch, output_dim] 转变为 [batch, 1, output_dim],torch.sum: [batch, output_dim]
                v2_neg_logits = torch.sum(J_visual_latent_p[:, None, :] * vis_K_u[:, :, None], dim=-1) / conf['temperature']

                # 计算InfoNCE损失
                infoNCE_v_loss_ps = -v2_logits + torch.logsumexp(torch.cat([v2_logits.unsqueeze(2), v2_neg_logits.unsqueeze(2)], dim=2), dim=2)
                infoNCE_v_loss_ps = infoNCE_v_loss_ps.mean()


                t2_logits = torch.sum(J_text_latent_p[:, None, :] * text_J_u[:, :, None], dim=-1) / conf['temperature']
                # 计算v和负样本的点积，然后除以温度
                t2_neg_logits = torch.sum(J_text_latent_p[:, None, :] * text_K_u[:, :, None], dim=-1) / conf['temperature']

                # 计算InfoNCE损失
                infoNCE_t_loss_ps = -t2_logits + torch.logsumexp(torch.cat([t2_logits.unsqueeze(2), t2_neg_logits.unsqueeze(2)], dim=2), dim=2)
                infoNCE_t_loss_ps = infoNCE_t_loss_ps.mean()

            else:
                infoNCE_v_loss_ps == 0 
                infoNCE_t_loss_ps == 0

        if self.with_visual and not self.with_text:
            if self.args.b_PC:
                cuj = self.vtbpr(Us, Js, J_visual_latent_p, None) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, K_visual_latent_p, None)
            else:
                cuj = self.vtbpr(Us, Js, J_visual_latent, None) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, K_visual_latent, None)

            p_ij = visual_ij 
            p_ik = visual_ik 

            # pred = self.uniform_value * p_ij + (1 - self.uniform_value) * cuj - (self.uniform_value * p_ik + (1 - self.uniform_value) * cuk)
            if self.use_weighted_loss:
                pred = ub_inter_weight * (p_ij - p_ik) + tb_inter_weight * (cuj - cuk)

            else:
                pred = self.uniform_value * (p_ij - p_ik) + (1 - self.uniform_value) * (cuj - cuk)

            if self.att:
                U_BuJ = self.args.uu_w *  Visual_UuJ 
                U_BuK = self.args.uu_w *  Visual_UuK 
                # U_BuJ = self.w1 *  Visual_UuJ 
                # U_BuK = self.w1 *  Visual_UuK 
                pred = pred + U_BuJ - U_BuK
        
        if not self.with_visual and self.with_text:
            if self.args.b_PC:
                cuj = self.vtbpr(Us, Js, None, J_text_latent_p) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, None, K_text_latent_p)
            else:
                cuj = self.vtbpr(Us, Js, None, J_text_latent) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, None, K_text_latent)

            p_ij = text_ij
            p_ik = text_ik

            # pred = self.uniform_value * p_ij + (1 - self.uniform_value) * cuj - (self.uniform_value * p_ik + (1 - self.uniform_value) * cuk)
            if self.use_weighted_loss:
                pred = ub_inter_weight * (p_ij - p_ik) + tb_inter_weight * (cuj - cuk)

            else:
                pred = self.uniform_value * (p_ij - p_ik) + (1 - self.uniform_value) * (cuj - cuk)

            if self.att:
                U_BuJ = self.args.uu_w * Text_UuJ
                U_BuK = self.args.uu_w * Text_UuK
                # U_BuJ = self.w1 * Text_UuJ
                # U_BuK = self.w1 * Text_UuK

                pred = pred + U_BuJ - U_BuK
                
        return pred, infoNCE_v_loss_pc, infoNCE_t_loss_pc, infoNCE_v_loss_ps, infoNCE_t_loss_ps