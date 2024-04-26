import torch
from torch.nn.init import uniform_
from torch.nn import *
import torch.nn as nn
import torch.nn.functional as F
from Models.BPRs.BPR import BPR


class VTBPR(BPR):
    def __init__(self, user_num, item_num, hidden_dim=512, theta_text = True, theta_visual = True, with_Nor=True, cos=True):
        super(VTBPR, self).__init__(user_num, item_num, hidden_dim=hidden_dim)
        if theta_visual:
            self.theta_user_visual = nn.Embedding(user_num, self.hidden_dim)
            nn.init.uniform_(self.theta_user_visual.weight, 0, 0.01)
        if theta_text:
            self.theta_user_text = nn.Embedding(user_num, self.hidden_dim)
            nn.init.uniform_(self.theta_user_text.weight, 0, 0.01)
        self.with_Nor = with_Nor
        self.cos = cos    

         
    def forward(self, users, items, visual_features=None, textural_features=None):
        ui_latent = BPR.forward(self, users, items)
    
        if visual_features is not None:
            theta_user_visual = self.theta_user_visual(users)
            if self.with_Nor:
                theta_user_visual = F.normalize(theta_user_visual,dim=0)

            if self.cos:
                ui_visual = F.cosine_similarity(theta_user_visual, visual_features, dim=-1)
            else:
                ui_visual = torch.sum(theta_user_visual * visual_features, dim=-1)

            ui_latent += ui_visual
        if textural_features is not None:
            theta_user_text = self.theta_user_text(users)
            if self.with_Nor:
                theta_user_text = F.normalize(theta_user_text,dim=0)
            if self.cos:
                ui_text = F.cosine_similarity(theta_user_text, textural_features, dim=-1)
            else:
                ui_text = torch.sum(theta_user_text * textural_features, dim=-1)
           
            ui_latent += ui_text
        
        return ui_latent
    
    def infer(self, bs, candi_num, Us, Js, Ks, J_visual_latent_p=None, J_text_latent_p=None, K_visual_latent_p=None, K_text_latent_p=None): 
        ujks_score = BPR.infer(self, bs, candi_num, Us, Js, Ks)
        if J_visual_latent_p is not None:
            theta_user_visual = self.theta_user_visual(Us)
            if self.with_Nor:
                theta_user_visual = F.normalize(theta_user_visual,dim=0)
            theta_user_visual, visual_features = self.wide_infer(bs, candi_num, J_visual_latent_p, K_visual_latent_p, theta_user_visual)
            if self.cos:
                ui_visual = F.cosine_similarity(theta_user_visual, visual_features, dim=-1)
            else:
                ui_visual = torch.sum(theta_user_visual * visual_features, dim=-1)
            ujks_score += ui_visual

        if J_text_latent_p is not None:
            theta_user_text = self.theta_user_text(Us)
            if self.with_Nor:
                theta_user_text = F.normalize(theta_user_text,dim=0)
            theta_user_text, textural_features = self.wide_infer(bs, candi_num, J_text_latent_p, K_text_latent_p, theta_user_text)
            if self.cos:
                ui_text = F.cosine_similarity(theta_user_text, textural_features, dim=-1)
            else:
                ui_text = torch.sum(theta_user_text * textural_features, dim=-1)
           
            ujks_score += ui_text   
        return ujks_score

    def wide_infer(self, bs, candi_num, J, K, I):
        J = J.unsqueeze(1) #256,1,512
        K = K.unsqueeze(0).expand(bs, -1, -1) #1,256,512->256,256,512
        Jks = torch.cat([J, K], dim=1) #256,257,512 # dim=1 里面第一个为postive target(1+256)
        I= I.unsqueeze(1).expand(-1, candi_num, -1) # 256,257,512
        return I, Jks