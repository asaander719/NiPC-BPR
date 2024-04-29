import torch
from torch.nn.init import uniform_
import torch.nn as nn
import torch.nn.functional as F
from Models.BPRs.BPR import BPR
from Models.BPRs.VTBPR import VTBPR
from Models.BPRs.TextCNN import TextCNN

class CRBPR(nn.Module):
    def __init__(self, args, embedding_weight, visual_features, text_features):        
        super(CRBPR, self) .__init__()
        self.args = args
        self.weight_P = args.weight_P
        self.hidden_dim = args.hidden_dim
        self.user_num = args.user_num
        self.item_num = args.item_num
        self.with_visual = args.with_visual
        self.with_text = args.with_text
        self.with_Nor = args.with_Nor
        self.cos = args.cos
        self.UC = args.UC
        self.GC = args.GC
        #for compatibility space
        self.visual_nn = nn.Sequential(nn.Linear(args.visual_feature_dim, self.hidden_dim),nn.Sigmoid())
        self.visual_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.visual_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))
   
        #for personalization space
        self.p_visual_nn = nn.Sequential(nn.Linear(args.visual_feature_dim, self.hidden_dim),nn.Sigmoid())
        self.p_visual_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.p_visual_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

        #for UC space
        self.s_visual_nn = nn.Sequential(nn.Linear(args.visual_feature_dim, self.hidden_dim),nn.Sigmoid())
        self.s_visual_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.s_visual_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

        #for IC space
        self.s3_visual_nn = nn.Sequential(nn.Linear(args.visual_feature_dim, self.hidden_dim),nn.Sigmoid())
        self.s3_visual_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.s3_visual_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

        self.sigmoid = nn.Sigmoid()
        
        if self.with_visual:
            self.visual_features = visual_features.to(args.device)
        if self.with_text:
            self.text_features = text_features.to(args.device)
            if self.args.dataset == 'IQON3000':
                self.max_sentense_length = args.max_sentence
                self.text_embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)
                self.textcnn = TextCNN(args.textcnn_layer, sentence_size=(args.max_sentence, args.text_feature_dim), output_size=self.hidden_dim)
                self.text_nn = nn.Sequential(nn.Linear(100 * args.textcnn_layer, self.hidden_dim),nn.Sigmoid()) 
                self.p_text_nn = nn.Sequential(nn.Linear(100 * args.textcnn_layer, self.hidden_dim),nn.Sigmoid())
                self.s_text_nn = nn.Sequential(nn.Linear(100 * args.textcnn_layer, self.hidden_dim),nn.Sigmoid())
                self.s3_text_nn = nn.Sequential(nn.Linear(100 * args.textcnn_layer, self.hidden_dim),nn.Sigmoid())
            elif self.args.dataset == 'Polyvore_519':
                self.text_nn = nn.Sequential(nn.Linear(args.text_feature_dim, self.hidden_dim),nn.Sigmoid()) 
                self.p_text_nn = nn.Sequential(nn.Linear(args.text_feature_dim, self.hidden_dim),nn.Sigmoid())
                self.s_text_nn = nn.Sequential(nn.Linear(args.text_feature_dim, self.hidden_dim),nn.Sigmoid())
                self.s3_text_nn = nn.Sequential(nn.Linear(args.text_feature_dim, self.hidden_dim),nn.Sigmoid())

        self.text_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.text_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))
        self.p_text_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.p_text_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))
        self.s_text_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.s_text_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))
        self.s3_text_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.s3_text_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

        self.vtbpr = VTBPR(self.user_num, self.item_num, hidden_dim=self.hidden_dim, 
            theta_text=self.with_text, theta_visual=self.with_visual, with_Nor=True, cos=True)
        print('Module already prepared, {} users, {} items'.format(self.user_num, self.item_num))
        self.bpr = BPR(self.user_num, self.item_num)
         
    def forward(self, batch, train =True, **args):
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

            if self.GC:    
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
            if self.args.dataset == 'IQON3000':
                text_I = self.text_embedding(self.text_features[Is].long()) #256,83,300 text_I = self.text_embedding(self.text_features[Is]) 
                text_J = self.text_embedding(self.text_features[Js].long())
                text_K = self.text_embedding(self.text_features[Ks].long())

                I_text_fea = self.textcnn(text_I.unsqueeze(1))  #256,400
                J_text_fea = self.textcnn(text_J.unsqueeze(1))
                K_text_fea = self.textcnn(text_K.unsqueeze(1))

                I_text_latent = self.text_nn(I_text_fea) #256,512
                J_text_latent = self.text_nn(J_text_fea)
                K_text_latent = self.text_nn(K_text_fea)

                J_text_latent_p = self.p_text_nn(J_text_fea)
                K_text_latent_p = self.p_text_nn(K_text_fea)

            elif self.args.dataset == 'Polyvore_519':
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
            #add similarity 
            if self.UC:
                if self.args.dataset == 'IQON3000':
                    text_bhis = self.text_embedding(self.text_features[bhis]) #torch.Size([64, 3, 83, 300])
                    bhis_text_fea = self.textcnn(text_bhis.reshape(bs * self.arg.num_his, self.arg.max_sentence, self.arg.text_feature_dim).unsqueeze(1))  #bs, 400(100*layers)
                    bhis_text_fea = self.s_text_nn(bhis_text_fea) #torch.Size([192, 512])
                    bhis_text_fea = bhis_text_fea.reshape(bs, self.arg.num_his, self.hidden_dim) #64, 3, 512
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
                elif self.args.dataset == 'Polyvore_519':
                    text_bhis = self.text_features[bhis]
                    text_bhis = self.s_text_nn(text_bhis)
                    text_J_p = self.s_text_nn(text_J)
                    text_K_p = self.s_text_nn(text_K)
                    b_his_text = torch.mean(text_bhis, dim=-2)  #bs, visual_feature_dim = 2048 #torch.Size([256, 2048])
                    if self.with_Nor:
                        b_his_text = F.normalize(b_his_text,dim=0)
                        text_J_p = F.normalize(text_J_p,dim=0)
                        text_K_p = F.normalize(text_K_p,dim=0)

                    if self.cos:
                        text_BuJ = F.cosine_similarity(b_his_text, text_J_p, dim=-1)
                        text_BuK = F.cosine_similarity(b_his_text, text_K_p, dim=-1)
                    else:
                        text_BuJ = torch.sum(b_his_text * text_J_p, dim=-1)
                        text_BuK = torch.sum(b_his_text * text_K_p, dim=-1)  

            if self.GC:
                if self.args.dataset == 'IQON3000':
                    text_this = self.text_embedding(self.text_features[this]) #torch.Size([64, 3, 83, 300])
                    this_text_fea = self.textcnn(text_this.reshape(bs * self.arg.num_his, self.arg.max_sentence, self.arg.text_feature_dim).unsqueeze(1))  #bs, 400(100*layers)
                    this_text_fea = self.s3_text_nn(this_text_fea) #torch.Size([192, 512])
                    this_text_fea = this_text_fea.reshape(bs, self.arg.num_his, self.hidden_dim) #64, 3, 512
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
                elif self.args.dataset == 'Polyvore_519':
                    text_this = self.text_features[this]#[64, 3, 2400])
                    text_this = self.s3_text_nn(text_this)
                    text_J_c = self.s3_text_nn(text_J)
                    text_K_c = self.s3_text_nn(text_K)
                    t_his_text = torch.mean(text_this, dim=-2)  #bs, visual_feature_dim = 2048 #torch.Size([256, 2048])
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

            if self.UC:
                C_BuJ = self.args.UC_w  * (self.args.UC_v_w  * Visual_BuJ + (1-self.args.UC_v_w) * text_BuJ)
                C_BuK = self.args.UC_w  * (self.args.UC_v_w  * Visual_BuK + (1-self.args.UC_v_w) * text_BuK)

                pred = pred + C_BuJ - C_BuK

            if self.GC:
                C_TuJ = self.args.GC_w * (self.args.GC_v_w * Visual_TuJ + (1-self.args.GC_v_w) * text_TuJ)
                C_TuK = self.args.GC_w * (self.args.GC_v_w * Visual_TuK + (1-self.args.GC_v_w) * text_TuK)

                pred = pred + C_TuJ - C_TuK   

        if self.with_visual and not self.with_text:
            if self.args.b_PC:
                cuj = self.vtbpr(Us, Js, J_visual_latent_p, None) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, K_visual_latent_p, None)
            else:
                cuj = self.vtbpr(Us, Js, J_visual_latent, None) #torch.Size(bs)
                cuk = self.vtbpr(Us, Ks, K_visual_latent, None)

            p_ij = 0.5 * visual_ij 
            p_ik = 0.5 * visual_ik 

            pred = self.weight_P * p_ij + (1 - self.weight_P) * cuj - (self.weight_P * p_ik + (1 - self.weight_P) * cuk)

            if self.UC:
                C_BuJ = self.args.UC_w * Visual_BuJ 
                C_BuK = self.args.UC_w * Visual_BuK 

                pred = pred + C_BuJ - C_BuK

            if self.GC:
                C_TuJ = self.args.GC_w * Visual_TuJ 
                C_TuK = self.args.GC_w * Visual_TuK 

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

            if self.UC:
                C_BuJ = self.args.UC_w * text_BuJ
                C_BuK = self.args.UC_w * text_BuK

                pred = pred + C_BuJ - C_BuK

            if self.GC:
                C_TuJ = self.args.GC_w * text_TuJ
                C_TuK = self.args.GC_w * text_TuK

                pred = pred + C_TuJ - C_TuK 

        if not self.with_visual and not self.with_text:
            cuj = self.vtbpr(Us, Js, None, None) #torch.Size(bs)
            cuk = self.vtbpr(Us, Ks, None, None)
            pred =  cuj - cuk
   
        return pred   

    def inference(self, batch, train=False, **args):
        Us = batch[0] #bs
        Is = batch[1]
        Js = batch[2]
        # Ks = batch[3]
        Ks_list = batch[3] #torch.Size([256, 1])
        bhis = batch[4]
        this = batch[5]
        tbhis = batch[6]
        bs = len(Us)
        candi_num = 1 + Ks_list.size(0)  # one positive + all negative #257
     
        if self.with_visual:
            vis_I = self.visual_features[Is] #bs,visual_feature_dim = 2048 = torch.Size([256, 2048])
            vis_J = self.visual_features[Js]
            if self.args.wide_evaluate:
                vis_K = self.visual_features[Ks_list.squeeze(1)] 
            else:
                vis_K = self.visual_features[Ks_list] 
  
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

            I_visual_latent, Jks_visual_latent = self.wide_infer(bs, candi_num, J_visual_latent, K_visual_latent, I_visual_latent)
            if self.cos:
                visual_ijs_score = F.cosine_similarity(I_visual_latent, Jks_visual_latent, dim=-1)    #256,257    
            else:
                visual_ijs_score = torch.sum(I_visual_latent * Jks_visual_latent, dim=-1)

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
                b_his_visual, vis_JK_p = self.wide_infer(bs, candi_num, vis_J_p, vis_K_p, b_his_visual)  
                if self.cos:
                    Visual_UC_score = F.cosine_similarity(b_his_visual, vis_JK_p, dim=-1)
                else:
                    Visual_UC_score  = torch.sum(b_his_visual * vis_JK_p, dim=-1)
                   

            if self.GC:    
                vis_this = self.visual_features[this]#bs,3,visual_feature_dim = 2048 torch.Size([256, 3, 2048])
                vis_this = self.s3_visual_nn(vis_this) #bs,3,512
                vis_J_c= self.s3_visual_nn(vis_J)
                vis_K_c = self.s3_visual_nn(vis_K)
                t_his_visual = torch.mean(vis_this, dim=-2)  #bs, visual_feature_dim = 2048 #torch.Size([256, 512])

                if self.with_Nor:
                    t_his_visual = F.normalize(t_his_visual,dim=0)
                    vis_J_c = F.normalize(vis_J_c,dim=0)
                    vis_K_c = F.normalize(vis_K_c,dim=0)

                t_his_visual, vis_JK_c = self.wide_infer(bs, candi_num, vis_J_c, vis_K_c, t_his_visual)  
                if self.cos:
                    Visual_GC_score = F.cosine_similarity(t_his_visual, vis_JK_c, dim=-1)
                else:
                    Visual_GC_score  = torch.sum(t_his_visual * vis_JK_c, dim=-1)

        if self.with_text:
            if self.args.dataset == 'IQON3000':
                text_I = self.text_embedding(self.text_features[Is].long()) #256,83,300 text_I = self.text_embedding(self.text_features[Is]) 
                text_J = self.text_embedding(self.text_features[Js].long())
                if self.args.wide_evaluate:
                    text_K = self.text_embedding(self.text_features[Ks_list.squeeze(1)].long())
                else:
                    text_K = self.text_embedding(self.text_features[Ks_list].long())

                I_text_fea = self.textcnn(text_I.unsqueeze(1))  #256,400
                J_text_fea = self.textcnn(text_J.unsqueeze(1))
                K_text_fea = self.textcnn(text_K.unsqueeze(1))

                I_text_latent = self.text_nn(I_text_fea) #256,512
                J_text_latent = self.text_nn(J_text_fea)
                K_text_latent = self.text_nn(K_text_fea)

                J_text_latent_p = self.p_text_nn(J_text_fea)
                K_text_latent_p = self.p_text_nn(K_text_fea)

            elif self.args.dataset == 'Polyvore_519':
                text_I = self.text_features[Is] #256,83,300
                text_J = self.text_features[Js]
                if self.args.wide_evaluate:
                    text_K = self.text_features[Ks_list.squeeze(1)]
                else:
                    text_K = self.text_features[Ks_list]

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

            I_text_latent, Jks_text_latent = self.wide_infer(bs, candi_num, J_text_latent, K_text_latent, I_text_latent)
            if self.cos:
                text_ijks = F.cosine_similarity(I_text_latent, Jks_text_latent, dim=-1)
            else:
                text_ijks = torch.sum(I_text_latent * Jks_text_latent, dim=-1)

            #add similarity 
            if self.UC:
                if self.args.dataset == 'IQON3000':
                    text_bhis = self.text_embedding(self.text_features[bhis]) #torch.Size([64, 3, 83, 300])
                    bhis_text_fea = self.textcnn(text_bhis.reshape(bs * self.args.num_his, self.args.max_sentence, self.args.text_feature_dim).unsqueeze(1))  #bs, 400(100*layers)
                    bhis_text_fea = self.s_text_nn(bhis_text_fea) #torch.Size([192, 512])
                    bhis_text_fea = bhis_text_fea.reshape(bs, self.args.num_his, self.hidden_dim) #64, 3, 512
                    bhis_text_fea_mean = torch.mean(bhis_text_fea, dim=-2) #torch.Size([bs, 512])
                    text_J_p = self.s_text_nn(J_text_fea)
                    text_K_p = self.s_text_nn(K_text_fea) 
                    if self.with_Nor:
                        bhis_text_fea_mean = F.normalize(bhis_text_fea_mean,dim=0)
                        text_J_p = F.normalize(text_J_p,dim=0)
                        text_K_p = F.normalize(text_K_p,dim=0)
                    bhis_text_fea_mean, text_JK_p = self.wide_infer(bs, candi_num, text_J_p, text_K_p, bhis_text_fea_mean)
                    if self.cos:
                        text_UC_score = F.cosine_similarity(bhis_text_fea_mean, text_JK_p, dim=-1)
                    else:
                        text_UC_score = torch.sum(bhis_text_fea_mean * text_JK_p , dim=-1)

                elif self.args.dataset == 'Polyvore_519':
                    text_bhis = self.text_features[bhis] # ([64, 3, 2400])
                    text_bhis = self.s_text_nn(text_bhis)
                    text_J_p = self.s_text_nn(text_J)
                    text_K_p = self.s_text_nn(text_K)
                    b_his_text = torch.mean(text_bhis, dim=-2) 
                    if self.with_Nor:
                        b_his_text = F.normalize(b_his_text,dim=0)
                        text_J_p = F.normalize(text_J_p,dim=0)
                        text_K_p = F.normalize(text_K_p,dim=0)
                    b_his_text, text_JK_p= self.wide_infer(bs, candi_num, text_J_p, text_K_p, b_his_text)
                    if self.cos:
                        text_UC_score = F.cosine_similarity(b_his_text, text_JK_p, dim=-1)
                    else:
                        text_UC_score = torch.sum(b_his_text * text_JK_p , dim=-1)

            if self.GC:
                if self.args.dataset == 'IQON3000':
                    text_this = self.text_embedding(self.text_features[this]) #torch.Size([64, 3, 83, 300])
                    this_text_fea = self.textcnn(text_this.reshape(bs * self.args.num_his, self.args.max_sentence, self.args.text_feature_dim).unsqueeze(1))  #bs, 400(100*layers)
                    this_text_fea = self.s3_text_nn(this_text_fea) #torch.Size([192, 512])
                    this_text_fea = this_text_fea.reshape(bs, self.args.num_his, self.hidden_dim) #64, 3, 512
                    this_text_fea_mean = torch.mean(this_text_fea, dim=-2) #torch.Size([bs, 512])
                    text_J_c = self.s3_text_nn(J_text_fea)
                    text_K_c = self.s3_text_nn(K_text_fea)
                    if self.with_Nor:
                        this_text_fea_mean = F.normalize(this_text_fea_mean,dim=0)
                        text_J_c = F.normalize(text_J_c,dim=0)
                        text_K_c = F.normalize(text_K_c,dim=0)
                    this_text_fea_mean, text_JK_c= self.wide_infer(bs, candi_num, text_J_c, text_K_c, this_text_fea_mean)
                    if self.cos:
                        text_GC_score = F.cosine_similarity(this_text_fea_mean, text_JK_c, dim=-1)
                    else:
                        text_GC_score = torch.sum(this_text_fea_mean * text_JK_c , dim=-1)

                elif self.args.dataset == 'Polyvore_519':
                    text_this = self.text_features[this] #[64, 3, 2400]
                    text_this = self.s3_text_nn(text_this)
                    text_J_c = self.s3_text_nn(text_J)
                    text_K_c = self.s3_text_nn(text_K)
                    t_his_text = torch.mean(text_this, dim=-2)  #bs, visual_feature_dim = 2048 #torch.Size([256, 2048])
                    if self.with_Nor:
                        t_his_text = F.normalize(t_his_text,dim=0)
                        text_J_c = F.normalize(text_J_c,dim=0)
                        text_K_c = F.normalize(text_K_c,dim=0)
                    t_his_text, text_JK_c= self.wide_infer(bs, candi_num, text_J_c, text_K_c, t_his_text)
                    if self.cos:
                        text_GC_score = F.cosine_similarity(t_his_text, text_JK_c, dim=-1)
                    else:
                        text_GC_score = torch.sum(t_his_text * text_JK_c , dim=-1)

        if self.with_visual and self.with_text:
            if self.args.b_PC:  
                if self.args.wide_evaluate: 
                    score_c = self.vtbpr.infer(bs, candi_num, Us, Js, Ks_list.squeeze(1), J_visual_latent_p, J_text_latent_p, K_visual_latent_p, K_text_latent_p) 
                else:
                    score_c = self.vtbpr.infer(bs, candi_num, Us, Js, Ks_list, J_visual_latent_p, J_text_latent_p, K_visual_latent_p, K_text_latent_p)      
            else:
                if self.args.wide_evaluate:
                    score_c = self.vtbpr.infer(bs, candi_num, Us, Js, Ks_list.squeeze(1), J_visual_latent, J_text_latent, K_visual_latent, K_text_latent) 
                else:
                    score_c = self.vtbpr.infer(bs, candi_num, Us, Js, Ks_list, J_visual_latent, J_text_latent, K_visual_latent, K_text_latent) 

            score_p = 0.5 * (visual_ijs_score + text_ijks)  
            score = self.weight_P * score_p  + (1 - self.weight_P) * score_c

            if self.UC:
                score_UC = self.args.UC_w  * (self.args.UC_v_w  * Visual_UC_score + (1-self.args.UC_v_w) * text_UC_score)
                score += score_UC

            if self.GC:
                score_GC = self.args.GC_w * (self.args.GC_v_w * Visual_GC_score + (1-self.args.GC_v_w) * text_GC_score)
                score += score_GC  

        if self.with_visual and not self.with_text:
            if self.args.b_PC:   
                if self.args.wide_evaluate: 
                    score_c = self.vtbpr.infer(bs, candi_num, Us, Js, Ks_list.squeeze(1), J_visual_latent_p, None, K_visual_latent_p, None)
                else:
                    score_c = self.vtbpr.infer(bs, candi_num, Us, Js, Ks_list, J_visual_latent_p, None, K_visual_latent_p, None)     
            else:
                if self.args.wide_evaluate: 
                    score_c = self.vtbpr.infer(bs, candi_num, Us, Js, Ks_list.squeeze(1), J_visual_latent, None, K_visual_latent, None) 
                else:
                    score_c = self.vtbpr.infer(bs, candi_num, Us, Js, Ks_list, J_visual_latent, None, K_visual_latent, None) 

            score_p = 0.5 * visual_ijs_score  
            score = self.weight_P * score_p  + (1 - self.weight_P) * score_c

            if self.UC:
                score_UC = self.args.UC_w * Visual_UC_score 
                score += score_UC

            if self.GC:
                score_GC = self.args.GC_w * Visual_GC_score
                score += score_GC   
        
        if not self.with_visual and self.with_text:
            if self.args.b_PC:   
                if self.args.wide_evaluate: 
                    score_c = self.vtbpr.infer(bs, candi_num, Us, Js, Ks_list.squeeze(1), None, J_text_latent_p, None, K_text_latent_p) 
                else:
                    score_c = self.vtbpr.infer(bs, candi_num, Us, Js, Ks_list, None, J_text_latent_p, None, K_text_latent_p)    
            else:
                if self.args.wide_evaluate: 
                    score_c = self.vtbpr.infer(bs, candi_num, Us, Js, Ks_list.squeeze(1), None, J_text_latent, None, K_text_latent) 
                else:
                    score_c = self.vtbpr.infer(bs, candi_num, Us, Js, Ks_list, None, J_text_latent, None, K_text_latent)

            score_p = 0.5 * text_ijks 
            score = self.weight_P * score_p  + (1 - self.weight_P) * score_c

            if self.UC:
                score_UC = self.args.UC_w * text_UC_score 
                score += score_UC

            if self.GC:
                score_GC = self.args.GC_w * text_GC_score
                score += score_GC 

        if not self.with_visual and not self.with_text: # without visual & text
            if self.args.wide_evaluate:   
                score_c = self.vtbpr.infer(bs, candi_num, Us, Js, Ks_list.squeeze(1), None, None, None, None)   
            else:
                score_c = self.vtbpr.infer(bs, candi_num, Us, Js, Ks_list, None, None, None, None)         
             
            score = score_c # NO comaptibility modeling
        return score
    
    def wide_infer(self, bs, candi_num, J, K, I):
        J = J.unsqueeze(1) #256,1,512
        K = K.unsqueeze(0).expand(bs, -1, -1) #1,256,512->256,256,512
        Jks = torch.cat([J, K], dim=1) #256,257,512 # dim=1 里面第一个为postive target(1+256)
        I= I.unsqueeze(1).expand(-1, candi_num, -1) # 256,257,512
        return I, Jks    