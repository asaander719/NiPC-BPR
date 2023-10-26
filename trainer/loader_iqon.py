import os
import os.path
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset


class Load_Data(Dataset):
    def __init__(self, args, test_data_ori, ub_his_dict, ut_his_dict, top_bottoms_dict, popular_bottoms,popular_tops, visual_features, text_features):
        self.test_data_ori = test_data_ori
        # self.his_lenth = his_lenth
        self.args = args
        self.visual_features = visual_features
        self.text_features = text_features
        self.user_bottom_dict = ub_his_dict
        self.user_top_dict = ut_his_dict
        self.top_bottoms_dict = top_bottoms_dict
        self.popular_bottoms = popular_bottoms
        self.popular_tops = popular_tops

    def __len__(self):
        return len(self.test_data_ori)

    def __getitem__(self, idx):
        test_data_ori = self.test_data_ori[idx]
        user_idx, top_idx, pos_bottom_idx, neg_bottom_idx = test_data_ori 

        bottom_his = self.user_bottom_dict[int(user_idx.numpy())]

        if self.args.repeated_interact:
            if int(pos_bottom_idx.numpy()) in bottom_his:
                bottom_his.remove(int(pos_bottom_idx.numpy()))
                bottom_his = list(set(bottom_his))
        else:
            while int(pos_bottom_idx.numpy()) in bottom_his:
                bottom_his.remove(int(pos_bottom_idx.numpy()))
        if self.args.num_his <= len(bottom_his):
            bottom_v_fea = self.visual_features[pos_bottom_idx]
            bottom_his_v_fea = self.visual_features[torch.LongTensor(bottom_his)]
            new_bottom_v_fea = bottom_v_fea.unsqueeze(0).expand(bottom_his_v_fea.size(0),-1) #torch.Size([99, 2048])
            b_v_score = torch.sum(new_bottom_v_fea * bottom_his_v_fea, dim=-1)
            b_score = b_v_score

            value, idx = torch.topk(b_score, k= self.args.num_his, dim=-1) #tensor([597.7301, 588.4199, 544.7933]) tensor([84, 52, 49])
            #此处的idx指的是当前user——bottom his （单条数据）对应的u_idx的第84个bottom 的idx，需要还原到item_map里面对应的bottomidx
            ub_his = [bottom_his[int(i.numpy())] for i in idx] #[68176, 100951, 8007] 

        elif len(bottom_his) == 0:
            ub_his = []
            # if self.args.popular_padding"]:
            for i in range(self.args.num_his):
                ub_his.append(popular_bottoms[i])#用popular item代替
            # else:
            #     for i in range(self.args.num_his"]):
            #         ub_his.append(-1)

        else:
            bottom_v_fea = self.visual_features[pos_bottom_idx]
            bottom_his_v_fea = self.visual_features[torch.LongTensor(bottom_his)]
            new_bottom_v_fea = bottom_v_fea.unsqueeze(0).expand(bottom_his_v_fea.size(0),-1) #torch.Size([99, 2048])
            b_v_score = torch.sum(new_bottom_v_fea * bottom_his_v_fea, dim=-1)
            b_score = b_v_score

            value, idx = torch.topk(b_score, k= len(bottom_his), dim=-1) 
            ub_his = [bottom_his[int(i.numpy())] for i in idx]       
            
            for i in range(self.args.num_his-len(ub_his)):
                ub_his.append(ub_his[0]) #选最相似的补齐

        #C2
        top_his = self.user_top_dict[int(user_idx.numpy())]

        if self.args.repeated_interact:
            if int(top_idx.numpy()) in top_his:
                top_his.remove(int(top_idx.numpy()))
                top_his = list(set(top_his))
        else:
            while int(top_idx.numpy()) in top_his:
                top_his.remove(int(top_idx.numpy()))

        if self.args.num_his <= len(top_his):
            top_v_fea = self.visual_features[top_idx]
            top_his_v_fea = self.visual_features[torch.LongTensor(top_his)]
            new_top_v_fea = top_v_fea.unsqueeze(0).expand(top_his_v_fea.size(0),-1) #torch.Size([99, 2048])
            t_v_score = torch.sum(new_top_v_fea * top_his_v_fea, dim=-1)
            t_score = t_v_score

            value, idx = torch.topk(t_score, k= self.args.num_his, dim=-1) #tensor([597.7301, 588.4199, 544.7933]) tensor([84, 52, 49])
            #此处的idx指的是当前user——bottom his （单条数据）对应的u_idx的第84个bottom 的idx，需要还原到item_map里面对应的bottomidx
            ut_his = [top_his[int(i.numpy())] for i in idx] #[68176, 100951, 8007] 

        elif len(top_his) == 0:
            ut_his = []
            if self.args.popular_padding:
                for i in range(self.args.num_his):
                    ut_his.append(popular_tops[i])#用popular item代替
            else:
                for i in range(self.args.num_his):
                    ut_his.append(-1)

        else:
            top_v_fea = self.visual_features[top_idx]
            top_his_v_fea = self.visual_features[torch.LongTensor(top_his)]
            new_top_v_fea = top_v_fea.unsqueeze(0).expand(top_his_v_fea.size(0),-1) #torch.Size([99, 2048])
            t_v_score = torch.sum(new_top_v_fea * top_his_v_fea, dim=-1)
            t_score = t_v_score

            if self.args.his_select_t:
                top_t_fea = self.text_features[int(top_idx.numpy())]
                top_his_t_fea = self.text_features[torch.LongTensor(top_his)]
                new_top_t_fea = top_t_fea.unsqueeze(0).expand(top_his_t_fea.size(0),-1)
                t_t_score = torch.sum(new_top_t_fea * top_his_t_fea, dim=-1)
                t_score += t_t_score
            value, idx = torch.topk(t_score, k= len(top_his), dim=-1) 
            ut_his = [top_his[int(i.numpy())] for i in idx]       
            
            for i in range(self.args.num_his-len(ut_his)):
                ut_his.append(ut_his[0]) #选最相似的补齐
                
        # S2 从当前的top交互历史里面选取n个和和target bottom（pos）最相似（原始空间的feature)的bottoms,但是与P2不同的是top 交互历史可能是0！
        if int(top_idx.numpy()) in self.top_bottoms_dict: #top 交互历史可能为0
            top_bottoms = self.top_bottoms_dict[int(top_idx.numpy())]

            if self.args.repeated_interact:
                if int(pos_bottom_idx.numpy()) in top_bottoms:
                    top_bottoms.remove(int(pos_bottom_idx.numpy()))
                    
                    top_bottoms=list(set(top_bottoms))
            else:
                while int(pos_bottom_idx.numpy()) in top_bottoms:
                    top_bottoms.remove(int(pos_bottom_idx.numpy()))
            if self.args.num_interact <= len(top_bottoms): 
                bottom_v_fea = self.visual_features[pos_bottom_idx]  
                top_bottoms_v_fea = self.visual_features[torch.LongTensor(top_bottoms)]    
                pos_bottoms_v_fea = bottom_v_fea.unsqueeze(0).expand(top_bottoms_v_fea.size(0),-1)
                b2_v_score = torch.sum(pos_bottoms_v_fea * top_bottoms_v_fea, dim=-1)
                b2_score = b2_v_score
               
                value_2, idx_2 = torch.topk(b2_score, k= self.args.num_interact, dim=-1) 
                tb_his = [top_bottoms[int(i.numpy())] for i in idx_2]
           
            elif len(top_bottoms) == 0:
                tb_his = []  
                # if self.args.popular_padding"]: 
                for i in range(self.args.num_interact):
                    tb_his.append(popular_bottoms[i])#用popular item代替
                # else:
                #     for i in range(self.args.num_interact"]):
                #         tb_his.append(-1) 
            else:
                bottom_v_fea = self.visual_features[pos_bottom_idx]  
                top_bottoms_v_fea = self.visual_features[torch.LongTensor(top_bottoms)]    
                pos_bottoms_v_fea = bottom_v_fea.unsqueeze(0).expand(top_bottoms_v_fea.size(0),-1)
                b2_v_score = torch.sum(pos_bottoms_v_fea * top_bottoms_v_fea, dim=-1)
                b2_score = b2_v_score
                
                value_2, idx_2 = torch.topk(b2_score, k= len(top_bottoms), dim=-1) 
                tb_his = [top_bottoms[int(i.numpy())] for i in idx_2]
                
                for i in range(self.args.num_interact -len(tb_his)):
                    tb_his.append(tb_his[0]) 
  
        else:    
            tb_his = []  
            for i in range(self.args.num_interact):
                # tb_his.append(-1)
                tb_his.append(popular_bottoms[i])#用popular item代替   
        
        return user_idx.long(), top_idx.long(), pos_bottom_idx.long(), neg_bottom_idx.long(), torch.LongTensor(ub_his), torch.LongTensor(ut_his),torch.LongTensor(tb_his)