import os
import os.path
# import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from collections import defaultdict

class Load_Data(Dataset):
    def __init__(self, args, data_ori, ub_his_dict, ut_his_dict, top_bottoms_dict, popular_bottoms, 
            popular_tops, bottom_user_dict, popular_users,ub_inter_weights_dict, tb_inter_weights_dict, ub_default_weight, tb_default_weight):
        self.data_ori = data_ori
        self.args = args
        self.user_bottom_dict = ub_his_dict
        self.user_top_dict = ut_his_dict
        self.top_bottoms_dict= top_bottoms_dict
        self.popular_bottoms = popular_bottoms
        self.popular_tops = popular_tops
        self.bottom_user_dict = bottom_user_dict
        self.popular_users = popular_users
        self.ub_inter_weights_dict = ub_inter_weights_dict
        self.tb_inter_weights_dict = tb_inter_weights_dict
        self.ub_default_weight = ub_default_weight
        self.tb_default_weight = tb_default_weight

    def __len__(self):
        return len(self.data_ori)

    def __getitem__(self, idx):
        data_ori = self.data_ori[idx]
        user_idx, top_idx, pos_bottom_idx, neg_bottom_idx = data_ori 

        sim_us = []
        # sim_us = self.u_userscf_dict[str(int(user_idx.numpy()))][:self.args.top_u]
        u_pb_his = []
        if int(pos_bottom_idx.numpy()) in self.bottom_user_dict: #有些pos b 只在test第一次出现，从training set提取的b_us dict可能不含有该posb
            pb_u = self.bottom_user_dict[int(pos_bottom_idx.numpy())]
        else:
            pb_u = self.popular_users[:self.args.top_u]

        if not self.args.with_self_his:
            if int(user_idx.numpy()) in pb_u: 
                pb_u.remove(int(user_idx.numpy()))  #不去除该sample的结果是正好保留该用户的历史
                
        if len(pb_u) >= self.args.top_u:
            sim_us = random.sample(pb_u, self.args.top_u)
        elif len(pb_u) == 0:
            sim_us = self.popular_users[:self.args.top_u]
        else: # sim_us 长度不一
            sim_us = pb_u

        for u in sim_us:
            u_pb = self.user_bottom_dict[int(u)]
            u_pb_his += u_pb 
        # u_pb_his = list(set(u_pb_his)) #本身出现次数多的应该被更高的概率抽中

        if self.args.repeated_interact:
            if int(pos_bottom_idx.numpy()) in u_pb_his:
                u_pb_his.remove(int(pos_bottom_idx.numpy()))            
        else:
            while int(pos_bottom_idx.numpy()) in u_pb_his:
                u_pb_his.remove(int(pos_bottom_idx.numpy()))

        if self.args.u_pb_num <= len(u_pb_his):
            u_pb_his = random.sample(u_pb_his, self.args.u_pb_num)

        else: 
            if self.args.popular_padding:
                u_pb_his += self.popular_bottoms[ : self.args.u_pb_num-len(u_pb_his)]
            else:
                u_pb_his += [-1] * (self.args.u_pb_num-len(u_pb_his))  #0 padding
        
        ub_key = (int(user_idx.numpy()), int(pos_bottom_idx.numpy()))
        tb_key = (int(top_idx.numpy()), int(pos_bottom_idx.numpy()))
        if ub_key in self.ub_inter_weights_dict:
            ub_weight = self.ub_inter_weights_dict.get(ub_key)
        else: 
            ub_weight = np.array(self.ub_default_weight)

        if tb_key in self.tb_inter_weights_dict: 
            tb_weight = self.tb_inter_weights_dict.get(tb_key)
        else:
            tb_weight = np.array(self.tb_default_weight)

        return user_idx.long(), top_idx.long(), pos_bottom_idx.long(), neg_bottom_idx.long(), torch.LongTensor(u_pb_his), torch.from_numpy(ub_weight), torch.from_numpy(tb_weight)  