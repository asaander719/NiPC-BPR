import os
import random
import time
import numpy as np
import json
import logging
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.nn.functional import logsigmoid
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from util import config
from tool.util import AverageMeter, poly_learning_rate, find_free_port
from trainer.loader_iqon import Load_Data
import csv
from torch.optim import Adam
from sys import argv
import json
import pdb
from torch.nn import *
import random
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

def get_parser(): 
    parser = argparse.ArgumentParser(description='Recommendation of Mix-and-Match Clothing by Modeling Indirect Personal Compatibility')
    parser.add_argument('--config', type=str, default='config/Polyvore_RB.yaml', help='config file')
    parser.add_argument('opts', help='see config/Polyvore_RB.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def load_csv_data(train_data_path):
    result = []
    with open(train_data_path,'r') as fp:
        for line in fp:
            t = line.strip().split(',')
            t = [int(i) for i in t]
            result.append(t)
    return result

def load_embedding_weight(textural_embedding_matrix):
    jap2vec = torch.load(textural_embedding_matrix)
    embeding_weight = []
    for jap, vec in jap2vec.items():
        embeding_weight.append(vec.tolist())
    embeding_weight.append(torch.zeros(300))
    embedding_weight = torch.tensor(embeding_weight).cuda()
    return embedding_weight

def Get_Data(train_data_file):
    user_history = pd.read_csv(train_data_file, header=None).astype('int')
    user_history.columns=["user_idx", "top_idx", "pos_bottom_idx", "neg_bottom_idx"]
    user_bottoms_dict = user_history.groupby("user_idx")["pos_bottom_idx"].agg(list).to_dict()
    user_tops_dict = user_history.groupby("user_idx")["top_idx"].agg(list).to_dict()
    top_bottoms_dict = user_history.groupby("top_idx")["pos_bottom_idx"].agg(list).to_dict()
    popular_bottoms = user_history["pos_bottom_idx"].value_counts().to_dict()
    popular_bottoms = list(popular_bottoms.keys())
    popular_tops = user_history["top_idx"].value_counts().to_dict()
    popular_tops = list(popular_tops.keys())
    return user_bottoms_dict, user_tops_dict, top_bottoms_dict, popular_bottoms, popular_tops

class F_TEST():
    """Performance comparison under different product interaction frequencies"""
    def __init__(self, args):
        self.args = args
        self.train_df = pd.read_csv(args.train_data,header=None).astype('int')
        self.test_df = pd.read_csv(args.test_data,header=None).astype('int')
        self.train_df.columns = ["user_idx", "top_idx", "pos_bottom_idx", "neg_bottom_idx"]
        self.train_bf_df = self.train_df["pos_bottom_idx"].value_counts().reset_index()
        self.train_bf_df.columns = ["pos_bottom_idx","frequent"]
        self.bottom_num = self.train_df["pos_bottom_idx"].nunique()

    def get_train_freq_list(self):
        f_0_2_df = self.train_bf_df[self.train_bf_df["frequent"] <= 2]
        f_0_2_list = list(f_0_2_df["pos_bottom_idx"])
        # logger.info('Unique Num of f-> [0,2] in training:{}'.format(len(f_0_2_list)))
        f_3_5_df = self.train_bf_df[self.train_bf_df["frequent"] >2]
        f_3_5_df = f_3_5_df[f_3_5_df["frequent"] <= 5]
        f_3_5_list = list(f_3_5_df["pos_bottom_idx"])
        # logger.info('Unique Num of f-> [3,5] in training:{}'.format(len(f_3_5_list)))
        f_6_10_df = self.train_bf_df[self.train_bf_df["frequent"] >5]
        f_6_10_df = f_6_10_df[f_6_10_df["frequent"] <= 10]
        f_6_10_list = list(f_6_10_df["pos_bottom_idx"])
        # logger.info('Unique Num of f-> [6,10] in training:{}'.format(len(f_6_10_list)))
        f_11___df = self.train_bf_df[self.train_bf_df["frequent"] > 10]
        f_11___list = list(f_11___df["pos_bottom_idx"])
        # logger.info('Unique Num of f-> [11,) in training:{}'.format(len(f_11___list)))
        return f_0_2_list, f_3_5_list, f_6_10_list, f_11___list

    def get_freq_test(self, bottom_list):
        result = []
        with open(self.args.test_data,'r') as fp:
            for line in fp:
                data = line.strip().split(',')
                data = [int(i) for i in data]
                i,t,pb,nb = data
                if pb in bottom_list:
                    result.append(data)
        return result

    def get_f_0_2(self, bottom_list):
        result = []
        train_bottom = list(self.train_df["pos_bottom_idx"].unique())
        with open(self.args.test_data,'r') as fp:
            for line in fp:
                data = line.strip().split(',')
                data = [int(i) for i in data]
                i,t,pb,nb = data
                if pb in bottom_list or pb not in train_bottom:
                    result.append(data)
        return result

    def f_evaluate(self, model, f_loader, f_len):      
        model.eval()
        end = time.time()
        pos = 0
        for i, aBatch in enumerate(f_loader):
            aBatch = [x.cuda(non_blocking=True) for x in aBatch]
            output = model.forward(aBatch, train=False)          
            pos += float(torch.sum(output.ge(0)))
        AUC = pos/f_len
        return AUC, pos

def Simple_Load_Data(args, data_ori, user_bottom_dict, user_top_dict, top_bottoms_dict, 
        popular_bottoms, popular_tops, visual_features_tensor, text_features_tensor):
    data_ori  = torch.LongTensor(data_ori)
    loaded_data = Load_Data(args, data_ori, user_bottom_dict, user_top_dict, top_bottoms_dict, popular_bottoms, popular_tops, visual_features_tensor, text_features_tensor)
    loaded_loader = torch.utils.data.DataLoader(loaded_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    t_len = len(loaded_data)
    return loaded_loader, t_len 

def test(model, test_loader, t_len):
    logger.info('>>>>>>>>>>>>>>>> Start Test >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.eval()
    end = time.time()
    pos = 0
    for i, aBatch in enumerate(test_loader):
        data_time.update(time.time() - end)
        aBatch = [x.cuda(non_blocking=True) for x in aBatch]
        output = model.forward(aBatch, train=False)          
        pos += float(torch.sum(output.ge(0)))
    AUC = pos/t_len
    batch_time.update(time.time() - end)
    end = time.time()
    logger.info('Test: [{}/{}] '
                'Right NUM {Right_NUM:.2f}. '
                'Accuracy {accuracy:.4f}.'.format(i + 1, len(test_loader),
                                                    Right_NUM = pos,
                                                    accuracy=AUC))
    logger.info('<<<<<<<<<<<<<<<<< End Test <<<<<<<<<<<<<<<<<')

def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    Fre_TEST = F_TEST(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)
    logger.info("=> loading features ...")

    visual_features_tensor = torch.load(args.visual_features_tensor, map_location= lambda a,b:a.cpu())#torch.Size([142737, 2048])
    
    if args.with_text:
        text_features_tensor = torch.load(args.textural_features_tensor, map_location= lambda a,b:a.cpu())#torch.Size([142737, 83])
        embedding_weight = load_embedding_weight(args.textural_embedding_matrix)#torch.Size([54276, 300])   
    else:
        text_features_tensor = None
        embedding_weight = None

    user_map = json.load(open(args.user_map))
    item_map = json.load(open(args.item_map))
  
    args.user_num = len(user_map)
    args.item_num = len(item_map)
    if args.test_distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    user_bottom_dict, user_top_dict, top_bottoms_dict, popular_bottoms, popular_tops = Get_Data(args.train_data) 
    test_data_ori = load_csv_data(args.test_data)
    # test_data_ori  = torch.LongTensor(test_data_ori)
    # test_data = Load_Data(args, test_data_ori, user_bottom_dict, user_top_dict, top_bottoms_dict, popular_bottoms, popular_tops, visual_features_tensor, text_features_tensor)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    # t_len = len(test_data)
    test_loader, t_len = Simple_Load_Data(args, test_data_ori, user_bottom_dict, user_top_dict, top_bottoms_dict, popular_bottoms, popular_tops, visual_features_tensor, text_features_tensor)
    print("test data len:", t_len)
    # names = [line.rstrip('\n') for line in open(args.names_path)]

    if args.arch == 'NiPCBPR':
        from Models.BPRs.NiPCBPR import NiPCBPR
        logger.info("=> loading model ...")
        model = NiPCBPR(args, embedding_weight, visual_features_tensor, text_features_tensor)
        logger.info(model)
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        if os.path.isfile(args.model_path):
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logger.info("=> loaded checkpoint '{}'".format(args.model_path))
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

        test(model, test_loader, t_len)

        if args.f_tset:
            logger.info('>>>>>>>> Start Evaluate under different product interaction frequencies >>>>>>>')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            f_0_2_list, f_3_5_list, f_6_10_list, f_11___list = Fre_TEST.get_train_freq_list()
            f_0_2_test = Fre_TEST.get_f_0_2(f_0_2_list)
            f_3_5_test = Fre_TEST.get_freq_test(f_3_5_list)
            f_6_10_test = Fre_TEST.get_freq_test(f_6_10_list)
            f_11___test = Fre_TEST.get_freq_test(f_11___list)

            f_0_2, f_0_2_len = Simple_Load_Data(args, f_0_2_test, user_bottom_dict, user_top_dict, top_bottoms_dict, popular_bottoms, popular_tops, visual_features_tensor, text_features_tensor)
            # logger.info('f_0_2_len:{}'.format(f_0_2_len))
            f_3_5, f_3_5_len = Simple_Load_Data(args, f_3_5_test, user_bottom_dict, user_top_dict, top_bottoms_dict, popular_bottoms, popular_tops, visual_features_tensor, text_features_tensor)
            # logger.info('f_3_5_len:{}'.format(f_3_5_len))
            f_6_10, f_6_10_len = Simple_Load_Data(args, f_6_10_test, user_bottom_dict, user_top_dict, top_bottoms_dict, popular_bottoms, popular_tops, visual_features_tensor, text_features_tensor)
            # logger.info('f_6_10_len:{}'.format(f_6_10_len))
            f_11__, f_11___len = Simple_Load_Data(args, f_11___test, user_bottom_dict, user_top_dict, top_bottoms_dict, popular_bottoms, popular_tops, visual_features_tensor, text_features_tensor)
            # logger.info('f_11___len:{}'.format(f_11___len))

            f_0_2_auc, f_0_2_num = Fre_TEST.f_evaluate(model, f_0_2, f_0_2_len)
            logger.info('Evaluation AUC of f-> [0, 2]:{accuracy:.4f}'.format(accuracy= f_0_2_auc))
            # print("f_0_2_auc: %.4f, pos_f_0_2_num: %d"%(f_0_2_auc, f_0_2_num))

            f_3_5_auc, f_3_5_num = Fre_TEST.f_evaluate(model, f_3_5,f_3_5_len)
            logger.info('Evaluation AUC of f-> [3, 5]:{accuracy:.4f}'.format(accuracy= f_3_5_auc))
            # print("f_3_5_auc: %.4f, pos_f_3_5_num: %d"%(f_3_5_auc, f_3_5_num))

            f_6_10_auc, f_6_10_num = Fre_TEST.f_evaluate(model, f_6_10,f_6_10_len)
            logger.info('Evaluation AUC of f-> [6, 10]:{accuracy:.4f}'.format(accuracy= f_6_10_auc))
            # print("f_6_10_auc: %.4f, pos_f_6_10_num: %d"%(f_6_10_auc, f_6_10_num))

            f_11__auc, f_11__num = Fre_TEST.f_evaluate(model, f_11__, f_11___len)
            logger.info('Evaluation AUC of f-> [11, ~]:{accuracy:.4f}'.format(accuracy= f_11__auc))
            # print("f_11__auc: %.4f, pos_f_11__num: %d"%(f_11__auc, f_11__num))

            logger.info('<<<<<<<<<<<<<<<<< End  <<<<<<<<<<<<<<<<<')

        
if __name__ == '__main__':
    main()