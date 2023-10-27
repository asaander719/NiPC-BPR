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

def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
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
    test_data_ori  = torch.LongTensor(test_data_ori)
    test_data = Load_Data(args, test_data_ori, user_bottom_dict, user_top_dict, top_bottoms_dict, popular_bottoms, popular_tops, visual_features_tensor, text_features_tensor)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    t_len = len(test_data)
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

if __name__ == '__main__':
    main()