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
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from util import config
from tool.util import * #AverageMeter, poly_learning_rate, find_free_port, EarlyStopping
from trainer.loader_APCL import Load_Data
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
    parser = argparse.ArgumentParser(description='APCL')
    parser.add_argument('--config', type=str, default='config/IQON3000_RB.yaml', help='config file')
    parser.add_argument('opts', help='see config/IQON3000_RB.yaml for all options', default=None, nargs=argparse.REMAINDER)
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

def training(device, w_infoNCE, model, train_data_loader, device, optimizer):
    r"""
        using data from Args to train model
        Args:
            mode: -
            train_data_loader: mini-batch iteration
            device: device on which model train
            visual_features: look up table for item visual features
            text_features: look up table for item textural features
            optimizer: optimizer of model
    """
    model.train()
    loss_scalar = 0.
    pos = 0
    for iteration, aBatch in enumerate(train_data_loader):
        aBatch = [x.to(device) for x in aBatch]
        # output = model.fit(aBatch[0], train=True, weight=False)  
        output, infoNCE_v_loss_pc, infoNCE_t_loss_pc, infoNCE_v_loss_ps, infoNCE_t_loss_ps = model.forward(aBatch, train=True)  
        pos += float(torch.sum(output.ge(0)))
        loss = (-logsigmoid(output)).sum() + w_infoNCE * (infoNCE_v_loss_pc + infoNCE_t_loss_pc + infoNCE_v_loss_ps + infoNCE_t_loss_ps)
        iteration += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_scalar += loss.detach().cpu()
        
    return loss_scalar/iteration, pos

def evaluating(device, model, testloader, t_len):
    model.eval()
    pos = 0
    for iteration, aBatch in enumerate(testloader):
        aBatch = [x.to(device) for x in aBatch]
        output, infoNCE_v_loss_pc, infoNCE_t_loss_pc, infoNCE_v_loss_ps, infoNCE_t_loss_ps = model.forward(aBatch, train=False)          
        pos += float(torch.sum(output.ge(0)))

    return pos/t_len

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

    popular_users = user_history["user_idx"].value_counts().to_dict()
    popular_users = list(popular_users.keys())

    bottom_user_dict = user_history.groupby("pos_bottom_idx")["user_idx"].agg(list).to_dict()
    return user_bottoms_dict, user_tops_dict, top_bottoms_dict, popular_bottoms, popular_tops, bottom_user_dict, popular_users

def main():
    args = get_parser()
    logger = get_logger()
    args.device = torch.device("cuda:%s"%args.train_gpu if torch.cuda.is_available() else "cpu")
    # logger.info(args)
    logger.info("=> creating model ...")

    visual_features_tensor = torch.load(args.visual_features_tensor, map_location= lambda a,b:a.cpu())#torch.Size([142737, 2048])
    v_zeros = torch.zeros(visual_features_tensor.size(-1)).unsqueeze(0)
    visual_features_tensor = torch.cat((visual_features_tensor,v_zeros),0)
    if args.with_text:
        text_features_tensor = torch.load(args.textural_features_tensor, map_location= lambda a,b:a.cpu())#torch.Size([142737, 83])
        embedding_weight = load_embedding_weight(args.textural_embedding_matrix)#torch.Size([54276, 300])   
        t_zeros = torch.zeros(text_features_tensor.size(-1)).unsqueeze(0)
        text_features_tensor = torch.cat((text_features_tensor,t_zeros),0)
    else:
        text_features_tensor = None
        embedding_weight = None
    
    user_map = json.load(open(args.user_map))
    item_map = json.load(open(args.item_map)) 
    args.user_num = len(user_map)
    args.item_num = len(item_map)

    if args.arch == 'APCL':
        from Models.BPRs.APCL import APCL
        model = APCL(args, embedding_weight, visual_features_tensor, text_features_tensor)
    elif args.arch == 'NiPCBPR':
        from Models.BPRs.NiPCBPR import NiPCBPR
        model = NiPCBPR(args, embedding_weight, visual_features_tensor, text_features_tensor)
    elif args.arch == 'GPBPR':
        from Models.BPRs.GPBPR import GPBPR
        model = GPBPR(args, embedding_weight, visual_features_tensor, text_features_tensor)
    elif args.arch == 'BPR':
        from Models.BPRs.BPR import BPR
        model = BPR(args.user_num, args.item_num)
    elif args.arch == 'VTBPR':
        from Models.BPRs.VTBPR import VTBPR
        model = VTBPR(args.user_num, args.item_num) 
    elif args.arch == 'CRBPR':
        from Models.BPRs.CRBPR import CRBPR
        model = CRBPR(args, embedding_weight, visual_features_tensor, text_features_tensor) 
    
    model.to(args.device)
    optimizer = Adam([{'params': model.parameters(),'lr': args.base_lr, "weight_decay": args.wd}])
    writer = SummaryWriter(args.save_path)
    logger.info(model)

    if args.weight:
        if os.path.isfile(args.weight):         
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])       
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:          
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):          
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])         
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:          
            logger.info("=> no checkpoint found at '{}'".format(args.resume)) 

    user_bottom_dict, user_top_dict, top_bottoms_dict, popular_bottoms, popular_tops = Get_Data(args.train_data) 

    train_data_ori = load_csv_data(args.train_data)
    train_data_ori  = torch.LongTensor(train_data_ori)
    train_data = Load_Data(args, train_data_ori, user_bottom_dict, user_top_dict, top_bottoms_dict, popular_bottoms, popular_tops, visual_features_tensor, text_features_tensor)


if __name__ == '__main__':
    main()
