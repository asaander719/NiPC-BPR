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
from trainer.loader_APCL import Load_Data, Test_Data
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
from torch.utils.data import DataLoader, Dataset
from tool.metrics import *
from config.configurator import parse_configure

# def get_parser(): 
#     parser = argparse.ArgumentParser(description='APCL')
#     parser.add_argument('--config', type=str, default='config/APCL_Polyvore_RB.yaml', help='config file') #APCL_IQON3000_RB.yaml #APCL_Polyvore_RB.yaml
#     parser.add_argument('opts', help='see config/APCL_Polyvore_RB.yaml for all options', default=None, nargs=argparse.REMAINDER)
#     args = parser.parse_args()
#     assert args.config is not None
#     cfg = config.load_cfg_from_cfg_file(args.config)
#     if args.opts is not None:
#         cfg = config.merge_cfg_from_list(cfg, args.opts)
#     return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def load_embedding_weight(textural_embedding_matrix, device):
    jap2vec = torch.load(textural_embedding_matrix)
    embeding_weight = []
    for jap, vec in jap2vec.items():
        embeding_weight.append(vec.tolist())
    embeding_weight.append(torch.zeros(300))
    embedding_weight = torch.tensor(embeding_weight, device=device)
    return embedding_weight

def training(device, w_infoNCE, model, train_data_loader, optimizer, epoch):
    model.train()
    loss_scalar = 0.
    pos = 0
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    end = time.time()

    for iteration, aBatch in enumerate(train_data_loader):
        aBatch = [x.to(device) for x in aBatch]
        # output = model.fit(aBatch[0], train=True, weight=False) 
        if args.arch == 'APCL':
            output, infoNCE_v_loss_pc, infoNCE_t_loss_pc, infoNCE_v_loss_ps, infoNCE_t_loss_ps = model.forward(aBatch, train=True)  
            loss = (-logsigmoid(output)).sum() + w_infoNCE * (infoNCE_v_loss_pc + infoNCE_t_loss_pc + infoNCE_v_loss_ps + infoNCE_t_loss_ps)
        else:
            output = model.forward(aBatch, train=True)
            loss = (-logsigmoid(output)).sum() 
        pos += float(torch.sum(output.ge(0)))
        iteration += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_scalar += loss.detach().cpu()

    end = time.time()
    logger.info('Epoch: [{}/{}][{}/{}] '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Loss {loss_meter:.4f} '
                'AUC_NUM: {AUC_NUM: }'.format(epoch+1, args.epochs, iteration + 1, len(train_data_loader),
                                                    data_time=data_time,
                                                    loss_meter=loss_scalar/iteration,
                                                    AUC_NUM=pos))    
    return loss_scalar/iteration, pos


def validate(device, model, val_loader, t_len): #for wide infer
    logger.info('>>>>>>>>>>>>>>>> Start Wide Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.eval()
    end = time.time()
    pos = 0
    preds = []
    for iteration, aBatch in enumerate(val_loader):
        aBatch = [x.to(device) for x in aBatch]
        scores = model.inference(aBatch, train=False)          
        # pos += float(torch.sum(output.ge(0)))
        _, tops = torch.topk(scores, k=args.metric_topk, dim=-1)
        preds.append(tops)

    preds = torch.cat(preds, dim=0)
    bs = preds.size(0)
    grd = [0] * bs
    grd_cnt = [1] * bs
    metrics = {}
    for topk in args.k: #args.k:
        metrics[topk] = {}
        REC, MRR, NDCG = get_metrics(grd, grd_cnt, preds.cpu().numpy(), topk)
        metrics[topk]['recall'] = REC
        metrics[topk]['mrr'] = MRR
        metrics[topk]['ndcg'] = NDCG
    # AUC = pos/t_len
    batch_time.update(time.time() - end)
    end = time.time()
    metric_strings = []
    for m in args.metrics:
        for k in args.k:
            metric_strings.append('{}@{}: {:.4f}'.format(m, k, metrics[k][m]))
    logger.info(', '.join(metric_strings))
    return metrics, preds

def validate_AUC(device, model, val_loader, t_len): #For AUC infer
    logger.info('>>>>>>>>>>>>>>>> Start AUC Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.eval()
    end = time.time()
    pos = 0
    for i, aBatch in enumerate(val_loader):
        data_time.update(time.time() - end)
        aBatch = [x.to(device) for x in aBatch]
        if args.arch == 'APCL':
            output, infoNCE_v_loss_pc, infoNCE_t_loss_pc, infoNCE_v_loss_ps, infoNCE_t_loss_ps = model.forward(aBatch, train=False) 
        else:
            output = model.forward(aBatch, train=False)             
        pos += float(torch.sum(output.ge(0)))
    AUC = pos/t_len
    # return pos/len(testloader)
    batch_time.update(time.time() - end)
    end = time.time()
    # if ((i + 1) % args.print_freq == 0) and main_process():
    logger.info('Test: [{}/{}] '
                'Accuracy {accuracy:.4f} '
                'AUC_NUM: {AUC_NUM: }'.format(i + 1, len(val_loader),
                                                    accuracy=AUC,
                                                    AUC_NUM=pos))
    return AUC, pos

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

def interaction_weight(train_data):
    interactions = pd.read_csv(train_data,header=None).astype('int')
    interactions.columns=["user_idx", "top_idx", "pos_bottom_idx", "neg_bottom_idx"]
    ub_counts = interactions.groupby(["user_idx", "pos_bottom_idx"]).size().reset_index(name='counts')
    ub_counts['inter_weights'] = 1 / np.sqrt(ub_counts['counts'])
    tb_counts = interactions.groupby(["top_idx", "pos_bottom_idx"]).size().reset_index(name='counts')
    tb_counts['inter_weights']  = 1 / np.sqrt(tb_counts['counts'])

    ub_inter_weights_dict = {(int(row['user_idx']), int(row["pos_bottom_idx"])): np.array(row['inter_weights']) for _, row in ub_counts.iterrows()}
    tb_inter_weights_dict = {(int(row["top_idx"]), int(row["pos_bottom_idx"])): np.array(row['inter_weights']) for _, row in tb_counts.iterrows()}
    # for cold-start problems, unseen data in test data, assign defualt median weight
    ub_default_weight = np.median(ub_counts['inter_weights'])
    tb_default_weight = np.median(tb_counts['inter_weights'])
    return ub_inter_weights_dict, tb_inter_weights_dict, ub_default_weight, tb_default_weight 

def main():
    global logger, writer, args
    # args = get_parser()
    args = parse_configure()
    logger = get_logger()
    
    # args.device = torch.device("cuda:%s"%args.cuda if torch.cuda.is_available() else "cpu")
    # logger.info(args)
    logger.info("=> creating model ...")

    visual_features_tensor = torch.load(args.visual_features_tensor, map_location= lambda a,b:a.cpu())#torch.Size([142737, 2048])
    v_zeros = torch.zeros(visual_features_tensor.size(-1)).unsqueeze(0)
    visual_features_tensor = torch.cat((visual_features_tensor,v_zeros),0)
    # visual_features_tensor.to(args.device)
    
    if args.with_text:
        text_features_tensor = torch.load(args.textural_features_tensor, map_location= lambda a,b:a.cpu())#torch.Size([142737, 83])       
        t_zeros = torch.zeros(text_features_tensor.size(-1)).unsqueeze(0)
        text_features_tensor = torch.cat((text_features_tensor,t_zeros),0)
        # text_features_tensor.to(args.device)
        if args.dataset == 'IQON3000':
            embedding_weight = load_embedding_weight(args.textural_embedding_matrix, args.device)#torch.Size([54276, 300])
        else:
            embedding_weight = None
    else:
        text_features_tensor = None
        embedding_weight = None

    user_map = json.load(open(args.user_map))
    item_map = json.load(open(args.item_map)) 
    args.user_num = len(user_map)
    args.item_num = len(item_map)
    ub_inter_weights_dict, tb_inter_weights_dict, ub_default_weight, tb_default_weight  = interaction_weight(args.train_data)
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
    # writer = SummaryWriter(args.save_path)
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

    user_bottoms_dict, user_tops_dict, top_bottoms_dict, popular_bottoms, popular_tops, bottom_user_dict, popular_users = Get_Data(args.train_data) 

    train_data_ori = load_csv_data(args.train_data)
    train_data_ori  = torch.LongTensor(train_data_ori)
    train_data = Load_Data(args, train_data_ori, user_bottoms_dict, user_tops_dict, top_bottoms_dict, popular_bottoms, popular_tops, 
        bottom_user_dict, popular_users, ub_inter_weights_dict, tb_inter_weights_dict, ub_default_weight, tb_default_weight)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_len = len(train_data_ori)

    test_data_ori = load_csv_data(args.test_data)
    test_data_ori  = torch.LongTensor(test_data_ori)
    test_data = Test_Data(args, test_data_ori, user_bottoms_dict, user_tops_dict, top_bottoms_dict, popular_bottoms, popular_tops, 
        bottom_user_dict, popular_users, ub_inter_weights_dict, tb_inter_weights_dict, ub_default_weight, tb_default_weight)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)
    t_len = len(test_data_ori)

    valid_data_ori = load_csv_data(args.valid_data)
    valid_data_ori  = torch.LongTensor(valid_data_ori)
    valid_data = Test_Data(args, valid_data_ori, user_bottoms_dict, user_tops_dict, top_bottoms_dict, popular_bottoms, popular_tops, 
        bottom_user_dict, popular_users, ub_inter_weights_dict, tb_inter_weights_dict, ub_default_weight, tb_default_weight)
    valid_loader = DataLoader(valid_data, batch_size=args.test_batch_size, shuffle=False)
    v_len = len(valid_data_ori)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("model para. Num:", params)
  
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')#动态调整学习率
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.97 ** epoch) 

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        epoch_log = epoch + 1
        loss_train, train_auc_num  = training(args.device, args.w_infoNCE, model, train_loader, optimizer, epoch)
        scheduler.step()

        # writer.add_scalar('loss_train', loss_train, epoch_log)

        if (epoch_log % args.save_freq == 0):
            args.save_path = './saved/' + args.dataset 
            filename = args.save_path + '/'+ args.arch  + '_' + args.mode  + '_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            if epoch_log / args.save_freq > 2:
                deletename = args.save_path + '/'+ args.arch + '_' + args.mode  + '_' + str(epoch_log - args.save_freq * 2) + '.pth'
                os.remove(deletename)

        metrics, preds = validate(args.device, model, test_loader, t_len)
        if args.evaluate:
            args.wide_evaluate = False
            AUC_v, pos_v = validate_AUC(args.device, model, valid_loader, v_len)
            # writer.add_scalar('AUC', AUC_v, epoch_log)
            if args.early_stop:
                early_stopping(AUC_v, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break 
          

if __name__ == '__main__':
    main()
