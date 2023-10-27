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
    parser.add_argument('--config', type=str, default='config/IQON3000_RT.yaml', help='config file')
    parser.add_argument('opts', help='see config/IQON3000_RT.yaml for all options', default=None, nargs=argparse.REMAINDER)
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

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 8
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # self.val_loss_min = np.Inf
        self.val_auc_max = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_auc, model):

        score = val_auc

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        '''Saves model when validation auc increase.'''
        if self.verbose:
            self.trace_func(f'Validation auc increase ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
        # torch.save(model, data_config['model_file'])
        self.val_auc_max = val_auc

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

def reindex_features(visual_features_ori, text_features_ori, item_map, args):
    visual_features = []
    text_features = []
    id_item_map = {}
    for item in item_map:
        id_item_map[item_map[item]] = item
        
    for iid in range(len(id_item_map)):
        item = str(id_item_map[iid])
        visual_fea = visual_features_ori[int(item)]
#         pdb.set_trace()
        visual_features.append(torch.Tensor(visual_fea))
        if text_features_ori is not None:
            text_fea = text_features_ori[item]
            text_features.append(text_fea)
    torch.save(torch.stack(visual_features, dim=0), args.visual_features_tensor)
    if text_features_ori is not None:
        torch.save(torch.stack(text_features, dim=0), args.textural_features_tensor)
    return visual_features, text_features

# def training(model, train_data_loader, device, optimizer):
#     r"""
#         using data from Args to train model
#         Args:
#             mode: -
#             train_data_loader: mini-batch iteration
#             device: device on which model train
#             visual_features: look up table for item visual features
#             text_features: look up table for item textural features
#             optimizer: optimizer of model
#     """
#     model.train()
#     model = model.to(device)
#     loss_scalar = 0.
#     for iteration, aBatch in enumerate(train_data_loader):
#         aBatch = [x.to(device) for x in aBatch]
#         # output = model.fit(aBatch[0], train=True, weight=False)
#         output = model.forward(aBatch, train=True)   
#         loss = (-logsigmoid(output)).sum() 
#         iteration += 1
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         loss_scalar += loss.detach().cpu()
        
#     return loss_scalar/iteration

def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # main_loss_meter = AverageMeter()
    # aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    # intersection_meter = AverageMeter()
    # union_meter = AverageMeter()
    # # target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    loss_scalar = 0.
    for i, aBatch in enumerate(train_loader):
        data_time.update(time.time() - end)
        # aBatch = [x.to(device) for x in aBatch]
        # output = model.fit(aBatch[0], train=True, weight=False)
        # input = input.cuda(non_blocking=True)
        aBatch = [x.cuda() for x in aBatch]
        output = model.forward(aBatch, train=True)         
        loss = (-logsigmoid(output)).sum() 
        i += 1
        if not args.multiprocessing_distributed:
            loss = torch.mean(loss)   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_scalar += loss.detach().cpu()
        
        if args.multiprocessing_distributed:
            n = len(aBatch[0]) #input.size(0)
            loss = loss.detach() * n 
            # count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss)#, dist.all_reduce(count)
            # n = count.item()
            loss = loss / n

            loss_meter.update(loss.item(), n)
        
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        # current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        # for index in range(0, args.index_split):
        #     optimizer.param_groups[index]['lr'] = current_lr
        # for index in range(args.index_split, len(optimizer.param_groups)):
        #     optimizer.param_groups[index]['lr'] = current_lr * 10
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        # if (i + 1) % args.print_freq == 0 and main_process():
    logger.info('Epoch: [{}/{}][{}/{}] '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'Remain {remain_time} '
                'Loss {loss_meter:.4f} '.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                    batch_time=batch_time,
                                                    data_time=data_time,
                                                    remain_time=remain_time,
                                                    loss_meter=loss_scalar/i))
                                                          
    if main_process():
        writer.add_scalar('loss_train_batch', loss_scalar/i, current_iter)
    return loss_meter.avg

def validate(model, val_loader, t_len):#,val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    model.eval()
    end = time.time()
    pos = 0
    for i, aBatch in enumerate(val_loader):
        data_time.update(time.time() - end)
        # aBatch = [x.to(device) for x in aBatch]
        # output = model.fit(aBatch[0], train=True, weight=False)
        # input = input.cuda(non_blocking=True)
        aBatch = [x.cuda(non_blocking=True) for x in aBatch]
        output = model.forward(aBatch, train=False)          
        pos += float(torch.sum(output.ge(0)))
    # pos_meter.update(pos)
    AUC = pos/t_len
    # return pos/len(testloader)
    batch_time.update(time.time() - end)
    end = time.time()
    # if ((i + 1) % args.print_freq == 0) and main_process():
    logger.info('Test: [{}/{}] '
                'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                    accuracy=AUC))
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

    return user_bottoms_dict, user_tops_dict, top_bottoms_dict, popular_bottoms, popular_tops


    
def main_worker(gpu, ngpus_per_node, argss):#多分布式
    global args
    args = argss
    visual_features_tensor = torch.load(args.visual_features_tensor, map_location= lambda a,b:a.cpu())#torch.Size([142737, 2048])
    # v_zeros = torch.zeros(visual_features_tensor.size(-1)).unsqueeze(0)
    # visual_features_tensor = torch.cat((visual_features_tensor,v_zeros),0)#torch.Size([142738, 2048])

    if args.with_text:
        text_features_tensor = torch.load(args.textural_features_tensor, map_location= lambda a,b:a.cpu())#torch.Size([142737, 83])
        embedding_weight = load_embedding_weight(args.textural_embedding_matrix)#torch.Size([54276, 300])

        # t_zeros = torch.zeros(text_features_tensor.size(-1)).unsqueeze(0)
        # text_features_tensor = torch.cat((text_features_tensor,t_zeros),0)#torch.Size([142738, 83])
        # e_zeros = torch.zeros(embedding_weight.size(-1)).unsqueeze(0).to(conf["device"])
        # embedding_weight = torch.cat((embedding_weight, e_zeros), 0).to(conf["device"])#torch.Size([54277, 300])
    
    else:
        text_features_tensor = None
        embedding_weight = None

    user_map = json.load(open(args.user_map))
    item_map = json.load(open(args.item_map))
  
    args.user_num = len(user_map)
    args.item_num = len(item_map)
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    if args.arch == 'NiPCBPR':#定义模型结构
        from Models.BPRs.NiPCBPR import NiPCBPR
        model = NiPCBPR(args, embedding_weight, visual_features_tensor, text_features_tensor)
    optimizer = Adam([{'params': model.parameters(),'lr': args.base_lr, "weight_decay": args.wd}])
    #     model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion)
    #     modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
    #     modules_new = [model.ppm, model.cls, model.aux]
    # elif args.arch == 'psa':
    #     from model.psanet import PSANet
    #     model = PSANet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, psa_type=args.psa_type,
    #                    compact=args.compact, shrink_factor=args.shrink_factor, mask_h=args.mask_h, mask_w=args.mask_w,
    #                    normalization_factor=args.normalization_factor, psa_softmax=args.psa_softmax, criterion=criterion)
    #     modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
    #     modules_new = [model.psa, model.cls, model.aux]
    # params_list = []#空列表放参数列表，为训练用
    # params_list.append(dict(params=model.parameters(), lr=args.base_lr))
    # # for module in modules_ori:
    # #     params_list.append(dict(params=module.parameters(), lr=args.base_lr))
    # # for module in modules_new:
    # #     params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
    # args.index_split = 5
    # optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # if args.sync_bn:
    #     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if main_process():#打印状态
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info(model)
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume)) 

    user_bottom_dict, user_top_dict, top_bottoms_dict, popular_bottoms, popular_tops = Get_Data(args.train_data) 

    train_data_ori = load_csv_data(args.train_data)
    train_data_ori  = torch.LongTensor(train_data_ori)
    train_data = Load_Data(args, train_data_ori, user_bottom_dict, user_top_dict, top_bottoms_dict, popular_bottoms, popular_tops, visual_features_tensor, text_features_tensor)
    # train_loader = DataLoader(train_data, batch_size=conf["batch_size"], shuffle=True, drop_last=True)

    # train_data = dataset.SemData(split='train', data_root=args.data_root, data_list=args.train_list)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    if args.evaluate:
        valid_data_ori = load_csv_data(args.valid_data)
        valid_data_ori  = torch.LongTensor(valid_data_ori)
        valid_data = Load_Data(args, valid_data_ori, user_bottom_dict, user_top_dict, top_bottoms_dict, popular_bottoms, popular_tops, visual_features_tensor, text_features_tensor)
        # valid_loader = DataLoader(valid_data, batch_size=conf["batch_size"], shuffle=False)
        v_len = len(valid_data_ori)
        # val_data = dataset.SemData(split='val', data_root=args.data_root, data_list=args.val_list, transform=val_transform)
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')#动态调整学习率
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.97 ** epoch) 

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        epoch_log = epoch + 1
        if args.distributed:
            train_sampler.set_epoch(epoch)
        loss_train = train(train_loader, model, optimizer, epoch)
        scheduler.step()

        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            if epoch_log / args.save_freq > 2:
                deletename = args.save_path + '/train_epoch_' + str(epoch_log - args.save_freq * 2) + '.pth'
                os.remove(deletename)
        if args.evaluate:
            AUC, pos = validate(model, val_loader, v_len)
            if main_process():
                writer.add_scalar('AUC', AUC, epoch_log)
                
            early_stopping(AUC, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break 

# def worker_init_fn(worker_id):
#     random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)#


if __name__ == '__main__':
    main()
