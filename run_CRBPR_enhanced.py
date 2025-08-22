import os
import time
import numpy as np
import json
import logging
import argparse
import torch
from torch.nn.functional import logsigmoid
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from util import config
from tool.util import *
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
from torch.utils.data import DataLoader, Dataset
from tool.metrics import *
from config.configurator import parse_configure
import psutil
import GPUtil
from thop import profile, clever_format


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


def count_model_parameters(model):
    """Count total and trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_details = {}
    for name, param in model.named_parameters():
        param_details[name] = {
            'shape': list(param.shape),
            'num_params': param.numel(),
            'requires_grad': param.requires_grad
        }
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_details': param_details
    }


def measure_computational_cost(model, sample_batch, device):
    """Measure FLOPs and memory usage"""
    model.eval()
    
    # Create sample input for FLOPs calculation
    with torch.no_grad():
        # Measure FLOPs
        try:
            flops, params = profile(model, inputs=(sample_batch,), verbose=False)
            flops_readable, params_readable = clever_format([flops, params], "%.3f")
        except Exception as e:
            logger.warning(f"Could not measure FLOPs: {e}")
            flops_readable, params_readable = "N/A", "N/A"
    
    # Measure GPU memory usage
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_memory_used = torch.cuda.memory_allocated(device) / 1024**3  # GB
        gpu_memory_cached = torch.cuda.memory_reserved(device) / 1024**3  # GB
    else:
        gpu_memory_used = 0
        gpu_memory_cached = 0
    
    # Measure CPU memory usage
    process = psutil.Process(os.getpid())
    cpu_memory_used = process.memory_info().rss / 1024**3  # GB
    
    return {
        'flops': flops_readable,
        'params_from_profile': params_readable,
        'gpu_memory_used_gb': gpu_memory_used,
        'gpu_memory_cached_gb': gpu_memory_cached,
        'cpu_memory_used_gb': cpu_memory_used
    }


def training(device, model, train_data_loader, optimizer, epoch):
    """Enhanced training function with timing and performance metrics"""
    model.train()
    loss_scalar = 0.
    pos = 0
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    
    end = time.time()
    epoch_start_time = time.time()

    for iteration, aBatch in enumerate(train_data_loader):
        data_load_time = time.time() - end
        data_time.update(data_load_time)
        
        aBatch = [x.to(device) for x in aBatch]
        
        # Forward pass timing
        forward_start = time.time()
        output = model.forward(aBatch, train=True)
        loss = (-logsigmoid(output)).sum()
        forward_end = time.time()
        forward_time.update(forward_end - forward_start)
        
        pos += float(torch.sum(output.ge(0)))
        iteration += 1
        
        # Backward pass timing
        backward_start = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_end = time.time()
        backward_time.update(backward_end - backward_start)
        
        loss_scalar += loss.detach().cpu()
        end = time.time()

    epoch_total_time = time.time() - epoch_start_time
    
    logger.info('Epoch: [{}/{}][{}/{}] '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Forward {forward_time.val:.3f} ({forward_time.avg:.3f}) '
                'Backward {backward_time.val:.3f} ({backward_time.avg:.3f}) '
                'Loss {loss_meter:.4f} '
                'AUC_NUM: {AUC_NUM: } '
                'Epoch Time: {epoch_time:.3f}s'.format(
                    epoch+1, args.epochs, iteration + 1, len(train_data_loader),
                    data_time=data_time,
                    forward_time=forward_time,
                    backward_time=backward_time,
                    loss_meter=loss_scalar/iteration,
                    AUC_NUM=pos,
                    epoch_time=epoch_total_time))
    
    return {
        'loss': loss_scalar/iteration,
        'auc_num': pos,
        'epoch_time': epoch_total_time,
        'avg_forward_time': forward_time.avg,
        'avg_backward_time': backward_time.avg,
        'avg_data_time': data_time.avg
    }


def validate(device, model, val_loader, t_len):
    """Enhanced validation with timing metrics"""
    logger.info('>>>>>>>>>>>>>>>> Start Wide Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    inference_time = AverageMeter()
    
    model.eval()
    validation_start_time = time.time()
    end = time.time()
    pos = 0
    preds = []
    
    for iteration, aBatch in enumerate(val_loader):
        data_load_time = time.time() - end
        data_time.update(data_load_time)
        
        aBatch = [x.to(device) for x in aBatch]
        
        # Inference timing
        inference_start = time.time()
        scores = model.inference(aBatch, train=False)
        inference_end = time.time()
        inference_time.update(inference_end - inference_start)
        
        _, tops = torch.topk(scores, k=args.metric_topk, dim=-1)
        preds.append(tops)
        
        batch_time.update(time.time() - end)
        end = time.time()

    total_validation_time = time.time() - validation_start_time
    
    preds = torch.cat(preds, dim=0)
    bs = preds.size(0)
    grd = [0] * bs
    grd_cnt = [1] * bs
    metrics = {}
    
    for topk in args.k:
        metrics[topk] = {}
        REC, MRR, NDCG = get_metrics(grd, grd_cnt, preds.cpu().numpy(), topk)
        metrics[topk]['recall'] = REC
        metrics[topk]['mrr'] = MRR
        metrics[topk]['ndcg'] = NDCG
    
    metric_strings = []
    for m in args.metrics:
        for k in args.k:
            metric_strings.append('{}@{}: {:.4f}'.format(m, k, metrics[k][m]))
    
    logger.info('Validation Time: {:.3f}s, Avg Inference Time: {:.4f}s, {}'.format(
        total_validation_time, inference_time.avg, ', '.join(metric_strings)))
    
    return metrics, preds, {
        'total_validation_time': total_validation_time,
        'avg_inference_time': inference_time.avg,
        'avg_data_time': data_time.avg
    }


def validate_AUC(device, model, val_loader, t_len):
    """Enhanced AUC validation with timing"""
    logger.info('>>>>>>>>>>>>>>>> Start AUC Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    inference_time = AverageMeter()
    
    model.eval()
    auc_start_time = time.time()
    end = time.time()
    pos = 0
    
    for i, aBatch in enumerate(val_loader):
        data_load_time = time.time() - end
        data_time.update(data_load_time)
        
        aBatch = [x.to(device) for x in aBatch]
        
        inference_start = time.time()
        output = model.forward(aBatch, train=False)
        inference_end = time.time()
        inference_time.update(inference_end - inference_start)
        
        pos += float(torch.sum(output.ge(0)))
        batch_time.update(time.time() - end)
        end = time.time()
    
    total_auc_time = time.time() - auc_start_time
    AUC = pos/t_len
    
    logger.info('AUC Test: [{}/{}] '
                'Accuracy {accuracy:.4f} '
                'AUC_NUM: {AUC_NUM: } '
                'Total Time: {total_time:.3f}s '
                'Avg Inference: {avg_inference:.4f}s'.format(
                    i + 1, len(val_loader),
                    accuracy=AUC,
                    AUC_NUM=pos,
                    total_time=total_auc_time,
                    avg_inference=inference_time.avg))
    
    return AUC, pos, {
        'total_auc_time': total_auc_time,
        'avg_inference_time': inference_time.avg
    }


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


def interaction_weight(train_data):
    interactions = pd.read_csv(train_data,header=None).astype('int')
    interactions.columns=["user_idx", "top_idx", "pos_bottom_idx", "neg_bottom_idx"]
    ub_counts = interactions.groupby(["user_idx", "pos_bottom_idx"]).size().reset_index(name='counts')
    ub_counts['inter_weights'] = 1 / np.sqrt(ub_counts['counts'])
    tb_counts = interactions.groupby(["top_idx", "pos_bottom_idx"]).size().reset_index(name='counts')
    tb_counts['inter_weights']  = 1 / np.sqrt(tb_counts['counts'])

    ub_inter_weights_dict = {(int(row['user_idx']), int(row["pos_bottom_idx"])): np.array(row['inter_weights']) for _, row in ub_counts.iterrows()}
    tb_inter_weights_dict = {(int(row["top_idx"]), int(row["pos_bottom_idx"])): np.array(row['inter_weights']) for _, row in tb_counts.iterrows()}
    ub_default_weight = np.median(ub_counts['inter_weights'])
    tb_default_weight = np.median(tb_counts['inter_weights'])
    return ub_inter_weights_dict, tb_inter_weights_dict, ub_default_weight, tb_default_weight


def save_performance_report(args, param_info, computational_cost, training_metrics, validation_metrics):
    """Save comprehensive performance report"""
    report = {
        'model_info': {
            'architecture': args.arch,
            'dataset': args.dataset,
            'mode': args.mode,
            'hidden_dim': args.hidden_dim,
            'batch_size': args.batch_size,
            'learning_rate': args.base_lr,
            'weight_decay': args.wd,
            'epochs': args.epochs
        },
        'parameter_info': param_info,
        'computational_cost': computational_cost,
        'training_metrics': training_metrics,
        'validation_metrics': validation_metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    # Save detailed report
    report_filename = f'reports/CRBPR_performance_report_{args.dataset}_{args.mode}_{time.strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Performance report saved to: {report_filename}")
    
    # Print summary
    logger.info("="*80)
    logger.info("CRBPR MODEL PERFORMANCE SUMMARY")
    logger.info("="*80)
    logger.info(f"Model: {args.arch}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Mode: {args.mode}")
    logger.info("-"*50)
    logger.info("PARAMETER STATISTICS:")
    logger.info(f"  Total Parameters: {param_info['total_params']:,}")
    logger.info(f"  Trainable Parameters: {param_info['trainable_params']:,}")
    logger.info(f"  Parameter Size: {param_info['trainable_params'] * 4 / 1024**2:.2f} MB (float32)")
    logger.info("-"*50)
    logger.info("COMPUTATIONAL COST:")
    logger.info(f"  FLOPs: {computational_cost['flops']}")
    logger.info(f"  GPU Memory Used: {computational_cost['gpu_memory_used_gb']:.3f} GB")
    logger.info(f"  CPU Memory Used: {computational_cost['cpu_memory_used_gb']:.3f} GB")
    logger.info("-"*50)
    logger.info("TIMING METRICS:")
    if training_metrics:
        logger.info(f"  Avg Training Time per Epoch: {np.mean([m['epoch_time'] for m in training_metrics]):.3f}s")
        logger.info(f"  Avg Forward Pass Time: {np.mean([m['avg_forward_time'] for m in training_metrics]):.4f}s")
        logger.info(f"  Avg Backward Pass Time: {np.mean([m['avg_backward_time'] for m in training_metrics]):.4f}s")
    if validation_metrics:
        logger.info(f"  Avg Validation Time: {np.mean([m['total_validation_time'] for m in validation_metrics]):.3f}s")
        logger.info(f"  Avg Inference Time: {np.mean([m['avg_inference_time'] for m in validation_metrics]):.4f}s")
    logger.info("="*80)


def main():
    global logger, writer, args
    args = parse_configure()
    logger = get_logger()
    
    logger.info("=> creating CRBPR model ...")
    
    # Load features
    visual_features_tensor = torch.load(args.visual_features_tensor, map_location=lambda a,b:a.cpu())
    v_zeros = torch.zeros(visual_features_tensor.size(-1)).unsqueeze(0)
    visual_features_tensor = torch.cat((visual_features_tensor,v_zeros),0)
    
    if args.with_text:
        text_features_tensor = torch.load(args.textural_features_tensor, map_location=lambda a,b:a.cpu())
        t_zeros = torch.zeros(text_features_tensor.size(-1)).unsqueeze(0)
        text_features_tensor = torch.cat((text_features_tensor,t_zeros),0)
        if args.dataset == 'IQON3000':
            embedding_weight = load_embedding_weight(args.textural_embedding_matrix, args.device)
        else:
            embedding_weight = None
    else:
        text_features_tensor = None
        embedding_weight = None

    user_map = json.load(open(args.user_map))
    item_map = json.load(open(args.item_map))
    args.user_num = len(user_map)
    args.item_num = len(item_map)
    
    ub_inter_weights_dict, tb_inter_weights_dict, ub_default_weight, tb_default_weight = interaction_weight(args.train_data)
    
    # Initialize CRBPR model
    from Models.BPRs.CRBPR import CRBPR
    model = CRBPR(args, embedding_weight, visual_features_tensor, text_features_tensor)
    model.to(args.device)
    
    # Count parameters
    param_info = count_model_parameters(model)
    logger.info(f"CRBPR Model Parameters:")
    logger.info(f"  Total: {param_info['total_params']:,}")
    logger.info(f"  Trainable: {param_info['trainable_params']:,}")
    
    optimizer = Adam([{'params': model.parameters(),'lr': args.base_lr, "weight_decay": args.wd}])
    logger.info(model)

    # Load weights if specified
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

    # Prepare data
    user_bottom_dict, user_top_dict, top_bottoms_dict, popular_bottoms, popular_tops = Get_Data(args.train_data)

    train_data_ori = load_csv_data(args.train_data)
    train_data_ori = torch.LongTensor(train_data_ori)
    train_data = Load_Data(args, train_data_ori, user_bottom_dict, user_top_dict, top_bottoms_dict, popular_bottoms, popular_tops, visual_features_tensor, text_features_tensor)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_len = len(train_data_ori)

    valid_data_ori = load_csv_data(args.valid_data)
    valid_data_ori = torch.LongTensor(valid_data_ori)
    valid_data = Load_Data(args, valid_data_ori, user_bottom_dict, user_top_dict, top_bottoms_dict, popular_bottoms, popular_tops, visual_features_tensor, text_features_tensor)
    valid_loader = DataLoader(valid_data, batch_size=args.test_batch_size, shuffle=False)
    v_len = len(valid_data_ori)

    test_data_ori = load_csv_data(args.test_data)
    test_data_ori = torch.LongTensor(test_data_ori)
    test_data = Load_Data(args, test_data_ori, user_bottom_dict, user_top_dict, top_bottoms_dict, popular_bottoms, popular_tops, visual_features_tensor, text_features_tensor)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)
    t_len = len(test_data_ori)

    # Measure computational cost with sample batch
    sample_batch = next(iter(train_loader))
    sample_batch = [x.to(args.device) for x in sample_batch]
    computational_cost = measure_computational_cost(model, sample_batch, args.device)
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.97 ** epoch)
    
    logger.info(args)
    
    # Training and validation metrics storage
    training_metrics = []
    validation_metrics = []
    
    total_training_start = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        epoch_log = epoch + 1
        
        # Training with metrics
        train_metrics = training(args.device, model, train_loader, optimizer, epoch)
        training_metrics.append(train_metrics)
        scheduler.step()

        # Validation with metrics
        test_metrics, preds, val_timing = validate(args.device, model, test_loader, t_len)
        validation_metrics.append(val_timing)
        
        if args.evaluate:
            args.wide_evaluate = False
            AUC_v, pos_v, auc_timing = validate_AUC(args.device, model, valid_loader, v_len)
            
            if args.early_stop:
                early_stopping(AUC_v, model)
                if early_stopping.early_stop:
                    logger.info("Early stopping triggered")
                    break
    
    total_training_time = time.time() - total_training_start
    logger.info(f"Total Training Time: {total_training_time:.3f}s")
    
    # Save comprehensive performance report
    save_performance_report(args, param_info, computational_cost, training_metrics, validation_metrics)


if __name__ == '__main__':
    main()