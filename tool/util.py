import os
import numpy as np
import torch
from torch import nn
import torch.nn.init as initer

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.97 ** epoch) 
    return lr


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


# def intersectionAndUnion(output, target, K, ignore_index=255):
#     # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
#     assert (output.ndim in [1, 2, 3])
#     assert output.shape == target.shape
#     output = output.reshape(output.size).copy()
#     target = target.reshape(target.size)
#     output[np.where(target == ignore_index)[0]] = ignore_index
#     intersection = output[np.where(output == target)[0]]
#     area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
#     area_output, _ = np.histogram(output, bins=np.arange(K+1))
#     area_target, _ = np.histogram(target, bins=np.arange(K+1))
#     area_union = area_output + area_target - area_intersection
#     return area_intersection, area_union, area_target


# def intersectionAndUnionGPU(output, target, K, ignore_index=255):
#     # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
#     assert (output.dim() in [1, 2, 3])
#     assert output.shape == target.shape
#     output = output.view(-1)
#     target = target.view(-1)
#     output[target == ignore_index] = ignore_index
#     intersection = output[output == target]
#     area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
#     area_output = torch.histc(output, bins=K, min=0, max=K-1)
#     area_target = torch.histc(target, bins=K, min=0, max=K-1)
#     area_union = area_output + area_target - area_intersection
#     return area_intersection, area_union, area_target


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)



# def group_weight(weight_group, module, lr):
#     group_decay = []
#     group_no_decay = []
#     for m in module.modules():
#         if isinstance(m, nn.Linear):
#             group_decay.append(m.weight)
#             if m.bias is not None:
#                 group_no_decay.append(m.bias)
#         elif isinstance(m, nn.modules.conv._ConvNd):
#             group_decay.append(m.weight)
#             if m.bias is not None:
#                 group_no_decay.append(m.bias)
#         elif isinstance(m, nn.modules.batchnorm._BatchNorm):
#             if m.weight is not None:
#                 group_no_decay.append(m.weight)
#             if m.bias is not None:
#                 group_no_decay.append(m.bias)
#     assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
#     weight_group.append(dict(params=group_decay, lr=lr))
#     weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
#     return weight_group


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port