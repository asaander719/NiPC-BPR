import os
import numpy as np
import torch
from torch import nn
import torch.nn.init as initer

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
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
        self.path = path
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

# def load_embedding_weight(textural_embedding_matrix):
#     jap2vec = torch.load(textural_embedding_matrix)
#     embeding_weight = []
#     for jap, vec in jap2vec.items():
#         embeding_weight.append(vec.tolist())
#     embeding_weight.append(torch.zeros(300))
#     embedding_weight = torch.tensor(embeding_weight).cuda()
#     return embedding_weight

# def load_embedding_weight(textural_embedding_matrix, device):
#     jap2vec = torch.load(textural_embedding_matrix)
#     embeding_weight = []
#     for jap, vec in jap2vec.items():
#         embeding_weight.append(vec.tolist())
#     embeding_weight.append(torch.zeros(300))
#     embedding_weight = torch.tensor(embeding_weight, device=device)
#     return embedding_weight

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