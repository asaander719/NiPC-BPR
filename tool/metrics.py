# import torch
# import numpy as np
# from config.configurator import configs


# class Metric(object):
#     def __init__(self):
#         self.metrics = configs['test']['metrics']
#         self.k = configs['test']['k']

#     def recall(self, test_data, r, k):
#         right_pred = r[:, :k].sum(1)
#         recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
#         recall = np.sum(right_pred / recall_n)
#         return recall

#     def precision(self, r, k):
#         right_pred = r[:, :k].sum(1)
#         precis_n = k
#         precision = np.sum(right_pred) / precis_n
#         return precision

#     def mrr(self, r, k):
#         pred_data = r[:, :k]
#         scores = 1. / np.arange(1, k + 1)
#         pred_data = pred_data * scores
#         pred_data = pred_data.sum(1)
#         return np.sum(pred_data)

#     def ndcg(self, test_data, r, k):
#         assert len(r) == len(test_data)
#         pred_data = r[:, :k]

#         test_matrix = np.zeros((len(pred_data), k))
#         for i, items in enumerate(test_data):
#             length = k if k <= len(items) else len(items)
#             test_matrix[i, :length] = 1
#         max_r = test_matrix
#         idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
#         dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
#         dcg = np.sum(dcg, axis=1)
#         idcg[idcg == 0.] = 1.
#         ndcg = dcg / idcg
#         ndcg[np.isnan(ndcg)] = 0.
#         return np.sum(ndcg)

#     def get_label(self, test_data, pred_data):
#         r = []
#         for i in range(len(test_data)):
#             ground_true = test_data[i]
#             predict_topk = pred_data[i]
#             pred = list(map(lambda x: x in ground_true, predict_topk))
#             pred = np.array(pred).astype("float")
#             r.append(pred)
#         return np.array(r).astype('float')

#     def eval_batch(self, data, topks):
#         sorted_items = data[0].numpy()
#         ground_true = data[1]
#         r = self.get_label(ground_true, sorted_items)

#         result = {}
#         for metric in self.metrics:
#             result[metric] = []

#         for k in topks:
#             for metric in result:
#                 if metric == 'recall':
#                     result[metric].append(self.recall(ground_true, r, k))
#                 if metric == 'ndcg':
#                     result[metric].append(self.ndcg(ground_true, r, k))
#                 if metric == 'precision':
#                     result[metric].append(self.precision(r, k))
#                 if metric == 'mrr':
#                     result[metric].append(self.mrr(r, k))

#         for metric in result:
#             result[metric] = np.array(result[metric])

#         return result

#     def eval(self, model, test_dataloader):
#         # for most GNN models, you can have all embeddings ready at one forward
#         if 'eval_at_one_forward' in configs['test'] and configs['test']['eval_at_one_forward']:
#             return self.eval_at_one_forward(model, test_dataloader)

#         result = {}
#         for metric in self.metrics:
#             result[metric] = np.zeros(len(self.k))

#         batch_ratings = []
#         ground_truths = []
#         test_user_count = 0
#         test_user_num = len(test_dataloader.dataset.test_users)
#         for _, tem in enumerate(test_dataloader):
#             if not isinstance(tem, list):
#                 tem = [tem]
#             test_user = tem[0].numpy().tolist()
#             batch_data = list(
#                 map(lambda x: x.long().to(configs['device']), tem))
#             # predict result
#             with torch.no_grad():
#                 batch_pred = model.full_predict(batch_data)
#             test_user_count += batch_pred.shape[0]
#             # filter out history items
#             batch_pred = self._mask_history_pos(
#                 batch_pred, test_user, test_dataloader)
#             _, batch_rate = torch.topk(batch_pred, k=max(self.k))
#             batch_ratings.append(batch_rate.cpu())
#             # ground truth
#             ground_truth = []
#             for user_idx in test_user:
#                 ground_truth.append(
#                     list(test_dataloader.dataset.user_pos_lists[user_idx]))
#             ground_truths.append(ground_truth)
#         assert test_user_count == test_user_num

#         # calculate metrics
#         data_pair = zip(batch_ratings, ground_truths)
#         eval_results = []
#         for _data in data_pair:
#             eval_results.append(self.eval_batch(_data, self.k))
#         for batch_result in eval_results:
#             for metric in self.metrics:
#                 result[metric] += batch_result[metric] / test_user_num

#         return result

#     def _mask_history_pos(self, batch_rate, test_user, test_dataloader):
#         if not hasattr(test_dataloader.dataset, 'user_history_lists'):
#             return batch_rate
#         for i, user_idx in enumerate(test_user):
#             pos_list = test_dataloader.dataset.user_history_lists[user_idx]
#             batch_rate[i, pos_list] = -1e8
#         return batch_rate
    
#     def eval_at_one_forward(self, model, test_dataloader):
#         result = {}
#         for metric in self.metrics:
#             result[metric] = np.zeros(len(self.k))

#         batch_ratings = []
#         ground_truths = []
#         test_user_count = 0
#         test_user_num = len(test_dataloader.dataset.test_users)

#         with torch.no_grad():
#             user_emb, item_emb = model.generate()

#         for _, tem in enumerate(test_dataloader):
#             if not isinstance(tem, list):
#                 tem = [tem]
#             test_user = tem[0].numpy().tolist()
#             batch_data = list(
#                 map(lambda x: x.long().to(configs['device']), tem))
#             # predict result
#             batch_u = batch_data[0]
#             batch_u_emb, all_i_emb = user_emb[batch_u], item_emb
#             with torch.no_grad():
#                 batch_pred = model.rating(batch_u_emb, all_i_emb)
#             test_user_count += batch_pred.shape[0]
#             # filter out history items
#             batch_pred = self._mask_history_pos(
#                 batch_pred, test_user, test_dataloader)
#             _, batch_rate = torch.topk(batch_pred, k=max(self.k))
#             batch_ratings.append(batch_rate.cpu())
#             # ground truth
#             ground_truth = []
#             for user_idx in test_user:
#                 ground_truth.append(
#                     list(test_dataloader.dataset.user_pos_lists[user_idx]))
#             ground_truths.append(ground_truth)
#         assert test_user_count == test_user_num

#         # calculate metrics
#         data_pair = zip(batch_ratings, ground_truths)
#         eval_results = []
#         for _data in data_pair:
#             eval_results.append(self.eval_batch(_data, self.k))
#         for batch_result in eval_results:
#             for metric in self.metrics:
#                 result[metric] += batch_result[metric] / test_user_num

#         return result




import math

import numpy as np


def getHIT_MRR(pred, target_items):
    hit = 0.
    mrr = 0.
    p_1 = []
    for p in range(len(pred)):
        pre = pred[p]
        if pre in target_items:
            hit += 1
            if pre not in p_1:
                p_1.append(pre)
                mrr = 1. / (p + 1)

    return hit, mrr


def getNDCG(rank_list, target_items):
    dcg = 0
    idcg = IDCG(len(target_items))
    for i in range(len(rank_list)):
        item_id = rank_list[i]
        if item_id not in target_items:
            continue
        rank = i + 1
        dcg += 1. / math.log(rank + 1, 2)

    return dcg / idcg


def IDCG(n):
    idcg = 0.
    for i in range(n):
        idcg += 1. / math.log(i + 2, 2)

    return idcg


def get_metrics(grd, grd_cnt, pred, topk):
    REC, MRR, NDCG = [], [], []
    for each_grd, each_grd_cnt, each_pred in zip(grd, grd_cnt, pred):
        NDCG.append(getNDCG(each_pred[:topk], [each_grd][:each_grd_cnt]))
        hit, mrr = getHIT_MRR(each_pred[:topk], [each_grd][:each_grd_cnt])
        REC.append(hit)
        MRR.append(mrr)

    REC = np.mean(REC)
    MRR = np.mean(MRR)
    NDCG = np.mean(NDCG)

    return REC, MRR, NDCG


# # class EarlyStopping:
# #     """Early stops the training if validation loss doesn't improve after a
# #     given patience."""

# #     def __init__(self,
# #                  pretrain_mode,
# #                  patience,
# #                  verbose=False,
# #                  delta=0,
# #                  trace_func=print):
# #         """
# #         Args:
# #             pretrain_mode (bool): Define the training mode, Default: True, pretrain_mode / if False, TransMatch training mode
# #             patience (int): How long to wait after last time validation loss improved.
# #                             Default: 8
# #             verbose (bool): If True, prints a message for each validation loss improvement.
# #                             Default: False
# #             delta (float): Minimum change in the monitored quantity to qualify as an improvement.
# #                             Default: 0
# #             path (str): Path for the checkpoint to be saved to.
# #             trace_func (function): trace print function.
# #                             Default: print
# #         """
# #         self.patience = patience
# #         self.pretrain_mode = pretrain_mode
# #         self.verbose = verbose
# #         self.counter = 0
# #         self.best_score = None
# #         self.early_stop = False
# #         self.val_auc_max = np.Inf
# #         self.delta = delta
# #         self.trace_func = trace_func

# #     def __call__(self, val_auc):

# #         score = val_auc

# #         if self.best_score is None:
# #             self.best_score = score
# #         elif score < self.best_score + self.delta:
# #             self.counter += 1
# #             self.trace_func(
# #                 f'EarlyStopping counter: {self.counter} out of {self.patience}'
# #             )
# #             if self.counter >= self.patience:
# #                 self.early_stop = True
# #                 # self.pretrain_mode = False
# #                 # if self.counter >= self.patience + self.patience:
# #                 #     self.early_stop = True
# #         else:
# #             self.best_score = score
# #             self.counter = 0

# #     # def save_checkpoint(self, val_auc, model):
# #     #     '''Saves model when validation auc increase.'''
# #     #     if self.verbose:
# #     #         self.trace_func(f'Validation auc increase ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
# #     #     self.val_auc_max = val_auc
