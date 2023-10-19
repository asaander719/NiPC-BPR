# @Time   : 2023/10/18

general_arguments = [
    'gpu_id', 'use_gpu',
    'seed',
    'state',
    'data_path',
    'benchmark_filename',
    'show_progress',
    'config_file',
    'save_dataset',
    'save_dataloaders',
]

training_arguments = [
    'epochs', 'train_batch_size',
    'learner', 'learning_rate',
    'training_neg_sample_num',
    'eval_step', 'stopping_step',
    'checkpoint_dir',
    'clip_grad_norm',
    'loss_decimal_place',
    'weight_decay'
]

evaluation_arguments = [
    'eval_args',
    'metrics', 'topk', 'valid_metric', 'valid_metric_bigger',
    'eval_batch_size',
    'metric_decimal_place'
]

dataset_arguments = [
    'USER_ID', 'GIVENITEM_ID', 'POSITIVEITEM_ID', 'NEGATIVE_ID',
    'LABEL_FIELD', 'threshold',
    'NEG_PREFIX',
    'preload_weight',
    'normalize_field', 'normalize_all'
]
