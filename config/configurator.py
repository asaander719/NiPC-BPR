import os
import yaml
import argparse
from util import config

def parse_configure():
    parser = argparse.ArgumentParser(description='APCL')
    parser.add_argument('--arch', type=str, help='Model name')
    parser.add_argument('--dataset', type=str, default='Polyvore_519', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size number')
    parser.add_argument('--test_batch_size', type=int, default=512, help='batch_size number')
    parser.add_argument('--mode', type=str, default='RB', help='given top, recommend bottom')
    parser.add_argument('--config', type=str, default='config/APCL_Polyvore_519_RB.yaml', help='config file') #APCL_IQON3000_RB.yaml #APCL_Polyvore_RB.yaml
    parser.add_argument('opts', help='see config/APCL_Polyvore_RB.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    else:
        args.device == "cpu"
    
    if args.arch == None:
        raise Exception("Please provide the model name through --model.")
    model_name = args.arch#.lower()
    args.config = 'config/APCL_' + str(args.dataset) + '_' + str(args.mode) + '.yaml'

    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
        cfg.device = args.device
        cfg.arch = args.arch
        cfg.dataset = args.dataset
        cfg.mode = args.mode
        cfg.cuda = args.cuda
        cfg.batch_size = args.batch_size
        cfg.test_batch_size = args.test_batch_size
    return cfg
    # if not os.path.exists('./config/{}_{}_{}.yaml'.format(args.arch, args.dataset, args.mode)): #config/APCL_Polyvore_RB.yaml
    #     raise Exception("Please create the yaml file for your model first.")
    
    # # with open('./config/{}_{}_{}.yaml'.format(model_name, args.dataset, args.mode), encoding='utf-8') as f:
    # with open('./config/APCL_Polyvore_519_RB.yaml', encoding='utf-8') as f:
    #     config_data = f.read()
    #     configs = yaml.safe_load(config_data)     
    #     configs['TRAIN']['arch'] = args.arch
    #     # # grid search
    #     # if 'tune' not in configs:
    #     #     configs['tune'] = {'enable': False}

    #     configs['device'] = args.device
    #     if args.dataset is not None:
    #         configs['dataset'] = args.dataset

    #     # if 'log_loss' not in configs['train']:
    #     #     configs['train']['log_loss'] = True
    #     if args.batch_size is not None:
    #         configs['TRAIN']['batch_size'] = args.batch_size
    #         configs['TEST']['test_batch_size'] = args.test_batch_size

    #     # early stop
    #     if 'patience' in configs['TRAIN']:
    #         if configs['TRAIN']['patience'] <= 0:
    #             raise Exception("'patience' should be greater than 0.")
    #         else:
    #             configs['TRAIN']['early_stop'] = True
    #     else:
    #         configs['TRAIN']['early_stop'] = False
    #     return configs
        

# configs = parse_configure()
