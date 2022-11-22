import gc
import time
import random
import torch
#import pynvml
import logging
import argparse
import os, os.path
import numpy as np
import pandas as pd
 
localtime = time.localtime(time.time())
str_time = f'{str(localtime.tm_year)}-{str(localtime.tm_mon)}-{str(localtime.tm_mday)}-{str(localtime.tm_hour)}-{str(localtime.tm_min)}'

from models.model import REMS
from src.train import REMSTrain
from data.dataloader import MMDataLoader
from src.config import ConfigRegression

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def setup_seed(seed):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.enabled = False
    #
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(args):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.model_save_path = os.path.join(args.model_save_dir, f'{args.modelname}-{args.dataset}-{args.name}-{args.seed}.pth')
    args.best_model_save_path = os.path.join(args.model_save_dir, f'{args.modelname}-{args.dataset}-{args.name}-{args.seed}-best.pth')

    print(args.best_model_save_path)
    
    using_cuda = torch.cuda.is_available()
    print(torch.version.cuda)
    print("using_cuda: ", using_cuda)

    if not torch.cuda.is_available():
        return None

    args.device = torch.device('cuda:0' if using_cuda else 'cpu')
    # data
    dataloader = MMDataLoader(args)

    train_loader, valid_loader, test_loader = dataloader

    model = REMS(args).to(args.device)

    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
                # print(p)
        return answer
    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    trainmodel = REMSTrain(args)#.getTrain(args)
    args.train_epoch = 0
    # do train
    if not args.test_only:
        trainmodel.do_train(model, dataloader)
    # load pretrained model
    assert os.path.exists(args.best_model_save_path)
    model.load_state_dict(torch.load(args.best_model_save_path))
    model.to(args.device)

    # do test
    if args.tune_mode:
        # using valid dataset to debug hyper parameters
        results = trainmodel.do_test(model, valid_loader, mode="VALID")
    else:
        results = trainmodel.do_test(model, test_loader, mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results

def run_normal(args):
    args.res_save_dir = args.res_save_dir
    init_args = args
    model_results = []
    seeds = args.seeds
    # run results
    for i, seed in enumerate(seeds):
        args = init_args
        # load config
        config = ConfigRegression(args)
        args = config.get_config()
        setup_seed(seed)
        args.seed = seed
        
        logger.info('Start running %s...' %(args.modelname))
        logger.info(args)
        # runnning
        args.cur_time = i+1
        test_results = run(args)
        # restore results
        if test_results is not None:
            model_results.append(test_results)

    criterions = list(model_results[0].keys())
    # load other results
    save_path = os.path.join(args.res_save_dir, \
                        f'{args.modelname}-{args.name}-{args.dataset}-{str_time}.csv')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    # save results
    # res = [args.modelname]
    res = []
    res_temp = []
    for c in criterions:
        values = [r[c] for r in model_results]
        res_temp.append(values)
    res_temp = np.array(res_temp)
    print(res_temp)

    for i in range(len(seeds)):
        res_tmp_tmp = [str(i)]
        res_tmp_tmp += list(res_temp[:,i])
        res.append(res_tmp_tmp)
        df.loc[len(df)] = res[i]
    df.to_csv(save_path, index=None)
    logger.info('Results are added to %s...' %(save_path))

def set_log(args):

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)

    log_file_path = os.path.join(args.logs_dir, f'{args.modelname}-{args.dataset}-{args.name}-{str_time}.log')

    # set logging
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    # add StreamHandler to terminal outputs
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, default='rems',
                        help='support rems')
    parser.add_argument('--dataset', type=str, default='iemocap',
                        help='support iemocap')
    parser.add_argument('--KeyEval', type=str, default='Loss',
                        help='support Loss')
    parser.add_argument('--name', type=str, default='',
                        help='the model name details')
    parser.add_argument('--test_only', action='store_true',
                        help='train+test or test only')
    parser.add_argument('--rems_use', action='store_true',
                        help='rems use ot not')
    parser.add_argument('--two_stage', action='store_true',
                        help='two-stage train or not')
    parser.add_argument('--bert_type',type=str, default='bert_base',
                        help='bert_type = bert_base/bert_large')
    parser.add_argument('--tasks', type=str, default='MTAV',
                        help='MATV')
    parser.add_argument('--unitask', type=str, default='M',
                        help='MTAV')
    parser.add_argument('--MAX_Epoch', type=int, default=30,
                        help='MAX_Epoch')
    parser.add_argument('--MIN_Epoch', type=int, default=0,
                        help='MIN_Epoch')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='num workers of loading data')
    parser.add_argument('--lr_text_bert', type=float, default=2e-5, #2e-5
                        help='lr_text_bert')
    parser.add_argument('--lr_text_other', type=float, default=5e-5,#1e-2, #1e-2
                        help='lr_text_other')
    parser.add_argument('--lr_other', type=float, default=5e-5, #2e-5
                        help='lr_other')
    parser.add_argument('--post_other_dropout', type=float, default=0.0, #0.0,
                        help='post_other_dropout')
    parser.add_argument('--post_text_dropout', type=float, default=0.0, #0.0,
                        help='post_text_dropout')
    # parser.add_argument('--data_dir', type=str, default='/path/to/iemocap/dataset/',
    parser.add_argument('--data_dir', type=str, default='/scratch/users/ntu/n2107335/Multimodal/Dataset/IEMOCAP/',
                        help='support rems')
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/results',
                        help='path to save results.')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='path to tensorboard results.')
    parser.add_argument('--logs_dir', type=str, default='results/logs',
                        help='path to log results.') # N
    parser.add_argument('--seeds', nargs='+', type=int,
                        help='set seeds for multiple runs!')
    parser.add_argument('--nargs-int-type', nargs='+', type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logger = set_log(args)
    print(args)

    # if args.dataset == 'iemocap':
    #     args.data_dir = os.path.join(args.data_dir, 'IEMOCAP/')

    run_normal(args)