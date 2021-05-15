from torch import cuda, optim
from torch.utils.data import DataLoader
import torch
import datetime
import os
import argparse
from tensorboardX import SummaryWriter
from dataset import load_datasets

def mk_dir(path_: str):
    os.makedirs(path_) if not os.path.isdir(path_) else None
    return path_

def log(string):
    global log_file
    print(string)
    with open(log_file, 'at', encoding='utf-8') as f:
        f.write(str(string))
        f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--description', type=str, default='pretrain')

    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--step_batch', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=128)

    parser.add_argument('--lr', type=float, default=5e-05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--drop_rate', type=float, default=0)

    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--m', type=int, default=24)
    parser.add_argument('--out_dim', type=int, default=50)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--task', type=str, default='pretrain')

    parser.add_argument('--save_vocab_path', type=str, default='')
    parser.add_argument('--save_vocab', type=bool, default=True)
    parser.add_argument('--pretrained_vocab_path', type=str, default='', help='load pretrained vocab path')
    parser.add_argument('--pretrained_vocab', type=bool, default=False, help='load pretrained vocab')

    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--pretrained_model_path', type=str, default='', help='load pretrained model path')
    parser.add_argument('--pretrained_model', type=bool, default=False, help='load pretrained model')

    parser.add_argument('--use_cuda', type=str, default='cuda')
    parser.add_argument('--multi_gpu', type=bool, default=True)

    args = parser.parse_args()

    log_root = mk_dir('./log/train')
    today = str(datetime.today().strftime('%y-%m-%d_%H-%M-%S'))
    log_dir = mk_dir(os.path.join(log_root, 'log_{}'.format(today)))
    log_file = os.path.join(log_dir, 'log.txt')
    model_save = mk_dir(os.path.join(log_dir, 'model_save')) if args.save_model else None
    vocab_save = mk_dir(os.path.join(log_dir, 'vocab_save')) if args.save_vocab else None
    writer = SummaryWriter(os.path.join(log_dir, 'tb'))

    device = args.use_cuda if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    for k, v in args.__dict__.items():
        log('{}: {}'.format(k, v))
    log('Real used device: {}'.format(device))

