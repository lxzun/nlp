import argparse
from ast import arg
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from dataset import Mydataset, Mydataset_spm
from models.Mymodel import MymodelForPretrain, MymodelForPretrain2
from sklearn.metrics import f1_score

from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime

from tensorboardX import SummaryWriter

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='pre-train H&M article')
parser.add_argument('-j', '--workers', default=5*3, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default="localhost:12355", type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')



parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--cpu', default=False, action='store_true',
                    help='CPU to use.')
parser.add_argument('--multiprocessing-distributed', default=True, action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=24*3, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when ' 
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--drop_rate', type=float, default=0.1)

parser.add_argument('--mode', type=int, default=1)
parser.add_argument('--new_vocab', type=int, default=1)
parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=768)
parser.add_argument('--m', type=int, default=12)
parser.add_argument('--out_dim', type=int, default=24)
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--n_layer', type=int, default=12)
parser.add_argument('--attd_mode', type=int, default=2)
parser.add_argument('--max_seq_length', type=int, default=512)
parser.add_argument('--task', type=str, default='pretrain')

parser.add_argument('--use_cuda', type=str, default='cuda')
parser.add_argument('--multi_gpu', type=bool, default=True)

best_acc1 = 0


def main():
    args = parser.parse_args()

    log_root = mk_dir('log')    
    today = str(datetime.today().strftime('%y-%m-%d_%H-%M-%S'))
    log_dir = mk_dir(os.path.join(log_root, f'{args.task}'))
    log_dir = mk_dir(os.path.join(log_dir, f'A{args.attd_mode}_V{args.new_vocab}_S{args.max_seq_length}_E{args.embedding_size}_H{args.hidden_size}_M{args.m}_K{args.k}_O{args.out_dim}_L{args.n_layer}'))
    log_dir = mk_dir(os.path.join(log_dir, f'log_{today}'))
    log_file = os.path.join(log_dir, 'log.txt')
    args.log_file = log_file
    args.log_dir = log_dir

    if args.attd_mode == 1:
        log(f'model size: {((args.new_vocab*1000+6 if args.new_vocab else 30522)*args.embedding_size+args.embedding_size*args.hidden_size+args.hidden_size+(args.k**2 * args.m * args.out_dim * 3 + args.out_dim * 3 + args.m * args.out_dim + args.m)*args.n_layer)/1000000:.2f} M', log_file)
        log(f'flops: {(args.embedding_size*args.hidden_size+(args.n_layer*(args.out_dim*args.max_seq_length*args.hidden_size*args.k*args.k*3+args.max_seq_length*(args.hidden_size//args.m)*2+args.out_dim*args.max_seq_length*args.hidden_size)))/1000000000:.2f} B', log_file)
    
    if args.attd_mode == 2:
        log(f'model size: {((args.new_vocab*1000+6 if args.new_vocab else 30522)*args.embedding_size+args.embedding_size*args.hidden_size+args.hidden_size+(args.k**2 * args.m * args.out_dim * 2 + args.out_dim * 2 + args.m * args.out_dim + args.m * args.out_dim + args.m)*args.n_layer)/1000000:.2f} M', log_file)
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                world_size=args.world_size, rank=args.rank)

    
    # Data loading code
    
    if args.new_vocab: 
        dataset = Mydataset_spm(task=args.task, vocab=f'vocab/vocab_{args.new_vocab}.model', max_length=args.max_seq_length, seq_mask=True)
    else: 
        dataset = Mydataset(task=args.task, max_length=args.max_seq_length, seq_mask=True)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    else:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        train_sampler = SubsetRandomSampler(indices)

    train_loader = torch.utils.data.DataLoader(dataset, args.batch_size, num_workers=args.workers, pin_memory=True, sampler=train_sampler)


    # create model
    if args.mode == 1:
        model = MymodelForPretrain(vocab_size=dataset.vocab_size,
                                embedding_size=args.embedding_size,
                                hidden_size=args.hidden_size,
                                m=args.m, out_dim=args.out_dim,
                                n_layer=args.n_layer,
                                pad_ids=dataset.pad_ids,
                                attd_mode=args.attd_mode,
                                drop_rate=args.drop_rate)
    elif args.mode == 2:
        model = MymodelForPretrain2(vocab_size=dataset.vocab_size,
                                embedding_size=args.embedding_size,
                                hidden_size=args.hidden_size,
                                m=args.m, out_dim=args.out_dim,
                                n_layer=args.n_layer,
                                pad_ids=dataset.pad_ids,
                                attd_mode=args.attd_mode,
                                drop_rate=args.drop_rate)

    if not torch.cuda.is_available() or args.cpu:
        print('using CPU, this will be slow')

    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            if args.mode == 1:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            elif args.mode == 2:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            if args.mode == 1:
                model = torch.nn.parallel.DistributedDataParallel(model)
            elif args.mode == 2:
                model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion1 = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion2 = nn.MSELoss().cuda(args.gpu)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, amsgrad=True)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=55, eta_min=1e-6)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            # best_acc1 = checkpoint['best_acc1']
            # if args.gpu is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion1, criterion2, optimizer, epoch, scheduler, args)

        # remember best acc@1 and save checkpoint

        if not args.multiprocessing_distributed:
            model.model_save(epoch, optimizer, scheduler, False, args)

        if args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0:
            model.module.model_save(epoch, optimizer, scheduler, False, args)

def train(train_loader, model, criterion1, criterion2, optimizer, epoch, scheduler, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc', ':6.2f')
    scores = AverageMeter('F1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, scores],
        prefix="Epoch: [{}]".format(epoch), log_file=args.log_file)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, target, mask) in enumerate(train_loader, 1):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None and not args.cpu:
            data = data.cuda(args.gpu, non_blocking=True)
        if args.gpu is not None and not args.cpu:
            mask = mask.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available() and not args.cpu:
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        if args.mode == 1:
            output = model(data)
            loss = criterion1(output[mask], target[mask])

        elif args.mode ==2:
            output, output2, output3 = model(data, target)
            loss = criterion1(output[mask], target[mask])
            loss += criterion2(output2[mask], output3[mask])

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))
        score = cal_f1(output, target)
        losses.update(loss.item(), data[0].size(0))
        top1.update(acc1[0].item(), data[0].size(0))
        scores.update(score, data[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        
        if i % 1000 == 0:
            # scheduler.step()

            if not args.multiprocessing_distributed:
                model.model_save(epoch, optimizer, scheduler, False, args)

            if args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0:
                model.module.model_save(epoch, optimizer, scheduler, False, args)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            if args.gpu is not None:
                data = [x.cuda(args.gpu, non_blocking=True) for x in data]
            if torch.cuda.is_available():
                target = [x.cuda(args.gpu, non_blocking=True) for x in target]

            # compute output
            output, _ = model(data)
            for j, logit in enumerate(output):
                if j == 0:
                    loss = criterion(logit, target[j])
                else:
                    loss += criterion(logit, target[j])

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data[0].size(0))
            top1.update(acc1[0], data[0].size(0))
            top5.update(acc5[0], data[0].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        progress.display_summary()

    return top1.avg


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_file=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_file = log_file

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        log('\t'.join(entries), self.log_file)

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        log(' '.join(entries), self.log_file)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def cal_f1(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)
        f1 = 0
        for i in range(batch_size):
            _, pred = output[i].topk(1, 1, True, True)
            pred = pred.t()
            score = f1_score(target[i].cpu(), pred.squeeze().cpu(), average='macro')
            f1 += score * (100.0 / batch_size)

        return f1


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        acc = [0] * len(topk)
        for i in range(batch_size):
            _, pred = output[i].topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target[i].view(1, -1).expand_as(pred))

            for j, k in enumerate(topk):
                correct_k = correct[:k].reshape(-1).float().mean(0, keepdim=True)
                acc[j] += (correct_k.mul_(100.0 / batch_size))

        return acc



def mk_dir(path_: str):
    os.makedirs(path_) if not os.path.isdir(path_) else None
    return path_

def log(string, log_file=""):
    print(string)
    with open(log_file, 'at', encoding='utf-8') as f:
        f.write(str(string))
        f.write('\n')

if __name__ == '__main__':
    main()
