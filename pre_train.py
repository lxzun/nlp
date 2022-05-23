from torch import optim, nn
from torch.utils.data import DataLoader
import torch
import numpy as np
from datetime import datetime
import os
import argparse
from tensorboardX import SummaryWriter
from dataset import Mydataset, Mydataset_spm
from models.Mymodel import MymodelForPretrain
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AdamW

def train(model, trainloader, criterion, optimizer, scheduler, epoch_idx, testloader, args, device):
    global best_loss
    num_batchs = len(trainloader)
    avg_loss = 0
    train_loss = 0

    for batch_idx, (data, labels) in enumerate(trainloader, 1):
        model.train()

        data, labels = data.to(device), labels.to(device)
        logit = model(data)
        loss = criterion(logit, labels.view(-1))
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        avg_loss += loss / num_batchs
        train_loss += loss.item()
        scheduler.step()

        if batch_idx % args.step_batch == 0:
            log('epoch: {:2d}/{}\t|\tbatch: {:2d}/{}\t|\tloss: {:.5f}\t|\t{:.8f}'.format(
                epoch_idx, args.num_epochs,
                batch_idx, num_batchs,
                train_loss/args.step_batch, scheduler.get_lr()[0]))

            if batch_idx % (args.step_batch*5) == 0:
                total_batch = int((epoch_idx-1) * len(trainloader) + batch_idx)
                eval_loss = evaluation(model, testloader, criterion, device)
                log(' >> epoch: {:2d}\t|\ttotal_batch: {:2d}\t|\teval_loss: {:.8f}'.format(
                    epoch_idx, total_batch, eval_loss))

                writer.add_scalars('loss', {'train loss': train_loss/args.step_batch, 'eval loss': eval_loss}, total_batch)

                if best_loss > eval_loss:
                    best_loss = eval_loss
                    if args.multi_gpu and torch.cuda.device_count() > 1:
                        model.module.save(vocab_save, model_save)
                    else: model.save(vocab_save, model_save)
                    writer.add_text('best loss', '{}b_{}'.format(total_batch, best_loss), total_batch)

            train_loss = 0
            
            
    return avg_loss


def evaluation(model, testloader, criterion, device):
    avg_loss = 0
    num_batchs = len(testloader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(testloader, 1):
            data, labels = data.to(device), labels.to(device)
            logit = model(data)
            loss = criterion(logit, labels.view(-1))
            avg_loss += loss / num_batchs
    return avg_loss


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
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--step_batch', type=int, default=200)
    parser.add_argument('--eval_batch_size', type=int, default=256)

    parser.add_argument('--lr', type=float, default=5e-04)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--drop_rate', type=float, default=0.)

    parser.add_argument('--new_vocab', type=int, default=0)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--m', type=int, default=32)
    parser.add_argument('--out_dim', type=int, default=128)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--attd_mode', type=int, default=2)
    parser.add_argument('--max_seq_length', type=int, default=512*1)
    parser.add_argument('--task', type=str, default='pretrain')

    parser.add_argument('--save_vocab', type=bool, default=True)
    parser.add_argument('--pretrained_vocab_path', type=str, default='', help='load pretrained vocab path')
    parser.add_argument('--pretrained_vocab', type=bool, default=False, help='load pretrained vocab')

    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--pretrained_model_path', type=str, default='', help='load pretrained model path')
    parser.add_argument('--pretrained_model', type=bool, default=False, help='load pretrained model')

    parser.add_argument('--use_cuda', type=str, default='cuda')
    parser.add_argument('--multi_gpu', type=bool, default=True)

    args = parser.parse_args()

    log_root = mk_dir('log_under_12')
    today = str(datetime.today().strftime('%y-%m-%d_%H-%M-%S'))
    log_dir = mk_dir(os.path.join(log_root, f'{args.task}'))
    log_dir = mk_dir(os.path.join(log_dir, f'A{args.attd_mode}_V{args.new_vocab}_S{args.max_seq_length}_E{args.embedding_size}_H{args.hidden_size}_M{args.m}_K{args.k}_O{args.out_dim}_L{args.n_layer}'))
    log_dir = mk_dir(os.path.join(log_dir, f'log_{today}'))
    log_file = os.path.join(log_dir, 'log_under_12.txt')
    model_save = mk_dir(os.path.join(log_dir, 'model_save')) if args.save_model else None
    model_save = model_save + '/model_weight' if model_save else None
    vocab_save = mk_dir(os.path.join(log_dir, 'vocab_save')) if args.save_vocab else None
    vocab_save = vocab_save + '/embedding_weight' if vocab_save else None
    writer = SummaryWriter(log_dir)

    device = args.use_cuda if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    for k, v in args.__dict__.items():
        log('{}: {}'.format(k, v))
    log('Real used device: {}'.format(device))

    if args.new_vocab: dataset = Mydataset_spm(task=args.task, vocab=f'vocab/vocab_{args.new_vocab}.model', max_length=args.max_seq_length)
    else: dataset = Mydataset(task=args.task, max_length=args.max_seq_length)

    dataset_size = len(dataset)
    validation_split = .00125
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    trainloader = DataLoader(dataset, args.batch_size, sampler=train_sampler)
    testloader = DataLoader(dataset, args.batch_size, sampler=valid_sampler)


    log('\n---- dataset info ----')
    log('\n* train data *')
    log('- num : {}'.format(len(train_indices)))
    log('\n* eval data *')
    log('- num : {}'.format(len(val_indices)))
    log('----------------------\n')

    model = MymodelForPretrain(vocab_size=dataset.vocab_size,
                               embedding_size=args.embedding_size,
                               hidden_size=args.hidden_size,
                               m=args.m, out_dim=args.out_dim,
                               n_layer=args.n_layer,
                               pad_ids=dataset.pad_ids,
                               attd_mode=args.attd_mode,
                               drop_rate=args.drop_rate)
        

    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, step_size_up=50, step_size_down=1000, max_lr=1e-3, mode='triangular', cycle_momentum=False)
    optimizer.zero_grad()

    best_loss = 9999

    for epoch_idx in range(1, args.num_epochs + 1):
        log('\n\n----------------------- {} epoch start! -----------------------'.format(epoch_idx))
        avg_loss = train(model, trainloader, criterion, optimizer, scheduler, epoch_idx, testloader, args, device)

    writer.close()