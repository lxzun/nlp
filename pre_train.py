from torch import optim, nn
from torch.utils.data import DataLoader
import torch
import numpy as np
from datetime import datetime
import os
import argparse
from tensorboardX import SummaryWriter
from dataset import Mydataset
from models.Mymodel import MymodelForPretrain, MysharemodelForPretrain
from torch.utils.data.sampler import SubsetRandomSampler


def train(model, trainloader, criterion, optimizer, epoch_idx, testloader, args, device):
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

        if batch_idx % args.step_batch == 0:
            log('epoch: {:2d}/{}\t|\tbatch: {:2d}/{}\t|\tloss: {:.5f}'.format(
                epoch_idx, args.num_epochs,
                batch_idx, num_batchs,
                train_loss/batch_idx))

        if batch_idx % (args.step_batch*10) == 0:
            total_batch = int((epoch_idx-1) * len(trainloader) + batch_idx)
            eval_loss = evaluation(model, testloader, criterion, device)
            log(' >> epoch: {:2d}\t|\ttotal_batch: {:2d}\t|\teval_loss: {:.8f}'.format(
                epoch_idx, total_batch, eval_loss))

            writer.add_scalars('loss', {'train loss': train_loss/batch_idx, 'eval loss': eval_loss}, total_batch)

            if best_loss > eval_loss:
                best_loss = eval_loss
                model.module.save(vocab_save + '/embedding_weight', model_save+'/model_weight')
                writer.add_text('best loss', '{}b_{}'.format(total_batch, best_loss), total_batch)

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
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--description', type=str, default='pretrain')

    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--step_batch', type=int, default=100)
    parser.add_argument('--eval_batch_size', type=int, default=22)

    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--drop_rate', type=float, default=0.)

    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--m', type=int, default=8)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--share', type=bool, default=True)
    parser.add_argument('--max_seq_length', type=int, default=512)
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

    log_root = mk_dir('./log/train')
    today = str(datetime.today().strftime('%y-%m-%d_%H-%M-%S'))
    log_dir = mk_dir(os.path.join(log_root, 'log_{}'.format(today)))
    log_file = os.path.join(log_dir, 'log.txt')
    model_save = mk_dir(os.path.join(log_dir, 'model_save')) if args.save_model else None
    vocab_save = mk_dir(os.path.join(log_dir, 'vocab_save')) if args.save_vocab else None
    writer = SummaryWriter(os.path.join(log_dir, args.description))

    device = args.use_cuda if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    for k, v in args.__dict__.items():
        log('{}: {}'.format(k, v))
    log('Real used device: {}'.format(device))

    dataset = Mydataset(task=args.task, max_length=args.max_seq_length)
    dataset_size = len(dataset)
    validation_split = .00125
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    trainloader = DataLoader(dataset, args.batch_size, num_workers=2, sampler=train_sampler, collate_fn=dataset.make_batch)
    testloader = DataLoader(dataset, args.batch_size, num_workers=2, sampler=valid_sampler, collate_fn=dataset.make_batch)


    log('\n---- dataset info ----')
    log('\n* train data *')
    log('- num : {}'.format(len(train_indices)))
    log('\n* eval data *')
    log('- num : {}'.format(len(val_indices)))
    log('----------------------\n')

    if args.share:
        model = MysharemodelForPretrain(args.m, args.out_dim, args.hidden_size, dataset.vocab_size, args.n_layer, dataset.pad_ids)
    else:
        model = MymodelForPretrain(args.m, args.out_dim, args.hidden_size, dataset.vocab_size, args.n_layer, dataset.pad_ids)
    if device == 'cuda':
        model.to(device)

    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer.zero_grad()

    best_loss = 9999

    for epoch_idx in range(1, args.num_epochs + 1):
        log('\n\n----------------------- {} epoch start! -----------------------'.format(epoch_idx))
        avg_loss = train(model, trainloader, criterion, optimizer, epoch_idx, testloader, args, device)

    writer.close()