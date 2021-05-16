from torch import optim, nn
from torch.utils.data import DataLoader
import torch
from datetime import datetime
import os
import argparse
from tensorboardX import SummaryWriter
from dataset import Mydataset
from models.Mymodel import Mymodelforpretrain


def train(model, trainloader, criterion, optimizer, epoch_idx, args, device):
    avg_loss = 0
    model.train()

    for batch_idx, (data, labels) in enumerate(trainloader, 1):

        imgs, labels = data.to(device), labels.to(device)
        logit = model(data)
        loss = criterion(logit, labels)
        loss.backward()

        avg_loss += loss / num_batchs
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % args.step_batch == 0:
            log('epoch: {:2d}/{}  |  batch: {:2d}/{}  |  loss: {:.4f}'.format(
                epoch_idx, args.num_epochs,
                batch_idx, num_batchs,
                loss.item()))
    return avg_loss


def evaluation(model, testloader, criterion, device):
    avg_loss = 0
    model.test()
    with torch.no_grad:
        for batch_idx, (data, labels) in enumerate(testloader, 1):
            imgs, labels = data.to(device), labels.to(device)
            logit = model(data)
            loss = criterion(logit, labels)
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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--step_batch', type=int, default=1000)
    parser.add_argument('--eval_batch_size', type=int, default=128)

    parser.add_argument('--lr', type=float, default=5e-05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--drop_rate', type=float, default=0)

    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--m', type=int, default=24)
    parser.add_argument('--out_dim', type=int, default=128)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--n_layer', type=int, default=12)
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
    writer = SummaryWriter(os.path.join(log_dir, 'tb'))

    device = args.use_cuda if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    for k, v in args.__dict__.items():
        log('{}: {}'.format(k, v))
    log('Real used device: {}'.format(device))

    trainset = Mydataset(task=args.task)
    testset = Mydataset(task=args.task, split='test')
    trainloader = DataLoader(trainset, args.batch_size, num_workers=2, shuffle=True)
    testloader = DataLoader(testset, args.batch_size, num_workers=2, shuffle=True)

    num_batchs = len(trainloader)
    log('\n---- dataset info ----')
    log('\n* train data *')
    log('- num : {}'.format(len(trainloader.dataset)))
    log('\n* eval data *')
    log('- num : {}'.format(len(testloader.dataset)))
    log('----------------------\n')

    model = Mymodelforpretrain(args.m, args.out_dim, args.hidden_size, trainset.vocab_size, args.n_layer, trainset.pad_ids)

    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer.zero_grad()

    best_loss = 9999
    best_epoch = 0

    for epoch_idx in range(1, args.num_epochs + 1):
        log('\n\n----------------------- {} epoch start! -----------------------'.format(epoch_idx))
        avg_loss = train(model, trainloader, criterion, optimizer, epoch_idx, args, device)
        eval_loss, eval_acc = evaluation(model, testloader, criterion, device)
        log(' >> epoch: {:2d}\t|\tavg_loss: {:.4f}\t|\teval_loss: {:.4f}\t'.format(
            epoch_idx, avg_loss, eval_loss))

        writer.add_scalars('loss', {'train loss': avg_loss, 'eval loss': eval_loss}, epoch_idx)

        if best_loss > eval_loss:
            best_loss = eval_loss
            best_epoch = epoch_idx
            writer.add_text('best loss', '{}e_{:.4f}%'.format(best_epoch, best_loss), epoch_idx)

    writer.close()