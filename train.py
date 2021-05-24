from torch import optim, nn
from torch.utils.data import DataLoader
import torch
import numpy as np
from datetime import datetime
import os
import argparse
from tensorboardX import SummaryWriter
from dataset import Mydataset
from models.Mymodel import MymodelForSequenceClassification


def train(model, trainloader, criterion, optimizer, epoch_idx, testloader, args, device):
    global best_acc
    num_batchs = len(trainloader)
    avg_loss = 0
    train_cor = 0
    train_n = 0
    train_loss = 0

    for batch_idx, (data, labels) in enumerate(trainloader, 1):
        model.train()

        data, labels = data.to(device), labels.to(device)
        logit = model(data)
        loss = criterion(logit, labels)
        loss.backward()

        avg_loss += loss / num_batchs
        optimizer.step()
        optimizer.zero_grad()

        train_cor += acc(logit, labels)
        train_n += len(labels)
        train_loss += loss.item()

        if batch_idx % args.step_batch == 0:
            train_acc = int(train_cor)/train_n
            log('epoch: {:2d}/{}\t|\tbatch: {:2d}/{}\t|\tloss: {:.5f}\t|\tacc: {:.2f}%'.format(
                epoch_idx, args.num_epochs,
                batch_idx, num_batchs,
                train_loss/args.step_batch, train_acc * 100))

            if batch_idx % (args.step_batch*10) == 0:
                total_batch = int((epoch_idx-1) * len(trainloader) + batch_idx)
                eval_loss, eval_acc = evaluation(model, testloader, criterion, device)
                log(' >> epoch: {:2d}\t|\ttotal_batch: {:2d}\t|\teval_loss: {:.5f}\t|\teval_acc: {:.2f}'.format(
                    epoch_idx, total_batch, eval_loss, eval_acc * 100))

                writer.add_scalars('loss', {'train loss': train_loss/args.step_batch, 'eval loss': eval_loss}, total_batch)
                writer.add_scalars('acc', {'train acc': train_acc, 'eval acc': eval_acc}, total_batch)

                if best_acc < eval_acc:
                    best_acc = eval_acc
                    if args.multi_gpu and torch.cuda.device_count() > 1:
                        model.module.save(vocab_save, model_save)
                    else:
                        model.save(vocab_save, model_save)
                    writer.add_text('best acc', '{}b_{:.3f}%'.format(total_batch, best_acc * 100), total_batch)

            train_loss = 0
            train_cor = 0
            train_n = 0


    return avg_loss


def evaluation(model, testloader, criterion, device):
    avg_loss = 0
    cor = 0
    n = 0
    num_batchs = len(testloader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(testloader, 1):
            data, labels = data.to(device), labels.to(device)
            logit = model(data)
            loss = criterion(logit, labels)
            cor += acc(logit, labels)
            n += len(labels)
            avg_loss += loss / num_batchs
    return avg_loss, int(cor) / n

def acc(logit, labels):
    return torch.sum(logit.argmax(1).eq(labels))

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
    parser.add_argument('--description', type=str, default='H128-M8-O64-12layer-pretrained-QQP')

    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--step_batch', type=int, default=25)
    parser.add_argument('--eval_batch_size', type=int, default=64)

    parser.add_argument('--lr', type=float, default=5e-05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--drop_rate', type=float, default=0.)

    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--m', type=int, default=24)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--task', type=str, default='qqp')

    parser.add_argument('--save_vocab', type=bool, default=True)
    parser.add_argument('--pretrained_vocab_path', type=str, default='/media/lxzun/HJ/Workdir/project_ing/EMNLP/model_save/H128-M8-O64-pretrain/vocab_save/embedding_weight', help='load pretrained vocab path')
    parser.add_argument('--pretrained_vocab', type=bool, default=True, help='load pretrained vocab')

    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--pretrained_model_path', type=str, default='/media/lxzun/HJ/Workdir/project_ing/EMNLP/model_save/H128-M8-O64-pretrain/model_save/model_weight', help='load pretrained model path')
    parser.add_argument('--pretrained_model', type=bool, default=True, help='load pretrained model')

    parser.add_argument('--use_cuda', type=str, default='cuda')
    parser.add_argument('--multi_gpu', type=bool, default=True)

    args = parser.parse_args()

    log_root = mk_dir('./log/train')
    today = str(datetime.today().strftime('%y-%m-%d_%H-%M-%S'))
    log_dir = mk_dir(os.path.join(log_root, 'log_{}'.format(today)))
    log_file = os.path.join(log_dir, 'log.txt')
    model_save = mk_dir(os.path.join(log_dir, 'model_save')) if args.save_model else None
    model_save = model_save + '/model_weight' if model_save else None
    vocab_save = mk_dir(os.path.join(log_dir, 'vocab_save')) if args.save_vocab else None
    vocab_save = vocab_save + '/embedding_weight' if vocab_save else None
    writer = SummaryWriter(os.path.join(log_dir, args.description))

    device = args.use_cuda if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    for k, v in args.__dict__.items():
        log('{}: {}'.format(k, v))
    log('Real used device: {}'.format(device))

    trainset = Mydataset(task=args.task, max_length=args.max_seq_length, split='train')
    testset = Mydataset(task=args.task, max_length=args.max_seq_length, split='validation')


    trainloader = DataLoader(trainset, args.batch_size, num_workers=2, shuffle=True, collate_fn=trainset.make_batch)
    testloader = DataLoader(testset, args.batch_size, num_workers=2, shuffle=False, collate_fn=testset.make_batch)


    log('\n---- dataset info ----')
    log('\n* train data *')
    log('- num : {}'.format(len(trainset)))
    log('\n* eval data *')
    log('- num : {}'.format(len(testset)))
    log('----------------------\n')
    num_classes = 0
    if args.task in ['qqp']:
        num_classes=2
        model = MymodelForSequenceClassification(vocab_size=trainset.vocab_size,
                                                 embedding_size=args.embedding_size,
                                                 hidden_size=args.hidden_size,
                                                 m=args.m, out_dim=args.out_dim,
                                                 n_layer=args.n_layer,
                                                 pad_ids=trainset.pad_ids,
                                                 num_classes=num_classes)

    if args.pretrained_vocab:
        model.load(vocab_path=args.pretrained_vocab_path)
    if args.pretrained_model:
        model.load(model_path=args.pretrained_model_path)

    if device == 'cuda':
        model.to(device)

    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer.zero_grad()

    best_acc = 0

    for epoch_idx in range(1, args.num_epochs + 1):
        log('\n\n----------------------- {} epoch start! -----------------------'.format(epoch_idx))
        avg_loss = train(model, trainloader, criterion, optimizer, epoch_idx, testloader, args, device)

    writer.close()