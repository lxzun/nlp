from torch import optim, nn
from torch.utils.data import DataLoader
import torch
import numpy as np
from datetime import datetime
import os
import argparse
from tensorboardX import SummaryWriter
from dataset import Mydataset, Mydataset_spm
from models.Mymodel import MymodelForSequenceClassification
from sklearn.metrics import classification_report as cr
from transformers import AdamW


def train(model, trainloader, criterion, optimizer, scheduler, epoch_idx, testloader, args, device):
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
        scheduler.step()

        if batch_idx % args.step_batch == 0:
            train_acc = int(train_cor)/train_n
            log('epoch: {:2d}/{}\t|\tbatch: {:2d}/{}\t|\tloss: {:.5f}\t|\tacc: {:.2f}%'.format(
                epoch_idx, args.num_epochs,
                batch_idx, num_batchs,
                train_loss/args.step_batch, train_acc * 100))

            if batch_idx % (args.step_batch*5) == 0:
                total_batch = int((epoch_idx-1) * len(trainloader) + batch_idx)
                eval_loss, eval_acc, report = evaluation(model, testloader, criterion, device)
                log(' >> epoch: {:2d}\t|\ttotal_batch: {:2d}\t|\teval_loss: {:.5f}\t|\teval_acc: {:.2f}\n'.format(
                    epoch_idx, total_batch, eval_loss, eval_acc * 100))
                log(report)

                writer.add_scalars('loss', {'train loss': train_loss/args.step_batch, 'eval loss': eval_loss}, total_batch)
                writer.add_scalars('acc', {'train acc': train_acc, 'eval acc': eval_acc}, total_batch)

                if best_acc < eval_acc:
                    best_acc = eval_acc
                    if args.multi_gpu and torch.cuda.device_count() > 1:
                        model.module.save(vocab_save, model_save)
                    else:
                        model.save(vocab_save, model_save)
                    writer.add_text('best acc', report, total_batch)

            train_loss = 0
            train_cor = 0
            train_n = 0


    return avg_loss


def evaluation(model, testloader, criterion, device):
    avg_loss = 0
    cor = 0
    n = 0
    pred_metric = []
    labels_metric = []
    num_batchs = len(testloader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(testloader, 1):
            data, labels = data.to(device), labels.to(device)
            logit = model(data)
            loss = criterion(logit, labels)
            pred_metric.append(np.array(logit.argmax(1).to('cpu')))
            labels_metric.append(np.array(labels.to('cpu')))
            cor += acc(logit, labels)
            n += len(labels)
            avg_loss += loss / num_batchs
    report = reports(np.hstack(pred_metric), np.hstack(labels_metric))
    return avg_loss, int(cor)/n, report

def acc(logit, labels):
    return torch.sum(logit.argmax(1).eq(labels))

def reports(pred, labels):
    return cr(labels, pred, zero_division=0)

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

    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--step_batch', type=int, default=50)
    parser.add_argument('--eval_batch_size', type=int, default=120)

    parser.add_argument('--lr', type=float, default=5e-05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--drop_rate', type=float, default=0.)

    parser.add_argument('--new_vocab', type=int, default=1)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--m', type=int, default=32)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--attd_mode', type=int, default=2)
    parser.add_argument('--max_seq_length', type=int, default=512*2)
    parser.add_argument('--task', type=str, default='qnli', help='qqp, mrpc, sst2, rte, qnli, mnli')
    parser.add_argument('--freeze', type=bool, default=False, help='freezing main model w/o ouput layer')
    

    parser.add_argument('--save_vocab', type=bool, default=True)
    parser.add_argument('--pretrained_vocab_path', type=str, default='log_under_12/train/log_21-06-11_23-15-50/vocab_save/embedding_weight', help='load pretrained vocab path')
    parser.add_argument('--pretrained_vocab', type=bool, default=True, help='load pretrained vocab')

    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--pretrained_model_path', type=str, default='log_under_12/train/log_21-06-11_23-15-50/model_save/model_weight', help='load pretrained model path')
    parser.add_argument('--pretrained_model', type=bool, default=True, help='load pretrained model')

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

    if args.new_vocab:
        trainset = Mydataset_spm(task=args.task, vocab=f'vocab/vocab_{args.new_vocab}.model', max_length=args.max_seq_length, split='train')
        testset = Mydataset_spm(task=args.task, vocab=f'vocab/vocab_{args.new_vocab}.model', max_length=args.max_seq_length, split='validation')

    else:
        trainset = Mydataset(task=args.task, max_length=args.max_seq_length, split='train')
        testset = Mydataset(task=args.task, max_length=args.max_seq_length, split='validation')


    trainloader = DataLoader(trainset, args.batch_size, num_workers=2, shuffle=True)
    testloader = DataLoader(testset, args.eval_batch_size, num_workers=2, shuffle=False)


    log('\n---- dataset info ----')
    log('\n* train data *')
    log('- num : {}'.format(len(trainset)))
    log('\n* eval data *')
    log('- num : {}'.format(len(testset)))
    log('----------------------\n')
    num_classes = 0
    if args.task in ['qqp', 'mrpc', 'sst2', 'rte', 'qnli']:
        num_classes=2
        model = MymodelForSequenceClassification(vocab_size=trainset.vocab_size,
                                                 embedding_size=args.embedding_size,
                                                 hidden_size=args.hidden_size,
                                                 m=args.m, out_dim=args.out_dim,
                                                 n_layer=args.n_layer,
                                                 pad_ids=trainset.pad_ids,
                                                 num_classes=num_classes,
                                                 attd_mode=args.attd_mode,
                                                 drop_rate=args.drop_rate,
                                                 freeze=args.freeze)

    elif args.task in ['mnli']:
        num_classes=3
        model = MymodelForSequenceClassification(vocab_size=trainset.vocab_size,
                                                 embedding_size=args.embedding_size,
                                                 hidden_size=args.hidden_size,
                                                 m=args.m, out_dim=args.out_dim,
                                                 n_layer=args.n_layer,
                                                 pad_ids=trainset.pad_ids,
                                                 num_classes=num_classes,
                                                 attd_mode=args.attd_mode,
                                                 drop_rate=args.drop_rate,
                                                 freeze=args.freeze)

    if args.pretrained_vocab:
        model.load(vocab_path=args.pretrained_vocab_path)
    if args.pretrained_model:
        model.load(model_path=args.pretrained_model_path)

    if device == 'cuda':
        model.to(device)

    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=5e-5, step_size_up=1000, step_size_down=1000, max_lr=2e-4, mode='triangular', cycle_momentum=False)
    optimizer.zero_grad()

    best_acc = 0

    for epoch_idx in range(1, args.num_epochs + 1):
        log('\n\n----------------------- {} epoch start! -----------------------'.format(epoch_idx))
        avg_loss = train(model, trainloader, criterion, optimizer, scheduler, epoch_idx, testloader, args, device)

    writer.close()