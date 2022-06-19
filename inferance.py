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
from models.Mymodel import MymodelForSequenceClassification 
from sklearn.metrics import f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime

path = './log'
chp = 'model_best.pth.tar'
# chp = 'checkpoint.pth.tar'
cudnn.deterministic = True
cudnn.benchmark = True

with torch.no_grad():
    for root, dirs, files in os.walk(path):
        if dirs == []:
            task, args = root.split(os.path.sep)[2:4]
            if task == 'pretrain':
                continue
            attd_mode, new_vocab, max_seq_length, embedding_size, hidden_size, m, k, out_dim, n_layer = [int(i[1:]) for i in args.split('_')]
            if new_vocab: 
                if task == 'mnli':
                    testset = Mydataset_spm(task=task, vocab=f'vocab/vocab_{new_vocab}.model', max_length=max_seq_length, split='test_matched')
                    testset2 = Mydataset_spm(task=task, vocab=f'vocab/vocab_{new_vocab}.model', max_length=max_seq_length, split='test_mismatched')
                else:
                    testset = Mydataset_spm(task=task, vocab=f'vocab/vocab_{new_vocab}.model', max_length=max_seq_length, split='test')


            else: 
                if task == 'mnli':
                    testset = Mydataset(task=task, max_length=max_seq_length, seq_mask=True, split='test_matched')
                    testset2 = Mydataset(task=task, max_length=max_seq_length, seq_mask=True, split='test_mismatched')
                else:
                    testset = Mydataset(task=task, max_length=max_seq_length, seq_mask=True, split='test')

            val_loader = torch.utils.data.DataLoader(testset, 1)
            if task == 'mnli':
                val_loader2 = torch.utils.data.DataLoader(testset2, 1)
            

            n_classes = 2
            if task in {'mnli', 'ax'}:
                n_classes = 3
            model = MymodelForSequenceClassification(vocab_size=testset.vocab_size,
                                                    embedding_size=embedding_size,
                                                    hidden_size=hidden_size,
                                                    m=m, out_dim=out_dim,
                                                    n_layer=n_layer,
                                                    pad_ids=testset.pad_ids,
                                                    num_classes=n_classes,
                                                    k=k,
                                                    attd_mode=attd_mode)

            if os.path.isfile(os.path.join(root, chp)):
                print("=> loading checkpoint...")
                checkpoint = torch.load(os.path.join(root, chp))
                checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}

                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint!")
            else:
                print("=> no checkpoint found at '{}'".format(chp))

            model.eval()

            if task == 'cola': file_name = os.path.join(root, 'CoLA.tsv')
            if task == 'sst2': file_name = os.path.join(root, 'SST-2.tsv')
            if task == 'mrpc': file_name = os.path.join(root, 'MRPC.tsv')
            if task == 'stsb': file_name = os.path.join(root, 'STS-B.tsv')
            if task == 'qqp': file_name = os.path.join(root, 'QQP.tsv')
            if task == 'mnli': file_name = os.path.join(root, 'MNLI-m.tsv')
            if task == 'qnli': file_name = os.path.join(root, 'QNLI.tsv')
            if task == 'rte': file_name = os.path.join(root, 'RTE.tsv')

            with open(file_name, 'w') as f:
                f.write('id\tlabel\n')
                for k, (data, target) in enumerate(val_loader, 0):
                    output = model(data)
                    if task in {'rte', 'qnli'}:
                        label = 'entailment' if output.argmax().item() == 0 else 'not_entailment'
                    elif task in {'mnli', 'ax'}:
                        if output.argmax().item() == 0:
                            label = 'entailment'
                        elif output.argmax().item() == 0:
                            label = 'neutral'
                        else:
                            label = 'contradiction'
                    else:
                        label = output.argmax().item()
                    f.write(f'{k}\t{label}\n')

                    if k % 100 == 0:
                        print(f'[{k}/{len(val_loader)}]')

            if task == 'mnli':
                file_name = os.path.join(root, f'MNLI-mm.tsv')
                with open(file_name, 'w') as f:
                    f.write('id\tlabel\n')
                    for k, (data, target) in enumerate(val_loader2, 0):
                        output = model(data)
                        if task in {'rte', 'qnli'}:
                            label = 'entailment' if output.argmax().item() == 0 else 'not_entailment'
                        elif task in {'mnli', 'ax'}:
                            if output.argmax().item() == 0:
                                label = 'entailment'
                            elif output.argmax().item() == 0:
                                label = 'neutral'
                            else:
                                label = 'contradiction'
                        else:
                            label = output.argmax().item()
                        f.write(f'{k}\t{label}\n')

                        if k % 100 == 0:
                            print(f'[{k}/{len(val_loader)}]')
