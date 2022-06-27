import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from dataset import Mydataset, Mydataset_spm
from models.Mymodel import MymodelForSequenceClassification 
import numpy as np

path = './log'
chp = 'model_best.pth.tar'
# chp = 'checkpoint.pth.tar'
cudnn.deterministic = True
cudnn.benchmark = True
ax_wnli = False

with torch.no_grad():
    for root, dirs, files in os.walk(path):
        if dirs == []:
            task, args = root.split(os.path.sep)[2:4]
            if task in {'pretrain', 'cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'qnli', 'rte'}:
                continue
            attd_mode, new_vocab, max_seq_length, embedding_size, hidden_size, m, k, out_dim, n_layer = [int(i[1:]) for i in args.split('_')]
            print(f'model size: {((new_vocab*1000+6 if new_vocab else 30522)*embedding_size+embedding_size*hidden_size+hidden_size+(k**2 * m * out_dim * 2 + out_dim * 2 + m*out_dim + m * out_dim + m )*n_layer)/1000000:.2f} M')
            
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

            val_loader = torch.utils.data.DataLoader(testset, 1024)
            if task == 'mnli':
                val_loader2 = torch.utils.data.DataLoader(testset2, 1024)
            

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

            if torch.cuda.is_available():
                model = model.to('cuda:0')
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
                for k, (data, _, idx) in enumerate(val_loader, 0):
                    if torch.cuda.is_available():
                        data = data.to('cuda:0')

                    output = model(data)
                    if task in {'rte', 'qnli'}:
                        label = np.asarray(output.argmax(-1).to('cpu').numpy(), dtype=str)
                        label[label=='0'] = 'entailment'
                        label[label=='1'] = 'not_entailment'
                    elif task in {'mnli', 'ax'}:
                        label = np.asarray(output.argmax(-1).to('cpu').numpy(), dtype=str)
                        label[label=='0'] = 'entailment'
                        label[label=='1'] = 'neutral'
                        label[label=='2'] = 'contradiction'
                    else:
                        label = np.asarray(output.argmax(-1).to('cpu').numpy(), dtype=str)
                    
                    for i in range(len(label)):
                        f.write(f'{idx[i].item()}\t{label[i]}\n')

                    if k % 5 == 0:
                        print(f'[{k}/{len(val_loader)}]')

            if task == 'mnli':
                file_name = os.path.join(root, f'MNLI-mm.tsv')
                with open(file_name, 'w') as f:
                    f.write('id\tlabel\n')
                    for k, (data, _, idx) in enumerate(val_loader2, 0):
                        if torch.cuda.is_available():
                            data = data.to('cuda:0')
                        
                        output = model(data)
                        if task in {'rte', 'qnli'}:
                            label = np.asarray(output.argmax(-1).to('cpu').numpy(), dtype=str)
                            label[label=='0'] = 'entailment'
                            label[label=='1'] = 'not_entailment'
                        elif task in {'mnli', 'ax'}:
                            label = np.asarray(output.argmax(-1).to('cpu').numpy(), dtype=str)
                            label[label=='0'] = 'entailment'
                            label[label=='1'] = 'neutral'
                            label[label=='2'] = 'contradiction'
                        else:
                            label = np.asarray(output.argmax(-1).to('cpu').numpy(), dtype=str)
                        
                        for i in range(len(label)):
                            f.write(f'{idx[i].item()}\t{label[i]}\n')

                        if k % 5 == 0:
                            print(f'[{k}/{len(val_loader)}]')

    if ax_wnli:
        for task in {'wnli', 'ax'}:
            testset = Mydataset(task=task, max_length=512, seq_mask=True, split='test')

            val_loader = torch.utils.data.DataLoader(testset, 128)

            if task == 'wnli': file_name = 'WNLI.tsv'
            if task == 'ax': file_name = 'AX.tsv'

            with open(file_name, 'w') as f:
                f.write('id\tlabel\n')
                for k, (data, _, idx) in enumerate(val_loader, 0):
                    label = np.asarray(torch.zeros_like(idx).numpy(), dtype=str)
                    if task in {'mnli', 'ax'}:
                        label[label=='0'] = 'entailment'
                    
                    for i in range(len(label)):
                        f.write(f'{idx[i].item()}\t{label[i]}\n')

                    if k % 5 == 0:
                        print(f'[{k}/{len(val_loader)}]')
