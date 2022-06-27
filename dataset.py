from datasets import load_dataset
from transformers import AutoTokenizer
from torch import nn
import random as rd
import numpy as np
import torch
import sentencepiece as spm

class Mydataset(nn.Module):
    def __init__(self, task='pretrain', max_length=512, split='train', seq_mask=False):
        super(Mydataset, self).__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_size = self.tokenizer.vocab_size
        self.mask_ids = self.tokenizer.mask_token_id
        self.pad_ids = self.tokenizer.pad_token_id
        self.task = task
        self.seq_mask = seq_mask
        if task == 'pretrain':
            self.data = load_dataset('openwebtext', split='train')
            self.length = self.data.num_rows
        else:
            self.data = load_dataset('glue', self.task, split=split)
            self.length = self.data.num_rows
        print('load Done!')

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.task == 'pretrain':
            data = self.tokenizer(self.data[item]["text"], return_tensors='pt', truncation=True, padding='max_length',
                                  max_length=self.max_length, return_attention_mask=False, return_token_type_ids=False)
            label = data['input_ids'].flatten()
            data = label.clone()
            indices_a = list(range(len(data)))
            k = int(len(data) * 0.25)
            if self.seq_mask:
                indices = set()
                while len(indices) < k:
                    i = np.random.randint(1, 5)
                    s = rd.sample(indices_a[:-i], k=1)[0]
                    indices.update(list(range(s, s+i)))

            else: indices = rd.sample(indices_a, k=k)
            indices = list(indices)
            data[indices] = self.mask_ids
            mask = torch.zeros_like(data, dtype=torch.bool)
            mask[indices] = True
            return data, label, mask

        if self.task == 'qqp':
            data = self.tokenizer(self.data[item]["question1"], text_pair=self.data[item]["question2"],
                                  truncation=True, max_length=self.max_length, padding='max_length',
                                  return_tensors='pt', return_attention_mask=False,
                                  return_token_type_ids=False)['input_ids'].flatten()
            label = self.data[item]['label']
            return data, label, self.data[item]['idx']

        
        elif self.task in {'mrpc', 'rte', 'stsb'}:
            data = self.tokenizer(self.data[item]["sentence1"], text_pair=self.data[item]["sentence2"],
                                   truncation=True, max_length=self.max_length, padding='max_length',
                                   return_tensors='pt', return_attention_mask=False,
                                   return_token_type_ids=False)['input_ids'].flatten()
            label = self.data[item]['label']
            return data, label, self.data[item]['idx']

        elif self.task in {'sst2', 'cola'}:
            data = self.tokenizer(self.data[item]["sentence"],
                                   truncation=True, max_length=self.max_length, padding='max_length',
                                   return_tensors='pt', return_attention_mask=False,
                                   return_token_type_ids=False)['input_ids'].flatten()
            label = self.data[item]['label']
            return data, label, self.data[item]['idx']

        elif self.task == 'mnli':
            data = self.tokenizer(self.data[item]["premise"], text_pair=self.data[item]["hypothesis"],
                                   truncation=True, max_length=self.max_length, padding='max_length',
                                   return_tensors='pt', return_attention_mask=False,
                                   return_token_type_ids=False)['input_ids'].flatten()
            label = self.data[item]['label']
            return data, label, self.data[item]['idx']

        elif self.task == 'qnli':
            data = self.tokenizer(self.data[item]["question"], text_pair=self.data[item]["sentence"],
                                   truncation=True, max_length=self.max_length, padding='max_length',
                                   return_tensors='pt', return_attention_mask=False,
                                   return_token_type_ids=False)['input_ids'].flatten()
            label = self.data[item]['label']
            return data, label, self.data[item]['idx']
        
        elif self.task == 'wnli':
            data = self.tokenizer(self.data[item]["sentence1"], text_pair=self.data[item]["sentence2"],
                                   truncation=True, max_length=self.max_length, padding='max_length',
                                   return_tensors='pt', return_attention_mask=False,
                                   return_token_type_ids=False)['input_ids'].flatten()
            label = self.data[item]['label']
            return data, label, self.data[item]['idx']

        elif self.task == 'ax':
            data = self.tokenizer(self.data[item]["premise"], text_pair=self.data[item]["hypothesis"],
                                   truncation=True, max_length=self.max_length, padding='max_length',
                                   return_tensors='pt', return_attention_mask=False,
                                   return_token_type_ids=False)['input_ids'].flatten()
            label = self.data[item]['label']
            return data, label, self.data[item]['idx']



class Mydataset_spm(nn.Module):
    def __init__(self, task='pretrain', vocab=None, max_length=512, split='train', seq_mask=False):
        super(Mydataset_spm, self).__init__()
        self.max_length = max_length
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(vocab)
        self.vocab_size = self.tokenizer.vocab_size()
        self.mask_ids = self.tokenizer.piece_to_id('[MASK]')
        self.sep_ids = self.tokenizer.piece_to_id('[SEP]')
        self.pad_ids = self.tokenizer.pad_id()
        self.task = task
        self.seq_mask = seq_mask
        if task == 'pretrain':
            self.data = load_dataset('openwebtext', split='train')
            self.length = self.data.num_rows
        else:
            self.data = load_dataset('glue', self.task, split=split)
            self.length = self.data.num_rows
        print('load Done!')

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.task == 'pretrain':
            data = self.tokenizer.encode_as_ids(self.data[item]['text'])
            if len(data) > self.max_length:
                indices = list(range(len(data)))
                s = rd.sample(indices[:-self.max_length], k=1)[0]
                data = data[s:s+self.max_length]
            else:
                data = data + [self.pad_ids] * (self.max_length - len(data))
            data = np.array(data, dtype=int)
            label = np.array(data, dtype=int)
            indices_a = list(range(len(data)))
            k = int(len(data) * 0.25)
            if self.seq_mask:
                indices = set()
                while len(indices) < k:
                    i = np.random.randint(1, 5)
                    s = rd.sample(indices_a[:-i], k=1)[0]
                    indices.update(list(range(s, s+i)))

            else: indices = rd.sample(indices, k=k)
            indices = list(indices)
            data[indices] = self.mask_ids
            data = torch.LongTensor(data)
            mask = torch.zeros_like(data, dtype=torch.bool)
            mask[indices] = True
            return data, torch.LongTensor(label), mask

        elif self.task == 'qqp':
            data1 = self.tokenizer.encode_as_ids(self.data[item]["question1"])
            data2 = self.tokenizer.encode_as_ids(self.data[item]["question2"])
            data = data1 + [self.sep_ids] + data2
            if len(data) > self.max_length:
                data = data[:self.max_length]
            else:
                data = data + [self.pad_ids] * (self.max_length - len(data))
            data = np.array(data, dtype=int)

            label = self.data[item]['label']
            return torch.LongTensor(data), label, self.data[item]['idx']

        elif self.task in {'mrpc', 'rte', 'stsb', 'wnli'}:
            data1 = self.tokenizer.encode_as_ids(self.data[item]["sentence1"])
            data2 = self.tokenizer.encode_as_ids(self.data[item]["sentence2"])
            data = data1 + [self.sep_ids] + data2
            if len(data) > self.max_length:
                data = data[:self.max_length]
            else:
                data = data + [self.pad_ids] * (self.max_length - len(data))
            data = np.array(data, dtype=int)

            label = self.data[item]['label']
            return torch.LongTensor(data), label, self.data[item]['idx']

        elif self.task in {'sst2', 'cola'}:
            data = self.tokenizer.encode_as_ids(self.data[item]["sentence"])
            if len(data) > self.max_length:
                data = data[:self.max_length]
            else:
                data = data + [self.pad_ids] * (self.max_length - len(data))
            data = np.array(data, dtype=int)

            label = self.data[item]['label']
            return torch.LongTensor(data), label, self.data[item]['idx']

        elif self.task in {'mnli', 'ax'}:
            data1 = self.tokenizer.encode_as_ids(self.data[item]["premise"])
            data2 = self.tokenizer.encode_as_ids(self.data[item]["hypothesis"])
            data = data1 + [self.sep_ids] + data2
            if len(data) > self.max_length:
                data = data[:self.max_length]
            else:
                data = data + [self.pad_ids] * (self.max_length - len(data))
            data = np.array(data, dtype=int)

            label = self.data[item]['label']
            return torch.LongTensor(data), label, self.data[item]['idx']

        elif self.task == 'qnli':
            data1 = self.tokenizer.encode_as_ids(self.data[item]["question"])
            data2 = self.tokenizer.encode_as_ids(self.data[item]["sentence"])
            data = data1 + [self.sep_ids] + data2
            if len(data) > self.max_length:
                data = data[:self.max_length]
            else:
                data = data + [self.pad_ids] * (self.max_length - len(data))
            data = np.array(data, dtype=int)

            label = self.data[item]['label']
            return torch.LongTensor(data), label, self.data[item]['idx']
            


if __name__=='__main__':
    a = Mydataset(task='qqp', return_attention_mask=True, return_token_type_ids=True)[1]