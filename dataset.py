from datasets import load_dataset
from transformers import AutoTokenizer
from torch import nn
import random as rd
import numpy as np
import torch
import sentencepiece as spm

class Mydataset(nn.Module):
    def __init__(self, task='pretrain', max_length=512, split='train', seq_mask=False, return_attention_mask=False, return_token_type_ids=False):
        super(Mydataset, self).__init__()
        self.return_attention_maks = return_attention_mask
        self.return_token_type_ids = return_token_type_ids
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
        if task == 'qqp':
            self.data = load_dataset('glue', 'qqp', split=split)
            self.length = self.data.num_rows
        print('load Done!')

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.task == 'pretrain':
            data = self.tokenizer(self.data[item]["text"], return_tensors='pt', truncation=True, max_length=self.max_length,
                                  return_attention_mask=False, return_token_type_ids=False)['input_ids']
            label = data[:, 1:].flatten()
            data = label.clone()
            indices = list(range(len(data)))
            k = int(len(data) * 0.15)
            if self.seq_mask:
                s = rd.sample(indices[:-k], k=1)[0]
                indices = list(range(s, s+k))
            else: indices = rd.sample(indices, k=k)
            data[indices] = self.mask_ids
            return data, label

        if self.task == 'qqp':
            data = self.tokenizer(self.data[item]["question1"], text_pair=self.data[item]["question2"],
                                  truncation=True, max_length=self.max_length,
                                  return_tensors='pt', return_attention_mask=self.return_attention_maks,
                                  return_token_type_ids=self.return_token_type_ids)['input_ids'][:, 1:].flatten()
            label = self.data[item]['label']
            return data, label

    def make_batch(self, samples):
        data = [i[0] for i in samples]
        label = [i[1] for i in samples]
        data = nn.utils.rnn.pad_sequence(data, True, self.pad_ids)
        if self.task == 'pretrain':
            label = nn.utils.rnn.pad_sequence(label, True, self.pad_ids)
        return data, torch.LongTensor(label)

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
            data = np.array(data, dtype=int)
            if len(data) > self.max_length:
                data = data[:self.max_length]
            label = np.array(data, dtype=int)
            indices = list(range(len(data)))
            k = int(len(data) * 0.15)
            if self.seq_mask:
                s = rd.sample(indices[:-k], k=1)[0]
                indices = list(range(s, s+k))
            else: indices = rd.sample(indices, k=k)
            data[indices] = self.mask_ids
            return torch.LongTensor(data), torch.LongTensor(label)

        elif self.task == 'qqp':
            data1 = self.tokenizer.encode_as_ids(self.data[item]["question1"])
            data2 = self.tokenizer.encode_as_ids(self.data[item]["question2"])
            data = data1 + [self.sep_ids] + data2
            if len(data) > self.max_length:
                data = data[:self.max_length]
            data = np.array(data, dtype=int)

            label = self.data[item]['label']
            return torch.LongTensor(data), label

        elif self.task in ['mrpc', 'rte']:
            data1 = self.tokenizer.encode_as_ids(self.data[item]["sentence1"])
            data2 = self.tokenizer.encode_as_ids(self.data[item]["sentence2"])
            data = data1 + [self.sep_ids] + data2
            if len(data) > self.max_length:
                data = data[:self.max_length]
            data = np.array(data, dtype=int)

            label = self.data[item]['label']
            return torch.LongTensor(data), label

        elif self.task == 'sst2':
            data = self.tokenizer.encode_as_ids(self.data[item]["sentence"])
            if len(data) > self.max_length:
                data = data[:self.max_length]
            data = np.array(data, dtype=int)

            label = self.data[item]['label']
            return torch.LongTensor(data), label

        elif self.task == 'mnli':
            data1 = self.tokenizer.encode_as_ids(self.data[item]["premise"])
            data2 = self.tokenizer.encode_as_ids(self.data[item]["hypothesis"])
            data = data1 + [self.sep_ids] + data2
            if len(data) > self.max_length:
                data = data[:self.max_length]
            data = np.array(data, dtype=int)

            label = self.data[item]['label']
            return torch.LongTensor(data), label

        elif self.task == 'qnli':
            data1 = self.tokenizer.encode_as_ids(self.data[item]["question"])
            data2 = self.tokenizer.encode_as_ids(self.data[item]["sentence"])
            data = data1 + [self.sep_ids] + data2
            if len(data) > self.max_length:
                data = data[:self.max_length]
            data = np.array(data, dtype=int)

            label = self.data[item]['label']
            return torch.LongTensor(data), label


    def make_batch(self, samples):
        data = [i[0] for i in samples]
        label = [i[1] for i in samples]
        data = nn.utils.rnn.pad_sequence(data, True, self.pad_ids)
        if self.task == 'pretrain':
            label = nn.utils.rnn.pad_sequence(label, True, self.pad_ids)
        return data, torch.LongTensor(label)


if __name__=='__main__':
    Mydataset(max_length=9999999)