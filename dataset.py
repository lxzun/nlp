from datasets import load_dataset
from transformers import AutoTokenizer
from torch import nn
import random as rd

class Mydataset(nn.Module):
    def __init__(self, task='pretrain', split='train', return_attention_mask=False, return_token_type_ids=False):
        super(Mydataset, self).__init__()
        self.return_attention_maks = return_attention_mask
        self.return_token_type_ids = return_token_type_ids
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_size = self.tokenizer.vocab_size
        self.mask_ids = self.tokenizer.mask_token_id
        self.pad_ids = self.tokenizer.pad_token_id
        self.task = task
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
            data = self.tokenizer(self.data[item]["text"], return_tensors='pt', return_attention_mask=False,
                                  return_token_type_ids=False)['input_ids']
            label = data[:, 1:].flatten()
            data = label.clone()
            indices = list(range(len(data)))
            indices = rd.sample(indices, k=int(len(data) * 0.15))
            data[indices] = self.mask_ids
            return data, label
        if self.task == 'qqp':
            data = self.tokenizer(self.data[item]["text"], return_tensors='pt', return_attention_mask=self.return_attention_maks,
                                  return_token_type_ids=self.return_token_type_ids)['input_ids']
            label = data[:, 1:].flatten()
            data = label.clone()
            indices = list(range(len(data)))
            indices = rd.sample(indices, k=int(len(data)*0.15))
            data[indices] = self.mask_ids
            return data, label

    def make_batch(self, samples):
        data = [i[0] for i in samples]
        label = [i[1] for i in samples]
        data = nn.utils.rnn.pad_sequence(data, True, self.pad_ids)
        label = nn.utils.rnn.pad_sequence(label, True, self.pad_ids)
        return data, label


if __name__=='__main__':
    print()