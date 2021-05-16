from datasets import load_dataset
from transformers import AutoTokenizer
from torch import nn

class Mydataset(nn.Module):
    def __init__(self, task='pretrain', max_length=1025, split='train', return_attention_mask=False, return_token_type_ids=False):
        super(Mydataset, self).__init__()
        self.return_attention_maks = return_attention_mask
        self.return_token_type_ids = return_token_type_ids
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_ids = self.tokenizer.pad_token_id
        self.task = task
        if task == 'pretrain':
            self.data = load_dataset('openwebtext', split='train')
            self.length = self.data.num_rows
            if split == 'train':
                self.length = int(self.length*0.9)
                self.data = self.data[:self.length]
            else:
                self.length = int(self.length*0.9)
                self.data = self.data[self.length:]
            print('load Done!')

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.task == 'pretrain':
            data = self.tokenizer(self.data[item]["text"], truncation=True, padding='max_length',
                                  max_length=self.max_length, return_tensors='pt', return_attention_mask=False,
                                  return_token_type_ids=False)['input_ids']
            data = data[:, 1:].flatten()
            return data, data



if __name__=='__main__':
    print()