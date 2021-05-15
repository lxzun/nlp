from datasets import load_dataset
from transformers import AutoTokenizer

def load_datasets(task='pretrain'):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    if task == 'pretrain':
        data = load_dataset('openwebtext')
        print('load Done!')
        data = data.map(pretrain_encode(tokenizer, data), batched=True)
        print('encode Done!')
        data = data.set_format(type='torch', columns=['text'])
        print('set format')
    return data

def pretrain_encode(tokenizer, data):
    return tokenizer(data["text"], truncation=True, padding='max_length')

if __name__=='__main__':
    a = load_datasets()
    print('done!')