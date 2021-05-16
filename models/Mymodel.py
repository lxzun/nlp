import torch
from torch import nn


class Myattention(nn.Module):
    def __init__(self, m, out_dim, k=3, drop_rate=0):
        super(Myattention, self).__init__()
        assert (k % 2 != 0), 'K must be Odd'
        self.q = nn.Conv2d(m, out_dim, k, padding=(k-1)//2)
        self.k = nn.Conv2d(m, out_dim, k, padding=(k-1)//2)
        self.v = nn.Conv2d(m, out_dim, k, padding=(k-1)//2)
        self.softmax = nn.Softmax(dim=0)
        self.gelu = nn.GELU()
        self.batchnorm = nn.BatchNorm2d(m)
        self.dropout = nn.Dropout(drop_rate)
        self.conv = nn.Conv2d(out_dim, m, 3, padding=1)

    def forward(self, x):
        input_shape = x.size() # b x m x seq_length x n
        m, seq_length, n = input_shape[1:]

        q = self.dropout(self.gelu(self.q(x)))          # b x out_dim x seq_lengh x n
        k = self.dropout(self.gelu(self.k(x)))
        v = self.dropout(self.gelu(self.v(x)))

        out_dim = q.size()[1]

        q = q.view(-1, seq_length, out_dim * n)
        k = k.view(-1, out_dim * n, seq_length)
        v = v.view(-1, seq_length, out_dim * n)

        score = self.softmax(torch.matmul(q, k))       # b x seq_length x seq_length
        out = torch.matmul(score, v)                   # b x seq_length x out_dim*n
        out = out.view(-1, out_dim, seq_length, n)     # b x out_dim x seq_length x n
        out = self.dropout(self.gelu(self.conv(out)))

        out = out + x
        out = self.batchnorm(out)

        return out

class Mymodel(nn.Module):
    def __init__(self, m, out_dim, hidden_size, vocab_size, n_layer, pad_ids, k=3, drop_rate=0):
        super(Mymodel, self).__init__()
        assert (hidden_size % m == 0), f"hidden_size: {hidden_size}, m: {m} ==> Must be hidden_size % m == 0!"

        self.m = m

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_ids)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(drop_rate)

        self.model = nn.ModuleList([Myattention(m, out_dim, k, drop_rate) for _ in range(n_layer)])

    def forward(self, x):
        # b, seq_length
        x = self.embedding(x)
        x = self.dropout(self.layernorm(x))

        input_shape = x.size()  # b, seq_length, hidden_size
        seq_length, hidden_size = input_shape[1:]

        x = x.view(-1, self.m, seq_length, hidden_size//self.m)  # b x m x seq_length x n

        all_outputs = ()
        for i, layer in enumerate(self.model):
            all_outputs = all_outputs + (x,)
            x = layer(x)

        return (x,) + all_outputs

    def save(self, vocab_path=None, model_path=None):
        if vocab_path: torch.save(self.embedding.state_dict(), vocab_path)
        if model_path: torch.save(self.model.state_dict(), model_path)

    def load(self, vocab_path=None, model_path=None):
        if vocab_path: self.embedding.load_state_dict(torch.load(vocab_path))
        if model_path: self.model.load_state_dict(torch.load(model_path))

class Mymodelforpretrain(nn.Module):
    def __init__(self, m, out_dim, hidden_size, vocab_size, n_layer, pad_ids, k=3, drop_rate=0):
        super(Mymodelforpretrain, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.model = Mymodel(m, out_dim, hidden_size, vocab_size, n_layer, pad_ids, k, drop_rate)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # b x seq_length

        outputs = self.model(x)

        hidden = outputs[0]                                      # b x m x seq_length x n
        hidden = hidden.view(-1, x.size()[1], self.hidden_size)  # b x seq_length x hidden_size

        out = self.linear(hidden)                                # b x seq_length x vocab_size
        out = out.view(-1, self.vocab_size)                      # b*seq_length x vocab_size

        return out

    def save(self, vocab_path, model_path):
        self.model.save(vocab_path, model_path)

    def load(self, vocab_path, model_path):
        self.model.load(vocab_path, model_path)

# class Mymodelforclassification(nn.Module):
#     def __init__(self, m, out_dim, hidden_size, vocab_size, n_layer, pad_ids, num_class, k=3, drop_rate=0):
#         super(Mymodelforclassification, self).__init__()
#         self.vocab_size = vocab_size
#         self.hidden_size = hidden_size
#         self.model = Mymodel(m, out_dim, hidden_size, vocab_size, n_layer, pad_ids, k, drop_rate)
#         self.linear = nn.Linear(hidden_size, vocab_size)