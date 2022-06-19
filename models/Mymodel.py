import torch
from torch import nn
import shutil
import os

class Myattention(nn.Module):
    def __init__(self, m, out_dim, hidden_size, k=3, drop_rate=0):
        super(Myattention, self).__init__()
        assert (k % 2 != 0), 'K must be Odd'
        self.hidden = hidden_size
        self.q = nn.Conv2d(m, out_dim, k, padding=(k-1)//2, groups=m)
        self.k = nn.Conv2d(m, out_dim, k, padding=(k-1)//2, groups=m)
        self.v = nn.Conv2d(m, out_dim, k, padding=(k-1)//2, groups=m)
        self.gelu = nn.GELU()
        self.layernorm = nn.LayerNorm(hidden_size // m)
        self.dropout = nn.Dropout(drop_rate)
        self.conv1= nn.Conv2d(out_dim, m, 1)
        # self.conv2 = nn.Conv2d(m, hidden_size, 1)
        # self.conv3 = nn.Conv2d(hidden_size, m, 1)

    def forward(self, x):
        input_shape = x.size() # b x m x seq_length x n
        m, seq_length, n = input_shape[1:]

        q = self.dropout(self.gelu(self.q(x)))          # b x out_dim x seq_length x n
        k = self.dropout(self.gelu(self.k(x)))
        v = self.dropout(self.gelu(self.v(x)))

        out_dim = q.size()[1]

        q = torch.transpose(q, 1, 2).contiguous()                    # b x seq_length x out_dim x n
        q = q.view(-1, seq_length, out_dim * n)                      # b x seq_length x out_dim*n

        k = torch.transpose(k, 2, 3).contiguous()                    # b x out_dim x n x seq_length
        k = k.view(-1, out_dim * n, seq_length)                      # b x out_dim*n x seq_length

        v = torch.transpose(v, 1, 2).contiguous()
        v = v.view(-1, seq_length, out_dim * n)

        score = torch.matmul(q, k)/(self.hidden**(1/2))        # b x seq_length x seq_length
        score = score.softmax(-1)
        out = torch.matmul(score, v)                                 # b x seq_length x out_dim*n
        out = out.view(-1, seq_length, out_dim, n)                   # b x seq_length x out_dim x n
        out = torch.transpose(out, 1, 2).contiguous()                # b x out_dim x seq_length x n
        out = self.conv1(out)                                        # b x m x seq_length x n
        out = self.layernorm(out)
        out = out + x

        # out1 = self.conv2(out)                                        # b x hidden_size x seq_length x n
        # out1 = self.conv3(out1)                                        # b x m x seq_length x n
        # out1 = self.layernorm(out1)
        
        # out = out + out1


        return out

class Myattention2(nn.Module):
    def __init__(self, m, out_dim, hidden_size, k=3, drop_rate=0):
        super(Myattention2, self).__init__()
        assert (k % 2 != 0), 'K must be Odd'
        self.hidden = hidden_size
        self.q = nn.Conv2d(m, out_dim, k, padding=(k-1)//2, groups=m)
        self.k = nn.Conv2d(m, out_dim, k, padding=(k-1)//2, groups=m)
        self.v = nn.Conv2d(m, out_dim, 1, bias=False, groups=m)
        self.gelu = nn.GELU()
        self.layernorm = nn.LayerNorm(hidden_size // m)
        self.dropout = nn.Dropout(drop_rate)
        self.conv1= nn.Conv2d(out_dim, m, 1)
        # self.conv2 = nn.Conv2d(m, hidden_size, 1)
        # self.conv3 = nn.Conv2d(hidden_size, m, 1)

    def forward(self, x):
        input_shape = x.size() # b x m x seq_length x n
        m, seq_length, n = input_shape[1:]

        q = self.dropout(self.gelu(self.q(x)))          # b x out_dim x seq_length x n
        k = self.dropout(self.gelu(self.k(x)))
        v = self.dropout(self.gelu(self.v(x)))

        out_dim = q.size()[1]

        q = torch.transpose(q, 1, 2).contiguous()                    # b x seq_length x out_dim x n
        q = q.view(-1, seq_length, out_dim * n)                      # b x seq_length x out_dim*n

        k = torch.transpose(k, 2, 3).contiguous()                    # b x out_dim x n x seq_length
        k = k.view(-1, out_dim * n, seq_length)                      # b x out_dim*n x seq_length

        v = torch.transpose(v, 1, 2).contiguous()
        v = v.view(-1, seq_length, out_dim * n)

        score = torch.matmul(q, k)/(self.hidden**(1/2))         # b x seq_length x seq_length
        score = score.softmax(-1)
        out = torch.matmul(score, v)                                 # b x seq_length x out_dim*n
        out = out.view(-1, seq_length, out_dim, n)                   # b x seq_length x out_dim x n
        out = torch.transpose(out, 1, 2).contiguous()                # b x out_dim x seq_length x n
        out = self.conv1(out)                                        # b x m x seq_length x n
        out = self.layernorm(out)
        out = out + x

        # out1 = self.conv2(out)                                        # b x hidden_size x seq_length x n
        # out1 = self.conv3(out1)                                        # b x m x seq_length x n
        # out1 = self.layernorm(out1)
        
        # out = out + out1


        return out

class Myattention3(nn.Module):
    def __init__(self, m, out_dim, hidden_size, k=3, drop_rate=0):
        super(Myattention3, self).__init__()
        assert (k % 2 != 0), 'K must be Odd'
        self.hidden = hidden_size
        self.q = nn.Conv2d(m, out_dim, k, padding=(k-1)//2, groups=m)
        self.k = nn.Conv2d(m, out_dim, k, padding=(k-1)//2, groups=m)
        self.v = nn.Linear(hidden_size, hidden_size // m * out_dim)
        self.gelu = nn.GELU()
        self.layernorm = nn.LayerNorm(hidden_size // m)
        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(hidden_size // m * out_dim, hidden_size)

    def forward(self, x):
        input_shape = x.size() # b x m x seq_length x n
        m, seq_length, n = input_shape[1:]

        q = self.dropout(self.gelu(self.q(x)))          # b x out_dim x seq_length x n
        k = self.dropout(self.gelu(self.k(x)))
        v = self.dropout(self.gelu(self.v(torch.transpose(x, 1, 2).contiguous().view(-1, seq_length, self.hidden))))

        out_dim = q.size()[1]

        q = torch.transpose(q, 1, 2).contiguous()                    # b x seq_length x out_dim x n
        q = q.view(-1, seq_length, out_dim * n)                      # b x seq_length x out_dim*n

        k = torch.transpose(k, 2, 3).contiguous()                    # b x out_dim x n x seq_length
        k = k.view(-1, out_dim * n, seq_length)                      # b x out_dim*n x seq_length

        v = torch.transpose(v, 1, 2).contiguous()
        v = v.view(-1, seq_length, out_dim * n)

        score = torch.matmul(q, k)/(self.hidden**(1/2))         # b x seq_length x seq_length
        score = score.softmax(-1)
        out = torch.matmul(score, v)                                 # b x seq_length x out_dim*n
        out = self.linear(out)                                       # b x seq_length x hidden_size
        out = out.view(-1, seq_length, m, n)                         # b x seq_length x m x n
        out = torch.transpose(out, 1, 2).contiguous()                # b x out_dim x seq_length x n
        out = self.layernorm(out)
        out = out + x

        # out1 = self.conv2(out)                                        # b x hidden_size x seq_length x n
        # out1 = self.conv3(out1)                                        # b x m x seq_length x n
        # out1 = self.layernorm(out1)
        
        # out = out + out1


        return out

class MyEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, pad_ids, drop_rate=0):
        super(MyEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_ids)
        self.linear = nn.Linear(embedding_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        # b, seq_length
        x = self.embedding(x)
        x = self.linear(x)
        x = self.dropout(self.layernorm(x))                                  # b x seq_length x hidden_size
        return x

class Mymodel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, m, out_dim, n_layer, pad_ids, k=3, attd_mode=1, drop_rate=0):
        super(Mymodel, self).__init__()
        assert (hidden_size % m == 0), f"hidden_size: {hidden_size}, m: {m} ==> Must be hidden_size % m == 0!"
        self.m = m
        self.embed = MyEmbedding(vocab_size, embedding_size, hidden_size, pad_ids, drop_rate)
        if attd_mode == 1:
            self.model = nn.ModuleList([Myattention(m, out_dim, hidden_size, k, drop_rate) for _ in range(n_layer)])
        elif attd_mode == 2:
            self.model = nn.ModuleList([Myattention2(m, out_dim, hidden_size, k, drop_rate) for _ in range(n_layer)])
        elif attd_mode == 3:
            self.model = nn.ModuleList([Myattention3(m, out_dim, hidden_size, k, drop_rate) for _ in range(n_layer)])

    def forward(self, x):
        # b, seq_length
        x = self.embed(x)                                                     # b x seq_length x hidden_size

        input_shape = x.size()
        seq_length, hidden_size = input_shape[1:]

        x = x.view(-1, seq_length, self.m, hidden_size//self.m)               # b x seq_length x m x n
        x = torch.transpose(x, 1, 2).contiguous()                             # b x m x seq_length x n


        for i, layer in enumerate(self.model):
            x = layer(x)

        return (x,)


class Mymodel2(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, m, out_dim, n_layer, pad_ids, k=3, attd_mode=1, drop_rate=0):
        super(Mymodel2, self).__init__()
        assert (hidden_size % m == 0), f"hidden_size: {hidden_size}, m: {m} ==> Must be hidden_size % m == 0!"
        self.m = m
        self.embed = MyEmbedding(vocab_size, embedding_size, hidden_size, pad_ids, drop_rate)
        if attd_mode == 1:
            self.model = nn.ModuleList([Myattention(m, out_dim, hidden_size, k, drop_rate) for _ in range(n_layer)])
        elif attd_mode == 2:
            self.model = nn.ModuleList([Myattention2(m, out_dim, hidden_size, k, drop_rate) for _ in range(n_layer)])
        elif attd_mode == 3:
            self.model = nn.ModuleList([Myattention3(m, out_dim, hidden_size, k, drop_rate) for _ in range(n_layer)])

    def forward(self, x, y):
        # b, seq_length
        with torch.no_grad():        
            self.embed = self.embed.eval()
            y = self.embed(y)                                                    # b x seq_length x hidden_size
            y = y.detach()

        self.embed = self.embed.train()
        x = self.embed(x)

        input_shape = x.size()
        seq_length, hidden_size = input_shape[1:]

        x = x.view(-1, seq_length, self.m, hidden_size//self.m)               # b x seq_length x m x n
        x = torch.transpose(x, 1, 2).contiguous()                             # b x m x seq_length x n

        for i, layer in enumerate(self.model):
            x = layer(x)

        return (x, y)


class MymodelForPretrain(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, m, out_dim, n_layer, pad_ids, k=3, attd_mode=1, drop_rate=0):
        super(MymodelForPretrain, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.model = Mymodel(vocab_size=vocab_size,
                             embedding_size=embedding_size, 
                             hidden_size=hidden_size, 
                             m=m, 
                             out_dim=out_dim, 
                             n_layer=n_layer, 
                             pad_ids=pad_ids, 
                             k=k, 
                             attd_mode=attd_mode, 
                             drop_rate=drop_rate)
        self.linear = nn.Linear(hidden_size, embedding_size)
        self.linear2 = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        # b x seq_length

        outputs = self.model(x)

        hidden = outputs[0]                                      # b x m x seq_length x n
        hidden = torch.transpose(hidden, 1, 2).contiguous()      # b x seq_length x m x n
        hidden = hidden.view(-1, x.size()[1], self.hidden_size)  # b x seq_length x hidden_size

        out = self.linear(hidden)                                # b x seq_length x embedding_size
        out = self.linear2(out)                                  # b x seq_length x vocab_size
        # out = out.view(-1, self.vocab_size)                      # b*seq_length x vocab_size

        return out
    
    def model_save(self, epoch, optimizer, scheduler, is_best, args):
        self.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, is_best, args)

    def save_checkpoint(self, state, is_best, args, filename='checkpoint.pth.tar'):
        filename = os.path.join(args.log_dir, filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(args.log_dir, 'model_best.pth.tar'))

    def model_load(self, checkpoint):
        self.model.load_state_dict(checkpoint)

class MymodelForPretrain2(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, m, out_dim, n_layer, pad_ids, k=3, attd_mode=1, drop_rate=0):
        super(MymodelForPretrain2, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.model = Mymodel2(vocab_size=vocab_size,
                             embedding_size=embedding_size, 
                             hidden_size=hidden_size, 
                             m=m, 
                             out_dim=out_dim, 
                             n_layer=n_layer, 
                             pad_ids=pad_ids, 
                             k=k, 
                             attd_mode=attd_mode, 
                             drop_rate=drop_rate)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.linear2 = nn.Linear(embedding_size, vocab_size)

    def forward(self, x, y):
        # b x seq_length

        outputs = self.model(x, y)

        hidden = outputs[0]                                      # b x m x seq_length x n
        hidden = torch.transpose(hidden, 1, 2).contiguous()      # b x seq_length x m x n
        hidden = hidden.view(-1, x.size()[1], self.hidden_size)  # b x seq_length x hidden_size

        out2 = self.linear(hidden)                                # b x seq_length x embedding_size
        out = self.linear2(out2)                                  # b x seq_length x vocab_size
        # out = out.view(-1, self.vocab_size)                      # b*seq_length x vocab_size

        return out, hidden, outputs[1]
    
    def model_save(self, epoch, optimizer, scheduler, is_best, args):
        self.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, is_best, args)

    def save_checkpoint(self, state, is_best, args, filename='checkpoint.pth.tar'):
        filename = os.path.join(args.log_dir, filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(args.log_dir, 'model_best.pth.tar'))

    def model_load(self, checkpoint):
        self.model.load_state_dict(checkpoint)

class MymodelForSequenceClassification(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, m, out_dim, n_layer, pad_ids, num_classes, k=3, attd_mode=1, drop_rate=0, freeze=False):
        super(MymodelForSequenceClassification, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.model = Mymodel(vocab_size, embedding_size, hidden_size, m, out_dim, n_layer, pad_ids, k, attd_mode, drop_rate)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.linear = nn.Linear(hidden_size, num_classes)
        self.drop_out = nn.Dropout(drop_rate)


    def forward(self, x):
        # b x seq_length

        outputs = self.model(x)

        hidden = self.drop_out(outputs[0])                                      # b x m x seq_length x n
        hidden = torch.transpose(hidden, 1, 2).contiguous()      # b x seq_length x m x n
        hidden = hidden.view(-1, x.size()[1], self.hidden_size)  # b x seq_length x hidden_size
        hidden = torch.mean(hidden, dim=1)                       # b x hidden_size
        # hidden = hidden[:,0,:]                                 # b x hidden_size
        out = self.linear(hidden)                 # b x num_classes

        return out

    def model_load(self, checkpoint):
        self.model.load_state_dict(checkpoint)
