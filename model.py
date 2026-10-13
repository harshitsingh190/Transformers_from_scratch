import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    # d_model is dim of model --> 512 
    def __init__(self,d_model:int , vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    # seq_len -> max length of each sentence
    def __init__(self,d_model:int , seq_len: int, dropout: float)->None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len , d_model)
        pe = torch.zeros(seq_len,d_model)
        # create vector of shape (seq_len,1)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) #  make rows into column
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
        # Apply the sin to the even postions 
        pe[:, 0::2] = torch.sin(position* div_term)
        pe[:, 1::2] = torch.cos(position* div_term)
        # our input dim are (batch_size, seq_len, d_model) so for adding pe vector we have make the dim similar so we add 1 at beginning so that it would add and 1 act as the batch_size of the input
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        # what it does--> It saves the pe tensor as a permanent part of the model (a "buffer") under the name 'pe'. You can later access it anywhere in your class using self.pe
        # why -->
        # It prevents training: Unlike a nn.Parameter, a buffer tells PyTorch that these numbers are fixed. The model will not calculate gradients or update these values during training, which is exactly what you want since positional encodings are calculated using a fixed mathematical formula.
        # It saves with the model: When you save your model using torch.save(model.state_dict()), this matrix is saved right along with the trained weights.
        # It moves with the model: If you move your model to a GPU using model.to('cuda'), this matrix automatically moves to the GPU memory alongside your other layers, preventing runtime errors caused by mismatched devices.
        self.register_buffer('pe',pe)

    def forward(self,x):
        # eg.. seq_len - 5000 but in input sentence length is 20 (x) so :x.shape[1] --> means 0 to 20 (x.shape[1]) 
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added

    def forward(self,x):
        mean = x.mean(dim=-1 , keepdim=True)
        std = x.std(dim=-1 , keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int , dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        x = torch.relu(self.linear_1(x))
        return self.linear_2(self.dropout(x))


    
    
        