import torch
from torch import nn
from torch.autograd import Variable

from pos import get_pos_onehot


class MultiHeadAttention(nn.Module):
    """
        A multihead attention module,
        using scaled dot-product attention.
    """

    def __init__(self, input_size, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.head_size = int(self.hidden_size / num_heads)
        # 加权求和
        self.q_linear = nn.Linear(self.input_size, self.hidden_size)
        self.k_linear = nn.Linear(self.input_size, self.hidden_size)
        self.v_linear = nn.Linear(self.input_size, self.hidden_size)
        #
        self.joint_linear = nn.Linear(self.hidden_size, self.hidden_size)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        # project the queries, keys and values by their respective weight matrices
        q_proj = self.q_linear(q).view(q.size(0), q.size(1), self.num_heads, self.head_size).transpose(1, 2)
        k_proj = self.k_linear(k).view(k.size(0), k.size(1), self.num_heads, self.head_size).transpose(1, 2)
        v_proj = self.v_linear(v).view(v.size(0), v.size(1), self.num_heads, self.head_size).transpose(1, 2)

        # calculate attention weights
        unscaled_weights = torch.matmul(q_proj, k_proj.transpose(2, 3))  # 转置第2,3维
        weights = self.softmax(unscaled_weights / torch.sqrt(torch.Tensor([self.head_size * 1.0]).to(unscaled_weights)))

        # weight values by their corresponding attention weights
        weighted_v = torch.matmul(weights, v_proj)
        # contiguous():返回一个内存连续的有相同数据的tensor，如果原tensor内存连续则返回原tensor
        weighted_v = weighted_v.transpose(1, 2).contiguous()

        # do a linear projection of the weighted sums of values
        joint_proj = self.joint_linear(weighted_v.view(q.size(0), q.size(1), self.hidden_size))

        # store a reference to attention weights, for THIS forward pass,
        # for visualisation purposes
        self.weights = weights

        return joint_proj


class Block(nn.Module):
    """
        One block of the transformer.
        Contains a multihead attention sublayer
        followed by a feed forward network.
    """

    def __init__(self, input_size, hidden_size, num_heads, activation=nn.ReLU, dropout=None):
        super(Block, self).__init__()
        self.dropout = dropout

        self.attention = MultiHeadAttention(input_size, hidden_size, num_heads)
        self.attention_norm = nn.LayerNorm(input_size)

        ff_layers = [
            nn.Linear(input_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, input_size),
        ]

        if self.dropout:
            self.attention_dropout = nn.Dropout(dropout)
            ff_layers.append(nn.Dropout(dropout))
        # nn.Sequential(): Modules will be added to it in the order they are passed in the constructor. Alternatively,
        # an ordered dict of modules can also be passed in.
        self.ff = nn.Sequential(*ff_layers)
        self.ff_norm = nn.LayerNorm(input_size)

    def forward(self, x):
        attended = self.attention_norm(self.attention_dropout(self.attention(x, x, x)) + x)
        return self.ff_norm(self.ff(attended) + x)


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, ff_size, num_blocks, num_heads, activation=nn.ReLU, dropout=None):
        """
            A single Transformer Network
            input_size: hidden weight
            hidden_size: hidden weight
            ff_size: hiden weight
        """
        super(Transformer, self).__init__()
        # construct num_blocks block, no residual structure
        self.blocks = nn.Sequential(*[Block(input_size, hidden_size, num_heads, activation, dropout=dropout)
                                      for _ in range(num_blocks)])

    def forward(self, x):
        """
            Sequentially applies the blocks of the Transformer network
        """
        return self.blocks(x)


class Net(nn.Module):
    """
        A neural network that encodes a sequence
        using a Transformer network
    """

    def __init__(self, embeddings, max_length, model_size=128, num_heads=4, num_blocks=1, dropout=0.1,
                 train_word_embeddings=True):
        super(Net, self).__init__()
        # Creates Embedding instance from given 2-dimensional FloatTensor.
        # embeddings (Tensor): FloatTensor containing weights for the Embedding.
        # First dimension is being passed to Embedding as 'num_embeddings', second as 'embedding_dim'.
        # freeze (boolean, optional): If ``True``, the tensor does not get updated in the learning process.
        # Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=not train_word_embeddings)
        self.model_size = model_size
        # Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
        # outputsize=[embedding.size(1), self.model_size]
        # embedding加权求和
        self.emb_ff = nn.Linear(embeddings.size(1), self.model_size)
        self.pos = nn.Linear(max_length, self.model_size)
        self.max_length = max_length
        self.transformer = Transformer(self.model_size, self.model_size, self.model_size, num_blocks, num_heads,
                                       dropout=dropout)
        # 2: biclass
        self.output = nn.Linear(self.model_size, 2)

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1)  # x.view(-1)将x展为一行
        x = self.emb_ff(self.embeddings(x))
        pos = self.pos(get_pos_onehot(self.max_length).to(x)).unsqueeze(0)
        x = x.view(*(x_size + (self.model_size,)))
        x += pos
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.output(x)


if __name__ == "__main__":
    """
        If run seperately, does a simple sanity check,
        by doing a random forward pass
    """
    t = Transformer(10, 20, 30, 3, 5)
    print(t)
    input = Variable(torch.rand(40, 20, 10))
    print(input)
