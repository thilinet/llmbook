from dataclasses import dataclass
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

@dataclass
class SLLMConfig:
    # Embedding dimension
    d_model: int = 128
    # Query key Value projection dimension
    d_head: int  = 128
    # bias for query,key and value projection matrices
    bias: bool = False
    dropout: int = 0.0
    # Number of input tokens
    context_window: int = 50
    # Number of attention heads
    n_heads: int = 2
    vocab_size: int = 52000
    n_layers: int=2

    
config = SLLMConfig()
assert config.d_model % config.n_heads == 0

    
class SingleHeadAttention(nn.Module):
    """
    Implements weighted self attention
    """
    def __init__(self, config):

        super().__init__()
        self.Wq =  nn.Linear(config.d_model, config.d_head, bias=config.bias)
        self.Wk =  nn.Linear(config.d_model, config.d_head, bias=config.bias)
        self.Wv =  nn.Linear(config.d_model, config.d_head, bias=config.bias)

        self.attn_drop = nn.Dropout(config.dropout)
        self.__init_weights()

    def __init_weights(self):

        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wv.weight)

    def forward(self, x, mask=None):

        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        attn_score = q @ k.transpose(-2,-1)
        
        if mask == None:
            mask = torch.triu(torch.ones(x.shape[-2], x.shape[-2],device=x.device), diagonal=1)
        
        masked = attn_score.masked_fill(mask.bool(), -torch.inf)
        attn_weights = torch.softmax(masked / math.sqrt(k.shape[-1]), dim=1)
        
        attn_weights = self.attn_drop(attn_weights)

        context_vector = attn_weights @ v

        return context_vector
    
class SingleHeadAttentionv1(nn.Module):
    """
    Implements weighted self attention
    """
    def __init__(self, config):

        super().__init__()
        self.Wq =  nn.Linear(config.d_model, config.d_head, bias=config.bias)
        self.Wk =  nn.Linear(config.d_model, config.d_head, bias=config.bias)
        self.Wv =  nn.Linear(config.d_model, config.d_head, bias=config.bias)

        self.attn_drop = config.dropout
        
        self.__init_weights()

    def __init_weights(self):

        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wv.weight)

    def forward(self, x, mask=None):

        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        
        context_vector = F.scaled_dot_product_attention(
                                query = q
                               ,key   = k
                               ,value = v
                               ,attn_mask=None
                               ,dropout_p=self.attn_drop
                               ,is_causal=True, scale=None)



        return context_vector
class MultiHeadAttention(nn.Module):
    """
    Multihead Attention Implementation
    """
    def __init__(self, config):
        
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SingleHeadAttention(config) for _ in range(config.n_heads)
            ]
        )
        self.projection_out = nn.Linear(config.n_heads * config.d_head, config.d_head)

    def forward(self, x):
        attentions = []
        for head in self.heads:
            attentions.append(head(x))

        context_vector = torch.cat(attentions, dim=-1)
        context_projected = self.projection_out(context_vector)
        return context_projected

class MultiHeadAttentionV1(nn.Module):
    """
    Multihead Attention Implementation
    """
    def __init__(self, config):
        
        super().__init__()

        self.projection_out = nn.Linear(config.n_heads * config.d_head, config.d_head)    
        
        self.Wq =  nn.Linear(config.d_model, config.d_head * config.n_heads, bias=config.bias)
        self.Wk =  nn.Linear(config.d_model, config.d_head * config.n_heads, bias=config.bias)
        self.Wv =  nn.Linear(config.d_model, config.d_head * config.n_heads, bias=config.bias)

        self.attn_drop  = config.dropout
        self.n_heads    = config.n_heads
        self.d_head     = config.d_head
        self.__init_weights()


    def __init_weights(self):

        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wv.weight)


    def forward(self, x):

        batch, length, d = x.shape
        is_causal = True
        
        if not self.train:
            is_causal = False
            self.attn_drop = 0.0
        
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        
        q = q.view(batch, length, self.n_heads, self.d_head)
        k = k.view(batch, length, self.n_heads, self.d_head)
        v = v.view(batch, length, self.n_heads, self.d_head)

        context_vector = F.scaled_dot_product_attention(
                                query = q
                               ,key   = k
                               ,value = v
                               ,attn_mask=None
                               ,dropout_p=self.attn_drop
                               ,is_causal=True, scale=None)

        context_vector = context_vector.contiguous().view(batch, length, self.d_head * self.n_heads)
        output = self.projection_out(context_vector)
        return output
    
class LayerNorm(nn.Module):

    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
    
    
    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.Linear(config.d_head, config.d_head, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.d_head, config.d_head, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.ln_1(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)

    
    
class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln1 = LayerNorm(config.d_model, bias=config.bias)
        self.mha = MultiHeadAttention(config)
        self.ln2 = LayerNorm(config.d_head, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):

        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x

    
class TransformerBlockV1(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln1 = LayerNorm(config.d_model, bias=config.bias)
        self.mha = MultiHeadAttentionV1(config)
        self.ln2 = LayerNorm(config.d_head, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):

        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x

class SLLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.token_embdgs = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embdgs   = nn.Embedding(config.context_window, config.d_model)
        self.droput       = nn.Dropout(config.dropout)

        self.transformer_blocks = nn.ModuleList(
        [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = LayerNorm(config.d_head)
        self.out_head = nn.Linear(config.d_head, config.vocab_size)

    def forward(self, x):

        batch_size, seq_length = x.shape
        token_embds = self.token_embdgs(x)
        pos_embds = self.pos_embdgs(torch.arange(seq_length, device=x.device))
        x = token_embds + pos_embds
        x = self.droput(x)

        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits 
    
class EmbeddingsBlock(nn.Module):
    """
    
    """
    def __init__(self, config):
        super().__init__()
        self.token_embdgs = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embdgs   = nn.Embedding(config.context_window, config.d_model)
        self.droput       = nn.Dropout(config.dropout)
    
    def forward(self, x):
        token_embds = self.token_embdgs(x)
        pos_embds = self.pos_embds(torch.arange(seq_length, device=x.device))
        x = token_embds + pos_embds
        x = self.dropout(x)
        return x

class SLLMv1(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.embedding_block    = EmbeddingsBlock(config)
        self.transformer_blocks = nn.ModuleList(
        [TransformerBlockV1(config) for _ in range(config.n_layers)]
        )
        self.final_norm = LayerNorm(config.d_head)
        self.out_head = nn.Linear(config.d_head, config.vocab_size)

    def forward(self, x):

        batch_size, seq_length = x.shape
        x = self.embedding_block(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits 