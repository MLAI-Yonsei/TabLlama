import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import LlamaForCausalLM

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        return cos_emb, sin_emb


class LlamaConfig:
    def __init__(self, hidden_size=1200, num_hidden_layers=2, num_attention_heads=16,
                 intermediate_size=2400):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size


class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # RoPE를 위한 RotaryEmbedding 초기화
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape

        # 쿼리, 키, 밸류 계산
        q = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE를 위한 cos 및 sin 임베딩 계산
        cos_emb, sin_emb = self.rotary_emb(seq_length, device=hidden_states.device)
        cos_emb = cos_emb[None, None, :, :].to(q.dtype)
        sin_emb = sin_emb[None, None, :, :].to(q.dtype)

        # 쿼리와 키에 RoPE 적용
        q = self.apply_rotary_pos_emb(q, cos_emb, sin_emb)
        k = self.apply_rotary_pos_emb(k, cos_emb, sin_emb)

        # 어텐션 스코어 계산
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            elif attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)

        return self.o_proj(attn_output)

    def apply_rotary_pos_emb(self, x, cos, sin):
        x_rotated = self.rotate_half(x)
        return (x * cos) + (x_rotated * sin)

    def rotate_half(self, x):
        x1 = x[..., :x.size(-1)//2]
        x2 = x[..., x.size(-1)//2:]
        return torch.cat((-x2, x1), dim=-1)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states
    
class HierarchicalLlama(nn.Module):
    def __init__(self, args, hidden_size, num_layers, num_heads, col_dim, reg_dim, num_classes, sequence_len, intent_num):
        super(HierarchicalLlama, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.col_dim = col_dim
        self.reg_dim = reg_dim
        self.sequence_len = sequence_len
        self.intent_num = intent_num
        

        self.transformer1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads),
            num_layers=num_layers
        )
        

        self.final_activation = nn.ReLU()


        config = LlamaConfig(hidden_size=col_dim * reg_dim * hidden_size,
                             num_hidden_layers = args.seq_num_layers,
                             num_attention_heads=args.seq_num_heads,
                             intermediate_size=args.intermediate_size
                             )
        
        self.transformer2 = LlamaModel(config)

        self.mlp = nn.Sequential(
            nn.Linear(
                col_dim * reg_dim * hidden_size,
                col_dim * reg_dim * hidden_size // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                col_dim * reg_dim * hidden_size // 2,
                col_dim * reg_dim,
            ),
        )

        self.intent_mlp = nn.Sequential(
                nn.Linear(col_dim * reg_dim * hidden_size, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.intent_num)
            )



        self.embedding = nn.Embedding(num_classes, hidden_size)


    def generate_causal_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-1e8')).masked_fill(mask == 1, float(0.0)) 
        return mask
    
    def create_causal_mask(self,seq_length, device):
        """
        Creates a causal mask for the attention mechanism that ensures
        each token can only attend to previous tokens (including itself).
        """
        mask = torch.tril(torch.ones((seq_length, seq_length), device=device)).unsqueeze(0).unsqueeze(0)
        return mask  # shape: (1, 1, seq_length, seq_length)

    def create_key_padding_mask(self,batch_size, seq_length, attention_mask, device):
        """
        Converts the input attention_mask (with 0 for padding and 1 for valid tokens)
        into a key padding mask compatible with the causal mask.
        """
        # attention_mask is assumed to be of shape (batch_size, seq_length)
        key_padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        key_padding_mask = key_padding_mask.expand(batch_size, 1, seq_length, seq_length)
        return key_padding_mask  # shape: (batch_size, 1, seq_length, seq_length)

    def combine_masks(self,causal_mask, key_padding_mask):
        """
        Combines the causal mask and key padding mask by applying the key padding mask
        to the causal mask.
        """
        combined_mask = causal_mask * key_padding_mask
        combined_mask = combined_mask.masked_fill(combined_mask == 0, float('-1e8'))
        return combined_mask
    


    def forward(self, x, masks):

        batch_size, seq_len, _ = x.size()

        embedding_output = self.embedding(x)

        embedding_output2 = embedding_output.view(batch_size, seq_len, self.col_dim * self.reg_dim, self.hidden_size)

        x = embedding_output2.permute(1, 0, 2, 3)
        

        for t in range(seq_len):
            x[t] = self.transformer1(x[t])
            

        x = x.permute(1, 0, 2, 3).contiguous()

        x = x.reshape(batch_size, seq_len, self.col_dim * self.reg_dim * self.hidden_size)
        
        # Create causal mask and key padding mask
        causal_mask = self.create_causal_mask(seq_len, x.device)
        key_padding_mask = self.create_key_padding_mask(batch_size, seq_len, masks, x.device)

        # Combine both masks
        final_causal_mask = self.combine_masks(causal_mask, key_padding_mask)

        x = self.transformer2(x, final_causal_mask)

        assert not torch.isnan(x).any(), f"NaN detected in {x}"

        x_seq = self.mlp(x.mean(dim=1))  # sequence_len * col_dim * reg_dim

        x_int = self.intent_mlp(x.mean(dim=1))

        x_seq = self.final_activation(x_seq)

        return x_seq, x_int
