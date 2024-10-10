import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(1)
        position_ids = torch.arange(sequence_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, sequence_length)
        position_embeddings = self.pe(position_ids)
        return x + position_embeddings

class CustomTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate=0.1):
        super(CustomTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        # Pre-norm
        normalized_x = self.norm1(x)
        normalized_x = normalized_x.transpose(0, 1)  # (seq_len, batch, hidden_size)
        attn_output, _ = self.attention(normalized_x, normalized_x, normalized_x, 
                                        key_padding_mask=key_padding_mask, 
                                        attn_mask=attn_mask)
        attn_output = attn_output.transpose(0, 1)  # (batch, seq_len, hidden_size)
        x = x + self.dropout1(attn_output)
        
        # Pre-norm for feed forward
        normalized_x = self.norm2(x)
        ff_output = self.feed_forward(normalized_x)
        x = x + self.dropout2(ff_output)
        
        return x

class HierarchicalGPT2(nn.Module):
    def __init__(self, args, hidden_size, num_layers, num_heads, col_dim, reg_dim, num_classes, sequence_len, intent_num):
        super(HierarchicalGPT2, self).__init__()
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

        self.positional_encoding = LearnablePositionalEncoding(d_model=col_dim * reg_dim * hidden_size)

        self.transformer2 = nn.ModuleList([
            CustomTransformerBlock(hidden_size=col_dim * reg_dim * hidden_size, num_heads=num_heads)
            for _ in range(num_layers)
        ])

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

        self.init_weights()

    def init_weights(self):
        def _xavier_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

        self.apply(_xavier_init)

        for layer in self.transformer1.layers:
            self._init_transformer_weights(layer)

    def _init_transformer_weights(self, layer):
        nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
        nn.init.xavier_uniform_(layer.linear1.weight)
        nn.init.xavier_uniform_(layer.linear2.weight)
        if layer.self_attn.in_proj_bias is not None:
            nn.init.constant_(layer.self_attn.in_proj_bias, 0.)
        if layer.self_attn.out_proj.bias is not None:
            nn.init.constant_(layer.self_attn.out_proj.bias, 0.)
        if layer.linear1.bias is not None:
            nn.init.constant_(layer.linear1.bias, 0.)
        if layer.linear2.bias is not None:
            nn.init.constant_(layer.linear2.bias, 0.)

    def generate_causal_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-1e8')).masked_fill(mask == 1, float(0.0)) 
        return mask

    def forward(self, x, masks):

        batch_size, seq_len, _ = x.size()

        embedding_output = self.embedding(x)

        embedding_output2 = embedding_output.view(batch_size, seq_len, self.col_dim * self.reg_dim, self.hidden_size)

        x = embedding_output2.permute(1, 0, 2, 3)

        for t in range(seq_len):
            x[t] = self.transformer1(x[t])

        x = x.permute(1, 0, 2, 3).contiguous()

        x = x.view(batch_size, seq_len, self.col_dim * self.reg_dim * self.hidden_size)
        x = self.positional_encoding(x)

        causal_mask = self.generate_causal_mask(seq_len).to(x.device)

        for block in self.transformer2:
            x = block(x, key_padding_mask=masks.float(), attn_mask=causal_mask)

        assert not torch.isnan(x).any(), f"NaN detected in {x}"

        x_seq = self.mlp(x.mean(dim=1))  # sequence_len * col_dim * reg_dim 

        x_int = self.intent_mlp(x.mean(dim=1))

        x_seq = self.final_activation(x_seq)

        return x_seq, x_int
