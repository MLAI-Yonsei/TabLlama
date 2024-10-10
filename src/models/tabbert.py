import torch.nn as nn
import torch
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


class HierarchicalTransformer(nn.Module):
    def __init__(self, args, hidden_size, num_layers, num_heads, col_dim, reg_dim, num_classes, sequence_len):
        super(HierarchicalTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.col_dim = col_dim
        self.reg_dim = reg_dim

        self.final_activation = nn.ReLU()

        self.transformer1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads),
            num_layers=num_layers
        )

        self.positional_encoding = LearnablePositionalEncoding(d_model=col_dim * reg_dim * hidden_size)
        self.transformer2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=col_dim * reg_dim * hidden_size, nhead=args.seq_num_heads),
            num_layers=args.seq_num_layers
        )

        self.mlp = nn.Sequential(
            nn.Linear( col_dim * reg_dim * hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, col_dim * reg_dim * 1)
        )

        self.mlp_intent = nn.Sequential(
            nn.Linear( col_dim * reg_dim * hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
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

        # Transformer 레이어의 가중치 초기화
        for layer in self.transformer1.layers:
            self._init_transformer_weights(layer)
        for layer in self.transformer2.layers:
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

    def forward(self, x, masks):
        batch_size, seq_len, data_dim = x.size()

        embedding_output = self.embedding(x)

        embedding_output2 = embedding_output.view(
            batch_size, seq_len, data_dim, self.hidden_size
        )

        x = embedding_output2.permute(1, 0, 2, 3)

        for t in range(seq_len):
            x[t] = self.transformer1(x[t])

        x = x.permute(1, 0, 2, 3).contiguous()

        x = x.view(batch_size, seq_len, data_dim * self.hidden_size)
        x = self.positional_encoding(x.permute(1, 0, 2))

        x = self.transformer2(x, src_key_padding_mask=masks)
        assert not torch.isnan(x).any(), f"NaN detected in {x}"
        x = x.permute(1, 0, 2)

        x = x.reshape(batch_size, seq_len, -1)
        x = x.mean(dim=1)
        
        x_seq = self.mlp(x)
        x_int = self.mlp_intent(x)

        x_seq = x_seq.view(batch_size, self.col_dim * self.reg_dim)

        x_seq = self.final_activation(x_seq)

        return x_seq, x_int
