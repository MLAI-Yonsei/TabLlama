import torch
import torch.nn as nn

class SimplifiedLSTMRNN(nn.Module):
    def __init__(self, args,hidden_size, num_layers, col_dim, reg_dim, num_classes,sequence_len,intent_num, model_type="lstm"):
        super(SimplifiedLSTMRNN, self).__init__()
        self.hidden_size = hidden_size
        self.model_type = model_type  # 'rnn' or 'lstm'
        self.col_dim = col_dim
        self.reg_dim = reg_dim
        self.num_classes = num_classes
        self.intent_num = intent_num

        self.embedding = nn.Embedding(num_classes, hidden_size)
        
        self.final_activation = nn.ReLU()

        # Select RNN or LSTM based on model_type
        if model_type == 'rnn':
            self.rnn = nn.RNN(input_size=self.col_dim*self.reg_dim*hidden_size, hidden_size=self.col_dim*self.reg_dim, num_layers=num_layers, batch_first=True)
        elif model_type == 'lstm':
            self.lstm = nn.LSTM(input_size=self.col_dim*self.reg_dim*hidden_size, hidden_size=self.col_dim*self.reg_dim, num_layers=num_layers, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(self.col_dim*self.reg_dim, self.col_dim*self.reg_dim),
            nn.ReLU(),
            nn.Linear(self.col_dim*self.reg_dim,self.col_dim*self.reg_dim)
        )
        
        self.intent_mlp = nn.Sequential(
                nn.Linear(col_dim * reg_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.intent_num)
            )

        self.init_weights()

    def init_weights(self):
        def _xavier_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, (nn.RNN, nn.LSTM)):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

        self.apply(_xavier_init)


    def forward(self, x, mask):
        batch_size, seq_len, data_dim = x.size()
                        
        x = self.embedding(x)
        x = x.view(batch_size, seq_len, data_dim * self.hidden_size)
        

        if self.model_type == 'rnn':
            x, _ = self.rnn(x)
        elif self.model_type == 'lstm':
            x, (h_n, c_n) = self.lstm(x)
            

        x_seq = self.mlp(x.mean(dim=1))
        
        
        x_int = self.intent_mlp(x.mean(dim=1))
        
        
        x_seq = self.final_activation(x_seq)
        

        return x_seq, x_int