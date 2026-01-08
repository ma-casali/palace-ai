from torch import nn
import torch

class PalacePlayer(nn.Module):
    def __init__(self, history_dim = 79, static_dim = 73, hidden_dim = 64, num_rnn_layers = 2):
        super(PalacePlayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_rnn_layers = num_rnn_layers
        
        # RNN branch (action history)
        self.rnn = nn.LSTM(
            input_size=history_dim,
            hidden_size=hidden_dim, 
            num_layers=2,
            batch_first=True)
        
        # Static branch (current player hands, draw pile length)
        self.static_mlp  = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Combined net
        self.combined_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 , hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 79)  # Output logits for 79 possible actions
        )

    def init_hidden(self, batch_size):
        # Get the device the model is currently on
        device = next(self.parameters()).device
        
        # LSTM hidden states are a tuple: (h_0, c_0)
        # Shape: (num_layers, batch_size, hidden_size)
        h_0 = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_dim, device=device)
    
        return (h_0, c_0)

    def forward(self, action_history, static_features, mask, hidden_state=None):
        # history sequence shape: (batch_size, seq_len, sequence_dim)
        # static features shape: (batch_size, static_dim)

        # 1. RNN Pass
        lstm_out, hidden_state = self.rnn(action_history, hidden_state)
        rnn_out = lstm_out[:, -1, :] 

        # 2. Static Pass
        static_out = self.static_mlp(static_features)
        
        # 3. Concatenate and Policy Head
        combined_out = torch.cat((rnn_out, static_out), dim=1)
        logits = self.combined_mlp(combined_out)
        
        # 4. Masking and Softmax
        masked_logits = logits.masked_fill(mask == 0, -1e9)
        probs = torch.softmax(masked_logits, dim=-1)

        return probs, hidden_state

