from torch import nn
import torch

class PalacePlayer(nn.Module):
    def __init__(self, history_dim = 79, static_dim = 73, hidden_dim = 256, num_rnn_layers = 2):
        super(PalacePlayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_rnn_layers = num_rnn_layers
        
        # RNN branch (action history)
        self.rnn = nn.LSTM(
            input_size=history_dim,
            hidden_size=hidden_dim, 
            num_layers=2,
            batch_first=True)
        
        dim_per_layer = hidden_dim // 8 # 32 if hidden_dim is 256
        
        # DenseNet Architecture for Static Features
        self.L1 = nn.Sequential(nn.Linear(static_dim, dim_per_layer), nn.LayerNorm(dim_per_layer), nn.ReLU())
        self.L2 = nn.Sequential(nn.Linear(dim_per_layer, dim_per_layer), nn.LayerNorm(dim_per_layer), nn.ReLU())
        self.L3 = nn.Sequential(nn.Linear(dim_per_layer, dim_per_layer), nn.LayerNorm(dim_per_layer), nn.ReLU())
        self.L4 = nn.Sequential(nn.Linear(dim_per_layer, dim_per_layer), nn.LayerNorm(dim_per_layer), nn.ReLU())
        self.L5 = nn.Sequential(nn.Linear(dim_per_layer, dim_per_layer), nn.LayerNorm(dim_per_layer), nn.ReLU())
        self.L6 = nn.Sequential(nn.Linear(dim_per_layer, dim_per_layer), nn.LayerNorm(dim_per_layer), nn.ReLU())
        self.L7 = nn.Sequential(nn.Linear(dim_per_layer, dim_per_layer), nn.LayerNorm(dim_per_layer), nn.ReLU())
        self.L8 = nn.Sequential(nn.Linear(dim_per_layer, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())

        # Combined net (RNN * L8 output + static features skip)
        self.combined_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
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

        # 1. DenseNet Static Feature Pass
        x1 = self.L1(static_features)
        x2 = self.L2(x1)
        x3 = self.L3(x2 + x1)
        x4 = self.L4(x3 + x2 + x1)
        x5 = self.L5(x4 + x3 + x2 + x1)
        x6 = self.L6(x5 + x4 + x3 + x2 + x1)
        x7 = self.L7(x6 + x5 + x4 + x3 + x2 + x1)
        x8 = self.L8(x7)
        # 1. RNN Pass
        lstm_out, hidden_state = self.rnn(action_history, hidden_state)
        rnn_out = lstm_out[:, -1, :] 
        
        # 3. Concatenate and Policy Head
        combined_out = torch.cat((rnn_out, x8, static_features), dim=1)
        logits = self.combined_mlp(combined_out)
        
        # 4. Masking and Softmax
        masked_logits = logits.masked_fill(mask == 0, -1e9)
        probs = torch.softmax(masked_logits, dim=-1)

        return probs, hidden_state

if __name__ == "__main__":
    # create dummy inputs
    model = PalacePlayer()
    model.eval()

    batch_size = 1
    dummy_history = torch.zeros(batch_size, 6, 79)  # (batch_size, seq_len, history_dim)
    dummy_static = torch.zeros(batch_size, 73)      # (batch_size, static_dim)
    dummy_mask = torch.ones(batch_size, 79)         # (batch_size, num_actions)

    torch.onnx.export(
        model,
        (dummy_history, dummy_static, dummy_mask),
        "palace_player.onnx",
        export_params=True,
        opset_version=12,
        input_names=['action_history', 'static_features', 'mask'],
        output_names=['probs', 'hidden_state']
    )