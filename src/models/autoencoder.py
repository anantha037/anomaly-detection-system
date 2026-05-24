import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, latent_dim, n_layers):
        super(LSTMAutoencoder, self).__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=latent_dim,
            num_layers=n_layers,
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=n_features,
            num_layers=n_layers,
            batch_first=True
        )
        
    def forward(self, x):
        # Encode
        _, (hidden, _) = self.encoder(x)
        
        # Get last hidden state
        last_hidden = hidden[-1]
        
        # Repeat context vector for seq_len
        seq_len = x.shape[1]
        decoder_input = last_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode
        reconstruction, _ = self.decoder(decoder_input)
        return reconstruction
