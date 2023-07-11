"""
Recurrent Autoencoder PyTorch implementation
"""

import torch
import torch.nn as nn
from easydict import EasyDict as edict
from functools import partial

class RecurrentEncoder(nn.Module):
    """Recurrent encoder"""

    def __init__(self, n_features, latent_dim, rnn):
        super().__init__()

        self.rec_enc1 = rnn(n_features, latent_dim, batch_first=True)  # （batch, seq_len,  input_size）
        # print(rnn)  # class 'torch.nn.modules.rnn.LSTM'

    def forward(self, x):
        _, h_n = self.rec_enc1(x)

        # for each in h_n:  # torch.Size([1, 128, 32])  torch.Size([1, 128, 32])
        #     print(each.shape)
        # assert 1==0

        return h_n

class RecurrentDecoder(nn.Module):
    """Recurrent decoder for RNN and GRU"""

    def __init__(self, latent_dim, n_features, rnn_cell, device):
        super().__init__()

        self.n_features = n_features
        self.device = device
        self.rec_dec1 = rnn_cell(n_features, latent_dim)
        self.dense_dec1 = nn.Linear(latent_dim, n_features)

    def forward(self, h_0, seq_len):
        # Initialize output
        x = torch.tensor([], device = self.device)

        # Squeezing
        h_i = h_0.squeeze()

        # Reconstruct first element with encoder output
        x_i = self.dense_dec1(h_i)

        # Reconstruct remaining elements
        for i in range(0, seq_len):
            h_i = self.rec_dec1(x_i, h_i)
            x_i = self.dense_dec1(h_i)
            x = torch.cat([x, x_i], axis=1)


        return x.view(-1, seq_len, self.n_features)


class RecurrentDecoderLSTM(nn.Module):
    """Recurrent decoder LSTM"""

    def __init__(self, latent_dim, n_features, rnn_cell, device):
        super().__init__()

        self.n_features = n_features
        self.device = device
        self.rec_dec1 = rnn_cell(n_features, latent_dim)
        self.dense_dec1 = nn.Linear(latent_dim, n_features)

    def forward(self, h_0, seq_len):
        # Initialize output
        x = torch.tensor([], device = self.device)

        # for i, each in enumerate(h_0):  # 0 torch.Size([1, 128, 32])  1 torch.Size([1, 128, 32])
        #     print(i, each.shape)
        # assert 1==0

        # Squeezing
        h_i = [h.squeeze(0) for h in h_0]  # 0 torch.Size([batchsize, 32])  1 torch.Size([batchsize, 32])

        # Reconstruct first element with encoder output
        x_i = self.dense_dec1(h_i[0])  # x_0 torch.Size([batchsize, 6])

        # Reconstruct remaining elements
        for i in range(0, seq_len):
            try:
                # if i==0:
                #     print(i, x_i.shape, len(h_i), h_i[0].shape)  # 0 torch.Size([128, 6]) 2 torch.Size([128, 32])
                #     assert 1==0
                h_i = self.rec_dec1(x_i, h_i)
                x_i = self.dense_dec1(h_i[0])
                x = torch.cat([x, x_i], axis=1)

            except:
                print(i, x_i.shape, len(h_i), h_i[0].shape)  # 0 torch.Size([6]) 2 torch.Size([32])
                assert 1==0

        return x.view(-1, seq_len, self.n_features)


class RecurrentAE(nn.Module):
    """Recurrent autoencoder"""

    def __init__(self, config):
        super().__init__()

        # Encoder and decoder configuration
        self.config = config
        self.rnn, self.rnn_cell = self.get_rnn_type(self.config.rnn_type, self.config.rnn_act)
        self.decoder = self.get_decoder(self.config.rnn_type)
        self.latent_dim = self.config.latent_dim
        self.n_features = self.config.n_features
        self.device = self.config.device

        # Encoder and decoder
        self.encoder = RecurrentEncoder(self.n_features, self.latent_dim, self.rnn)
        self.decoder = self.decoder(self.latent_dim, self.n_features, self.rnn_cell, self.device)

    def forward(self, x):
        seq_len = x.shape[1]
        h_n = self.encoder(x)
        out = self.decoder(h_n, seq_len)

        return torch.flip(out, [1])

    @staticmethod
    def get_rnn_type(rnn_type, rnn_act=None):
        """Get recurrent layer and cell type"""
        if rnn_type == 'RNN':
            rnn = partial(nn.RNN, nonlinearity=rnn_act)
            rnn_cell = partial(nn.RNNCell, nonlinearity=rnn_act)

        else:
            rnn = getattr(nn, rnn_type)
            rnn_cell = getattr(nn, rnn_type + 'Cell')

        return rnn, rnn_cell

    @staticmethod
    def get_decoder(rnn_type):
        """Get recurrent decoder type"""
        if rnn_type == 'LSTM':
            decoder = RecurrentDecoderLSTM
        else:
            decoder = RecurrentDecoder
        return decoder

if __name__ == '__main__':

    # Configuration
    config = {}
    config['n_features'] = 3
    config['latent_dim'] = 4
    config['rnn_type'] = 'GRU'
    config['rnn_act'] = 'relu'
    config['device'] = 'cpu'
    config = edict(config)

    # Adding random data
    # X = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.float32).unsqueeze(2)
    X = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.float32).unsqueeze(2)
    X = torch.ones((2, 10, 3))  # （batch, seq_len,  input_size）
    print(X)
    print(X.shape)

    # Model
    model = RecurrentAE(config)

    # Encoder
    h = model.encoder(X)
    out = model.decoder(h, seq_len = 10)
    print(out.shape)
    # print(out)
    out = torch.flip(out, [1])
    # print(out)

    # Loss
    loss = nn.L1Loss(reduction = 'mean')
    l = loss(X, out)


