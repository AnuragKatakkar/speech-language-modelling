"""
Base Ecnoder/Decoder Models that can be used as part of other architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

class FullyConnectedEncoder(nn.Module):
    def __init__(self, dims, dropout=0.2):
        super(FullyConnectedEncoder, self).__init__()
        self.layers = []
        for idx, dim in enumerate(dims[:-1]):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(dim, dims[idx+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

class FullyConnectedDecoder(nn.Module):
    def __init__(self, dims, dropout=0.2):
        super(FullyConnectedDecoder, self).__init__()
        self.layers = []
        for idx, dim in enumerate(dims[:-1]):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(dim, dims[idx+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class ConvEncoder(nn.Module):
    def __init__(self, hidden_channels, embedding_dim, input_size=(32, 80)):
        super(ConvEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_channels = hidden_channels
        self.kernel_dims, self.output_size = self.get_kernel_dims()
        self.embedding_dim = embedding_dim

        self.modules = []
        for idx, kernel in enumerate(self.kernel_dims):
            if idx == 0:
                self.modules.append(
                    nn.Sequential(
                    nn.Conv2d(1, hidden_channels[idx],
                              kernel_size=kernel, stride=(2, 1)),
                    nn.BatchNorm2d(hidden_channels[idx]),
                    nn.LeakyReLU())
                )
            else:
                self.modules.append(
                  nn.Sequential(
                  nn.Conv2d(hidden_channels[idx-1], hidden_channels[idx],
                            kernel_size=kernel, stride=(2, 1)),
                  nn.BatchNorm2d(hidden_channels[idx]),
                  nn.LeakyReLU())
                )
        self.encoder = nn.Sequential(*self.modules)
        self.embedding = nn.Linear(self.output_size, self.embedding_dim)

    def forward(self, x):
        conv_output = self.encoder(x)
        return self.embedding(conv_output.reshape(-1, self.output_size))

    def get_kernel_dims(self):
        # Assume a convolution size of 3 or 4
        kernel_dims = []
        input = self.input_size[0]

        for i in range(len(self.hidden_channels)):
            if input < 3:
                break
            if i == 0:
                kernel_dim2 = 80
            else:
                kernel_dim2 = 1
            
            if input%2==0:
                kernel_dim1 = 4
            else:
                kernel_dim1 = 3

            kernel_dims.append((kernel_dim1, kernel_dim2))
            input = self.get_conv_output_size(input, kernel_dim1, stride=2)
            output_size = input*self.hidden_channels[i]

        return kernel_dims, int(output_size)

    def get_conv_output_size(self, input, kernel, stride):
        return (input - kernel)//stride + 1


class ConvDecoder(nn.Module):
    def __init__(self, hidden_channels, embedding_dim, input_size=(32, 80)):
        super(ConvDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_channels = hidden_channels
        self.kernel_dims, self.output_size = self.get_kernel_dims()
        self.kernel_dims.reverse()
        self.hidden_channels = self.hidden_channels[:len(self.kernel_dims)]
        self.hidden_channels.reverse()
        self.embedding_dim = embedding_dim

        self.modules = []
        for idx, kernel in enumerate(self.kernel_dims):
            if idx == len(self.kernel_dims) - 1:
                self.modules.append(
                    nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_channels[idx], 1,
                              kernel_size=kernel, stride=(2, 1)),
                    nn.LeakyReLU())
                )
            else:
                self.modules.append(
                  nn.Sequential(
                  nn.ConvTranspose2d(self.hidden_channels[idx], self.hidden_channels[idx+1],
                            kernel_size=kernel, stride=(2, 1)),
                  nn.BatchNorm2d(self.hidden_channels[idx+1]),
                  nn.LeakyReLU())
                )
        self.decoder = nn.Sequential(*self.modules)
        self.embedding = nn.Linear(self.embedding_dim, self.output_size)

    def forward(self, x):
        output_C = self.hidden_channels[0]
        output_H = self.output_size//output_C
        embedding = self.embedding(x).reshape(-1, output_C, output_H, 1)
        return self.decoder(embedding).squeeze(1)

    def get_kernel_dims(self):
        # Assume a convolution size of 3 or 4
        kernel_dims = []
        input = self.input_size[0]

        for i in range(len(self.hidden_channels)):
            if input < 3:
                break
            if i == 0:
                kernel_dim2 = 80
            else:
                kernel_dim2 = 1
            
            if input%2==0:
                kernel_dim1 = 4
            else:
                kernel_dim1 = 3

            kernel_dims.append((kernel_dim1, kernel_dim2))
            input = self.get_conv_output_size(input, kernel_dim1, stride=2)
            output_size = input*self.hidden_channels[i]

        return kernel_dims, int(output_size)

    def get_conv_output_size(self, input, kernel, stride):
        return (input - kernel)//stride + 1


class LSTMEncoder(nn.Module):
    def __init__(self, hidden_size=512, num_layers=3, dropout=0.2):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=80, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, x, frames):
        """
            Since we pad each syllable with 0 valued frames, 
            we want to index into the hidden states and only
            use the last hidden state of the non-padded part
            of each syllable
        """
        out, (h, c) = self.lstm(x)
        return out[range(out.shape[0]), frames]

class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size=512, output_size=80, num_layers=3, output_frames=32):
        """
            Args:
                - output_frames: number of time steps to unroll the decoder for,
        """
        super(LSTMDecoder, self).__init__()
        
        self.layers = nn.ModuleList()
        self.output_frames = output_frames
        self.num_layers = num_layers

        for i in range(num_layers):
            if i==num_layers-1:
                output=output_size
            else:
                output=hidden_size
            self.layers.append(
                nn.LSTMCell(input_size=hidden_size, hidden_size=output)
            )
        
    def forward(self, x):
        output = []
        hiddens = [None]*self.num_layers
        for time_step in range(self.output_frames):
            if time_step==0:
                inp = x
            else:
                inp = hiddens[0][0]

            for layer_idx in range(self.num_layers):
                hiddens[layer_idx] = self.layers[layer_idx](inp, hiddens[layer_idx])
                inp = hiddens[layer_idx][0]

                if layer_idx==self.num_layers-1:
                    output.append(hiddens[layer_idx][0].unsqueeze(1))

        return torch.cat(output, axis=1)


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, keys, values, queries, lens):

        energy = torch.bmm(keys, queries.unsqueeze(2)).squeeze(2)

        attention = F.softmax(energy, dim=1)

        out = torch.bmm(attention.unsqueeze(1), values).squeeze(1)

        return attention, out


class PanphonMLP(nn.Module):
    def __init__(self, input_dimension=1024, output_dim=66, hidden_dims=[512, 512, 256], objective="classification", share_classifier=True):
        super(PanphonMLP, self).__init__()

        self.objective = objective
        self.share_classifier = share_classifier
        self.input_dimension = input_dimension
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.layers = []
        self.classifiers = []
        layer_dims = [input_dimension] + hidden_dims + [output_dim]
        for idx, layer_dim in enumerate(layer_dims[:-1]):
            if idx == 3:
                if self.objective=="classification":
                    if share_classifier:
                        self.classifiers = nn.Linear(layer_dim, 3)
                    else:
                        self.classifiers = nn.ModuleList([nn.Linear(layer_dim, 3) for i in range(output_dim)])
                else:
                    self.layers.append(
                        nn.Sequential(
                            nn.Linear(layer_dim, layer_dims[idx+1])
                        )
                    )
            else:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(layer_dim, layer_dims[idx+1]),
                        nn.Dropout(0.2),
                        nn.GELU()
                    )
                )

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        if self.objective=="classification":
            latent = self.layers(x)
            if self.share_classifier:
                # Repeat x 66 times
                latent_repeated = latent.unsqueeze(1).repeat(1, 66, 1)
                return self.classifiers(latent_repeated)
            else:
                output = []
                for i in range(self.output_dim):
                    output.append(self.classifiers[i](latent).unsqueeze(1))
                return torch.cat(output, dim=1)

        else:
            return self.layers(x)


class PostNet(nn.Module):
    def __init__(self, num_layers=3):
        super(PostNet, self).__init__()
        self.layers = []
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(1, 512, kernel_size=(5, 5), stride=1, padding=(5//2, 5//2)), 
                nn.Dropout(0.2),
                nn.ReLU()
            )
        )

        self.layers.append(
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(5, 5), stride=1, padding=(5//2, 5//2)), 
                nn.Dropout(0.2),
                nn.ReLU()
            )
        )

        self.layers.append(
            nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=(5, 5), stride=1, padding=(5//2, 5//2)), 
                nn.Dropout(0.2),
                nn.ReLU()
            )
        )

        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        output = self.layers(x.unsqueeze(1))
        return output.squeeze(1)

class LanguageModel(nn.Module):
    def __init__(self, vocab_size=7792, embedding_type="panphon"):
        super(LanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_type = embedding_type
        self.embedding_dim = 300
        self.hidden_dim = 300
        self.tie_weights = True
        self.projection = nn.Linear(66, 300)
        nlayers = 3
        self.embedding_layer_dim = 66 if embedding_type=="panphon" else 300
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_layer_dim)
        self.rnns  = [
                nn.LSTM(
                    self.embedding_dim if l == 0 else self.hidden_dim,
                    self.hidden_dim if l != nlayers - 1 else self.embedding_dim,
                    1,
                    dropout=0 if l!= nlayers-1 else 0, batch_first=True) for l in range(nlayers)
            ]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(self.embedding_dim, self.vocab_size)
        if self.embedding_type=="w2v":
          self.decoder.weight = self.embedding.weight


    def forward(self, x, lens):
        # Feel free to add extra arguments to forward (like an argument to pass in the hiddens)
        out = self.embedding(x)
        if self.embedding_type == "panphon":
            out = F.relu(self.projection(out))
        for layer in self.rnns:
          packed_x = pack_padded_sequence(out, lens, batch_first=True, enforce_sorted=False)
          packed_out = layer(packed_x.float())[0]
          out, out_lens = pad_packed_sequence(packed_out, batch_first=True)
          out, (h, c) = layer(out)
        decoder_out = self.decoder(out)
        return out, decoder_out
