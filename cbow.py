"""
This file defines variants of the CBOW model used to predict
the target melspectrogram from a given context of melspectrograms.
"""

import torch
import torch.nn as nn

import numpy as np

from models import FullyConnectedEncoder, FullyConnectedDecoder, ConvEncoder, ConvDecoder, LSTMEncoder, LSTMDecoder, Attention, PanphonMLP, PostNet
from tacotron_layers import CBHG
from vae import _make_vae

class CBOW(nn.Module):
    def __init__(self, context, encoder_dims, decoder_dims, share_encoder=False, encoder_type="FC",
                 decoder_type="FC", embedding_dim=512, batch_size=32, num_layers=3, lstm_lm=False, postnet=False, cbhg=False):
        super(CBOW, self).__init__()
        self.context = context
        self.share_encoder = share_encoder
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.num_layers = num_layers
        self.lstm_lm = lstm_lm
        # self.postnet = PostNet()
        self.use_postnet = postnet
        self.use_cbhg = cbhg
        if self.use_cbhg:
            self.cbhg = CBHG(in_dim=256)

        if encoder_type=="FC":
            self.encoder = FullyConnectedEncoder(encoder_dims)
            if not share_encoder:
                self.encoder = nn.ModuleList([FullyConnectedEncoder(encoder_dims) for i in range(context*2)])
        
        elif encoder_type=="Conv":
            self.encoder = ConvEncoder(encoder_dims, embedding_dim)
            if not share_encoder:
                self.encoder = nn.ModuleList([ConvEncoder(encoder_dims, embedding_dim) for i in range(context*2)])
        
        elif encoder_type=="LSTM":
            # By default we will only support shared encoder for LSTM
            lstm_encoder_dim = 256 # New for concatenative models
            self.encoder = LSTMEncoder(hidden_size=lstm_encoder_dim, num_layers=self.num_layers)


        if decoder_type=="FC":
            self.decoder = FullyConnectedDecoder(decoder_dims)
        elif decoder_type=="Conv":
            self.decoder = ConvDecoder(decoder_dims, embedding_dim)
        elif decoder_type=="LSTM":
            lstm_decoder_dim = (256 * 2 * self.context) + 300
            if self.use_cbhg:
                output_dim = 256
            else:
                output_dim = 80
            self.decoder = LSTMDecoder(hidden_size=lstm_decoder_dim, output_size=output_dim, num_layers=self.num_layers)

        # if panphon:
        #     self.panphon_model = PanphonMLP()

    def forward(self, x, frames=None, lstm_lm_ftrs=None):
        if self.encoder_type=="FC":
            if self.share_encoder:
                embedding = self.encoder(x[:, 0, :])
                for i in range(1, self.context*2):
                    embedding += self.encoder(x[:, i, :])
            else:
                embedding = self.encoder[0](x[:, 0, :])
                for i in range(1, self.context*2):
                    embedding += self.encoder[i](x[:, i, :])
        
        elif self.encoder_type=="Conv":
            if self.share_encoder:
                embedding = self.encoder(x[:, 0, :, :].unsqueeze(1))
                for i in range(1, self.context*2):
                    embedding += self.encoder(x[:, i, :, :].unsqueeze(1))
            else:
                embedding = self.encoder[0](x[:, 0, :, :].unsqueeze(1))
                for i in range(1, self.context*2):
                    embedding += self.encoder[i](x[:, i, :, :].unsqueeze(1))

        elif self.decoder_type=="LSTM":
            embedding = []
            # embedding = self.encoder(x[:, 0, :, :], frames[:, 0])
            for i in range(self.context*2):
                embedding.append(self.encoder(x[:, i, :, :], frames[:, i]))

            embedding = torch.cat(embedding, axis=1)
            embedding = torch.cat([embedding, lstm_lm_ftrs.squeeze(1)], axis=1)

        # if self.roberta:
        #     roberta_prediction = self.panphon_model(embedding)

        #     if self.use_postnet:
        #         return self.postnet(self.decoder(embedding)), roberta_prediction
        #     if self.use_cbhg:
        #         return self.cbhg(self.decoder(embedding)), roberta_prediction
        #     return self.decoder(embedding), roberta_prediction

        if self.use_postnet:
            return self.postnet(self.decoder(embedding))
        if self.use_cbhg:
                return self.cbhg(self.decoder(embedding))
        return self.decoder(embedding)


class VariationalCBOW(nn.Module):
    """
        A joint model of CBOW and VAE. The VAE is used to auto-encode all
        the context syllables and reconstruction+vae loss is minimised on those,
        The latent mu_i of context variables are summed and used in ConvDecoder
        to predict the next syllable. Reconstruction loss is minimized over the latter.
    """
    def __init__(self, context, decoder_dims, embedding_dim):
        super(VariationalCBOW, self).__init__()
        self.embedding_dim = embedding_dim
        self.context = context

        self.vae = _make_vae()
        self.decoder = ConvDecoder(decoder_dims, embedding_dim)

    def forward(self, x):
        context_reconstruction = []
        mu_i = []
        logvar_i = []
        
        vae_outputs = self.vae(x[:, 0, :, :].unsqueeze(1))
        context_reconstruction.append(vae_outputs[0].unsqueeze(1))
        embedding = vae_outputs[2]
        mu_i.append(vae_outputs[2])
        logvar_i.append(vae_outputs[3])

        for i in range(1, self.context*2):
            vae_outputs = self.vae(x[:, i, :, :].unsqueeze(1))
            context_reconstruction.append(vae_outputs[0].unsqueeze(1))
            embedding += vae_outputs[2]
            mu_i.append(vae_outputs[2])
            logvar_i.append(vae_outputs[3])

        # Predict next syllable using embedding
        prediction = self.decoder(embedding)

        return torch.cat(context_reconstruction, axis=1), mu_i, logvar_i, prediction


class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder_hidden_size=256, decoder_hidden_state=256, embedding_dim=256, context=20, isTrain=True):
        super(Seq2SeqAttention, self).__init__()

        self.context = context

        self.embedding_layer = nn.Linear(80, embedding_dim)

        self.encoder1 = nn.LSTMCell(input_size=80, hidden_size=encoder_hidden_size)
        self.encoder2 = nn.LSTMCell(input_size=encoder_hidden_size, hidden_size=encoder_hidden_size)
        self.encoder3 = nn.LSTMCell(input_size=encoder_hidden_size, hidden_size=encoder_hidden_size)

        self.decoder1 = nn.LSTMCell(input_size=encoder_hidden_size*2, hidden_size=encoder_hidden_size)
        self.decoder2 = nn.LSTMCell(input_size=encoder_hidden_size, hidden_size=encoder_hidden_size)
        self.decoder3 = nn.LSTMCell(input_size=encoder_hidden_size, hidden_size=80)

        self.key_network = nn.Linear(encoder_hidden_size, embedding_dim)
        self.value_network = nn.Linear(encoder_hidden_size, embedding_dim)

        self.attention = Attention()

        self.isTrain = isTrain

        self.max_len = 400

    def forward(self, x, epoch):
        max_len = self.max_len
        encoder_hiddens = [None, None, None]

        keys, values = [], []

        decoder_hiddens = [None, None, None]
        decoder_outputs = []

        teacher_forcing_prob = 0.95
        if epoch > 10:
            teacher_forcing_prob -= (epoch-10)*0.01

        if teacher_forcing_prob < 0.5:
            teacher_forcing_prob = 0.5

        for i in range(max_len):
            # Perform operations through encoder layers
            if i <= self.context:
                encoder_input = x[:, i, :]
            else:
                if self.isTrain:
                    if np.random.uniform(0, 1) < teacher_forcing_prob:
                        encoder_input = x[:, i, :]    
                    else:
                        encoder_input = decoder_outputs[-1].squeeze(1)
                else:
                    encoder_input = decoder_outputs[-1].squeeze(1)

            encoder_hiddens[0] = self.encoder1(encoder_input, encoder_hiddens[0])
            
            input2 = encoder_hiddens[0][0]
            encoder_hiddens[1] = self.encoder2(input2, encoder_hiddens[1])

            input3 = encoder_hiddens[1][0]
            encoder_hiddens[2] = self.encoder3(input3, encoder_hiddens[2])

            key = self.key_network(encoder_hiddens[2][0])
            value = self.value_network(encoder_hiddens[2][0])

            keys.append(key.unsqueeze(1))
            values.append(value.unsqueeze(1))

            if i >= self.context:
                # Perform operations through decoder
                if i - self.context == 0:
                    emb = self.embedding_layer(x[:, i-1, :].unsqueeze(1))
                    context = values[-1]
                else:
                    if self.isTrain:
                        if np.random.uniform(0, 1) < teacher_forcing_prob:
                            emb = self.embedding_layer(x[:, i, :].unsqueeze(1))    
                        else:
                            emb = self.embedding_layer(decoder_outputs[-1])
                    else:
                        emb = self.embedding_layer(decoder_outputs[-1])

                    attn_keys = torch.cat(keys[i-self.context:i], axis=1)
                    attn_values = torch.cat(values[i-self.context:i], axis=1)
                    
                    attention, context = self.attention(attn_keys, attn_values, decoder_hiddens[1][0], None)
                    context = context.unsqueeze(1)

                decoder_input1 = torch.cat([emb, context], dim=2).squeeze(1)
                decoder_hiddens[0] = self.decoder1(decoder_input1, decoder_hiddens[0])

                decoder_input2 = decoder_hiddens[0][0]
                decoder_hiddens[1] = self.decoder2(decoder_input2, decoder_hiddens[1])

                decoder_input3 = decoder_hiddens[1][0]
                decoder_hiddens[2] = self.decoder3(decoder_input3, decoder_hiddens[2])

                decoder_outputs.append(decoder_hiddens[2][0].unsqueeze(1))

        return torch.cat(decoder_outputs, dim=1)



def _make_model(input_size=32*80, context=2, model_type="FC", share_encoder=False, num_layers=3, isTrain=True, 
                panphon=False, postnet=False, cbhg=False, lstm_lm=False):
    """
        Instantiates desired model and returns and instance of that model

        Args:
            - input_size: int, currently hard-coded to 32*80 (num_frames*num_mels)
    """

    if model_type=="FC":
        encoder_dims = [input_size, 1024, 1024, 768, 512, 512]
        decoder_dims = [512, 512, 768, 1024, 1024, input_size]

        return CBOW(context=context, encoder_dims=encoder_dims, decoder_dims=decoder_dims, share_encoder=share_encoder, panphon=panphon)
    
    elif model_type=="Conv":
        hidden_channels = [32, 64, 128, 512, 512]
        embedding_dim = 512
        return CBOW(context=context, encoder_dims=hidden_channels, decoder_dims=hidden_channels, embedding_dim=512,
                    encoder_type="Conv", decoder_type="Conv", share_encoder=share_encoder, postnet=postnet, cbhg=cbhg, lstm_lm=lstm_lm)
    
    elif model_type=="LSTM":
        return CBOW(context=context, encoder_dims=None, decoder_dims=None, embedding_dim=512,
                    encoder_type="LSTM", decoder_type="LSTM", num_layers=num_layers, postnet=postnet, cbhg=cbhg, lstm_lm=lstm_lm)

    elif model_type=="VAE_CBOW":
        hidden_channels = [32, 64, 128, 512, 512]
        embedding_dim = 512
        return VariationalCBOW(context=context, decoder_dims=hidden_channels, embedding_dim=embedding_dim)

    elif model_type=="frame_attention":
        return Seq2SeqAttention(context=context, isTrain=isTrain)
