"""
Main training routine for various models
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

import cbow
from datasets import make_dataset
from models import LanguageModel

import numpy as np
import wandb
import argparse
import time
import sys
import pickle5 as pickle

DATA_FOLDER_PATH = "/content/"

def parse_args(args):
    """
        Parse command line arguments using the argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="LJSPeech")
    parser.add_argument("--compseq", type=str, default="LJSpeech_syllables_left_awb_collated_utterances")
    parser.add_argument("--model", type=str, default="CBOW", help="One of CBOW, CBOW_Left, or VAE")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--context", type=int, default=2)
    parser.add_argument("--model_path", type=str, default="/home1/akatakka/models/ljspeech/cbow")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--encoder_decoder", type=str, default="FC")
    parser.add_argument("--chkpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--num_layers", type=int, default=3, help="number of layers in encoder and decoder")
    parser.add_argument('--pad_context', default=False, action='store_true')
    parser.add_argument('--no_pad_context', dest='pad_context', action='store_false')
    parser.add_argument('--postnet', default=False, action='store_true')
    parser.add_argument('--cbhg', default=False, action='store_true')
    parser.add_argument('--lstm_lm_type', type=str, default="panphon", 
                            help="type of embedding used in the LSTM LM")

    args = parser.parse_args(args)
    return args

def train(model, lstm_lm_model, optimizer, criterion, train_loader, device="cuda:0", encoder_decoder="Conv"):
    """
        Main training routine for the model

        Args:
            - model: an instance of the PyTorch model to train, assumed to 
                    be on device already,
            - optimizer: optimizer to use,
            - criterion: criterion for the loss,
            - train_loader: DataLoader, 
            - device: cuda or cpu, 
            - encoder_decoder: decides whether to unsqueeze outputs

        Returns:
            - train_loss: float, loss at the end of the epoch
    """
    model.train()
    batch_loss = 0
    train_loss = 0
    for idx, (x, y, _, frames, _, lstm_lm_ids) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        lstm_lm_ids = lstm_lm_ids.to(device)[:, :-1]

        x_lens = torch.tensor([len(xx) for xx in lstm_lm_ids])

        lstm_lm_ids = pad_sequence(lstm_lm_ids, batch_first=True, padding_value=0.0)
        frames = frames.numpy()
        frames -= 1
        frames = np.clip(frames, -1, 31)

        lstm_lm_output = lstm_lm_model(lstm_lm_ids, x_lens)[0][:, -1, :].detach()

        outputs = model(x.float(), frames, lstm_lm_output)
        if encoder_decoder=="FC":
            loss = criterion(outputs.unsqueeze(1), y)
        else:
            loss = criterion(outputs.unsqueeze(1), y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        batch_loss += loss.item()

        if idx+1%500 == 0:
            print("Minibatch {}, Loss: {}".format(idx+1, batch_loss/500))
            batch_loss = 0

        del x, y, outputs, loss, frames, lstm_lm_ids, lstm_lm_output

    return train_loss/len(train_loader)

def validate(model, lstm_lm_model, optimizer, criterion, dev_loader, device="cuda:0", encoder_decoder="Conv"):
    """
        Main evaluation routine for the model

        Args:
            - model: an instance of the PyTorch model to train, assumed to 
                    be on device already,
            - optimizer: optimizer to use,
            - criterion: criterion for the loss,
            - dev_loader: DataLoader, 

        Returns:
            - dev_loss: float, loss at the end of the epoch
    """
    model.eval()
    val_loss = 0
    for idx, (x, y, _, frames, _, lstm_lm_ids) in enumerate(dev_loader):
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            lstm_lm_ids = lstm_lm_ids.to(device)[:, :-1]

            x_lens = torch.tensor([len(xx) for xx in lstm_lm_ids])

            lstm_lm_ids = pad_sequence(lstm_lm_ids, batch_first=True, padding_value=0.0)
            frames = frames.numpy()
            frames -= 1
            frames = np.clip(frames, -1, 31)

            lstm_lm_output = lstm_lm_model(lstm_lm_ids, x_lens)[0][:, -1, :].detach()

            outputs = model(x.float(), frames, lstm_lm_output)

            if encoder_decoder=="FC":
                loss = criterion(outputs.unsqueeze(1), y)
            else:
                loss = criterion(outputs.unsqueeze(1), y.unsqueeze(1))
            val_loss += loss.item()

            del x, y, outputs, loss, frames, lstm_lm_ids, lstm_lm_output

    return val_loss/len(dev_loader)

def init_weights(m):
    """
        Initialize model weights
    """
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight.data)
    elif type(m)==nn.LSTM:
        for name, param in m.named_parameters():
            if "bias" in name:
                torch.nn.init.constant_(param, 0.0)
            elif "weight" in name:
                torch.nn.init.xavier_uniform_(param)

if __name__=="__main__":
    """
        Main routine for end to end training, evaluation, checkpointing, etc.
    """

    # Parse arguments and define constants
    args = parse_args(sys.argv[1:])
    CUDA = torch.cuda.is_available()
    DEVICE = "cuda:{}".format(args.gpu) if CUDA else "cpu"
    # MODEL_FOLDER_PATH = args.model_path
    MODEL_FOLDER_PATH = "/content/gdrive/My Drive/Capstone/Fall 2020/Experiments/Models/LJSpeech AWB Alignments/" # To make sure we don't write in incorrect locations

    # First define all hyperparameters
    BATCH_SIZE = args.batch
    CONTEXT = args.context
    OPTIM = "Adam"
    LEARNING_RATE = 1e-5
    MODEL_NAME = "{}_k{}_{}_syllable_depth{}_lstm_lm_to_decoder_{}".format(args.model, CONTEXT, args.encoder_decoder, args.num_layers, args.lstm_lm_type)
    NUM_EPOCHS = args.epochs
    if NUM_EPOCHS == 0:
        NUM_EPOCHS = 1000

    COMPSEQ_PATH = DATA_FOLDER_PATH + args.compseq

    #Define config dict for wandb
    config_dict = dict(
        CONTEXT = CONTEXT,
        CUDA = CUDA,
        BATCH_SIZE=BATCH_SIZE,
        MODEL_FOLDER_PATH = MODEL_FOLDER_PATH,
        MODEL_NAME = MODEL_NAME,
        dataset = args.compseq,
        model_class = args.model,
        unit = "syllables",
        optim = OPTIM,
        lr = LEARNING_RATE,
        encoder_decoder_type=args.encoder_decoder,
        padded_dataset = args.pad_context,
        depth = args.num_layers,
    )
    if args.model=="CBOW" and args.encoder_decoder=="LSTM":
        print("WARNING: Have you fixed the frame indexing in dataset?")

    #init and log in wandb
    run = wandb.init(project="capstone", config=config_dict)

    # Open translation dict
    translation_dict = {}
    with open("/content/translation_dict_roberta_w2v_pp.pkl", "rb") as f:
        translation_dict = pickle.load(f)

    # Define dataset and dataloaders
    train_dataset, dev_dataset = make_dataset(args.model, COMPSEQ_PATH, context=CONTEXT, model_type=args.encoder_decoder, 
                                                pad_context=args.pad_context, panphon=False, translation_dict=translation_dict)

    train_dataloader = DataLoader(train_dataset, shuffle=True, pin_memory=True, 
                                  batch_size=BATCH_SIZE, drop_last=True, num_workers=8 if CUDA else 1)
    dev_dataloader = DataLoader(dev_dataset, shuffle=True, pin_memory=True, 
                                  batch_size=BATCH_SIZE, drop_last=True, num_workers=8 if CUDA else 1)
    
    print("Initialising Model....{}".format(MODEL_NAME))
    model = cbow._make_model(context=args.context, model_type=args.encoder_decoder, num_layers=args.num_layers, lstm_lm=True)
    print(model)
    model.apply(init_weights)
    model.to(DEVICE)

    # Load and initialise LSTM LMmodel

    lstm_lm_model = LanguageModel(embedding_type=args.lstm_lm_type)

    lstm_lm_model.load_state_dict(torch.load("/content/LSTM_PANPHON.pt"))
    lstm_lm_model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    epoch_start = 0 # Epoch counter start

    if args.chkpt is not None:
        print("Loading Checkpoint: {}".format(args.chkpt))
        chkpt = torch.load(args.chkpt)
        model.load_state_dict(chkpt["model_state_dict"])
        optimizer.load_state_dict(chkpt["optimizer_state_dict"])
        epoch_start = chkpt["epoch"] + 1
        NUM_EPOCHS += epoch_start
    
    print("Training....")
    train_start_time = time.time()
    for i in range(epoch_start, NUM_EPOCHS):
        epoch_start_time = time.time()
        train_loss = train(model, lstm_lm_model, optimizer, criterion, train_dataloader, DEVICE, args.encoder_decoder)
        dev_loss = validate(model, lstm_lm_model, optimizer, criterion, dev_dataloader, DEVICE, args.encoder_decoder)

        print("Epoch {}, Train Loss: {}, Val Loss: {}, Time Taken: {}".format(i, train_loss, dev_loss, time.time() - epoch_start_time))
        
        #Log on wandb
        wandb.log({"train_loss":train_loss, "val_loss":dev_loss})

        if i%10==0:
            print("========== TOTAL TIME ELAPSED : {:.4f}s ==========".format(time.time()-train_start_time))
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_loss,
                'val_losses': dev_loss,
                }, MODEL_FOLDER_PATH + MODEL_NAME + "_epoch{}.pt".format(i))
