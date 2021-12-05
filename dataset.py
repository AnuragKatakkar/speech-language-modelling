"""
Defines various datasets as well as util functions
for returning the desired dataset to the training routine.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import h5py
import pickle5 as pickle

class CBOWDataset(Dataset):
    """
        Builds the desired CBOW dataset. Note that we are agnostic 
        to the unit of alignment (syllable vs phoneme vs word)

    """
    def __init__(self, samples, context, sample_ids=None, sample_names=None,
                 frame_lens=None, cbow_type="CBOW", flatten=True, pad_context=True, panphon=None,
                 translation_dict=None):
        """
            Builds a dict datastructure that points index->sample_num
            This way we don't need to slice the dataset beforehand into
            samples of size 2*context+1, and can instead slice that at
            runtime in __getitem__

            Args:
                - samples: List[np.ndarray], list of utterances,
                - context: int, size of the context window,
                - sample_ids: List[int], list of int indexes of unit idx,
                - sample_names: List[str], sample names,
                - frame_lens: List[int], number of frames in the original unit 
                        before being padded with 0s (sil),
                - cbow_type: str, whether we are predicting the middle syllable, 
                        or rightmost,
                - flatten: bool, whether or not to flatten the melspectrogram. True for FC
                        encoder-decoder architectures,
                - pad_context: bool, whether or not to add 0 padded silence syllables to 
                        utterances
        """
        self.samples = samples
        self.sample_ids = sample_ids #Used to map from SYLL2IDX back to syllable
        self.sample_names = sample_names
        self.context = context
        self.len = 0
        self.sample_lens = []
        self.indices = {}
        self.cbow_type = cbow_type
        self.frame_lens = frame_lens
        self.flatten = flatten
        self.pad_context = pad_context
        self.panphon = True if panphon is not None else False
        self.syll2panphon = panphon["syll_2_panphon"] if panphon is not None else None
        self.idx2syll = panphon["idx2syll"] if panphon is not None else None
        self.translation_dict = translation_dict
        self.lstm_lm = True
        for idx, samp in enumerate(samples):
            self.sample_lens.append(self.len)
            uttr_len = samp.shape[0]
            if not self.pad_context:
                uttr_len -= 2*self.context
            for i in range(self.len, self.len+uttr_len):
                self.indices[i] = idx
            self.len += uttr_len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        sample_idx = self.indices[index]

        X = None
        Y = None
        if self.pad_context:
            if self.cbow_type == "CBOW":
                unit_idx = index - self.sample_lens[sample_idx] + self.context
                # pad context size on both sides
                utterance = np.pad(self.samples[sample_idx], ((self.context, self.context), (0,0), (0, 0)), mode='constant', constant_values=0)
                X = np.concatenate([utterance[unit_idx-self.context:unit_idx, :, :], 
                                    utterance[unit_idx+1:unit_idx+self.context+1, :, :]], axis=0)
                Y = utterance[unit_idx, :, :]
            else:
                # pad 2*context-1 0s on the left
                unit_idx = index - self.sample_lens[sample_idx] + 2*self.context - 1
                utterance = np.pad(self.samples[sample_idx], ((self.context*2 - 1, 1), (0,0), (0, 0)), mode='constant', constant_values=0)
                X = utterance[unit_idx-2*self.context+1:unit_idx+1, :, :]
                Y = utterance[unit_idx+1, :, :]
        else:
            unit_idx = index - self.sample_lens[sample_idx]
            utterance = self.samples[sample_idx]
            if self.cbow_type == "CBOW":
                X = np.concatenate([utterance[unit_idx:unit_idx+self.context, :, :], 
                                    utterance[unit_idx+self.context+1:unit_idx+2*self.context+1, :, :]], axis=0)
                Y = utterance[unit_idx+self.context, :, :]
            else:
                X = utterance[unit_idx:unit_idx+2*self.context, :, :]
                Y = utterance[unit_idx+2*self.context, :, :]


        if self.flatten:
            X = X.reshape(2*self.context, -1)
            Y = Y.reshape(1, -1)

        ids = self.sample_ids[sample_idx].tolist()
        frame_lens = self.frame_lens[sample_idx].tolist()
        #Pad ids with -1 for sil
        if self.pad_context:
            if self.cbow_type=="CBOW":
                ids = [-1]*self.context + ids + [-1]*self.context
                frame_lens = [0]*self.context + frame_lens + [0]*self.context
                ids = ids[unit_idx-self.context:unit_idx+self.context+1]
                frame_lens = frame_lens[unit_idx-self.context:unit_idx+self.context+1]
            else:
                ids =  [-1]*(self.context*2-1) + ids + [-1]
                frame_lens = [0]*(self.context*2-1) + frame_lens + [0]
                ids = ids[unit_idx-2*self.context+1:unit_idx+2]
                frame_lens = frame_lens[unit_idx-2*self.context+1:unit_idx+2]
        else:
            ids = ids[unit_idx:unit_idx+2*self.context+1]
            frame_lens = frame_lens[unit_idx:unit_idx+2*self.context+1]


        panphon_ftr = None
        if self.panphon:
            if self.cbow_type=="CBOW":
                panphon_syll = self.idx2syll[ids[self.context]]
                panphon_context_syll = None
            else:
                panphon_syll = self.idx2syll[ids[-1]]
                panphon_context_syll = [self.idx2syll[syll_idx] for syll_idx in ids[:-1]]
            panphon_ftr = self.syll2panphon[panphon_syll]
            # Concatenate context ftrs for now - can return as list later if needed
            panphon_context_ftrs = torch.cat([torch.FloatTensor(self.syll2panphon[syll]).squeeze(0) for syll in panphon_context_syll])
        
            return torch.tensor(X), torch.tensor(Y), torch.tensor(ids), torch.tensor(frame_lens), self.sample_names[sample_idx], torch.FloatTensor(panphon_ftr), panphon_context_ftrs

        if self.lstm_lm:
            lstm_lm_ids = [self.translation_dict['w2v'][idx] for idx in ids] # 0 at beginning for <s>
        
            return torch.tensor(X), torch.tensor(Y), torch.tensor(ids), torch.tensor(frame_lens), self.sample_names[sample_idx], torch.tensor(roberta_ids)


        return torch.tensor(X), torch.tensor(Y), torch.tensor(ids), torch.tensor(frame_lens), self.sample_names[sample_idx]


class CBOWInferenceDataset(Dataset):
    """
        Builds the desired CBOW dataset. Note that we are agnostic 
        to the unit of alignment (syllable vs phoneme vs word)

    """
    def __init__(self, samples, context, sample_ids=None, sample_names=None, frame_lens=None,
                 cbow_type="middle", flatten=True, pad_context=True, panphon=None, translation_dict=None):
        """
            Builds a dict datastructure that points index->sample_num
            This way we don't need to slice the dataset beforehand into
            samples of size 2*context+1, and can instead slice that at
            runtime in __getitem__

            Args:
                - samples: List[np.ndarray], list of utterances,
                - context: int, size of the context window,
                - sample_ids: List[int], list of int indexes of unit idx, used to map with SYLL2IDX
                - sample_names: List[str], sample names,
                - frame_lens: List[int], number of frames in the original unit 
                        before being padded with 0s (sil),
                - cbow_type: str, whether we are predicting the middle syllable, 
                        or rightmost
        """
        self.samples = samples
        self.context = context
        self.sample_ids = sample_ids
        self.sample_names = sample_names
        self.frame_lens = frame_lens
        self.cbow_type = cbow_type
        self.flatten = flatten
        self.pad_context = pad_context
        self.panphon = True if panphon is not None else False
        self.syll2panphon = panphon["syll_2_panphon"] if panphon is not None else None
        self.idx2syll = panphon["idx2syll"] if panphon is not None else None
        self.lstm_lm = True
        self.translation_dict = translation_dict
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        X = None
        if self.pad_context:
            if self.cbow_type == "CBOW":
                # pad context size on both sides
                X = np.pad(self.samples[index], ((self.context, self.context), (0,0), (0, 0)), mode='constant', constant_values=0)
            else:
                X = np.pad(self.samples[index], ((self.context*2 - 1, 0), (0,0), (0, 0)), mode='constant', constant_values=0)
        else:
            X = self.samples[index]

        if self.flatten:
            X = X.reshape(X.shape[0], -1)

        ids = self.sample_ids[index].tolist()
        frame_lens = self.frame_lens[index].tolist()
        
        if self.pad_context:
            #Pad ids with -1 for sil
            ids = [-1]*self.context + ids + [-1]*self.context if self.cbow_type=="CBOW" else [-1]*(self.context*2-1) + ids
            #Pad frame_lens with -1 for sil
            frame_lens = [0]*self.context + frame_lens + [0]*self.context if self.cbow_type=="CBOW" else [0]*(self.context*2-1) + frame_lens

        panphon_ftr = None
        if self.panphon:
            if self.cbow_type=="CBOW":
                panphon_syll = self.idx2syll[ids[self.context]]
            else:
                panphon_syll = self.idx2syll[ids[-1]]
            panphon_ftr = self.syll2panphon[panphon_syll]
        
            return torch.tensor(X), torch.tensor(ids), torch.tensor(frame_lens), self.sample_names[index], torch.FloatTensor(panphon_ftr)

        if self.lstm_lm:
            lstm_lm_ids = [self.translation_dict['w2v'][idx] for idx in ids] # 0 at beginning for <s>
        
            return torch.tensor(X), torch.tensor(ids), torch.tensor(frame_lens), self.sample_names[index], torch.tensor(lstm_lm_ids)


        return torch.tensor(X), torch.tensor(ids), torch.tensor(frame_lens), self.sample_names[index]


class VAEDataset(Dataset):
    def __init__(self, samples, sample_ids=None, sample_names=None, frame_lens=None):
        """
            Builds a dict datastructure that points index->sample_num
            This way we don't need to slice the dataset beforehand into
            samples of size 2*context+1, and can instead slice that at
            runtime in __getitem__

            Args:
                - samples: List[np.ndarray], list of utterances,
                - context: int, size of the context window,
                - sample_ids: List[int], list of int indexes of unit idx,
                - sample_names: List[str], sample names,
                - frame_lens: List[int], number of frames in the original unit 
                        before being padded with 0s (sil),
        """
        self.samples = samples
        self.sample_ids = sample_ids #Used to map from SYLL2IDX back to syllable
        self.sample_names = sample_names
        self.len = 0
        self.sample_lens = []
        self.indices = {}
        self.frame_lens = frame_lens
        for idx, samp in enumerate(samples):
            self.sample_lens.append(self.len)
            uttr_len = samp.shape[0]
            for i in range(self.len, self.len+uttr_len):
                self.indices[i] = idx
            self.len += uttr_len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        sample_idx = self.indices[index]
        unit_idx = index - self.sample_lens[sample_idx]

        X = self.samples[sample_idx][unit_idx, :, :]
        ids = self.sample_ids[sample_idx].tolist()[unit_idx]
        frame_lens = self.frame_lens[sample_idx].tolist()[unit_idx]

        return X, ids, frame_lens, self.sample_names[sample_idx]


class FrameLevelDataset(Dataset):
    def __init__(self, utterances, sample_names, max_frames=400):
        self.utterances = utterances
        self.sample_names = sample_names
        self.max_frames = max_frames

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index):
        X = self.utterances[index]
        if X.shape[0] < self.max_frames:
            X = np.pad(X, ((0, self.max_frames - X.shape[0]), (0, 0)), mode='constant', constant_values=0)[:400, :]
        return torch.tensor(X), self.sample_names[index]


def make_dataset(task, raw_data, dataset="LJSpeech", context=None, inference=False, model_type="FC", pad_context=True, panphon=False, translation_dict=None):
    """
    Constructs and returns the appropriate type of train, dev Datasets

    Args:
        - task: str, CBOW vs VAE, determines the type of Dataset built.
        - raw_data: str, path to the raw .h5 data,
        - dataset: str, name of the dataset used, and key of the h5py object,
        - context: int, the amount of context to use for CBOW datasets,
        - mode: str, defined for CBOW dataset, middle, or rightmost prediction,
    """

    # Open and process the compseq
    cs = h5py.File(raw_data, 'r')
    dset = cs[dataset]
    keys = list(dset['data'].keys()) #These are also the sample names in the dataset

    all_utterances = []
    utterance_ids = keys #names of the utterances
    unit_ids = [] #unit ids within the utterance
    og_lens = [] #num frames for each individual unit
    #Preprocess - Unpack the compseq to create a numpy dataset
    for key in keys:
        arr = dset['data'][key]['features'].value
        if model_type!="frame_attention":
            ids = dset['data'][key]['intervals'][:, 2]
            og_len = dset['data'][key]['intervals'][:, 3]

        if pad_context:
            all_utterances.append(arr)
            if model_type!="frame_attention":
                og_lens.append(og_len)
                unit_ids.append(ids)
        else:
            # If no padding is used, only include utterances that have 2K+1 syllables
            if arr.shape[0] >= 2*context + 1:
                all_utterances.append(arr)
                if model_type!="frame_attention":
                    og_lens.append(og_len)
                    unit_ids.append(ids)
    
    # Close h5 File
    cs.close()

    num_train = int(0.9 * len(all_utterances))
    flatten = False if model_type=="Conv" or model_type=="LSTM" or model_type=="VAE_CBOW" else True

    # If predicting panphon features, open syll2idx and syll2panphon dictionaries
    panphon_dicts = None
    if panphon:
        syll2idx = {}
        with open("/content/SYLL2IDX_nopau_none_2.pkl", "rb") as f:
            syll2idx = pickle.load(f)

        idx2syll = {syll2idx[k]:k for k in syll2idx}

        panphon_ftr_compseq = h5py.File("/content/LJSpeech_panphon_syllables", 'r')
        pp_dset = panphon_ftr_compseq["panphon_syllables"]
        pp_keys = list(pp_dset['data'].keys())

        syll_2_panphon = {}
        for key in pp_keys:
            syll_2_panphon[key] = pp_dset['data'][key]['features'].value

        panphon_dicts = {
            "idx2syll" : idx2syll,
            "syll_2_panphon" : syll_2_panphon
        }


    if task=="CBOW" or task=="CBOW_Left":

        if model_type == "frame_attention":
            train_dataset = FrameLevelDataset(all_utterances[num_train:], sample_names=keys[num_train:])
            dev_dataset = FrameLevelDataset(all_utterances[:num_train], sample_names=keys[:num_train])
            return train_dataset, dev_dataset

        if inference:
            return CBOWInferenceDataset(all_utterances[num_train:], context=context, 
                                    sample_ids=unit_ids[num_train:], sample_names=keys[num_train:], 
                                    frame_lens=og_lens[num_train:], cbow_type=task, flatten=flatten,
                                    pad_context=pad_context, panphon=panphon_dicts, translation_dict=translation_dict)

        # samples, context, sample_ids=None, sample_names=None, frame_lens=None, cbow_type="middle")
        train_dataset = CBOWDataset(all_utterances[:num_train], context=context, 
                                    sample_ids=unit_ids[:num_train], sample_names=keys[:num_train], 
                                    frame_lens=og_lens[:num_train], cbow_type=task, flatten=flatten,
                                    pad_context=pad_context, panphon=panphon_dicts, translation_dict=translation_dict)

        dev_dataset = CBOWDataset(all_utterances[num_train:], context=context, 
                                    sample_ids=unit_ids[num_train:], sample_names=keys[num_train:], 
                                    frame_lens=og_lens[num_train:], cbow_type=task, flatten=flatten,
                                    pad_context=pad_context, panphon=panphon_dicts, translation_dict=translation_dict)

        return train_dataset, dev_dataset

    elif task=="VAE":
        # task is VAE
        train_dataset = VAEDataset(all_utterances[:num_train], sample_ids=unit_ids[:num_train],
                                    sample_names=keys[:num_train], frame_lens=og_lens[:num_train])
        dev_dataset = VAEDataset(all_utterances[num_train:], sample_ids=unit_ids[num_train:],
                                    sample_names=keys[num_train:], frame_lens=og_lens[num_train:])
        return train_dataset, dev_dataset
