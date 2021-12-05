# speech-language-modelling

### Towards Language Modelling in the Speech Domain Using Sub-word Lingustic Units [arXiv](https://arxiv.org/abs/2111.00610)
---

This repository contains code for reproducing the experiments and methods in our paper, titled as above. A simple diagram of the overall model is shown below:

![Speech Language Model Diagram](/assets/images/model.png)

---

### Running Experiments

To replicate our experiments from the paper, run:

```bash
python train.py --model CBOW_Left --encoder_decoder LSTM --batch 32 \
--num_layers 3 --compseq LJSpeech_phones_left_collated_utterances --no_pad_context 
```

To train your own model from scratch, and to experiment with other model architectures, use the following complete set of parameters:

```
- model: CBOW (use left and right context), CBOW_Left(left context only), or VAE,
- context: how many units (syllables/phonemes/words) of context to use,
- batch: batch size,
- epochs: number of epochs to train for,
- encoder_decoder: type of encoder-decoder model, one of FC (Fully Connected), LSTM, or Conv,
- chkpt: frequency of saving model checkpoints, in number of epochs,
- num_layers: number of layers to use in each encoder and decoder,
- pad_context: if true, pads utterances with < 2*context number of units with 0s,
- no_pad_context: if true, skips utterances with < 2*context number of units,
- postnet: if true, applies a tacotron-style PostNet architecture after the decoder,
- cbhg: if true, , applies a tacotron-style CBHG architecture after the decoder,
- lstm_lm_type: panphon, or w2v (wave2vec), depending on the type of auxiliary LSTM LM trained
```

Note on Data Format: In an effort to standardise the storage and sharing of datasets across research communities, we store our data in the [CMU-Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK) format. Eessentially, all melspectrograms are stored in a h5py dict (h5py has it's own advantages such as allowing the slicing of numpy arrays directly from disk without loading into memory). Read the code under [dataset.py](https://github.com/AnuragKatakkar/speech-language-modelling/blob/main/dataset.py#L305) to see how we read data, and see "Section 3 Dataset" of our paper for details on preprocessing and extracting melspectrograms.


---

