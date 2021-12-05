"""
Utils for various signal processing requirements, file handling, etc.
"""

import numpy as np
import librosa
import soundfile as sf

def invert_and_save(mel, output_dir, name):
    """
        Invert a melspectrogram and write to file as a .wav
    """
    y = librosa.feature.inverse.mel_to_audio(mel.T, n_fft=1024, hop_length=128, win_length=1024)
    sf.write(output_dir+name+".wav", y, 16000, subtype='PCM_16')
