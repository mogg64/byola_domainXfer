#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import librosa
import multiprocessing
import numpy as np
import soundfile as sf
import random
import torch
import torchaudio     
import pandas as pd 

from byol_a.common import *
from byol_a.augmentations import PrecomputedNorm
from byol_a.models import AudioNTT2020

repo_path = '/Users/oggmw1/Documents/XHLT'
os.chdir(repo_path)

device = torch.device('cpu')
cfg = load_yaml_config('config.yaml')
print(cfg)

model_path = sys.argv[1]
data_path = sys.argv[2] 

stats = pd.read_csv( os.path.join(os.path.dirname(model_path), 'stats.csv') ) 
stats = [stats['mean'][0], stats['std'][0]]

# Preprocessor and normalizer.
to_melspec = torchaudio.transforms.MelSpectrogram(
    sample_rate=cfg.sample_rate,
    n_fft=cfg.n_fft,
    win_length=cfg.win_length,
    hop_length=cfg.hop_length,
    n_mels=cfg.n_mels,
    f_min=cfg.f_min,
    f_max=cfg.f_max,
)
normalizer = PrecomputedNorm(stats)

# Load pretrained weights.
model = AudioNTT2020(d=cfg.feature_d)
model.load_weight(model_path, device)
model.eval()

# collect audio files
print('starting: ',data_path)
print('collecting audio...')

audio_files = []
for xx in Path(data_path).glob("**/*.wav"):
    audio_files.append(xx)
for xx in Path(data_path).glob("**/*.mp3"):
    audio_files.append(xx)
for xx in Path(data_path).glob("**/*.flac"):
    audio_files.append(xx)
random.shuffle(audio_files) 
    
print('collected audio files: ',str(len(audio_files)))

print('extracting embeds from audio files!')
for idx, f in enumerate(audio_files):
    
    in_path = str(audio_files[idx])
    _, ext = os.path.splitext(in_path)
    out_path = in_path.replace(ext,'_byola_embed.npy')
    print('starting!',out_path)

    # Load your audio file.
    wav, sr = torchaudio.load(in_path) 
    # confirm sample rate matches what model expects. 
    # if needed, convert sample rate ahead of time if needed see resample_data.py
    assert sr == cfg.sample_rate 

    if wav.shape[1] < sr:
        pad_len = sr - wav.shape[1]
        wav = torch.cat((wav,torch.zeros(1,pad_len)),axis=1)

    # Convert to a log-mel spectrogram, then normalize.
    lms = normalizer((to_melspec(wav) + torch.finfo(torch.float).eps).log())

    # Now, convert the audio to the representation.
    rep = model(lms.unsqueeze(0))

    rep_np = torch.squeeze(rep).detach().numpy()
    np.save(out_path,rep_np)
    print('done! ',in_path)

                                

