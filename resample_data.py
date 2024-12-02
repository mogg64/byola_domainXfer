#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import librosa
import multiprocessing
import numpy as np
import soundfile as sf
import sys

data_path = sys.argv[1] 
if not data_path.endswith(os.path.sep):
    data_path += os.path.sep

print('starting: ',data_path)
print('collecting audio...')

audio_files = []
for xx in Path(data_path).glob("**/*.wav"):
    audio_files.append(xx)
for xx in Path(data_path).glob("**/*.mp3"):
    audio_files.append(xx)
for xx in Path(data_path).glob("**/*.flac"):
    audio_files.append(xx)
    
    
print('collected audio files: ',str(len(audio_files)))
print('resampling audio files!')

for idx, f in enumerate(audio_files):
    
    in_path = str(audio_files[idx])
    in_ext = os.path.splitext(in_path)[1]
    out_path = in_path.replace(os.path.dirname(data_path),os.path.dirname(data_path)+'_16k')
    
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    
    y, sr = librosa.load(in_path,sr = 16000,mono=True)

    sf.write(out_path, y, sr)
        
    print('done! ',in_path)



    
        