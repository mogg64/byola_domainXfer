# byola_domainXfer
Models from the paper: Self-Supervised Convolutional Audio Models are Flexible Acoustic Feature Learners: A Domain Specificity and Transfer-Learning Study
See this report for full details.

# Installation
Install from requirements file and install original BYOL-A repo https://github.com/nttcslab/byol-a 

See also getting_started.txt

A small batch of test audio to check functionality (referenced below) [can be found here](https://www.dropbox.com/scl/fi/6pmb19i02er79j489lnsy/test_audio.zip?rlkey=g0yde5bgu0kignwhvkq03eo17&st=z4xfrzfz&dl=0)

# Usage
We provide scripts to extract embeddings for batches of audio files (organized into folders).

Models are designed to operate on 16k sampling rate audio data (we assume .wav files). We provide a script to do this with librosa and this operates as a preprocessor.
````
python resample_data.py /path/to/test_audio
````

Then extract embeddings for the preprocessed audio. We have released BYOLA checkpoints for each of the training diets. Download a model and extract the .zip file inside the checkpoints folder. Then you should see checkpoints/checkpoints_all (or ending in '_sp' or '_ns' for '_all' data, speech or non-speech respectively). Each folder contains a checkpoint.pth and a stats.csv that extract_embeds.py reads to load the model and the metadata for normalization. 
````
python extract_embeds.py /path/to/checkpoint/file.pth /path/to/test_audio_16k
````

# Checkpoints
Model checkpoints can be downloaded at the following links:

Model trained on [speech data](https://www.dropbox.com/scl/fi/sgk7r4cs38666olllfnei/checkpoints_sp.zip?rlkey=rvzh3sacb1lvwqicydluurcfr&st=m72dnah2&dl=0)

Model trained on [non-speech data](https://www.dropbox.com/scl/fi/bt6fo5popfbyx94lrpfpv/checkpoints_ns.zip?rlkey=2k41yye6y3ebw7t1jjtzd3bth&st=5jt1vm03&dl=0)

Model trained on [speech and non-speech data](https://www.dropbox.com/scl/fi/jg1s99561lrt1ty7n5app/checkpoints_all.zip?rlkey=a2d612zxzmhj9rbs21nf59wy0&st=571waac7&dl=00)

