# pytorch-playground
My PyTorch playground for NLP

## Setup commands I used
* Installing Anaconda (Python3.6 will also be installed)<br>
wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh<br>
bash Anaconda3-4.2.0-Linux-x86_64.sh # prefix: $ANACONDA_PATH

* Installing PyTorch (0.2.0) in an Anaconda env<br>
conda create --name pytorch_test<br>
source $ANACONDA_PATH/envs/pytorch_test/bin/activate pytorch_test<br>
conda install pytorch torchvision cuda80 -c soumith

## Models
* Text Classifier (./text_classifier)<br>
Classifying input text (wrods, phrases, sentences, or documents) using LSTM

## Questions or comments?
Feel free to e-mail me (hassy@logos.t.u-tokyo.ac.jp).
