#!/bin/sh
echo 'computer hacked...'
sleep 3
echo 'just kidding, going to install pytorch lightning and scann'
pip install -q git+https://github.com/PytorchLightning/pytorch-lightning-bolts.git@master --upgrade
pip install -q scann
echo 'Installed Libraries'
#download data
echo 'Downloading UC Merced Data'
gdown http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip -O /content/UCMerced_LandUse.zip
wait
unzip -qq /content/UCMerced_LandUse.zip

wait

python /content/SpaceForce-DataSearch/SSLTrainer.py --DATA_PATH /content/UCMerced_LandUse/Images --epoch 1 --num_workers 2 --version 0 --pretrain_encoder True
wait
python /content/SpaceForce-DataSearch/EvalEmbeddings.py --MODEL_PATH /content/models/SSL/SIMCLR_SSL_0/SIMCLR_SSL_0.pt --DATA_PATH /content/UCMerced_LandUse/Images --val_split 0.2
wait
python /content/SpaceForce-DataSearch/finetuner.py --MODEL_PATH /content/models/SSL/SIMCLR_SSL_0/SIMCLR_SSL_0.pt --DATA_PATH /content/UCMerced_LandUse/Images --val_split 0.2 --num_workers 4 --epochs 1 --version 1
