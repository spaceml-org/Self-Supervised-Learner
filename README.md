<div align="center">

<img src="https://github.com/spaceml-org/Self-Supervised-Learner/blob/main/readme_icons/curator_logo_wide.PNG" >


<!--**Rapidly curate a dataset for scientific studies without the need for machine learning, self supervised ML coding expertise.** -->

---

<p align="center">
  Published by <a href="http://spaceml.org/">SpaceML</a> •
  <a href="https://arxiv.org/abs/2012.10610">About SpaceML</a> •
  <a href="https://github.com/spaceml-org/Self-Supervised-Learner/blob/simsiam/tutorials/PythonColabTutorial_Merced.ipynb">Quick Colab Example</a> 
</p>


[![Python Version](https://img.shields.io/badge/python-3.5%20|%203.6%20|%203.7%20|%203.8-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/Cuda-10%20|%2011.0-4dc71f.svg)](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html)
[![Pip Package](https://img.shields.io/badge/Pip%20Package-Coming%20Soon-0073b7.svg)](https://pypi.org/project/pip/)
[![Docker](https://img.shields.io/badge/Docker%20Image-Coming%20Soon-34a0ef.svg)](https://www.docker.com/)

[![Google Colab Notebook Example](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/spaceml-org/Self-Supervised-Learner/blob/simsiam/tutorials/PythonColabTutorial_Merced.ipynb)

</div>

# Curator :earth_americas:

Curator can be used to train a classifier with fewer labeled examples needed using self-supervised learning. This repo is for you if you have a lot of unlabeled images and a small fraction (if any) of them labeled.


<ins> **What is Self-Supervised Learning?** </ins> \
Self-supervised learning is a subfield of machine learning focused on developing representations of images without any labels, which is useful for reverse image searching, categorization and filtering of images, especially so when it would be infeasible to have a human manually inspect each individual image. It also has downstream benefits for classification tasks. For instance, training SSL on 100% of your data and finetuning the encoder on the 5% of data that has been labeled significantly outperforms training a model from scratch on 5% of data or transfer learning based approaches typically.

### How To Use SSL Curator
Step 1) **Self-Supervied Learning (SSL): Training an encoder without labels**
   - The first step is to train a self-supervised encoder. Self-supervised learning does not require labels and lets the model learn from purely unlabeled data to build an image encoder.
```bash
python train.py --technique SIMCLR --model imagenet_resnet18 --DATA_PATH myDataFolder/AllImages  --epochs 100 --log_name ssl 
```

Step 2) **Fine tuning: Training a classifier with labels**
   - With the self-supervised training done, the encoder is used to initialize a classifier (finetuning). Because the encoder learned from the entire unlabeled dataset previously, the classifier is able to achieve higher classification accuracy than training from scratch or pure transfer learning.

```bash
python train.py --technique CLASSIFIER --model ./models/SIMCLR_ssl.ckpt --DATA_PATH myDataFolder/LabeledImages  --epochs 100 --log_name finetune 
```

__Requirements__: GPU with CUDA 10+ enabled, [requirements.txt](https://github.com/spaceml-org/Self-Supervised-Learner/blob/main/requirements.txt)


<table border="0">
 <tr>
    <td><b style="font-size:30px">Most Recent Release</b></td>
    <td><b style="font-size:30px">Update</b></td>
    <td><b style="font-size:30px">Model</b></td>
    <td><b style="font-size:30px">Processing Speed</b></td>
 </tr>
 <tr>
    <td>:heavy_check_mark: 1.0.3</td>
    <td>Package Documentation Improved</td>
    <td>Support for SIMSIAM</td>
    <td>Multi-GPU Training Supported</td>
 </tr>
</table>

## TL;DR Quick example
Run [`sh example.sh`](https://github.com/spaceml-org/Self-Supervised-Learner/blob/main/example.sh) to see the tool in action on the [UC Merced land use dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html).

## Arguments to train.py
You use train.py to train an SSL model and classifier. There are multiple arguments available for you to use: 

__Mandatory Arguments__

```--model```: The architecture of the encoder that is trained. All encoder options can be found in the models/encoders.py. Currently resnet18, imagenet_resnet18, resnet50, imagenet_resnet50 and minicnn are supported. You would call minicnn with a number to represent output embedding size, for example ```minicnn32```

```--technique```: What type of SSL or classification to do. Options as of 1.0.4 are SIMCLR, SIMSIAM or CLASSIFIER

```--log_name```: What to call the output model file (prepended with technique). File will be a .ckpt file, for example SIMCLR_mymodel2.ckpt

```--DATA_PATH```: The path to your data. If your data does not contain a train and val folder, a copy will automatically be created with train & val splits

Your data must be in the following folder structure as per pytorch ImageFolder specifications:
```
/Dataset
    /Class 1
        Image1.png
        Image2.png
    /Class 2
        Image3.png
        Image4.png

#When your dataset does not have labels yet you still need to nest it one level deep
/Dataset
    /Unlabelled
        Image1.png
        Image2.png

```

__Optional Arguments__

```--batch_size```: batch size to pass to model for training

```--epochs```: how many epochs to train

```--learning_rate```: learning rate for the encoder when training

```--cpus```: how many cpus you have to use for data reading

```--gpus```: how many gpus you have to use for training

```--seed```: random seed for reproducibility

```-patience```: early stopping if validation loss does not go down for (patience) number of epochs

```--image_size```: 3 x image_size x image_size input fed into encoder

```--hidden_dim```: hidden dimensions in projection head or classification layer for finetuning, depending on the technique you're using

```--OTHER ARGS```: each ssl model and classifier have unique arguments specific to that model. For instance, the classifier lets you select a ```linear_lr``` argument to specify a different learning rate for the classification layer and the encoder. These optional params can be found by looking at the ```add_model_specific_args``` method in each model contained in the models folder.



__Optional:__ To optimize your environment for deep learning, run this repo on the pytorch nvidia docker:

```bash
docker pull nvcr.io/nvidia/pytorch:20.12-py3
mkdir docker_folder
docker run --user=root -p 7000-8000:7000-8000/tcp --volume="/etc/group:/etc/group:ro" --volume="/etc/passwd:/etc/passwd:ro" --volume="/etc/shadow:/etc/shadow:ro" --volume="/etc/sudoers.d:/etc/sudoers.d:ro" --gpus all -it --rm -v /docker_folder:/inside_docker nvcr.io/nvidia/pytorch:20.12-py3
apt update
apt install -y libgl1-mesa-glx
#now clone repo inside container, install requirements as usual, login to wandb if you'd like to
```

## How to access models after training in python environment
Both self-supervised models and finetuned models can be accessed and used normally as `pl_bolts.LightningModule` models. They function the same as a pytorch nn.Module but have added functionality that works with a pytorch lightning Trainer.

For example:
```python
from models import SIMCLR, CLASSIFIER
simclr_model = SIMCLR.SIMCLR.load_from_checkpoint('/content/models/SIMCLR_ssl.ckpt') #Used like a normal pytorch model
classifier_model = CLASSIFIER.CLASSIFIER.load_from_checkpoint('/content/models/CLASSIFIER_ft.ckpt') #Used like a normal pytorch model
```

## Using Your Own Encoder
If you don't want to use the predefined encoders in models/encoders.py, you can pass your own encoder as a .pt file to the --model argument and specify the --embedding_size arg to tell the tool the output shape from the model.

## Releases
- :heavy_check_mark: (0.7.0) Dali Transforms Added
- :heavy_check_mark: (0.8.0) UC Merced Example Added
- :heavy_check_mark: (0.9.0) Model Inference with Dali Supported
- :heavy_check_mark: (1.0.0) SIMCLR Model Supported
- :heavy_check_mark: (1.0.1) GPU Memory Issues Fixed
- :heavy_check_mark: (1.0.1) Multi-GPU Training Enabled
- :heavy_check_mark: (1.0.2) Package Speed Improvements
- :heavy_check_mark: (1.0.3) Support for SimSiam and Code Restructuring
- :ticket: (1.0.4) Cluster Visualizations for Embeddings 
- :ticket: (1.1.0) Supporting numpy, TFDS datasets
- :ticket: (1.2.0) Saliency Maps for Embeddings

## Citation
If you find Curator useful in your research, please consider citing the tool:
```
@article{
  title={Curator: A No-Code, Self-Supervised Learning and Active Labeling Tool to Create Labeled Image Datasets from Petabyte-Scale Imagery
},
  author={Rudy Venguswamy},
  journal={European Geosciences Union 2021},
  year={2021}
}
```

</div>

