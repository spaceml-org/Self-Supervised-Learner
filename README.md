<div align="center">

<img src="https://github.com/RudyVenguswamy/SpaceForceDataSearch/blob/main/readme/curator_logo_wide.PNG" >


<!--**Rapidly curate a dataset for scientific studies without the need for machine learning, self supervised ML coding expertise.** -->

---

<p align="center">
  Published by <a href="http://spaceml.org/">SpaceML</a> •
  <a href="https://arxiv.org/abs/2012.10610">About SpaceML</a> •
  <a href="https://colab.research.google.com/github/RudyVenguswamy/SpaceForceDataSearch/blob/main/PythonColabTutorial_Merced.ipynb">Quick Colab Example</a> 
</p>


[![Python Version](https://img.shields.io/badge/python-3.5%20|%203.6%20|%203.7%20|%203.8-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/Cuda-10%20|%2011.0-4dc71f.svg)](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html)
[![Pip Package](https://img.shields.io/badge/Pip%20Package-Coming%20Soon-0073b7.svg)](https://pypi.org/project/pip/)
[![Docker](https://img.shields.io/badge/Docker%20Image-Coming%20Soon-34a0ef.svg)](https://www.docker.com/)

[![Google Colab Notebook Example](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/RudyVenguswamy/SpaceForceDataSearch/blob/main/PythonColabTutorial_Merced.ipynb)
<!--
[![CodeFactor](https://www.codefactor.io/repository/github/pytorchlightning/pytorch-lightning/badge)](https://www.codefactor.io/repository/github/pytorchlightning/pytorch-lightning)
-->
</div>

# Curator :earth_americas:

Curator can be used to train a classifier with fewer labeled examples needed.

### How Curator Works
Step 1) **Self-Supervied Learning (SSL): Training an encoder without labels**
   - The first step is to train a self-supervised encoder. Self-supervised learning does not require labels and lets the model learn from purely unlabeled data to build an image encoder.

Step 2) **Fine tuning (FT): Training a classifier with labels**
   - With the self-supervised training done, the encoder is used to initialize a classifier (finetuning). Because the encoder learned from the entire unlabeled dataset previously, the classifier is able to achieve higher classification accuracy than training from scratch or pure transfer learning.

__Requirements__: GPU with CUDA 10+ enabled, [requirements.txt](https://github.com/spaceml-org/Self-Supervised-Learner/blob/main/requirements.txt)


<table border="0">
 <tr>
    <td><b style="font-size:30px">Most Recent Release</b></td>
    <td><b style="font-size:30px">Update</b></td>
    <td><b style="font-size:30px">Model</b></td>
    <td><b style="font-size:30px">Processing Speed</b></td>
 </tr>
 <tr>
    <td>:heavy_check_mark: 1.0.2</td>
    <td>Package Speed Improvements</td>
    <td>Support for SIMCLR</td>
    <td>Multi-GPU Training Supported</td>
 </tr>
</table>

## TL;DR Quick example
Run [`sh example.sh`](https://github.com/spaceml-org/Self-Supervised-Learner/blob/main/example.sh) to see the tool in action on the [UC Merced land use dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html).

## Using Your Own Data Set
```bash
SSL: python ssl_dali_distrib.py --ARGUMENTS
FT: python finetuner_dali_distrib.py --ARGUMENTS
```
To run it with your own data, please put your data in the following folder structure:
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
from SpaceForceDataSearch.ssl_dali_distrib import SIMCLR 
from SpaceForceDataSearch.finetuner_dali_distrib import finetuner

simclr_model = SIMCLR.load_from_checkpoint('PATH_TO_SSL_MODEL.ckpt')
finetuned_model = finetuner.load_from_checkpoint('PATH_TO_SSL_MODEL.ckpt')

print(simclr_model.encoder)
print(finetuned_model.encoder)

datapoint = my_load_and_transform_function('example.jpg')

embedding = simclr_model(datapoint)
prediction = finetuned_model(datapoint)
```

## Using Your Own Encoder
If you don't want to use the predefined encoders in encoders_dali.py, it's very easy to modify the code. To add your own model:
1) Modify encoders_dali.py and add your custom model, which needs to be of type torch.nn.Module.
2) Modify the load_encoder method of encoders_dali.py to have your model as an option. You will need to specify the output size (vector length - int) of the embedding generated for a single datapoint as well. Add another elif statement like so to do so:
```
 elif encoder_name == 'my_custom_encoder':
        model, embedding_size = my_custom_encoder(my_init_params), my_model_output_embedding_size: int
```

## Releases
- :heavy_check_mark: (0.7.0) Dali Transforms Added
- :heavy_check_mark: (0.8.0) UC Merced Example Added
- :heavy_check_mark: (0.9.0) Model Inference with Dali Supported
- :heavy_check_mark: (1.0.0) SIMCLR Model Supported
- :heavy_check_mark: (1.0.1) GPU Memory Issues Fixed
- :heavy_check_mark: (1.0.1) Multi-GPU Training Enabled
- :heavy_check_mark: (1.0.2) Package Speed Improvements
- :ticket: (1.0.3) Cluster Visualizations for Embeddings 
- :ticket: (1.1.0) Supporting numpy, TFDS datasets
- :ticket: (1.2.0) Saliency Maps for Embeddings

## Citation
If you find Curator useful in your research, please consider citing
```
@article{,
  title={},
  author={},
  journal={},
  year={}
}
```
