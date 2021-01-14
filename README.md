![alt text](https://github.com/RudyVenguswamy/SpaceForceDataSearch/blob/main/curator.PNG)
# SpaceForce-DataSearch
A tool to quickly train a Self-Supervised Learning model and finetune the model on labeled data in order to help researchers rapidly curate a dataset for scientific studies without the need for machine learning, self supervised ML coding expertise.

__Requirements__: GPU with CUDA 10+ enabled, requirements.txt

Run `sh example.sh` to see the tool in action on the UC Merced land use dataset

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
docker run --user=root -p 7000-8000:7000-8000/tcp --volume="/etc/group:/etc/group:ro" --volume="/etc/passwd:/etc/passwd:ro" --volume="/etc/shadow:/etc/shadow:ro" --volume="/etc/sudoers.d:/etc/sudoers.d:ro" --gpus all -it --rm -v /home/rudyvenguswamy/docker_folder:/inside_docker nvcr.io/nvidia/pytorch:20.12-py3
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

## Coming Updates
- Cluster Visualizations for Embeddings 
- Supporting numpy, TFDS datasets
- Saliency Maps for Embeddings


