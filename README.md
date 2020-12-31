# SpaceForce-DataSearch
A tool to quickly train a Self-Supervised Learning model and finetune the model on labeled data in order to help researchers rapidly curate a dataset for scientific studies without the need for machine learning, self supervised ML coding expertise.

__Requirements__: GPU with CUDA 10+ enabled, requirements.txt

Run `sh example.sh` to see the tool in action on the UC Merced land use dataset

To run it with your own data, please put your data in the following folder structure:
```
/Dataset
    /Class 1
        Image1.png
        Image2.png
    /Class 2
        Image3.png
        Image4.png
```
To optimize your environment for deep learning, run this repo on the pytorch nvidia docker:
`docker pull nvcr.io/nvidia/pytorch:20.12-py3`

```bash
mkdir docker_folder
docker run --user=root -p 7000-8000:7000-8000/tcp --volume="/etc/group:/etc/group:ro" --volume="/etc/passwd:/etc/passwd:ro" --volume="/etc/shadow:/etc/shadow:ro" --volume="/etc/sudoers.d:/etc/sudoers.d:ro" --gpus all -it --rm -v /home/rudyvenguswamy/docker_folder:/inside_docker nvcr.io/nvidia/pytorch:20.12-py3`
#now clone repo inside container, install requirements as usual
```

##How to access models after training
Both self-supervised models and finetuned models can be accessed and used normally as pl_bolts.LightningModule models. They function the same as a pytorch nn.Module but have added functionality that works with a pytorch lightning Trainer.
For example:
`from SpaceForceDataSearch import SIMCLR, finetuner`

`simclr_model = model.load_from_checkpoint('PATH_TO_SSL.ckpt')`

