# SpaceForce-DataSearch
A tool to help scientists using satellite imagery of specific phenomena to find similar images to rapidly curate a dataset for scientific studies

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
To optimize your environment for deep learning, run this repo on docker:
`docker pull nvcr.io/nvidia/pytorch:20.12-py3`

`mkdir docker_folder`

`docker run --user=root -p 7000-8000:7000-8000/tcp --volume="/etc/group:/etc/group:ro" --volume="/etc/passwd:/etc/passwd:ro" --volume="/etc/shadow:/etc/shadow:ro" --volume="/etc/sudoers.d:/etc/sudoers.d:ro" --gpus all -it --rm -v /home/rudyvenguswamy/docker_folder:/inside_docker nvcr.io/nvidia/pytorch:20.12-py3`

`clone repo inside container, install requirements as usual`
