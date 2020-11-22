# Nvidia Cuda Docker Images

If one wishes to use cuda tools in a docker container, it is best to download nvidia-cuda images.

## 1. Set up Nvidia container toolkit
This is necessary for docker containers to support cuda functions. We follow the Nvidia official guide [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to set up the container toolkit.

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

Then install the `nvidia-docker2` package.
```
apt update
apt install nvidia-docker2
```

and restart docker service
```
systemctl restart docker
```

## 2. Test if the environment works

The following command executes a run-and-toss container to see if `nvidia-smi` work. One should note that the **cuda version in the tag should match the original cuda version** installed on the host machine. Surely remember to use `--gpus all`.

```
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```
## 3. Use the container of your choice
For example,
```
nvidia/cuda:11.1-devel-ubuntu20.04
```

These docker images come with preinstalled `nvcc` and cuda libraries. One **should NOT** install `nvidia-driver` in the container.