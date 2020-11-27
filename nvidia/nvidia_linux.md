# Installing Nvidia driver and CUDA toolkit

This records some problems I've encountered during installing Nvidia drivers and CUDA toolkit.

## On Linux PC

It is highly discouraged that one uses `apt` to install Nvidia drivers and libraries. Because `apt` determines library versions from the existing repos, it could result in version mismatch. For example, if you have installed CUDA 10.2 and Nvidia driver 430, and want to update to Nvidia driver 450, you would run `apt install nvidia-driver-450` as would be suggested by most online answers. But the installed CUDA 10.0 would record in system the cuda 10.0 repo. And an attempt such as `apt install nvidia-cuda-toolkit` would result in installing from this older repo. Hence there would be a mismatch of versions between the CUDA libraries and driver version.

The correct and safe way would be to download the installation files from the [official CUDA toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive) and install as instructed.

#### Potential problem 1: Local installer and network installer

It is recommended to use the local installer. It can be seen from the installation guide of the network installer that it first adds a new repo to the list of `apt` repos, and then install from it using `apt`. However, if there was a previous installation of CUDA, there would be an entry of a repo for an older version of Nvidia libraries, which takes higher priority when running `apt update` because it is installed early. Hence you would end up with a new driver but old libraries.

If you are sure to do it this way, delete the old repos first manually. Even if you don't plan to do it this way, delete them anyway after you have successfully installed the new version.

## On Linux docker

If one wishes to use cuda tools in a docker container, it is best to download nvidia-cuda images. 

### 1. Set up Nvidia container toolkit
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

### 2. Test if the environment works

The following command executes a run-and-toss container to see if `nvidia-smi` work. One should note that the **cuda version in the tag should match the original cuda version** installed on the host machine. Surely remember to use `--gpus all`.

```
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```
### 3. Use the container of your choice
For example,
```
nvidia/cuda:11.0-cudnn8-devel-ubuntu20.04
```

These docker images come with preinstalled `nvcc` and cuda libraries. One **should NOT** install `nvidia-driver` or `nvidia-cuda-toolkit` in the container.

For a complete list of usable tags and their descriptions see its [docker hub page](https://hub.docker.com/r/nvidia/cuda) or the [gitlab page](https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md)

#### Potential problem 2: GPG key error

This could be the problem of the CDN that hosts the Nvidia repos (at [developer.download.nvidia.com](developer.download.nvidia.com)). Current solutions:

 1. `gpg --keyserver keyserver.ubuntu.com --recv-keys F60F4B3D7FA2AF80`
 2. `apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub`
 3. Use `apt update --allow-unauthenticated`
 4. Use `apt install apt-transport-https`

Some related discussion are at [here](https://github.com/NVIDIA/nvidia-docker/issues/1369), [here](https://github.com/NVIDIA/nvidia-docker/issues/613), [here](https://github.com/NVIDIA/nvidia-docker/issues/969), [here](https://github.com/NVIDIA/nvidia-docker/issues/658) and [here](https://blog.csdn.net/weixin_43545898/article/details/108960744).

#### Potential problem 3: nvcc not found

`nvcc` is supposed to exist at the creation of the container. However it is possible that after installing some other apps (or maybe updating libraries, I don't know), `nvcc` becomes unavailable. Don't install with `apt install nvidia-cuda-toolkit`. Check if `nvcc` exists in the local CUDA path, which is located at `/usr/local/cuda-x.x`. If so, then the problem is with `PATH`. Add the following to `~/.bashrc`.

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-x.x/lib64
export PATH=$PATH:/usr/local/cuda-x.x/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-x.x
```

Change `cuda-x.x` to your version of installtion.

#### Potential problem 4: Nvidia driver version mismatch

This happens if the docker has a newer Nvidia driver version than the host machine, but they have the same CUDA version. The reason is that in `/usr/lib/x86_64-linux-gnu`, we have a symbolic link `libcuda.so.1 -> libcuda.so.xxx.xxx.xx`, where `xxx.xxx.xx` is the default version that comes with the docker image. However, if the Nvidia driver of the host machine has a lower version, this could cause a mismatch. The solution is to relink `libcuda.so.1`.