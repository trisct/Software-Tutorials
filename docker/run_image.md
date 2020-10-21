## `docker run` command

Suppose you have an image call 'image_example'. The following command allows you to start a container from this image.

```
docker run image_example
```

Some commonly used flags are as follows.

`-i` or `--interactive` runs the container in interactive mode, even if not attached

`-p xx:yy` or `--publish xx:yy` forwards port xx of the host to port yy of the container. (port 22: ssh protocol)

`-d` or `--detach` runs the container in the background

`--name container_name` names the container as 'container_name'

`-v path_on_host:path_in_container` maps two paths in and outside a container

`--shm-size="4g"` specifies the size of the shared memory (size of /dev/shm)

### `docker stop` command

This stops a running container
```
docker stop container_name
```


### `docker rm` command

This removes a container

```
docker rm container_name
```
