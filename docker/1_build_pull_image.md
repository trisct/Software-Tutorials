## `docker pull` command

You can pull an existing image from the docker hub by
```
docker pull ubuntu:16.04
```
where "16.04" is a tag that specifies the version. If no tag is provided, the default tag is "latest".

## Managing images

Pulled images can be used to build new containers at any time. But saving the images locally can take a lot of space. To check the pulled images one can use
```
docker images -a
```

This command lists all images saved locally. One can then delete images by ID, for example,
```
docker rmi image_id
```
