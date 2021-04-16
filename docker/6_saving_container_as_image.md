# Saving a container as an image

If you make changes in a running container, you can save it as an image via
```
docker commit -m "message" -a "author" [container name] [image name]
```

Reference: https://www.mirantis.com/blog/how-do-i-create-a-new-docker-image-for-my-application/

You can use the saved image to run a new container.
