## `docker start` command

Starts a existing but exited container.

## `docker stop` command

Stops (exits) a running container.

## `docker rm` command

Removes a container. Use `--force` to force the removal (if it's running).

## `docker exec` command

Executes a command inside the container, as follows.
```
docker exec container_name command_in_container
```
For example,
```
docker exec container_name bash
```
runs the `bash` command in the container. One must use `-i` option to interact with the bash program, and `-t` option to open a pseudo-tty (teletypewriter) to exhibit bash prompts.
