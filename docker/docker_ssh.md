# SSH into docker

This teachers you how to ssh into docker

## 1. Publish a port

This should be done during creating the container with a `-p [foreign_port:22]` option. Of course, you can configure ssh to listen to ports other than 22.

## 2. Install ssh in container

```
apt install ssh
```

## 3. Allow login

Go to `/etc/ssh/sshd_config` and set

```
PermitRootLogin yes
```

## 4. Login with the correct IP

If you are running the container on your local machine, the loop-back IP 127.0.0.1 should be enough.
```
ssh root@127.0.0.1 -p foreign_port
```

If the container is on a remote machine, use the foreign IP of that remote host.
```
ssh root@foreign_host_IP -p foreign_port
```

If you don't know the IP. Use
```
ifconfig
```
which can be install by
```
apt install net-tools
```