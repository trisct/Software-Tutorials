# ssh: Secure shell

## ssh with RSA key

Make sure that the ssh service is installed on both sides.

### 1. Check if there is existing RSA key pair

Go to `~/.ssh/` and check if there is existing RSA keys (default: `id_rsa` and `id_rsa.pub`). If not, generate use `ssh-keygen -b 4096`.

### 2. Put the public key on the remote host

Create file (unless there already is one) `username@remotehost:~/.ssh/authorized_keys`. Copy the contents of `id_rsa.pub` to `authorized_keys`.