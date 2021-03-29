# Mounting

This needs to be done manually for some headless machines.

### Check for available drives

```
sudo fdisk -l
```

Usually a USB drive is at the end of this list and is called something like `/dev/sda1`

### Create a mount point

A mount point is a folder where you access your drive (the door to your drive).

```
cd /mnt/
sudo mkdir mountpoint_name
```

### Mount it

```
sudo mount /dev/sda1 /mnt/mountpoint_name
```

### Unmount it

Finally unmount it by either unmounting the device or the mount point. Note that the unmount command is umount

```
sudo umount /mnt/mountpoint_name
```

