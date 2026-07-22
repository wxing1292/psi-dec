macos layer:

```
limactl start mvm

limactl stop mvm

limactl edit mvm

limactl shell mvm
```

host OS layer:

- terminal 0:

```
sudo rm -f /tmp/firecracker.socket
sudo firecracker --api-sock /tmp/firecracker.socket --enable-pci
```

- terminal 1:

```
start-vm.sh
```
