# Revisiting Deep Models for Image Super Resolution

## Environment Setup

Create a new conda environment and install PyTorch:

```shell
conda create -n revisit_sr python=3.8
source activate revisit_sr
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
```

Install the required packages:

```shell
pip install -r requirements.txt
```

## Multi-processing Distributed Data Parallel Training

### Single Node

Single GPU training:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
--master_port=9922 main.py --distributed
```

Single node with multiple (e.g., 4) GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
--master_port=9922 main.py --distributed
```

### Multiple Nodes

After activating the virtual environment with PyTorch>=1.8.0, run `hostname -I | awk '{print $1}'` to get the ip address of the master node. Suppose the master ip address is `10.31.133.85`, and we want to train the model on two nodes with multiple GPUs, then the commands are:

Node 0 (master node):

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
--nnodes=2 --node_rank=0 --master_addr="10.31.133.85" --master_port=9922 main.py \
--distributed
```

Node 1:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
--nnodes=2 --node_rank=1 --master_addr="10.31.133.85" --master_port=9922 main.py \
--distributed
```

Description of the options:

- `--nproc_per_node`: number of processes on each node. Set this to the number of GPUs on the node to maximize the training efficiency.
- `--nnodes`: total number of nodes for training.
- `--node_rank`: rank of the current node within all nodes.
- `--master_addr`: the ip address of the master (rank 0) node.
- `--master_port`: a free port to communicate with the master node.
- `--distributed`: multi-processing Distributed Data Parallel (DDP) training.
- `--local_world_size`: number of GPUs on the current node.

For a system with Slurm Workload Manager, please load required modules: `module load cuda cudnn`.
