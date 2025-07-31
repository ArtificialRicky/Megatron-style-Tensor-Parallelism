import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from mpi4py import MPI
from mpi_wrapper import Communicator
from logger import log_args, log_stats
import argparse, os
from data.data_parallel_preprocess import split_data
from model.MLP_pytorch import MLPModel

def lr_schedule(init_lr, iter_num, decay=0.9, stage_num=100):
    return init_lr * (decay ** (np.floor(iter_num / stage_num)))

def train_mlp_pytorch(
    x_train, y_train, x_test, y_test, model, device, num_epoch=3, batch_size=60, init_lr=0.1, comm=None
):
    iter_num = 0
    num_examples = x_train.shape[0]
    rank = model.get_rank()
    dp_size = model.dp_size
    mp_size = model.mp_size

    # Use a standard PyTorch optimizer
    optimizer = optim.SGD(model.parameters(), lr=init_lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        # Train
        model.train()
        if rank == 0:
            print("*" * 40 + "Training" + "*" * 40)
        
        # Shuffle training data
        perm = torch.randperm(num_examples)
        x_train, y_train = x_train[perm], y_train[perm]

        for i in range(0, num_examples, batch_size):
            optimizer.zero_grad()
            
            x_batch = x_train[i : i + batch_size].to(device)
            y_batch = y_train[i : i + batch_size].to(device)
            
            output = model(x_batch)
            loss = criterion(output, y_batch)
            
            loss.backward()

            # Average gradients for data parallelism
            if dp_size > 1:
                for param in model.parameters():
                    if param.grad is not None:
                        # The communicator needs to handle PyTorch tensors on GPU
                        comm.All_reduce_mean(param.grad)

            lr = lr_schedule(init_lr, iter_num, stage_num=100 / dp_size)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.step()
            
            iter_num += 1

            if (iter_num + 1) % 10 == 0 and rank == 0:
                predict = torch.argmax(output, dim=1)
                acc = (predict == y_batch).sum().item() / y_batch.shape[0]
                print(
                    f"Epoch:{epoch+1} iter_num:{i}/{num_examples}: Train Loss: {loss.item():.4f}, Train Acc: {acc:.4f}, lr_rate: {lr:.4f}"
                )
        
        if rank == 0:
            print("*" * 88)

        # Evaluate
        model.eval()
        if rank // mp_size == 0: # Only DP rank 0 of each MP group evaluates
            eval_acc = 0
            total_test_samples = x_test.shape[0]

            if rank % mp_size == 0:
                print("\n" + "*" * 40 + "Evaluating" + "*" * 40)

            with torch.no_grad():
                for i in range(0, total_test_samples, batch_size):
                    x_batch = x_test[i : i + batch_size].to(device)
                    y_batch = y_test[i : i + batch_size].to(device)
                    
                    output = model(x_batch)
                    predict = torch.argmax(output, dim=1)
                    eval_acc += (predict == y_batch).sum().item()

            if rank % mp_size == 0:
                print(f"Test Acc: {eval_acc / total_test_samples:.4f}")
                print("*" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp_size", type=int, help="model parallel size", default=1)
    parser.add_argument("--dp_size", type=int, help="data parallel size", default=1)
    parser.add_argument(
        "--megatron-mp",
        action="store_true",
        help="Use this flag to enable Megatron-style model parallelism",
    )
    args = parser.parse_args()

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    comm = MPI.COMM_WORLD
    comm = Communicator(comm)
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    assert args.dp_size * args.mp_size == nprocs, "Total processes must equal dp_size * mp_size"

    # Set up GPU device for this process
    # This assumes one process per GPU.
    # The user needs to ensure this with `mpirun` and CUDA_VISIBLE_DEVICES.
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda")

    batch_size = 60
    init_lr = 0.01

    if rank == 0:
        log_args(
            batch_size=batch_size,
            init_lr=init_lr,
            dp_size=args.dp_size,
            mp_size=args.mp_size,
            megatron_mp=args.megatron_mp,
        )

    # load MNIST data
    MNIST_data = h5py.File("./data/MNISTdata.hdf5", "r")

    x_train_np = np.float32(MNIST_data["x_train"])
    y_train_np = np.int32(np.array(MNIST_data["y_train"][:, 0]))

    x_train_np, y_train_np = split_data(
        x_train=x_train_np,
        y_train=y_train_np,
        mp_size=args.mp_size,
        dp_size=args.dp_size,
        rank=rank,
    )
    
    # Convert to PyTorch tensors
    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(y_train_np).long()
    x_test = torch.from_numpy(np.float32(MNIST_data["x_test"][:]))
    y_test = torch.from_numpy(np.int32(np.array(MNIST_data["y_test"][:, 0]))).long()
    MNIST_data.close()
    
    mlp_model = MLPModel(
        comm=comm,
        dp_size=args.dp_size,
        mp_size=args.mp_size,
        megatron_mp=args.megatron_mp,
        feature_dim=784,
        hidden_dim=256,
        output_dim=10,
    ).to(device)

    train_mlp_pytorch(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        model=mlp_model,
        device=device,
        num_epoch=3,
        batch_size=int(batch_size / args.dp_size),
        init_lr=init_lr,
        comm=comm,
    ) 