import torch.nn as nn
from .Layers_pytorch import FCLayer

class MLPModel(nn.Module):
    def __init__(
        self,
        comm,
        dp_size: int = 1,
        mp_size: int = 1,
        megatron_mp: bool = False,
        feature_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
    ):
        """
        Defines a MLP block using PyTorch.

        :param comm:  The MPI communicator wrapper
        :param dp_size: Data parallelism size
        :param mp_size: Model parallelism size
        :param megatron_mp: Whether to use Megatron-style model parallelism
        :param feature_dim: The feature dimension
        :param hidden_dim: The hidden dimension
        :param output_dim: The output dimension
        """
        super().__init__()
        self.comm = comm
        self.rank = comm.Get_rank()
        self.mp_size = mp_size
        self.dp_size = dp_size

        # FC1 is a Column-Parallel Linear layer
        self.fc1 = FCLayer(
            comm=comm,
            in_dim=feature_dim,
            out_dim=hidden_dim,
            bias=True,
            dp_size=dp_size,
            mp_size=mp_size,
            megatron_mp=megatron_mp,
            is_fc1=True,
        )
        
        self.relu = nn.ReLU()
        
        # FC2 is a Row-Parallel Linear layer
        self.fc2 = FCLayer(
            comm=comm,
            in_dim=hidden_dim,
            out_dim=output_dim,
            bias=True,
            dp_size=dp_size,
            mp_size=mp_size,
            megatron_mp=megatron_mp,
            is_fc1=False,
        )

    def forward(self, x):
        """
        :param x: input images of shape (batch_size, feature_dim)
        :return: output tensor of shape (batch_size, output_dim)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def get_rank(self):
        return self.rank 