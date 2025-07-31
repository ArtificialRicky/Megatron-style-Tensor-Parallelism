import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np

# This is a custom autograd function to handle the all-reduce operation
# for the output of the row-parallel linear layer in the forward pass,
# and to pass the gradients correctly in the backward pass.
class _ReduceForwardPass(Function):
    @staticmethod
    def forward(ctx, input, comm):
        ctx.comm = comm
        input_clone = input.clone()
        comm.All_reduce(input_clone)
        return input_clone

    @staticmethod
    def backward(ctx, grad_output):
        # The gradient is just passed through. The all-reduce in the backward
        # pass of the *preceding* column-parallel layer will handle the gradient summation.
        return grad_output, None

# This is a custom autograd function to handle the all-gather operation
# for the input of the row-parallel linear layer in the backward pass.
class _GatherBackwardPass(Function):
    @staticmethod
    def forward(ctx, input, comm):
        ctx.comm = comm
        return input

    @staticmethod
    def backward(ctx, grad_output):
        comm = ctx.comm
        grad_output_clone = grad_output.clone()
        # All-reduce the gradients for the weights of the column-parallel layer.
        comm.All_reduce(grad_output_clone)
        return grad_output_clone, None


class FCLayer(nn.Module):
    def __init__(
        self,
        comm,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        dp_size: int = 1,
        mp_size: int = 1,
        megatron_mp: bool = False,
        is_fc1: bool = True, # Determines if it's column or row parallel
    ):
        super().__init__()
        self.comm = comm
        self.rank = comm.Get_rank()
        self.mp_rank = self.rank % mp_size
        self.dp_rank = self.rank // mp_size
        self.mp_size = mp_size
        self.dp_size = dp_size
        self.megatron_mp = megatron_mp
        self.is_fc1 = is_fc1 # FC1 is Column-Parallel, FC2 is Row-Parallel

        self.in_dim = in_dim
        self.out_dim = out_dim
        
        if self.megatron_mp:
            if self.is_fc1: # Column-Parallel Linear Layer
                assert self.out_dim % self.mp_size == 0
                self.partitioned_out_dim = self.out_dim // self.mp_size
                self.W = nn.Parameter(torch.empty(self.in_dim, self.partitioned_out_dim))
                if bias:
                    self.b = nn.Parameter(torch.empty(self.partitioned_out_dim))
                else:
                    self.b = None
            else: # Row-Parallel Linear Layer
                assert self.in_dim % self.mp_size == 0
                self.partitioned_in_dim = self.in_dim // self.mp_size
                self.W = nn.Parameter(torch.empty(self.partitioned_in_dim, self.out_dim))
                if bias:
                     self.b = nn.Parameter(torch.empty(self.out_dim))
                else:
                    self.b = None
        else: # Standard or Data Parallel
            self.W = nn.Parameter(torch.empty(self.in_dim, self.out_dim))
            if bias:
                self.b = nn.Parameter(torch.empty(self.out_dim))
            else:
                self.b = None

        self.init_weights()

    def init_weights(self):
        # Initialization similar to the numpy version
        nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))
        if self.b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b, -bound, bound)
    
    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.W.device)

        if self.megatron_mp:
            if self.is_fc1: # Column-Parallel
                # (g o f) where f is identity and g is identity in fwd pass
                # In bwd pass, f is all-gather and g is identity
                # We apply the custom function to handle the backward pass all-gather
                x = _GatherBackwardPass.apply(x, self.comm)
                output = torch.matmul(x, self.W)
            else: # Row-Parallel
                # (g o f) where f is identity and g is all-reduce in fwd pass
                # In bwd pass, f is identity and g is identity
                output_partial = torch.matmul(x, self.W)
                # Sum the partial outputs across all ranks
                output = _ReduceForwardPass.apply(output_partial, self.comm)
        else: # Data Parallel or Single GPU
            output = torch.matmul(x, self.W)

        if self.b is not None:
            output += self.b
        
        # Handle Data Parallelism Gradient Averaging
        if self.dp_size > 1 and self.training:
            # This should be handled by the optimizer logic, not here.
            # But for simplicity in this refactor, we can do it here.
            # This requires access to gradients, which is only after .backward()
            # So this logic should be moved to the training loop.
            pass

        return output 