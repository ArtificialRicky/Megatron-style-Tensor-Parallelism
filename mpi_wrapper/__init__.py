import numpy as np
from mpi4py import MPI
import torch

class Communicator:
    def __init__(self, comm):
        self.comm = comm

    def All_reduce_mean(self, x):
        is_tensor = torch.is_tensor(x)
        if is_tensor:
            # Ensure tensor is on a CUDA device for direct GPU communication
            if not x.is_cuda:
                raise TypeError("Input tensor must be a CUDA tensor for MPI operations.")
            
            # The buffer for mpi4py is (pointer, MPI_datatype)
            # We assume float32 tensors.
            buffer = [x.contiguous().data_ptr(), MPI.FLOAT]
            self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)
            x /= self.size
        else: # Original numpy logic
            # Ensure x is a numpy array for the original logic
            if not isinstance(x, np.ndarray):
                raise TypeError("Input must be a numpy array or a torch CUDA tensor.")
            self.comm.Allreduce(MPI.IN_PLACE, [x, MPI.FLOAT], op=MPI.SUM)
            x /= self.size

    def All_reduce(self, x):
        is_tensor = torch.is_tensor(x)
        if is_tensor:
            if not x.is_cuda:
                raise TypeError("Input tensor must be a CUDA tensor for MPI operations.")
            
            buffer = [x.contiguous().data_ptr(), MPI.FLOAT]
            self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)
        else:
            if not isinstance(x, np.ndarray):
                raise TypeError("Input must be a numpy array or a torch CUDA tensor.")
            self.comm.Allreduce(MPI.IN_PLACE, [x, MPI.FLOAT], op=MPI.SUM)

    def Get_rank(self):
        return self.comm.Get_rank()
