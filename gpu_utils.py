import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

def init_rng(total_elements, seed=1234):
    """
    Initializes the RNG states for the GPU threads.
    """
    return create_xoroshiro128p_states(total_elements, seed=seed)

@cuda.jit
def zero_array_3d(arr):
    """
    Fast kernel to zero out a 3D array on the device.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = arr.shape
    if i < nx and j < ny and k < nz:
        arr[i, j, k] = 0

@cuda.jit
def init_centroid_kernel(cx, cy, cz):
    """
    Initialize centroids to the center of the cell (0.5).
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = cx.shape
    if i < nx and j < ny and k < nz:
        cx[i, j, k] = 0.5
        cy[i, j, k] = 0.5
        cz[i, j, k] = 0.5