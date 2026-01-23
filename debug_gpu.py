from numba import cuda
import numpy as np

try:
    print("Detecting GPU...")
    print(cuda.detect())
    
    print("Creating test array...")
    x = np.array([1, 2, 3], dtype=np.float32)
    
    print("Moving to Device...")
    d_x = cuda.to_device(x)
    
    print("Success! Device array created.")
except Exception as e:
    print(f"FAILED: {e}")