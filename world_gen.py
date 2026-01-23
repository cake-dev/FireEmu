import numpy as np
import math
from numba import cuda, njit, prange
import os

# --- GPU NOISE KERNELS ---

@cuda.jit(device=True)
def hash12(x, y, seed):
    """
    GLSL-style hash function for pseudo-random numbers on GPU.
    Returns float between 0.0 and 1.0
    """
    val = math.sin(x * 12.9898 + y * 78.233 + seed) * 43758.5453
    return val - math.floor(val)

@cuda.jit(device=True)
def mix(a, b, t):
    return a * (1.0 - t) + b * t

@cuda.jit(device=True)
def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)

@cuda.jit(device=True)
def noise_2d_gpu(x, y, seed):
    """
    Value Noise implementation on GPU.
    """
    ix = math.floor(x)
    iy = math.floor(y)
    fx = x - ix
    fy = y - iy

    # Four corners
    a = hash12(ix, iy, seed)
    b = hash12(ix + 1.0, iy, seed)
    c = hash12(ix, iy + 1.0, seed)
    d = hash12(ix + 1.0, iy + 1.0, seed)

    # Smooth interpolation
    ux = smoothstep(fx)
    uy = smoothstep(fy)

    # Mix
    return mix(mix(a, b, ux), mix(c, d, ux), uy)

@cuda.jit
def generate_terrain_kernel(terrain, nx, ny, scale, octaves, persistence, lacunarity, seed):
    """
    Generates FBM terrain directly on GPU.
    """
    x, y = cuda.grid(2)
    
    if x < nx and y < ny:
        amplitude = 1.0
        frequency = 1.0
        max_val = 0.0
        total = 0.0
        
        # FBM Loop
        for i in range(octaves):
            sx = (x + seed) / scale * frequency
            sy = (y + seed) / scale * frequency
            
            val = noise_2d_gpu(sx, sy, seed)
            total += val * amplitude
            max_val += amplitude
            
            amplitude *= persistence
            frequency *= lacunarity
            
        # Normalize and Curve
        if max_val > 0:
            norm = total / max_val
            terrain[x, y] = norm ** 1.5

# --- CPU HELPERS (Trees) ---

@njit(parallel=True)
def place_trees_cpu(fuel, terrain_z, nx, ny, nz, num_trees, seed):
    """
    Places trees on the CPU using Numba for speed.
    Parallelized with prange to handle large grid sizes (512+) efficiently.
    """
    np.random.seed(seed)
    
    # We loop over trees in parallel. 
    # Since write operations (assignments) to fuel are simple overrides, 
    # race conditions (two trees overlapping) are benign for generation.
    for i in prange(num_trees):
        # Re-seeding per thread or just using global random state in Numba
        # Numba handles thread-local RNG automatically in parallel loops
        
        tx = np.random.randint(0, nx)
        ty = np.random.randint(0, ny)
        z_base = terrain_z[tx, ty]
        
        # Don't place trees if ground is too high or out of bounds
        if z_base >= nz - 5: 
            continue
            
        tree_height = np.random.randint(8, 15)
        crown_base = np.random.randint(3, 6)
        crown_radius = np.random.randint(2, 4)
        
        top = min(z_base + tree_height, nz - 1)
        trunk_top = min(z_base + crown_base, top)
        
        # 1. Trunk
        if trunk_top > z_base:
            for z in range(z_base, trunk_top):
                fuel[z, tx, ty] = 2.0
        
        # 2. Crown
        center_z = (trunk_top + top) / 2
        
        z_start = trunk_top
        z_end = top
        
        for z in range(z_start, z_end):
            for dx in range(-crown_radius, crown_radius + 1):
                for dy in range(-crown_radius, crown_radius + 1):
                    dist_sq = dx*dx + dy*dy + (z - center_z)**2
                    if dist_sq < crown_radius**2:
                        cx = tx + dx
                        cy = ty + dy
                        
                        if 0 <= cx < nx and 0 <= cy < ny:
                            fuel[z, cx, cy] = 0.8 + (np.random.random() * 0.4)

def generate_world(nx, ny, nz, scale=60.0, max_height=12.0, flat=False):
    """
    Generates terrain and fuel.
    """

    seed = np.random.randint(0, 10000)

    # 1. Terrain
    # ----------
    if flat:
        # Fully flat ground at z = 0
        terrain_z = np.zeros((nx, ny), dtype=np.int32)

    else:
        threadsperblock = (8, 8)
        blockspergrid_x = (nx + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (ny + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        terrain_dev = cuda.device_array((nx, ny), dtype=np.float32)

        generate_terrain_kernel[blockspergrid, threadsperblock](
            terrain_dev, nx, ny, scale, 3, 0.5, 2.0, float(seed)
        )

        norm_terrain = terrain_dev.copy_to_host()
        del terrain_dev

        terrain_z = (norm_terrain * max_height).astype(np.int32)
        terrain_z = np.clip(terrain_z, 0, nz - 1)

    # 2. Fuel
    # -------
    fuel = np.zeros((nz, nx, ny), dtype=np.float32)

    grass_density = np.random.uniform(0.5, 0.8, (nx, ny)).astype(np.float32)
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')

    valid_mask = terrain_z < nz
    z_indices = terrain_z[valid_mask]
    x_indices = X[valid_mask]
    y_indices = Y[valid_mask]
    densities = grass_density[valid_mask]

    fuel[z_indices, x_indices, y_indices] = densities

    # 3. Trees
    # --------
    base_density = 35 / (128 * 128)
    total_cells = nx * ny
    calc_num_trees = int(total_cells * base_density)
    num_trees = max(20, calc_num_trees)

    place_trees_cpu(fuel, terrain_z, nx, ny, nz, num_trees, seed)

    return fuel, terrain_z.astype(np.float32)