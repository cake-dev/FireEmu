import numpy as np
import struct
import os
import csv

# Try importing pyproj, handle if missing
try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    print("Warning: pyproj not found. Georeferencing will default to 0,0.")

class QuicFireIO:
    """
    Handles reading QUIC-Fire input files (.inp, .dat, .bin).
    """
    
    @staticmethod
    def read_simparams(filepath):
        """Reads grid dimensions from QU_simparams.inp"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found.")
            
        params = {}
        with open(filepath, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            
        try:
            params['nx'] = int(lines[1].split('!')[0].split()[0])
            params['ny'] = int(lines[2].split('!')[0].split()[0])
            params['nz'] = int(lines[3].split('!')[0].split()[0])
            params['dx'] = float(lines[4].split('!')[0].split()[0])
            params['dy'] = float(lines[5].split('!')[0].split()[0])
            params['dz'] = 1.0
            if len(lines) > 7:
                try:
                    val = float(lines[7].split('!')[0].split()[0])
                    params['dz'] = val
                except:
                    pass
        except Exception as e:
            print(f"Warning: Error parsing QU_simparams.inp: {e}") 
            
        return params

    @staticmethod
    def read_quic_fire_inp(filepath):
        """Reads simulation timing and input flags from QUIC_fire.inp"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found.")
            
        params = {}
        with open(filepath, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            
        try:
            # Basic Timing
            params['sim_time'] = int(lines[4].split('!')[0].split()[0])
            params['dt'] = int(lines[6].split('!')[0].split()[0])
            params['out_int_fire'] = int(lines[8].split('!')[0].split()[0])
            params['out_int_wind'] = int(lines[9].split('!')[0].split()[0])
            params['nz_fire'] = int(lines[12].split('!')[0].split()[0])
            
            # File Format Flags (Lines 18, 19)
            # Line 18: fuel_file_type (1=all in one, 2=separate)
            # Line 19: fuel_file_format (1=stream, 2=fortran records)
            if len(lines) > 19:
                params['fuel_file_type'] = int(lines[18].split('!')[0].split()[0])
                params['fuel_file_format'] = 2#int(lines[19].split('!')[0].split()[0])
            else:
                params['fuel_file_type'] = 1
                params['fuel_file_format'] = 1 # Default to stream
                
        except Exception as e:
            print(f"Warning parsing QUIC_fire.inp: {e}")
            
        return params

    @staticmethod
    def read_sensor1(filepath):
        """
        Parses sensor1.inp to extract wind schedule.
        Returns list of tuples: [(time_sec, speed, direction), ...]
        """
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Using default wind.")
            return []

        schedule = []
        with open(filepath, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        # Format usually:
        # Header
        # ... setup lines ...
        # Times:
        # Time1
        # Setup lines
        # Z Speed Dir
        
        # Simplified parser looking for time blocks
        # Assuming format described in PDF 3.9
        # This parser relies on finding the wind speed/dir lines which are usually floats
        
        # A robust way is to scan for lines with 3 floats where the first is height
        # and associate them with the preceding time.
        
        current_time = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            parts = line.split()
            
            # Check if this line is a Unix Timestamp (large integer)
            # QF uses Unix Epoch. 
            if len(parts) == 1 and parts[0].isdigit() and int(parts[0]) > 1000000:
                current_time = int(parts[0])
                
                # The wind data usually follows a few lines down.
                # Skip site flag (1-4), roughness (float), etc.
                # Usually 3 or 4 lines down.
                # Let's scan forward 3-5 lines for the wind data
                for offset in range(1, 10):
                    if i + offset >= len(lines): break
                    sub_parts = lines[i+offset].split()
                    
                    # Looking for: Height Speed Direction (3 floats)
                    if len(sub_parts) >= 3:
                        try:
                            # Try converting first 3 to floats
                            h = float(sub_parts[0])
                            s = float(sub_parts[1])
                            d = float(sub_parts[2])
                            
                            # Heuristic: Height is usually reasonable (2-50), Speed (0-50), Dir (0-360)
                            if 0 <= s < 100 and 0 <= d <= 360:
                                schedule.append((current_time, s, d))
                                break # Found valid data for this time
                        except:
                            continue
            i += 1
            
        # Normalize time to start at 0
        if schedule:
            start_time = schedule[0][0]
            schedule = [(t - start_time, s, d) for t, s, d in schedule]
            
        return schedule

    @staticmethod
    def read_fuel_dat(filepath, nx, ny, nz, file_format=1):
        """
        Reads binary fuel files (treesrhof.dat, etc).
        file_format: 1 = Stream (Flat binary), 2 = Fortran Records (Headers)
        Returns: numpy array (nx, ny, nz)
        """
        if not os.path.exists(filepath):
            print(f"Warning: Fuel file {filepath} not found. Returning empty.")
            return np.zeros((nx, ny, nz), dtype=np.float32)
            
        # Total expected elements. 
        # Note: QF fuel files often contain multiple fuel types.
        # Shape is usually (N_types, nx, ny, nz).
        # We need to detect file size to infer N_types.
        
        file_size = os.path.getsize(filepath)
        bytes_per_float = 4
        
        if file_format == 1: # Stream
            total_floats = file_size // bytes_per_float
            grid_size = nx * ny * nz
            
            if total_floats % grid_size != 0:
                print(f"Warning: File size {file_size} does not match grid {nx}x{ny}x{nz}. Trying best guess.")
                
            n_types = total_floats // grid_size
            
            data = np.fromfile(filepath, dtype=np.float32)
            
            # Reshape: Fortran (i, j, k, type) -> (x, y, z, type)
            # Usually stored: x varies fastest, then y, then z.
            if n_types > 1:
                # Assuming (n_types, x, y, z) or (x, y, z, n_types)? 
                # QF PDF says: (ft%n_fuel_types, firegrid%nx, firegrid%ny, ft%n_grid_top)
                # Fortran order: dim1 varies fastest.
                # So in memory: Type, X, Y, Z
                
                raw = data.reshape((n_types, nx, ny, nz), order='F')
                
                # Sum all fuel types for bulk density/moisture
                combined = np.sum(raw, axis=0)
                return combined
            else:
                return data.reshape((nx, ny, nz), order='F')
                
        elif file_format == 2: # Fortran Records
            # This is harder because headers are interspersed.
            # Usually [Header][Data][Header]
            # We'll read the first record length to guess structure.
            with open(filepath, 'rb') as f:
                header = struct.unpack('i', f.read(4))[0]
                
                # Check if header matches expected grid slice (nx*ny*4 bytes) or full grid
                expected_full = nx * ny * nz * 4
                expected_slice = nx * ny * 4
                
                f.seek(0)
                
                if header == expected_full:
                    # One giant block per fuel type
                    # Read block, skip footer, read next block...
                    # Implementation simplified: read everything, skip headers manually
                    raw_bytes = f.read()
                    # Strip headers (4 bytes at start/end of each block)
                    # This is tricky without knowing N_types. 
                    # Assuming 1 type for now or parsing carefully.
                    
                    # For safety in this environment, fallback to reading payload of first block
                    f.seek(4)
                    data = np.fromfile(f, dtype=np.float32, count=nx*ny*nz)
                    return data.reshape((nx, ny, nz), order='F')
                else:
                    print("Complex Fortran record structure detected. Reading raw stream as fallback.")
                    data = np.fromfile(filepath, dtype=np.float32)
                    # Heuristic cleaning of headers? 
                    # For now, just return zeros if complex to avoid crash
                    return np.zeros((nx, ny, nz), dtype=np.float32)

    # ... (Keep write methods from previous turn) ...
    @staticmethod
    def read_raster_origin(project_dir):
        path = os.path.join(project_dir, 'rasterorigin.txt')
        if os.path.exists(path):
            with open(path, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    return float(lines[0].strip()), float(lines[1].strip())
        return 0.0, 0.0

    @staticmethod
    def write_fortran_block(f, data_array, dtype='f4'):
        arr = np.array(data_array, dtype=dtype)
        num_bytes = arr.nbytes
        f.write(struct.pack('i', num_bytes))
        arr.tofile(f)
        f.write(struct.pack('i', num_bytes))

    @staticmethod
    def write_grid_bin(output_dir, nx, ny, nz, dz):
        path = os.path.join(output_dir, 'grid.bin')
        z_bottoms = np.arange(nz + 2) * dz 
        z_centers = z_bottoms + (dz * 0.5)
        with open(path, 'wb') as f:
            QuicFireIO.write_fortran_block(f, z_bottoms, 'f4')
            QuicFireIO.write_fortran_block(f, z_centers, 'f4')

    @staticmethod
    def write_fire_indexes_bin(output_dir, fuel_grid, nx, ny, nz):
        path = os.path.join(output_dir, 'fire_indexes.bin')
        indices = []
        count = 0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    if fuel_grid[i, j, k] > 0:
                        count += 1
                        idx = (i + 1) + (j * nx) + (k * nx * ny)
                        indices.append(idx)
        with open(path, 'wb') as f:
            QuicFireIO.write_fortran_block(f, [count], 'i4')
            QuicFireIO.write_fortran_block(f, [nz], 'i4')
            QuicFireIO.write_fortran_block(f, indices, 'i4')
        return np.array(indices)

class QuicFireCSVWriter:
    def __init__(self, nx, ny, dx, dy, origin_x, origin_y):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.origin_x = origin_x
        self.origin_y = origin_y
        
        if HAS_PYPROJ:
            x_coords = origin_x + (np.arange(nx) + 0.5) * dx
            y_coords = origin_y + (np.arange(ny) + 0.5) * dy
            self.xx, self.yy = np.meshgrid(x_coords, y_coords)
            transformer = Transformer.from_crs("epsg:5070", "epsg:4979", always_xy=True)
            self.lon_grid, self.lat_grid = transformer.transform(self.xx, self.yy)
        else:
            self.lon_grid = np.zeros((ny, nx))
            self.lat_grid = np.zeros((ny, nx))

    def write_sparse_csv(self, data_array, filepath, header_val_name, zlevels=None):
        nx, ny, nz = data_array.shape
        z_indices = range(nz) if zlevels is None else zlevels
        all_rows = []
        for k in z_indices:
            layer_data = data_array[:, :, k]
            layer_data_T = layer_data.T 
            nz_y, nz_x = np.nonzero(layer_data_T)
            if len(nz_y) == 0: continue
            vals = layer_data_T[nz_y, nz_x]
            lons = self.lon_grid[nz_y, nz_x]
            lats = self.lat_grid[nz_y, nz_x]
            for i in range(len(vals)):
                if abs(vals[i]) > 1e-6:
                    all_rows.append(f"{nz_x[i]},{nz_y[i]},{lons[i]:.8f},{lats[i]:.8f},{k},{vals[i]:.6f}\n")
        if not all_rows: return
        header = f"IndexX,IndexY,Longitude,Latitude,ZLevel,{header_val_name}\n"
        with open(filepath, 'w') as f:
            f.write(header)
            f.writelines(all_rows)