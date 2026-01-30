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
    def read_topo_inputs(filepath):
        """
        Parses QU_TopoInputs.inp to determine terrain type and filename.
        Returns a dict with 'filename', 'topo_flag', etc.
        """
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Defaulting to flat.")
            return {'topo_flag': 0, 'filename': ''}

        params = {}
        with open(filepath, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        try:
            # Line 0: Header (skip)
            # Line 1: Filename (quoted)
            # Remove quotes if present
            raw_fname = lines[1].split('!')[0].strip()
            params['filename'] = raw_fname.replace('"', '').replace("'", "")
            
            # Line 2: Topo flag
            # Format: 5 ! Comment
            params['topo_flag'] = int(lines[2].split('!')[0].strip())
            
        except Exception as e:
            print(f"Error parsing QU_TopoInputs.inp: {e}")
            return {'topo_flag': 0, 'filename': ''}
            
        return params

    @staticmethod
    def read_topo_dat(filepath, nx, ny):
        """
        Reads binary topography file (e.g. usgs_dem.dat).
        Handles both flat float32 binary and Fortran-record binary (8 byte overhead).
        Returns numpy array (nx, ny).
        """
        if not os.path.exists(filepath):
            print(f"Warning: Topo file {filepath} not found. Returning flat.")
            return np.zeros((nx, ny), dtype=np.float32)
            
        try:
            file_size = os.path.getsize(filepath)
            expected_bytes = nx * ny * 4
            data = None
            
            # 1. Check for Fortran Record (4 byte header + data + 4 byte footer)
            if file_size == expected_bytes + 8:
                try:
                    with open(filepath, 'rb') as f:
                        header = struct.unpack('i', f.read(4))[0]
                        if header == expected_bytes:
                            print(f"  Detected Fortran record format for {os.path.basename(filepath)}")
                            data = np.fromfile(f, dtype=np.float32, count=nx*ny)
                        else:
                            # Not a valid header, reset
                            f.seek(0)
                except:
                    pass

            # 2. Check for Flat Binary
            if data is None:
                if file_size != expected_bytes:
                    print(f"Warning: Topo file size {file_size} bytes does not match grid {nx}x{ny} ({expected_bytes} bytes). Attempting read...")
                
                data = np.fromfile(filepath, dtype=np.float32)
            
            # 3. Validation and Resizing
            if data.size != nx * ny:
                print(f"Resizing topo data from {data.size} to {nx*ny}")
                if data.size > nx * ny:
                    data = data[:nx*ny]
                else:
                    data = np.pad(data, (0, nx*ny - data.size), 'constant')
            
            # Reshape: Fortran order (x varies fastest, then y)
            return data.reshape((nx, ny), order='F')
            
        except Exception as e:
            print(f"Error reading topo file: {e}")
            return np.zeros((nx, ny), dtype=np.float32)

    @staticmethod
    def read_ignite_dat(filepath):
        """
        Parses ignite.dat. Supports ATV ignition (Type 5).
        Returns a dict: {'type': int, 'lines': []}
        """
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Returning None.")
            return None
            
        ignition_data = {'type': 0, 'lines': []}
        
        with open(filepath, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        try:
            # Basic parsing state
            reading_atv = False
            
            for line in lines:
                # Check for ignition type
                if "igntype=" in line.lower():
                    # Handle spaces like igntype= 5 or igntype=5
                    val = line.lower().split("igntype=")[1].split()[0]
                    ignition_data['type'] = int(val)
                
                # Check for list start
                if "&atvlist" in line.lower():
                    reading_atv = True
                    continue
                
                # Check for list end
                if line.startswith("/"):
                    reading_atv = True # Ensure we keep reading if data follows /
                    continue
                    
                # Read data lines (ATV)
                # Format: x_start y_start x_end y_end t_start t_end
                if reading_atv:
                    parts = line.split()
                    # Filter out purely text/header lines (like natv=, targettemp=)
                    if "=" in line:
                        continue
                        
                    # ATV data lines usually have 6 floats
                    if len(parts) >= 6:
                        try:
                            # Verify they are numbers
                            vals = [float(p) for p in parts[:6]]
                            ignition_data['lines'].append({
                                'x_start': vals[0],
                                'y_start': vals[1],
                                'x_end': vals[2],
                                'y_end': vals[3],
                                't_start': vals[4],
                                't_end': vals[5]
                            })
                        except ValueError:
                            # Not a data line
                            continue
                            
            return ignition_data

        except Exception as e:
            print(f"Error parsing ignite.dat: {e}")
            return None

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

        # print(lines)
        
        # just grab the last line, looks like '6, 3.0, 225' where 1st is height, second speed, third dir
        if len(lines) > 0:
            last_line = lines[-1]
            parts = last_line.replace(',', ' ').split()
            if len(parts) >= 3:
                try:
                    s = float(parts[1])
                    d = float(parts[2])
                    schedule.append((0, s, d)) # time 0
                except:
                    pass
            
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
            
        file_size = os.path.getsize(filepath)
        bytes_per_float = 4
        
        if file_format == 1: # Stream
            total_floats = file_size // bytes_per_float
            grid_size = nx * ny * nz
            n_types = total_floats // grid_size
            data = np.fromfile(filepath, dtype=np.float32)
            
            if n_types > 1:
                # Fortran order: dim1 varies fastest -> Type, X, Y, Z
                raw = data.reshape((n_types, nx, ny, nz), order='F')
                combined = np.sum(raw, axis=0)
                return combined
            else:
                return data.reshape((nx, ny, nz), order='F')
                
        elif file_format == 2: # Fortran Records
            with open(filepath, 'rb') as f:
                header = struct.unpack('i', f.read(4))[0]
                expected_full = nx * ny * nz * 4
                
                if header == expected_full:
                    f.seek(4)
                    data = np.fromfile(f, dtype=np.float32, count=nx*ny*nz)
                    return data.reshape((nx, ny, nz), order='F')
                else:
                    print("Complex Fortran record structure detected. Reading raw stream as fallback.")
                    data = np.fromfile(filepath, dtype=np.float32)
                    return np.zeros((nx, ny, nz), dtype=np.float32)

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