# Grid Dimensions
NX = 128
NY = 128
NZ = 32
DX = 2.0  # meters
DY = 2.0  # meters
DZ = 1.0  # meters

# Time
DT = 0.5 # seconds
RUN_SECONDS = 256
TOTAL_TIME = RUN_SECONDS
SAVE_INTERVAL = 1

# Physics Constants
G = 9.81  # m/s^2
RHO_AIR = 1.225  # kg/m^3
CP_AIR = 1005.0  # J/(kg*K)
T_AMBIENT = 300.0  # K

# Fuel Properties
CM = 1.0
H_WOOD = 18.62e6 
CP_WOOD = 1700.0 
T_CRIT = 500.0 
N_F = 0.4552
N_O2 = 0.5448

EEP = 50000.0 # Watts (J/s)
C_RAD_LOSS = 0.2
T_BURNOUT = 30.0 

# --- WATER / MOISTURE PHYSICS ---
# Latent heat of vaporization for water (J/kg)
L_V = 2.26e6 
# Specific heat of water (J/kg*K)
CP_WATER = 4186.0
# Boiling point (K)
T_BOIL = 373.15

# Total energy required to evaporate 1kg of water from T_AMBIENT
# Heat to boiling + Phase change
H_H2O_EFF = (CP_WATER * (T_BOIL - T_AMBIENT)) + L_V

# Wind Solver Constants
K_VON_KARMAN = 0.4
Z0 = 0.1