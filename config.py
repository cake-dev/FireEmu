# Grid Dimensions
NX = 512
NY = 512
NZ = 32
DX = 2.0  # meters (min 1 meter for stability)
DY = 2.0  # meters (min 1 meter for stability)
DZ = 1.0  # meters (vertical resolution can be stretched, but keeping uniform for v1)

# Time
DT = 1.0 # seconds
RUN_SECONDS = 1024  # Total simulation time in seconds
TOTAL_TIME = RUN_SECONDS# * DT
SAVE_INTERVAL = 1  # Save every N steps (can adjust higher to save space, but lose temporal res)

# Physics Constants
G = 9.81  # m/s^2
RHO_AIR = 1.225  # kg/m^3
CP_AIR = 1005.0  # J/(kg*K)
T_AMBIENT = 300.0  # K

# Fuel Properties (Standard Grass/Pine mix approximation)
# Reaction rate constant (Cm) - estimated
CM = 1.0 
# Heat of combustion (Hwood) - J/kg
H_WOOD = 18.62e6 
# Specific heat of wood
CP_WOOD = 1700.0 # J/(kg*K)
# Critical temperature for combustion
T_CRIT = 500.0 # K
# Stoichiometric coefficients (approximate for wood)
N_F = 0.4552
N_O2 = 0.5448
# Energy per Packet (EEP) - Watts (J/s) or Joules? Paper says "Energy per unit time per EP" = 50kW
EEP = 50000.0 # Watts
# Radiation loss fraction
C_RAD_LOSS = 0.2
# Burnout time (t_burnout)
T_BURNOUT = 30.0 # seconds

# Wind Solver Constants
# Von Karman constant
K_VON_KARMAN = 0.4
# Roughness length (z0)
Z0 = 0.1 # meters

# --- SLOPE PHYSICS ---
# Controls how much the terrain gradient pushes the fire uphill.
# A value of 0.0 means no slope effect (only wind).
# A value of 2.0 - 5.0 is typical to emulate "flame attachment".
SLOPE_FACTOR = 4.0 

MOD_DT = False
JUMP_HACK = False