"""
config_qf.py - Configuration for QUIC-Fire GPU Simulation

Contains all physical constants and default parameters based on:
- Linn et al. (2020) QUIC-Fire paper
- Briggs (1984) plume theory
- Cionco (1965) canopy profiles
- Drysdale (2011) combustion properties
"""

# =============================================================================
# GRID DIMENSIONS (Defaults - typically overridden by input files)
# =============================================================================
NX = 128
NY = 128
NZ = 32
DX = 2.0  # meters
DY = 2.0  # meters
DZ = 1.0  # meters

# =============================================================================
# TIME STEPPING
# =============================================================================
DT = 0.5  # seconds
RUN_SECONDS = 256
TOTAL_TIME = RUN_SECONDS
SAVE_INTERVAL = 1

# =============================================================================
# ATMOSPHERIC CONSTANTS
# =============================================================================
G = 9.81           # Gravitational acceleration [m/s²]
RHO_AIR = 1.225    # Air density at sea level [kg/m³]
CP_AIR = 1005.0    # Specific heat of air [J/(kg·K)]
T_AMBIENT = 300.0  # Ambient temperature [K] (~27°C)

# =============================================================================
# FUEL PROPERTIES (Paper Section 2.4)
# =============================================================================

# Reaction rate constant (Eq. 1)
CM = 1.0

# Heat of combustion of wood [J/kg]
# From Drysdale (2011): 18.62 MJ/kg
H_WOOD = 18.62e6

# Specific heat of wood [J/(kg·K)]
CP_WOOD = 1700.0

# Critical ignition temperature [K]
T_CRIT = 500.0  # ~227°C

# Stoichiometry coefficients for wood combustion
# N_f(wood) + N_o2(oxygen) → products + heat
N_F = 0.4552
N_O2 = 0.5448

# Energy per EP packet [W] (Paper Section 2.4)
# "EEP is set to 50 kW as a balance between computational cost 
#  and representation of the energy transport"
EEP = 50000.0

# Radiative loss fraction (Paper Section 2.4)
# "We estimated Crad_loss to be 0.2 (20% of the net energy)"
C_RAD_LOSS = 0.2

# Burnout time [s] (Paper Section 2.4)
# "tburnout is the assumed time for fine fuel particles to burn out,
#  which is currently estimated to be 30 s"
T_BURNOUT = 30.0

# =============================================================================
# MOISTURE PHYSICS (Paper Section 2.7)
# =============================================================================

# Latent heat of vaporization for water [J/kg]
L_V = 2.26e6

# Specific heat of water [J/(kg·K)]
CP_WATER = 4186.0

# Boiling point [K]
T_BOIL = 373.15

# Effective energy to evaporate water from ambient [J/kg]
# Includes sensible heating (T_ambient to T_boil) + latent heat
H_H2O_EFF = (CP_WATER * (T_BOIL - T_AMBIENT)) + L_V

# Moisture of extinction (typical for wildland fuels)
MOISTURE_EXTINCTION = 0.30  # 30%

# =============================================================================
# WIND PROFILE CONSTANTS
# =============================================================================

# Von Kármán constant
K_VON_KARMAN = 0.4

# Surface roughness length [m]
# Typical values: grass 0.01-0.1, shrubs 0.1-0.5, forest 0.5-2.0
Z0 = 0.1

# Reference height for wind measurements [m]
Z_REF = 10.0

# =============================================================================
# CANOPY PARAMETERS (Cionco 1965)
# =============================================================================

# Default canopy height [m]
CANOPY_HEIGHT = 10.0

# Cionco attenuation coefficient range
ALPHA_CANOPY_MIN = 0.5  # Sparse canopy
ALPHA_CANOPY_MAX = 3.0  # Dense canopy

# Reference fuel density for attenuation scaling [kg/m³]
RHO_FUEL_REF = 2.0

# =============================================================================
# PLUME PARAMETERS (Briggs 1984)
# =============================================================================

# Entrainment coefficient
BETA_ENTRAINMENT = 0.5

# Maximum plume-induced updraft [m/s]
W_PLUME_MAX = 25.0

# Plume decay rate per vertical level
PLUME_DECAY_BASE = 0.92

# =============================================================================
# TERRAIN PARAMETERS
# =============================================================================

# Height scale for terrain influence decay [m]
TERRAIN_INFLUENCE_SCALE = 50.0

# =============================================================================
# EP TRANSPORT PARAMETERS (Paper Section 2.5)
# =============================================================================

# Fraction of EPs transported by wind-dominated process
WIND_EP_FRACTION = 0.7

# Creeping length scale [m]
L_CREEP = 2.0

# =============================================================================
# COMPUTATIONAL PARAMETERS
# =============================================================================

# Thread block sizes for GPU kernels
THREADS_PER_BLOCK_3D = (8, 8, 8)
THREADS_PER_BLOCK_2D = (8, 8)

# Small values for numerical stability
EPSILON = 1e-6
MIN_FUEL = 1e-4