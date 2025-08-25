'''
 Model of Backi et al. 2018
 Modifications:
 2 phase: l-l
 no mixer - assumed droplet distribution
 physical properties from Henschke 1995
 MiBK in water
 
 constants to be imported into model_backi.py
'''
import numpy as np
# import physical properties
import utils.Properties.properties_Butylacetate as prop

# number of segments
N_S = 200 # -
# number of droplet classes
N_D = 10 # -
EPS = 1e-12 # epsilon for numerical stability
POS_IN = np.zeros(N_D) # position of droplet at inlet in vertical direction m

# Geometry data
R = 0.1 # radius of separator m
L = 1.8 # length of separator m
D_STIRRER = 0.1 # diameter of stirrer m
# parameter definition
G = 9.81 # gravity constant m/s^2

# property data from properties file
RHO_O= prop.RHO_O # density of organic phase kg/m3
RHO_W = prop.RHO_W # density of water phase kg/m3
ETA_O = prop.ETA_O # viscosity of organic phase Pa*s
ETA_W = prop.ETA_W # viscosity of water phase Pa*s
DELTA_RHO = prop.DELTA_RHO # density difference in kg/m3
SIGMA = prop.SIGMA # interfacial tension N/m
R_V = prop.R_V # asymetric film drainage parameter

# R_IG = 8.314 # ideal gas constant J/mol*K
# RHO_G = 1.2 # density of gas kg/m3
# M_G = 28.97e-3 # molar mass of gas kg/mol

## Henschke
HA = 1e-20 # Hamaker constant J
EPSILON_DI = 1 # holdup at interface -
EPSILON_DP = 0.9 # holdup in dense-packed zone -