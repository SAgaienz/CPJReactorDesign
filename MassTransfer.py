from PressureDrop import rho_m, mu_m_simple
from StreamData import Q0, F0
import numpy as np 

rho_c = 1.411e3
r0 = 122.4
e_cat = 60.2216e-2
rho_b = 0.64026e3
dp90 = 475e-6
dp10 = 873.2e-6
dp = np.mean([dp90, dp10])


############ Ext. Mass Transfer ############
# Thoenes-Kramer
D_AB = 10e-6 * 100e-2 # m2/s
def kc_TK(T, P, Q, Fls, D, D_AB=D_AB, e=e_cat, SF=1, dp=dp):
    Ac = np.pi*D**2/4
    U = Q/Ac
    rho = rho_m(T, P, Q, Fls)  # kg/m3
    mu = mu_m_simple(T, P, Q, Fls) # Pa.s (kg/m.s) 
    Re = (U*dp*rho/(mu*(1-e)*SF))
    Sc = (mu/(rho*D_AB))
    kc = (Re**0.5)*(Sc**(1/3))*((D_AB*SF*(1-e))/(e*dp))
    return kc

print(kc_TK(80+273.15, 2000, Q0['value'], F0, 0.314))