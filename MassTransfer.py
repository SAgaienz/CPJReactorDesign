from PressureDrop import rho_m, mu_m_simple, mu_i
from StreamData import Q0, F0
import numpy as np 

rho_c = 1.411e3
r0 = 122.4
e_cat = 60.2216e-2
rho_b = 0.64026e3
dp90 = 475e-6
dp10 = 873.2e-6
dp = np.mean([dp90, dp10])


De_EtOH = lambda T:  9.5551255e-8  * 100e-2 * np.exp(-43100/(8.314*T))
############ Ext. Mass Transfer ############
# Thoenes-Kramer
D_AB = 10e-6 * 100e-2 # m2/s
def kc_TK(T, P, Q, Fls, D, D_AB, e=e_cat, SF=1, dp=dp):
    Ac = np.pi*D**2/4
    U = Q/Ac  # m/s
    rho = rho_m(T, P, Q, Fls)  # kg/m3
    mu = mu_m_simple(T, P, Q, Fls) # Pa.s (kg/m.s) 
    Re = (U*dp*rho/(mu*(1-e)*SF))
    Sc = (mu/(rho*D_AB))
    kc = (Re**0.5)*(Sc**(1/3))*((D_AB*SF*(1-e))/(e*dp))
    return kc

def Tort_Dogu(e_cat=e_cat):
    return e_cat/(1 - np.pi*(((1-e_cat)*(3/(4*np.pi))**(2/3) )) )

def D_IB_C(T, P):
    De_IB_358 = 0.0018e-4  # m2/s
    mu2 = mu_i(T, P)[1]
    mu1 = mu_i(358, P)[1]
    tort = Tort_Dogu()
    sig = 0.8
    D_IB_cat_358 = De_IB_358*tort/(sig*e_cat)
    return D_IB_cat_358*(mu1/mu2)*(T/358)

kc_R1 = lambda T, P, Q, Fls, D: kc_TK(T, P, Q, Fls, 0.5, D_IB_C(T, P))
print(kc_R1(80+273.15, 2000, Q0['value'], F0, 0.314))
# print(D_IB_cat)