#%%
import numpy as np 
from thermo import Chemical
from StreamData import sn_ls, fn_ls, P0
import matplotlib.pyplot as plt
from Rates import EB_rates , rate_ETBE_Thy_act, rate_TBA_Honk, rate_TBA_Umar, rate_diB_Honk, rate_TriB_Honk
#%%
def Cpi(T, P, f_ls = fn_ls, s_ls = sn_ls): ## takes T in K, P in kPa, returns J/mol/K
    P = P*1000
    Cp_ls = [Chemical(f, T, P).Cpm for f in fn_ls ]
    return Cp_ls

def Hf(T, P, f_ls = fn_ls, s_ls = sn_ls, update =False): ## takes T in K, P in kPa, returns J/mol
    if update:
        P = P*1000
        Hf_ls = [Chemical(f, T, P).Hf for f in fn_ls ]
        return Hf_ls
    else:
        return [-134990.0, -16900.0, None, 110300.0, -125650.0, -11170.0, -6990.0, -241820.0, -234440.0, -312410.0, None, -104900.0, None]

#%%
def dCprx(T, P):
    _e, Cp_IB, _, _, _, _, _, Cp_water, Cp_EtOH, Cp_TBA, Cp_ETBE, Cp_di_IB, Cp_tri_IB = Cp_ls = Cpi(T, P)
    dCprx1 = Cp_ETBE - Cp_EtOH - Cp_IB
    dCprx2 = Cp_IB + Cp_water - Cp_TBA
    dCprx3 = Cp_di_IB - 2*Cp_IB
    dCprx4 = Cp_tri_IB - Cp_di_IB - Cp_IB
    dCprx5 = Cp_ETBE + Cp_water - Cp_TBA - Cp_EtOH
    return [dCprx1, dCprx2, dCprx3, dCprx4, dCprx5]

def dHrx(T, P): ## takes T in K, P in kPa, returns J/mol rx
    Hf_ls = Hf(T, P) # standard heats of formation
    dCprx_ls = dCprx(T, P) # change in Cp of reaction j at temp T (K) and P (kPa)
    ## standard heats of reactions ####
    dHrx1_0 = -44.3e3 #J/mol
    dHrx2_0 = Hf_ls[7] + Hf_ls[1] - Hf_ls[9] #J/mol
    dHrx3_0 = Hf_ls[11] - 2*Hf_ls[1] #J/mol
    dHrx4_0 = -192.87e3 - Hf_ls[11] - Hf_ls[1] #J/mol
    dHrx5_0 = -316.5e3 + Hf_ls[7] - Hf_ls[9] - Hf_ls[8] #J/mol
    Tr = 25 + 273.15
    dHrx1 = dHrx1_0 +  dCprx_ls[0]*(T - Tr)
    dHrx2 = dHrx2_0 +  dCprx_ls[1]*(T - Tr)
    dHrx3 = dHrx3_0 +  dCprx_ls[2]*(T - Tr)
    dHrx4 = dHrx4_0 +  dCprx_ls[3]*(T - Tr)
    dHrx5 = dHrx5_0 +  dCprx_ls[4]*(T - Tr)
    return [dHrx1, dHrx2, dHrx3, dHrx4, dHrx5]  

def EB_ada(T, P, Q, arr):
    
    r_ETBE_t = rate_ETBE_Thy_act(T, P, Q, arr)
    r_TBA_1 = rate_TBA_Honk(T, P, Q, arr)
    r_TBA_2 = rate_TBA_Umar(T, P, Q, arr)
    r_di_IB_t = rate_diB_Honk(T, P, Q, arr)
    r_tri_IB_t = rate_TriB_Honk(T, P, Q, arr)

    dHrx_ls = dHrx(T, P)
    Hrx1, Hrx2, Hrx3, Hrx4, Hrx5 = [r*dH for r, dH in zip([r_ETBE_t, -r_TBA_1, r_di_IB_t/2, r_tri_IB_t,  -r_TBA_2], dHrx_ls)]
    Cp_ls = Cpi(T, P)
    sum_FiCpi = sum([Fi*Cpi for Fi, Cpi in zip(arr, Cp_ls)])
    print('-------------')
    print(sum([Hrx1, Hrx2, Hrx3, Hrx4, Hrx5 ]))
    print(sum_FiCpi)
    return (0 - Hrx1 - Hrx2 - Hrx3 - Hrx4 - Hrx5)/sum_FiCpi
# %%

def phase_checker(Tspan, Pspan):
    return [Cpi(T, P) for T, P in zip(Tspan, Pspan)]
# %%
