#%%
import numpy as np 
from thermo import Chemical
from StreamData import sn_ls, fn_ls, P0
import matplotlib.pyplot as plt
P0 =  P0['value']
T0 = 80+273.15
#%%
def Cpi(T, P, f_ls = fn_ls, s_ls = sn_ls): ## takes T in K, P in kPa, returns J/mol/K
    P = P*1000
    Cp_ls = [Chemical(f, T, P).Cpm for f in fn_ls ]
    return Cp_ls

IB_ane, Cp_IB, Cp_1B, Cp_B_diene, Cp_NB_ane, Cp_trans_B, Cp_cis_B, Cp_water, Cp_EtOH, Cp_TBA, Cp_ETBE, Cp_di_IB, Cp_tri_IB = Cp_ls = Cpi(260, P0)
# %%
dCprx1 = Cp_ETBE - Cp_EtOH - Cp_IB
dCprx2 = Cp_IB + Cp_water - Cp_TBA
dCprx3 = Cp_di_IB - 2*Cp_IB
dCprx4 = Cp_tri_IB - Cp_di_IB - Cp_IB


# %%
