#%%
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Rates import RATE, rate_TBA_Honk
from StreamData import CF_dict, mt0, P0, Ft0, Q0, sn_ls, fn_ls, F0
#%%
F_IB_ane, F_IB, F_1B, F_B_diene, F_NB_ane, F_trans_B, F_cis_B, F_water, F_EtOH, F_TBA, F_ETBE, F_di_IB, F_tri_IB = F0
#%%
T0, P0, Q0 = [343, 1600, 15] # K, kPa, m3/h

# %%
def PBR(W, arr):
    "T, P, Q, IB_ane', 'IB', '1B', 'B_diene', 'NB_ane', 'trans_B', 'cis_B', 'water', 'EtOH', 'TBA', 'ETBE', 'di_IB', 'tri_IB"
    T, P, Q = arr[:3]
    Fls = arr[3:]
    dT, dP, dQ = 0,0,0
    r_IB_ane, r_IB, r_1B, r_B_diene, r_NB_ane, r_trans_B, r_cis_B, r_water, r_EtOH, r_TBA, r_ETBE, r_di_IB, r_tri_IB  = RATE(T, P, Q, Fls)
    return [dT, dP, dQ, r_IB_ane, r_IB, r_1B, r_B_diene, r_NB_ane, r_trans_B, r_cis_B, r_water, r_EtOH, r_TBA, r_ETBE, r_di_IB, r_tri_IB]
#%%
Wtot = 36000
Wspan = np.linspace(0.001, Wtot, 100)
y0 = [T0, P0, Q0, *F0]
ans = solve_ivp(PBR, [0, Wtot], y0, dense_output=True).sol(Wspan)
for n, Fls in zip(sn_ls, ans[3:]):
    plt.plot(Wspan, Fls, label = n)
plt.legend(loc = 'best')
plt.show()


# %%
for n, i in zip(sn_ls, PBR(1, y0)[3:]):
    print('r ' + n + ' = ', i)
# %%
rate_TBA_Honk(T0, P0, Q0, F0)
# %%

print(F0)
# %%
