#%%
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
from Rates import RATE
from EnergyBalance import EB_ada
from StreamData import mt0, Ft0, Q0, P0, sn_ls, fn_ls, F0
#%%
def PBR(V, arr):
    T, P, Q = arr[:3]
    Fls = arr[3:]
    r_IB_ane, r_IB, r_1B, r_B_diene, r_NB_ane, r_trans_B, r_cis_B, r_water, r_EtOH, r_TBA, r_ETBE, r_di_IB, r_tri_IB = [rhob*r for r in RATE(T, P, Q, Fls)]
    dT = EB_ada(T, P, Q, Fls)
    dP, dQ = 0,0

    return [dT, dP, dQ, r_IB_ane, r_IB, r_1B, r_B_diene, r_NB_ane, r_trans_B, r_cis_B, r_water, r_EtOH, r_TBA, r_ETBE, r_di_IB, r_tri_IB]

#%%
rhob = 610
Wtot = 50000 #kg
Vspan = np.linspace(0, Wtot/rhob, 100)
T0 = 80+273.15
Pt0, Qt0 = [v['value'] for v in [P0, Q0] ]
y0 = [T0,Pt0, Qt0, *F0]
ans = solve_ivp(PBR, [0, Wtot/rhob], y0, dense_output = True).sol(Vspan)

fig, (ax1, ax2) = plt.subplots(2,1, sharex = True)
for n, Fls in zip(fn_ls, ans[3:]):
    ax1.plot(Vspan, Fls, label = n)
ax1.legend(loc = 'best')
ax2.plot(Vspan, ans[0])
plt.show()

# %%



# %%
