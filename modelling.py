#%%
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
from Rates import RATE, EB_rates
from EnergyBalance import EB_ada
from StreamData import mt0, Ft0, Q0, P0, sn_ls, fn_ls, F0, phase_check
from PressureDrop import Ergun
from matplotlib import cm
from scipy.optimize import fsolve
#%%
def PBR(V, arr, Vtot, LD):
    T, P, Q = arr[:3]
    Fls = arr[3:]
    rhob = 610
    r_IB_ane, r_IB, r_1B, r_B_diene, r_NB_ane, r_trans_B, r_cis_B, r_water, r_EtOH, r_TBA, r_ETBE, r_di_IB, r_tri_IB = rate_ls = [rhob*r for r in RATE(T, P, Q, Fls)]
    dT = EB_ada(T, P, Q, Fls)
    dQ = 0
    L = LD*(4*Vtot/LD*np.pi)**(-3) # m
    dP = Ergun(T, P, Q, Fls, LD, L) #kPa/m3_rx
    # dP = 0
    if phase_check(T, P).count('g') != 0:
        print('------------ Vapour Phase in Reactor!! --------------')
        print('P = ' + str(np.round(P, 3)) + ' kPa')
        print('T = ' + str(np.round(T, 3)) + ' K')
        print('L/D = ' + str(LD))
        KeyboardInterrupt
    return [dT, dP, dQ, r_IB_ane, r_IB, r_1B, r_B_diene, r_NB_ane, r_trans_B, r_cis_B, r_water, r_EtOH, r_TBA, r_ETBE, r_di_IB, r_tri_IB]

#%%
rhob = 610
Wtot = 16000 #kg
Vtot = Wtot/rhob
print(Vtot)
Vspan = np.linspace(0, Vtot, 100)
T0 = 80+273.15
Pt0, Qt0 = [v['value'] for v in [P0, Q0]]
y0 = [T0, 3500, Qt0, *F0]
LD = 500
ans = solve_ivp(lambda V, arr: PBR(V, arr, Vtot, LD), [0, Wtot/rhob], y0, dense_output = True).sol(Vspan)

cols = cm.rainbow(np.linspace(0, 1, len(ans[3:])))
fig, (ax1, ax2) = plt.subplots(2,1, sharex = True)
for n, Fls, col in zip(fn_ls, ans[3:], cols):
    ax1.plot(Vspan, Fls, label = n, color = col)
ax1.legend(loc = 'best')
ax2.plot(Vspan, ans[0])
axP = ax2.twinx()
axP.plot(Vspan, ans[1])
plt.show()


# %%
