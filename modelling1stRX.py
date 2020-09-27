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
def PBR(V, arr, L, LD):
    T, P, Q = arr[:3]
    Fls = arr[3:]
    rhob = 610
    D = L/LD
    Ac = (np.pi*D**2)/4
    r_IB_ane, r_IB, r_1B, r_B_diene, r_NB_ane, r_trans_B, r_cis_B, r_water, r_EtOH, r_TBA, r_ETBE, r_di_IB, r_tri_IB = rate_ls = [rhob*r*Ac for r in RATE(T, P, Q, Fls, L, D)]
    dT = EB_ada(T, P, Q, Fls)
    dQ = 0
    dP = Ergun(T, P, Q, Fls, LD, L) #kPa/m3_rx
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
W1 = 7423 #kg
Vtot = W1/rhob
LD = 500
D = (4*Vtot/(LD*np.pi))**(1/3)
L = LD*D
print('Catalyst mass = ' + str(np.round(Wtot, 3)) + ' kg')
print('reactor volume = ' + str(np.round(Vtot, 3)) + ' m3')
print('reactor length = ' + str(np.round(L, 3)) + ' m')
print('reactor diameter = ' + str(np.round(D, 3)) + ' m')
print('L/D = ' + str(np.round(LD, 3)))

Lspan = np.linspace(0, L, 100)
T1 = 363
Pt0, Qt0 = [v['value'] for v in [P0, Q0]]
y0 = [T1, 2000, Qt0, *F0]
ans = solve_ivp(lambda V, arr: PBR(V, arr, L, LD), [0, L], y0, dense_output = True).sol(Lspan)
F_span_out = ans.T[-1]

def plot():
    cols = cm.rainbow(np.linspace(0, 1, len(ans[3:])))
    fig, (ax1, ax2) = plt.subplots(2,1, sharex = True)
    for n, Fls, col in zip(fn_ls, ans[3:], cols):
        ax1.plot(Lspan, Fls, label = n, color = col)
    ax1.legend(loc = 'best')
    ax2.plot(Lspan, ans[0])
    axP = ax2.twinx()
    axP.plot(Lspan, ans[1])
    plt.show()
plot()
