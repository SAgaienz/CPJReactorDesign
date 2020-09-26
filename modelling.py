#%%
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
from Rates import RATE, EB_rates
from EnergyBalance import EB_ada, phase_checker
from StreamData import mt0, Ft0, Q0, P0, sn_ls, fn_ls, F0
from matplotlib import cm
from scipy.optimize import fsolve
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
Wtot = 1600000 #kg
Vspan = np.linspace(0, Wtot/rhob, 100)
T0 = 80+273.15
Pt0, Qt0 = [v['value'] for v in [P0, Q0]]
y0 = [T0,Pt0, Qt0, *F0]
ans = solve_ivp(PBR, [0, Wtot/rhob], y0, dense_output = True).sol(Vspan)

cols = cm.rainbow(np.linspace(0, 1, len(ans[3:])))
fig, (ax1, ax2) = plt.subplots(2,1, sharex = True)
for n, Fls, col in zip(fn_ls, ans[3:], cols):
    ax1.plot(Vspan, Fls, label = n, color = col)
ax1.legend(loc = 'best')
ax2.plot(Vspan, ans[0])
plt.show()
# %%
def phase_checker_plot():
    for n, ls, col in zip(fn_ls, np.array(phase_checker(ans[0], ans[1])).T, cols):
        plt.plot(Vspan, ls , label = n, color = col)
    plt.legend(loc = 'best')
    plt.show()
# phase_checker_plot()


# %%
def rate_vs_T_plot(Tspan = np.linspace(323.15, 400, np.shape(ans)[1])):
    rates =np.array([ EB_rates(T, y0[1], y0[2], y0[3:]) for T in Tspan]).T
    for n, r, col in zip(['ETBE', 'TBA dehy', 'TBA ether', 'di-IB', 'tri-IB'], rates, cols):
        plt.plot(Tspan, r, color = col, label = n)
    plt.legend(loc = 'best')
    plt.show()

rate_vs_T_plot()
# %%
# %%
np.shape(ans)[1]
# %%
def Equilibrium_Conditions(T, P, Q):
    def PBR(V, arr):
        T, P, Q = arr[:3]
        Fls = arr[3:]
        r_IB_ane, r_IB, r_1B, r_B_diene, r_NB_ane, r_trans_B, r_cis_B, r_water, r_EtOH, r_TBA, r_ETBE, r_di_IB, r_tri_IB = [rhob*r for r in RATE(T, P, Q, Fls)]
        # dT = EB_ada(T, P, Q, Fls)
        dT = 0
        dP, dQ = 0,0

        return [dT, dP, dQ, r_IB_ane, r_IB, r_1B, r_B_diene, r_NB_ane, r_trans_B, r_cis_B, r_water, r_EtOH, r_TBA, r_ETBE, r_di_IB, r_tri_IB]

    #%%
    rhob = 610
    Wtot = 1600000 #kg0
    # T0 = 80+273.15
    # Pt0, Qt0 = [v['value'] for v in [P0, Q0]]
    y0 = [T,P, Q, *F0]
    ans = solve_ivp(PBR, [0, Wtot/rhob], y0)['y']
        # ans = ans.T
    return ans.T[-1]
Fe_span = [Equilibrium_Conditions(T, 1600, Qt0) for T in np.linspace(323.15, 373.15, 10)]
# %%
for n, col, Fls in zip(fn_ls, cols, np.array(Fe_span).T[3:]):
    plt.plot(np.linspace(323.15, 373.15, 10), Fls, color = col, label = n)
plt.legend(loc = 'best')
plt.show()
# %%
