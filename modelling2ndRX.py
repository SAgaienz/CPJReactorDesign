#%%
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
from Rates import RATE, EB_rates
from EnergyBalance import EB
from StreamData import mt0, Ft0, Q0, P0, sn_ls, fn_ls, F0, phase_check
from PressureDrop import Ergun
from matplotlib import cm
from scipy.optimize import fsolve
from OptimumTemperature import EQ_plot
from modelling1stRX import Total_RX1_outlet_vals, TotalR1Values
#%%
def PBR(z, arr, LD, L, U, Tu, debug=False):
    T, P, Q = arr[:3]
    Fls = arr[3:]
    rhob = 610
    D = L/LD
    Ac = (np.pi*D**2)/4
    r_IB_ane, r_IB, r_1B, r_B_diene, r_NB_ane, r_trans_B, r_cis_B, r_water, r_EtOH, r_TBA, r_ETBE, r_di_IB, r_tri_IB = rate_ls = [rhob*r*Ac for r in RATE(T, P, Q, Fls, L, D)] # mol/s.m = mol/s.kg * kg/m3 * m2
    dT, Duty, RX_Heats = EB(T, P, Q, Fls, L, LD, U, Tu) # K/m, W/
    dQ = 0
    dP = Ergun(T, P, Q, Fls, LD, L) #kPa/m_rx
    if debug:
        debug_output(z, T, dT, P, dP, r_ETBE, Duty, RX_Heats)
    
    if phase_check(T, P).count('g') != 0:
        print('------------ Vapour Phase in Reactor!! --------------')
        print('P = ' + str(np.round(P, 3)) + ' kPa')
        print('T = ' + str(np.round(T, 3)) + ' K')
        print('L/D = ' + str(LD))
        print(fn_ls[phase_check(T, P).index('g')] + ' has vapourised.')
        quit()
    return [dT, dP, dQ, r_IB_ane, r_IB, r_1B, r_B_diene, r_NB_ane, r_trans_B, r_cis_B, r_water, r_EtOH, r_TBA, r_ETBE, r_di_IB, r_tri_IB]

def debug_output(z, T, dT, P, dP, r_ETBE, Duty, RX_Heats):
    print('z = ' + str(np.round(z, 3)) + ' | T = ' + str(np.round(T, 3)) + ' | dT = ' + str(np.round(dT, 3)) +' | P = ' + str(np.round(P, 3)) + ' | dP = ' + str(np.round(dP, 3)) +
    ' | r_ETBE = ' + str(np.round(r_ETBE, 3)) + ' | Duty = ' + str(np.round(Duty, 3)) + 
    ' | Hrx = ' + str(np.round(RX_Heats, 3)))

def per_tube_cond(T, P, Q, Fls, Nt):
    Q0 = Q/Nt
    Fls_0 = [F/Nt for F in Fls]
    return [T, P, Q0, *Fls_0]

rhob = 610
Wtot = 16000 #kg
def reactor_params(Nt, D, Wtot=Wtot, rhob=rhob):
    # total reactor values
    Vtot = Wtot/rhob
    Ac = (np.pi*D**2)/4
    Ltot = Vtot/Ac
    LD_tot = Ltot/D
    Tot_vals = [Wtot, Vtot, Ltot, D, LD_tot]
    # per tube values
    Lp = Ltot/Nt
    Vp = Lp*((np.pi*D**2)/4)
    Wp = Vp*rhob
    LDp = Lp/D
    per_tube_vals = [Wp, Vp, Lp, D, LDp]
    return [Tot_vals, per_tube_vals]
#%%
Nt = 500
D = 75e-3
T1, P1, Qt1, F1 = Total_RX1_outlet_vals
T2_0 = 50 + 273.15
P2_0 = 2000
Qt2_0 = Qt1
y0 = per_tube_cond(T2_0, P2_0, Qt2_0, F1, Nt)
U, Tu = 0, 55+273.15
[Wtot, Vtot, Ltot, D, LDtot], [Wp, Vp, Lp, D, LDp] = reactor_params(Nt, D)

def Reactor_conditions_output():
    print('-------------------- Total Reactor Parameters --------------------')
    print('Total Catalyst mass = ' + str(np.round(Wtot, 3)) + ' kg')
    print('Total reactor volume = ' + str(np.round(Vtot, 3)) + ' m3')
    print('Total reactor length = ' + str(np.round(Ltot, 3)) + ' m')
    print('reactor diameter = ' + str(np.round(D, 3)) + ' m')
    print('L/D tot = ' + str(np.round(LDtot, 3)))
    print('--------------- Individual Tube-Reactor Parameters ---------------')
    print('Tube Catalyst mass = ' + str(np.round(Wp, 3)) + ' kg')
    print('Tube reactor volume = ' + str(np.round(Vp, 3)) + ' m3')
    print('Tube reactor length = ' + str(np.round(Lp, 3)) + ' m')
    print('Tube L/D = ' + str(np.round(LDp, 3)))
Reactor_conditions_output()

Lspan = np.linspace(0, Lp, 1000)
debug=True
ans = solve_ivp(lambda V, arr: PBR(V, arr, LDp, Lp, U, Tu, debug), [0, Lp], y0, dense_output = True).sol(Lspan)
#%%
Tspan = ans[0]
Pspan = ans[1]
Qspan = ans[2]
Fspan = ans[3:]

Tt_ls = Tspan
Pt_ls = Pspan
Qt_ls = [Q*Nt for Q in Qspan]
Ft_ls = []
for Fls in Fspan:
    Fls_i = []
    for F in Fls:
        Fls_i.append(F*Nt)
    Ft_ls.append(Fls_i)

T1 = Tt_ls[-1]
P1 = Pt_ls[-1]
Q1 = Qt_ls[-1]
F1_ls = [Fls[-1] for Fls in Ft_ls]
TotalR2Values = [Tt_ls, Pt_ls, Qt_ls, Ft_ls]
Total_RX2_outlet_vals = [T1, P1, Q1, F1_ls]
print('Outlet Values:' + str(Total_RX2_outlet_vals))
#%%
Tspan_plot = [T - 273.15 for T in Tspan]
Pspan_plot = [P for P in Pspan] 
Fspan_plot = [F*3600 for F in  Fspan]

def plot():
    Pcol = 'k'
    Tcol = 'r'
    cols = cm.rainbow(np.linspace(0, 1, len(Fspan_plot)))
    fig, (ax1, axP) = plt.subplots(2,1, sharex = True)
    for n, Fls, col in zip(fn_ls, Fspan_plot, cols):
        ax1.plot(Lspan, Fls, label = n, color = col)
    ax1.legend(loc = 'best')
    ax1.set_ylabel('Molar flow (mol/h)')
    
    axP.plot(Lspan, Pspan_plot, color = Pcol)
    axP.set_xlabel('Reactor Length (m)')
    axP.tick_params(axis = 'y', labelcolor = Pcol)
    axP.set_ylabel('Pressure (kPa)', color = Pcol)

    axT = axP.twinx()
    axT.plot(Lspan, Tspan_plot, color = Tcol)
    axT.tick_params(axis='y', labelcolor = Tcol)
    axT.set_ylabel(r'Temperature ($^\circ C$)', color = Tcol)
    plt.show()
plot()

#%%
def TPlot():
    EQ_plot(False)
    plt.plot(TotalR1Values[0], TotalR1Values[3][10])
    plt.plot(TotalR2Values[0], TotalR2Values[3][10])
    plt.show()
TPlot()    
# %%
