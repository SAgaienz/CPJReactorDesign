
#### Run this to simulate a PBR for the production of ETBE from IB and EtOH...
#%%\
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Rates import RATE, EB_rates
from EnergyBalance import EB
from StreamData import  Q0, P0, sn_ls, fn_ls, F0, phase_check
from PressureDrop import Ergun
from matplotlib import cm
from OptimumTemperature import EQ_plot
import pandas
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
Nt = 780
Fac = 1.5
D = 52.5e-3
Ac = (np.pi*D**2)/4
T1 = 70+273.15
Pt0, Qt0 = [v['value'] for v in [P0, Q0]]
Qt0 = Qt0*Fac
y0 = per_tube_cond(T1, 2200, Qt0, [F*Fac for F in F0], Nt)
U, Tu = 3.37, 35+273.15
[Wtot, Vtot, Ltot, D, LDtot], [Wp, Vp, Lp, D, LDp] = reactor_params(Nt, D)

def Reactor_conditions_output():
    print('-------------------- Total Reactor Parameters --------------------')
    print('Total Catalyst mass = ' + str(np.round(Wtot, 3)) + ' kg')
    print('Total reactor volume = ' + str(np.round(Vtot, 3)) + ' m3')
    print('Total reactor length = ' + str(np.round(Ltot, 3)) + ' m')
    print('reactor diameter = ' + str(np.round(D, 3)) + ' m')
    print('L/D tot = ' + str(np.round(LDtot, 3)))
    print('Ac = ' + str(np.round(Ac, 3)))
    print('--------------- Individual Tube-Reactor Parameters ---------------')
    print('Tube Catalyst mass = ' + str(np.round(Wp, 3)) + ' kg')
    print('Tube reactor volume = ' + str(np.round(Vp, 3)) + ' m3')
    print('Tube reactor length = ' + str(np.round(Lp, 3)) + ' m')
    print('Tube L/D = ' + str(np.round(LDp, 3)))
    print('Tube Ac = ' + str(np.round(Ac, 3)))
    print('Qtube = ' + str(y0[2]))
    print('utube = ' + str(y0[2]/Ac))
Reactor_conditions_output()

Lspan = np.linspace(0, Lp, 10000)
debug=True
ans = solve_ivp(lambda V, arr: PBR(V, arr, LDp, Lp, U, Tu, debug), [0, Lp], y0, dense_output = True).sol(Lspan)
print('Done R1!')
#%%
Tspan = ans[0]
Pspan = ans[1]
Qspan = ans[2]
Fspan = ans[3:]
Rspan = np.array([EB_rates(T, P, Q, Fls) for T, P, Q, Fls in zip(Tspan, Pspan, Qspan, np.array(Fspan).T)]).T
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
TotalR1Values = [Tt_ls, Pt_ls, Qt_ls, Ft_ls]
op_vals = [Tt_ls, Pt_ls, Qt_ls, *Ft_ls]
get_vals = True
Total_RX1_outlet_vals = [T1, P1, Q1, F1_ls]
print('Outlet Values:' + str([T1, P1, Q1, F1_ls[10], F1_ls[9]]))
# if get_vals:
#     kc_R1_span = [kcam_R1(T, P, Q, Fls, D) for T, P, Q, Fls in zip(Tspan, Pspan, Qspan, np.array(Fspan).T)]
#     df = pandas.DataFrame(kc_R1_span)
#     df.to_csv('kcam.csv', index = False)
#%%

df_single = pandas.DataFrame(np.array([Lspan, Tspan, Pspan, Qspan, *Fspan]).T,  columns = ['z', 'T', 'P', 'Q', *sn_ls])
df_single.to_csv('Single_reactor_values.csv', index = False)
df_single
#%%
Tspan_plot = [T - 273.15 for T in Tspan]
Pspan_plot = [P for P in Pspan] 
Fspan_plot = [F*3600 for F in  Fspan]

def plot(Wf = 10, save = True):
    W =4*2.54
    H = 1.5*W
    Pcol = 'k'
    Tcol = 'r'
    cols = cm.rainbow(np.linspace(0, 1, len(Fspan_plot)))
    fig, (ax1, axP) = plt.subplots(2,1, sharex = True)
    fig.set_figheight(H)
    fig.set_figwidth(W)
    for n, Fls, col in zip(fn_ls, Ft_ls, cols):
        ax1.plot(Lspan, Fls, label = n, color = col)
    ax1.legend(loc = 'best')
    ax1.set_ylabel('Molar flow (mol/s)')
    
    axP.plot(Lspan, Pspan_plot, color = Pcol)
    axP.set_xlabel('Total Equivalent Reactor Length (m)')
    axP.tick_params(axis = 'y', labelcolor = Pcol)
    axP.set_ylabel('Pressure (kPa)', color = Pcol)

    axT = axP.twinx()
    axT.plot(Lspan, Tspan_plot, color = Tcol)
    axT.tick_params(axis='y', labelcolor = Tcol)
    axT.set_ylabel(r'Temperature ($^\circ C$)', color = Tcol)
    plt.tight_layout()
    if save:
        plt.savefig('Plots/Main.pdf', dpi = 500)
    plt.show()
plot(True)

#%%
def TPlot():
    EQ_plot(False)
    plt.plot(TotalR1Values[0], TotalR1Values[3][10])
    plt.show()
# TPlot()    
# %%

### determination of where EMT effects are negligable #####
# for Q, D in zip([])
nls = ['1. ETBE', '2. TBA1', '3. di-IB', '4. tri-IB', '5. TBA2']
plt.figure(figsize = [6, 6])
for rls, n in zip(Rspan, nls):
    plt.plot(Lspan, rls, label = n)
# plt.plot(Lspan, kc_R1_span, label =  'kcam')
plt.xlabel('Length along single reactor tube, m')
plt.ylabel('Catalyst mass-based rate, mol/kg.s')
plt.legend(loc = 'best')
plt.savefig('Plots/AllLines.pdf', dpi = 500)
plt.show()
# %%
def Conversion(Fls):
    Fls = np.array(Fls).T
    x_IB = [] # 1
    x_ETBE = []  # 10
    x_TBA = [] # 9
    S_TBA_ETBE = [] 
    S_DIB_ETBE = []
    _, F_IB0, _, _, _, _, _, _, F_EtOH0, F_TBA0, F_ETBE0, F_DIB0, _ = Fls[0]
    # F_ETBE_pos = F_IB + F_ETBE0
    for _, F_IB, _, _, _, _, _,_,  F_EtOH, F_TBA, F_ETBE, F_DIB, _  in Fls:
        x_IB.append((F_IB0 - F_IB)/F_IB0)
        x_ETBE.append(1-((F_IB0 -(F_ETBE - F_ETBE0))/F_IB0) )
        x_TBA.append(1-((F_IB0 -(F_TBA - F_TBA0))/F_IB0) )
        S_TBA_ETBE.append((F_TBA - F_TBA0)/(F_ETBE - F_ETBE0))
        S_DIB_ETBE.append((F_DIB0 - F_DIB0)/(F_ETBE - F_ETBE0))
    return [x_IB, x_ETBE,x_TBA ,  S_TBA_ETBE, S_DIB_ETBE]


# %%
plt.plot(Lspan, Conversion(Ft_ls)[0], label = 'Total Conversion')
plt.plot(Lspan, Conversion(Ft_ls)[1], label = 'ETBE Conversion')
plt.plot(Lspan, Conversion(Ft_ls)[2], label = 'TBA Conversion')
plt.xlabel('Reactor Length, m')
plt.ylabel('Conversion')
plt.show()

# %%
plt.plot(Lspan, Conversion(Ft_ls)[2])
plt.plot(Lspan, Conversion(Ft_ls)[3])
plt.xlabel('Reactor Length, m')
plt.ylabel(r'Selectivitiy relative to ETBE, $mol_i/mol ETBE$')
plt.show()

# %%
Wtot, Vtot, Ltot, Q0['value']*1.5
# %%
input_vals = {}
output_vals = {}
for i, n in enumerate(['T', 'P', 'Q', 'IB_ane', 'IB', '1B', 'B_diene', 'NB_ane', 'trans_B', 'cis_B', 'water', 'EtOH', 'TBA', 'ETBE', 'di_IB', 'tri_IB']):
    input_vals[n] = y0[i]
    output_vals[n] = op_vals[i]
print(input_vals)
# %%
print(sn_ls)
# %%
print(sum([input_vals[key] for key in ['IB_ane', 'IB', '1B', 'B_diene', 'NB_ane', 'trans_B', 'cis_B', 'water', 'EtOH', 'TBA', 'ETBE', 'di_IB', 'tri_IB']])*3600)
# %%
print(input_vals['Q'])
# %%
sum(F0)
# %%
np.mean(Tspan)-273.15
# %%
Conversion(Ft_ls)[0][-1]
# %%
phase_check(134.9+273.15, 19e5)
# %%
sn_ls
# %%
