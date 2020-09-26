#%%
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Rates import RATE
from StreamData import mt0, Ft0, Q0, P0, sn_ls, fn_ls, F0, fnr_ls, snr_ls, Q0
from matplotlib import cm
import csv
from pandas import read_csv, DataFrame
from scipy.interpolate import interp1d

Qt0 = Q0['value']
#%%
def OptiFunc(T0, Wtot = 20000):
    def PBR(V, arr):
        T, P, Q = arr[:3]
        Fls = arr[3:]
        r_IB_ane, r_IB, r_1B, r_B_diene, r_NB_ane, r_trans_B, r_cis_B, r_water, r_EtOH, r_TBA, r_ETBE, r_di_IB, r_tri_IB = [rhob*r for r in RATE(T, P, Q, Fls)]
        dT, dP, dQ = 0,0,0
        return [dT, dP, dQ, r_IB_ane, r_IB, r_1B, r_B_diene, r_NB_ane, r_trans_B, r_cis_B, r_water, r_EtOH, r_TBA, r_ETBE, r_di_IB, r_tri_IB]

    def selectivity(Fls):
        F_IB, F_water, F_EtOH, F_TBA, F_ETBE, F_di_IB, F_tri_IB = Fls
        dF_IB = F_IB[0] - F_IB[-1]
        dF_TBA = F_TBA[-1] - F_TBA[0]
        dF_ETBE= F_ETBE[-1] - F_ETBE[0] 
        dF_di_IB = F_di_IB[-1] - F_di_IB[0] 
        dF_tri_IB = F_tri_IB[-1] - F_tri_IB[0]

        S_ETBE = dF_ETBE/dF_IB
        S_TBA = dF_TBA/dF_IB
        S_di_IB = 2*dF_di_IB/dF_IB
        S_tri_IB = dF_tri_IB/dF_IB

        return np.array([(F_IB[0] - F_IB[-1])/F_IB[0], S_ETBE , S_TBA, S_di_IB, S_tri_IB])

    rhob = 610
    Vspan = np.linspace(0, Wtot/rhob, 2)
    Pt0, Qt0 = [v['value'] for v in [P0, Q0] ]
    y0 = [T0,Pt0, Qt0, *F0]
    _, _, _, _, F_IB, _, _, _, _, _, F_water, F_EtOH, F_TBA, F_ETBE, F_di_IB, F_tri_IB = solve_ivp(PBR, [0, Wtot/rhob], y0)['y']
    return selectivity([F_IB, F_water, F_EtOH, F_TBA, F_ETBE, F_di_IB, F_tri_IB])

# %%
n = 7
T0i, T0f = 40+273.15, 100+273.15
Wi, Wf = 13000, 30000
def data_collect(n = 7, T0_cond = [T0i, T0f], W_cond = [Wi, Wf]):
    Tspan = np.linspace(T0i, T0f, n)
    Wspan = np.linspace(Wi, Wf, n)
    x = np.zeros(shape = [n, n])
    S_ETBE = np.zeros(shape = [n, n])
    S_TBA = np.zeros(shape = [n, n])
    S_di_IB = np.zeros(shape = [n, n])
    S_tri_IB = np.zeros(shape = [n, n])
    x_c = np.zeros(shape = [n, n])
    Tspan, Wspan = np.meshgrid(Tspan, Wspan)
    for i in range(n):
        for j in range(n):
            W = Wspan[i, j]
            T0 = Tspan[i, j]
            print('----------------' + str(i) + ', ' + str(j) +'---------------')
            print(T0, W)
            a = OptiFunc(T0, W)
            x[i, j] = a[0]
            S_ETBE[i, j] = a[1]
            S_TBA[i, j] = a[2]
            S_di_IB[i, j] = a[3]
            S_tri_IB[i, j] = a[4]
            x_c[i, j] = 1 - a[1] - a[2] - a[3] - a[4]
    for n, ls in zip(['T', 'W', 'x', 'S_ETBE', 'S_TBA', 'S_di_IB', 'S_tri_IB'], [Tspan, Wspan, x, S_ETBE, S_TBA, S_di_IB, S_tri_IB]):
        with open('OptimumTempData/' + n + '.csv',"w+") as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(ls)
# data_collect(n = 25)
#%%
def import_data():
    ret = []
    for n in ['T', 'W', 'x', 'S_ETBE', 'S_TBA', 'S_di_IB', 'S_tri_IB']:
        df = read_csv('OptimumTempData/' + n + '.csv').values.tolist()
        ret.append(np.array(df))
    return ret
Tspan, Wspan, x, S_ETBE, S_TBA, S_di_IB, S_tri_IB = import_data()
#%%
def plot_3d():
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1 = fig.gca(projection='3d')
    
    surf1 = ax1.plot_surface(Tspan, Wspan, x, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(Tspan, Wspan, S_ETBE, cmap=cm.viridis, linewidth=0, antialiased=False)
    # surf3 = ax2.plot_surface(Tspan, Wspan, S_TBA, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # surf4 = ax2.plot_surface(Tspan, Wspan, S_di_IB, cmap=cm.magma, linewidth=0, antialiased=False)

    plt.show()
# plot_3d()

##########################
#%%
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

def EQ_func(Tspan, P, Q):
    ret = []
    for i, T in enumerate(Tspan):
        print('run ' + str(i+1) + '/' + str(len(Tspan)) + ' | T = ' + str(T))
        ret.append(Equilibrium_Conditions(T, 1600, 14/3600))
    Fe_span = np.array(ret).T[3:]
    # Fe_span = np.array([Equilibrium_Conditions(T, 1600, 14/3600) for T in Tspan]).T[3:]
    return Fe_span
# %%
def EQ_data_collect(Tspan, P, Q):
    Fe_vals = EQ_func(Tspan2, P, Q)

    df = DataFrame([Tspan2, *Fe_vals]).T
    df.columns = ['T', *sn_ls]
    df.to_csv('OptimumTempData/Equilibrium_data_vs_T.csv', index = False)
Tspan2 = np.linspace(30+273.15, 373.15, 101)
# EQ_data_collect(Tspan2, 1600, Qt0)
# %%
df = read_csv('OptimumTempData/Equilibrium_data_vs_T.csv')

interp_Fe_T = interp1d(df['ETBE'], df['T'])
max_ETBE = max(df['ETBE'])
T_opt = interp_Fe_T(max_ETBE)

cols = cm.rainbow(np.linspace(0, 1, len(snr_ls)))
for n, l, col in zip(snr_ls, fnr_ls, cols):
    plt.plot(df['T'], df[n], label = l , color = col)
plt.plot([T_opt, T_opt], [0, max_ETBE], label = 'Optimum Temp. = ' + str(T_opt) + 'K', color = 'r')
plt.plot([min(df['T']), T_opt], [max_ETBE, max_ETBE], label = 'Max. Equil. ETBE = ' + str(max_ETBE) + ' mol/s', color = 'r', ls = '--')
plt.legend(loc = 'best')
plt.show()

# %%
