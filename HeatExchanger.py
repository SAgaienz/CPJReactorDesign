#%%
from thermo import Chemical
import numpy as np
from pandas import read_csv
from PressureDrop import rho_m,  rho_if, mu_m_simple, mu_i
from matplotlib import pyplot as plt
from StreamData import fn_ls
#%%

def Cpi_mass(T, P, f_ls = fn_ls, s_ls = sn_ls): ## takes T in K, P in kPa, returns J/kg.K
    P = P*1000
    Cp_ls = [Chemical(f, T, P).Cp for f in fn_ls ]
    return Cp_ls


def k(T, P):
    k_ls = [Chemical(f, T, P).ThermalConductivityLiquid(T, P) for f in fn_ls]
    return k_ls

df_single = read_csv('Single_reactor_values.csv')
dt_i = 52.5e-3
dt_o = 60.5e-3
u_t = df_single['Q'].tolist()[0]/(0.25*np.pi*dt_o**2)
u_t, df_single['Q']
#%%
mu_ls = []
rho_ls = []
u_ls = []
Cp_ls = []
k_ls = []
for i in range(len(df_single)):
    T, P, Q, Fls = df_single.iloc[i][1], df_single.iloc[i][2], df_single.iloc[i][3], df_single.iloc[i][4:]
    x_ls = [F/sum(Fls) for F in Fls]

    rho_ls.append(rho_m(T, P, Q, Fls))
    mu_ls.append(mu_m_simple(T, P, Q, Fls))
    u_ls.append(Q/(0.25*np.pi*dt_i**2))
    Tot_Cp = sum([Cp*xi for Cp, xi in zip(Cpi_mass(T, P), x_ls)])
    Cp_ls.append(Tot_Cp)
    k_ls.append(sum([k*x for k, x in zip(k(T, P), x_ls)]))
    print(i/len(df_single))
mu_t = np.mean(mu_ls)
rho_t = np.mean(rho_ls)
u_t = np.mean(u_ls)
Cp_t = np.mean(Cp_ls)
k_t = np.mean(k_ls)
#%%
L =  15.5 #m
# [k_t, Cp_t, mu_t, u_t, rho_t] = [0.16109164289057082, 173.01751400722304, 0.00021444115223357086, 0.005145535961510567, 625.6507175436316]
k_t, Cp_t, mu_t, u_t, rho_t = [0.1619805174612345,
 2868.299247405188,
 0.00024144422043755132,
 0.0038591519711329274,
 642.0767150248047]

#%%

N_t_tot = 780 #tubes
Nt = N_t_tot/13 # 13 HE with 200 tubes
#60 tubes per bundle
def Dbf(Nt):
    n, K = 2.142, 0.319
    return dt_o*(Nt/K)**(1/n)
Db = Dbf(Nt) #bundle diameter
Dshell = Db + 15e-3
pt = 1.25*dt_o
lb = Dshell/5

de = 1.1/dt_o*(pt**2 - 0.917*dt_o**2)
## tube side:

# %%

# %%
T = 48
rho_s, mu_s, k_s, Cp_s = Chemical('water', 48+273.15, 200000).rho, Chemical('water', 48+273.15, 200000).mu, Chemical('water', 48+273.15, 200000).ThermalConductivityLiquid(48+273.15, 200000),  Chemical('water', 48+273.15, 200000).Cp
rho_s, mu_s, k_s, Cp_s
#%%
def Re_f(rho, dp, mu, u):
    return rho*u*dp/mu
Re_t = Re_f(rho_t, dt_o, mu_t, u_t) 
print(Re_t)
def Pr_f(Cp, mu, k):
    return Cp*mu/k
print(Pr_f(Cp_t, mu_t, k_t))
def h_tube(di, do, jh, rho, mu, u, k, Cp, L): ## in laminar
    Re = Re_f(rho, di, mu, u)
    Pr = Cp*mu/k
    Nu = (1.86*(Re*Pr)**0.33)*((di/L)**0.33)*(1)
    return k/di*Nu
    # return k/di*jh*Re*Pr**(1/3)
jh = 1e-2
ht = h_tube(dt_i, dt_o, jh, rho_t, mu_t, u_t, k_t, Cp_t ,L)
ht

def Re_sf(Qw):
    As = (pt-dt_o)*Dshell*lb/pt
    u = Qw/As
    Re_s = u*rho_s*de/mu_s
    return Re_s
Qw = 0.017
Re_s = Re_sf(Qw)
print(Re_s)
Pr_s = Cp_s*mu_s/k_s
# %%
jsh = 6e-4
jsf = 1e-2

h_s = k_s/dt_o*jsh*Re_s*Pr_s**(1/3)
h_s, ht
# %%
k_steel = 45
hf = 2000
U = (1/h_s + (dt_o*np.log(dt_o/dt_i)/2*k_steel) + ((dt_i/dt_o)/hf) + (dt_o/dt_i)/ht)**-1
U
# %%
print(1/h_s, dt_o*np.log(dt_o/dt_i)/2*k_steel, (dt_i/dt_o)/hf, (dt_o/dt_i)/ht)
# %%
u_s = Qw/0.25*np.pi*(de**2)
dP_s = 8*jsf*(Dshell/de)*(L/lb)*(rho_s*(u_s**2)/2)*(1)
dP_s
k_s, rho_s, mu_s, Cp_s

h_s
dP_t = 6.88
#%%
Q = Q*N_t_tot
# pump sizing
def pumpPower(dP, Q = Q, eta = 0.7):
    return dP*Q/(eta*3.6e6)
pumpPower((20-7.5+dP_t)*1e5)
# %%
Qw*13
# %%
