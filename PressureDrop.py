#%%
import numpy as np 
from thermo import Chemical, phase_change
from StreamData import sn_ls, fn_ls, P0, F0, P_min_120C, Q0
import matplotlib.pyplot as plt

e_cat = 60.2216e-2
rho_b = 0.64026e3
dp90 = 475e-6
dp10 = 873.2e-6
dp = np.mean([dp90, dp10])


def mu_i(T, P, f_ls = fn_ls, s_ls = sn_ls): ## takes T in K, P in kPa, returns liquid viscosity (Pa.s)
    P = P*1000
    mu_ls = [Chemical(f, T, P).mu for f in fn_ls ]
    return mu_ls

def mu_m_simple(T, P, Q, Fls): # returns approximate liquid viscosity (Pa.s)
    "Assumes no interactions"
    mu_ls = mu_i(T, P)
    xi_ls = [F/sum(Fls) for F in Fls]
    ln_mu_m = sum([xi*np.log(mu_i) for xi, mu_i in zip(xi_ls, mu_ls)])
    return np.exp(ln_mu_m)

def rho_if(T, P, f_ls = fn_ls, s_ls = sn_ls): ## takes T in K, P in kPa, returns kg/m3
    P = P*1000
    rho_ls = [Chemical(f, T, P).rho for f in fn_ls]
    return rho_ls

def MM(f_ls = fn_ls, s_ls = sn_ls, update = False):
    if update:
        mm_ls = [Chemical(f, 273.15, 101325).MW for f in fn_ls]
        return mm_ls
    else:
        return [58.124, 56.10632, 56.10632, 54.09044, 58.124, 56.10632, 56.10632, 18.01528, 46.06844, 74.1216, 102.17476, 112.21264, 168.31896]

def Mass_Flow(Fls):
    m_ls = [MM_i*F_i/1000 for MM_i, F_i in zip(MM(), Fls)] ## g/mol * mol/s * 1kg/1000g --> kg/s
    return m_ls

def rho_m(T, P, Q, Fls):
    rho_ls = rho_if(T, P)
    m_ls = Mass_Flow(Fls)
    mf_ls = [m_i/(sum(m_ls)) for m_i in m_ls]
    a_ls = []
    for mf_i, rho_i in zip(mf_ls, rho_ls):
        a_ls.append(mf_i/rho_i)
    rho_m = sum(a_ls)**-1
    return rho_m

def reactor_length_dia(Vtot, LD):
    D = (Vtot*4/(np.pi*LD))**(1/3)
    L = LD*D
    return [L, D]

def Ergun(T, P, Q, Fls, L, LD): # K, kPa, m3/s, mol/s, [-], m
    rho = rho_m(T, P, Q, Fls) # kg/m3
    mu = mu_m_simple(T, P, Q, Fls) # Pa.s
    D  = L/LD # m
    Ac = (np.pi*D**2)/4 # m2
    G = sum(Mass_Flow(Fls))/Ac # kg/s.m2
    e = e_cat
    return (-G/(rho*dp))*((1-e)/(e**3))*((150*mu*(1-e)/dp) + 1.75*G)/1000 # kPa/m3
