#%%
import numpy as np 
from scipy.optimize import curve_fit, fsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ipywidgets import interact
conc = {'t': [0, 450, 850, 1256, 1664, 1969, 2343, 2751, 3159, 3600, 4381, 5026, 5875, 6928] , 
    'IB':  [1.85,1.519, 1.25, 1, 0.835, 0.675, 0.55, 0.47, 0.4, 0.345, 0.325, 0.325, 0.325, 0.325], 
    'EtOH': [1.8, 1.425, 1.175, 0.98, 0.79, 0.635, 0.5, 0.435, 0.350, 0.29, 0.275, 0.275, 0.275, 0.275], 
    'ETBE': [0, 0.325, 0.6, 0.81, 1, 1.15, 1.29, 1.375, 1.43, 1.475, 1.519, 1.519, 1.519, 1.519] 
    }

def Ka(T, Ka_r = 16.5, T_r =  343, dH = -44.3e3): # dH: kJ/kmol
    R = 8.314
    return Ka_r*np.exp((-dH/R)*((T**-1)- (T_r**-1)))

def kf3(T, k3_r = 4.7e-4, Tr = 343, Ea = 81.2e3): #k3_r: dm3/g.s, Ea: kJ/kmol
    R = 8.314
    k0 = k3_r/np.exp(-Ea/(R*Tr))
    return k0*np.exp(-Ea/(R*T)) ## not verified yet...

#%%
################## verification ##########################
def Verifunc(β = 20, F = 85, W = 1):
    tspan = np.linspace(0, conc['t'][-1], 100)
    def Ka(T, Ka_r = 16.5, T_r =  343, dH = -44.3e3): # dH: kJ/kmol
        R = 8.314
        return Ka_r*np.exp((-dH/R)*((T**-1)- (T_r**-1)))

    def kf3(T, k3_r = 4.7e-4, Tr = 343, Ea = 81.2e3): #k3_r: dm3/g.s, Ea: kJ/kmol
        R = 8.314
        k0 = k3_r/np.exp(-Ea/(R*Tr))
        return k0*np.exp(-Ea/(R*T)) ## not verified yet...

    def rate_french_ETBE_conc(T, P, Q, arr, Ct = 4.7e-3): #Ct: mol/g cat, returns cat mass-based rate (mol/s.g)
        CIB, CEtOH, CETBE = arr
        k, Keq = kf3(T), Ka(T)
        α = 1/Keq
        return k*(CIB - α*(CETBE/CEtOH) + ((β*(CIB*CEtOH - α*CETBE))/(CIB + F*(CEtOH**2) + CETBE)))

    def batch(t, arr):
        r = rate_french_ETBE_conc(343, 1 , 1 , arr)*W
        return [-r, -r, r]

    
    y0 = [1.85, 1.8, 0]
    ans = solve_ivp(batch, [0, tspan[-1]], y0, dense_output=True).sol(tspan)

    plt.scatter(conc['t'], conc['IB'])    
    plt.scatter(conc['t'], conc['EtOH'])
    plt.scatter(conc['t'], conc['ETBE'])
    plt.plot(tspan, ans[0])
    plt.plot(tspan, ans[1])
    plt.plot(tspan, ans[2])
    plt.show()
Verifunc()
#%%
########## T approximate dependence of beta and F ##########
# IB, EtOH, ETBE
Ci1 = [j[-1] for a, j in conc.items()][1:] #mol/dm3


def solver(s):
    Z1, Z2 = s
    T = 323
    Xer = 0.91
    def TAp(T, Z1, Z2):
        def BetaFunc(T):
            return 20*np.exp(Z1*((T**-1) - 343**-1))

        def FFunc(T):
            return 85*np.exp(Z2*((T**-1) - 343**-1))

        def Ka(T, Ka_r = 16.5, T_r =  343, dH = -44.3e3): # dH: kJ/kmol
            R = 8.314
            return Ka_r*np.exp((-dH/R)*((T**-1)- (T_r**-1)))

        def kf3(T, k3_r = 4.7e-4, Tr = 343, Ea = 81.2e3): #k3_r: dm3/g.s, Ea: kJ/kmol
            R = 8.314
            k0 = k3_r/np.exp(-Ea/(R*Tr))
            return k0*np.exp(-Ea/(R*T)) ## not verified yet...

        def rate_french_ETBE_conc(T, P, Q, arr, Ct = 4.7e-3): #Ct: mol/g cat, returns cat mass-based rate (mol/s.g)
            CIB, CEtOH, CETBE = arr
            k, Keq = kf3(T), Ka(T)
            α = 1/Keq
            F = FFunc(T)
            β = BetaFunc(T)
            return k*(CIB - α*(CETBE/CEtOH) + ((β*(CIB*CEtOH - α*CETBE))/(CIB + F*(CEtOH**2) + CETBE)))

        def batch(t, arr):
            r = rate_french_ETBE_conc(T, 1 , 1 , arr)
            return [-r, -r, r]

        y0 = [1.85, 1.8, 0]
        ans = solve_ivp(batch, [0, 20000], y0, dense_output=True)
        Xspan = []
        for CIB in ans['y'][0]:
            Xspan.append((ans['y'][0][0] - CIB)/ans['y'][0][0])
        return ans['t'], Xspan
    Xe = TAp(T, Z1, Z2)[1][-1]
    return [Xe - Xer, Xe - Xer]
print(fsolve(solver, x0 = [10000,20000]))

#%%


# %%
