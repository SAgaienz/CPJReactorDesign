import numpy as np 
from thermo import Chemical, UNIFAC

def activity(T, P, Q, arr): # takes concentration
    xIB, xEt, xw, xTBA, xETBE, xDIB, xTRIB, x1B, xIBane = x_ls = [C/sum(arr) for C in arr]
    names = ['isobutene', 'ethyl alcohol', 'water', 'tert-butanol', 'ethyl tert-butyl ether', '2,4,4-trimethyl-2-pentene', '4,4-dimethyl-2-neopentyl-1-pentene', '1-butene', 'isobutane']
    UNIFAC_DG = [{1: 2, 7: 1}, {1: 1, 2: 1, 14: 1}, {16: 1}, {1: 3, 4: 1, 14: 1}, {1: 4, 4: 1, 25: 1}, {1: 5, 4: 1, 8: 1}, {1: 6, 2: 2, 4: 2, 7: 1}, {1: 1, 2: 1, 5: 1}, {1: 3, 3: 1}]
    # for name in names:
    #     a = Chemical(name, T, P).UNIFAC_groups
    #     UNIFAC_DG.append(a)
    # return UNIFAC_DG
    return [g*x for g, x, in zip(UNIFAC(T, x_ls, UNIFAC_DG), x_ls)]
########### Adsorption Constants ############
def B_Et(T, TrKr = [358, 1.05e4], dH = -8.3e3, rho_b = 640.26): # 323<T<358
    R = 8.314
    Tr, Kr = TrKr
    return Kr*np.exp((-dH/R)*((T**-1) - (Tr**-1)))/ rho_b

def B_IB(T, TrKr = [340, 260], dH = -54.2e3, rho_b = 640.26):
    R = 8.314
    Tr, Kr = TrKr
    return Kr*np.exp((-dH/R)*((T**-1) - (Tr**-1)))/ rho_b

def B_ETBE(T, Tr = [313, 323], Kadr = [7.3e-7, 3.23]):
    R = 8.314
    dH = -R*np.log(Kadr[1]/Kadr[0])/((Tr[1]**-1)-(Tr[0]**-1))
    Kr, Tr = Kadr[0], Tr[0]
    return Kr*np.exp((-dH/R)*((T**-1) - (Tr**-1)))

########### ETBE rate (Thyrion et al) ############

def Ka_ETBE(T, Ka_r = 16.5, T_r =  343, dH = -44.3e3): # dH: kJ/kmol
    R = 8.314
    return Ka_r*np.exp((-dH/R)*((T**-1)- (T_r**-1)))

def k_ETBE(T, k3_r = 4.7e-4, Tr = 343, Ea = 81.2e3): #k3_r: dm3/g.s, Ea: kJ/kmol
    R = 8.314
    k0 = k3_r/np.exp(-Ea/(R*Tr))
    return k0*np.exp(-Ea/(R*T))

def rate_ETBE_Thy(T, P, Q, arr, Beta = 20, F = 85): #returns cat mass-based rate (mmol/s.g)
        CIB, CEt, _, _, CETBE, _, _, _, _ = arr
        β = Beta
        k, Keq = k_ETBE(T), Ka_ETBE(T)
        α = 1/Keq
        return k*(CIB - α*(CETBE/CEt) + ((β*(CIB*CEt - α*CETBE))/(CIB + F*(CEt**2) + CETBE)))


############      TBA (M.Honkela)       #################
def Ka_TBA(T): # T in K, dimensionless
    return np.exp((-3111.9/T) + 7.6391)

def k_TBA(T): #T in K , returns mol/s.kg_cat
    E = 18e3
    R = 8.314
    Fref = 0.2 
    Tref = 343
    return Fref*np.exp(-(E/R)*((T**-1) - (Tref**-1)))

def rate_TBA_Honk(T, P, Q, arr):
    aIB, _, aw, aTBA, _, _, _, _, _ = activity(T, P, Q, arr)
    k, Ka = k_TBA(T), Ka_TBA(T)
    KwKTBA = 1.5
    return k*(aw*aIB - Ka*aTBA)/(aTBA + (KwKTBA*aw))

############      Di-IB  (M.Honkela)       #################

def k_diB(T): ## T in K,  mol/s.kg_cat
    Fdib = 0.82 * (1000/3600) # mol/h.g_cat --> mol/s.kg_cat
    Edib = 30e3
    R = 8.314
    Tr = 373.15 
    return Fdib*np.exp((-Edib/R)*((T**-1) - (Tr**-1)))

def rate_diB_Honk(T, P, Q, arr):
    aIB, _, _, aTBA, _, _, _, _, _ = activity(T, P, Q, arr)
    k = k_diB(T)*1e-3
    K_TBA_K_IB = 7
    # factor = ()
    return (k*aIB**2)/(aIB + K_TBA_K_IB*aTBA)**2


############      Tri-IB  (M.Honkela)       #################
# def k_diB(T):
#     Fdib = 
#     Edib = 
#     R = 8.314
#     Tr = 
#     return Fdib*np.exp((-Edib/R)*((T**-1) - (Tr**-1)))