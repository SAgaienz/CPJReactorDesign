import numpy as np 
from thermo import Chemical, UNIFAC

def activity(T, P, Q, arr): # takes concentration
    x_ls = [C/sum(arr) for C in arr]
    names = ['isobutene', 'ethyl alcohol', 'water', 'tert-butanol', 'ethyl tert-butyl ether', '2,4,4-trimethyl-2-pentene', '4,4-dimethyl-2-neopentyl-1-pentene', '1-butene', 'isobutane']
    UNIFAC_DG = [{1: 2, 7: 1}, {1: 1, 2: 1, 14: 1}, {16: 1}, {1: 3, 4: 1, 14: 1}, {1: 4, 4: 1, 25: 1}, {1: 5, 4: 1, 8: 1}, {1: 6, 2: 2, 4: 2, 7: 1}, {1: 1, 2: 1, 5: 1}, {1: 3, 3: 1}]
    
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

def rate_ETBE_Thy_act(T, P, Q, arr, Beta = 20, F = 85): #returns cat mass-based rate (mol/s.g) (mol/s.kg_cat)
        aIB, aEtOH, aw, aTBA, aETBE, _, _, _, _ = activity(T, P, Q, arr)
        B = 50
        F = 7
        D = 8
        k, Keq = k_ETBE(T), Ka_ETBE(T)
        k2, Keq2 = 20*k, 20*Keq
        α = 1/Keq
        return ((k*aIB*aEtOH - (k/Keq)*B*aETBE)/(aEtOH + B*aETBE)) + ((k2*aIB*aEtOH - (k2/Keq2)*D*aETBE)/(aIB + F*aEtOH**2 + D*aETBE))


############      TBA (M.Honkela)       #################
def Ka_TBA(T): # T in K, dimensionless
    return np.exp((-3111.9/T) + 7.6391)

def k_TBA(T): #T in K , returns mol/s.kg_cat
    E = 18e3
    R = 8.314
    Fref = 0.2  * (1000/3600) # mol/h.g_cat --> mol/s.kg_cat
    Tref = 343
    return Fref*np.exp(-(E/R)*((T**-1) - (Tref**-1)))

def rate_TBA_Honk(T, P, Q, arr):
    aIB, _, aw, aTBA, _, _, _, _, _ = activity(T, P, Q, arr)
    k, Ka = k_TBA(T), Ka_TBA(T)
    KwKTBA = 1.5
    K_TBA_K_IB = 7
    return k*(aw*aIB - Ka*aTBA)/(aIB + K_TBA_K_IB*aTBA + aw*(K_TBA_K_IB*KwKTBA))

############      Di-IB  (M.Honkela)       #################

def k_diB(T): ## T in K,  mol/s.kg_cat
    Fdib = 0.82 * (1000/3600) # mol/h.g_cat --> mol/s.kg_cat
    Edib = 30e3
    R = 8.314
    Tr = 373.15 
    return Fdib*np.exp((-Edib/R)*((T**-1) - (Tr**-1)))

def rate_diB_Honk(T, P, Q, arr):
    aIB, _, aw, aTBA, _, _, _, _, _ = activity(T, P, Q, arr)
    k = k_diB(T)
    KwKTBA = 1.5
    K_TBA_K_IB = 7
    return (k*aIB**2)/(aIB + K_TBA_K_IB*aTBA + aw*(K_TBA_K_IB*KwKTBA))**2


############      Tri-IB  (M.Honkela)       #################
def k_TriiB(T): # T in K, returns mol/s.kg_cat
    Ftrib = 0.065 * 1000 / 3600
    Etrib = 1.8e6
    R = 8.314
    Tr = 373.15
    return Ftrib*np.exp((-Etrib/R)*((T**-1) - (Tr**-1)))

def rate_TriB_Honk(T, P, Q, arr): # takes F in mol/s, returns mol/s.kg_cat
    # arr = check_zero(arr)
    _ , a_IB , _, _, _, _, _, a_water , _ , a_TBA , _ , a_diB , _ = activity(T, P, Q, arr)
    k = k_TriiB(T) # mol/s.kg_cat
    K_TBA_K_IB = 7
    KwKTBA = 1.5
    return k*a_IB*a_diB/((a_IB + K_TBA_K_IB*a_TBA + a_water*(K_TBA_K_IB*KwKTBA))**3)