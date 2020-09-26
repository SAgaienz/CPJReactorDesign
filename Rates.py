import numpy as np 
from thermo import Chemical, UNIFAC
from StreamData import CF_dict, fn_ls, sn_ls

def fetch_UNIFAC_DG(full_name_list = fn_ls, update =False):
    UNIFAC_DG = []
    if update:
        for name in full_name_list:
            print(name)
            a = Chemical(name, 200, 100).UNIFAC_groups
            UNIFAC_DG.append(a)
        return UNIFAC_DG
    else:
        return [{1: 3, 3: 1}, {1: 2, 7: 1}, {1: 1, 2: 1, 5: 1}, {5: 2}, {1: 2, 2: 2}, {1: 2, 6: 1}, {1: 2, 6: 1}, {16: 1}, {1: 1, 2: 1, 14: 1}, {1: 3, 4: 1, 14: 1}, {1: 4, 4: 1, 25: 1}, {1: 5, 4: 1, 8: 1}, {1: 6, 2: 2, 4: 2, 7: 1}]

##### convert F to a ######
def activity(T, P, Q, arr): # takes Fi (mol/s)
    x_ls = [C/sum(arr) for C in arr]
    UNIFAC_DG = fetch_UNIFAC_DG()
    a_ls = [g*x for g, x, in zip(UNIFAC(T, x_ls, UNIFAC_DG), x_ls)]
    # print(len(a_ls))
    # print(a_ls)
    # print(T)
    return a_ls

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
    _ , a_IB , _ , _, _, _, _ , a_water , a_EtOH , a_TBA , a_ETBE , a_di_IB , _ = activity(T, P, Q, arr)
    B = 50
    F = 7
    D = 8
    k, Keq = k_ETBE(T), Ka_ETBE(T)
    k2, Keq2 = 20*k, 20*Keq
    α = 1/Keq
    return ((k*a_IB*a_EtOH - (k/Keq)*B*a_ETBE)/(a_EtOH + B*a_ETBE)) + ((k2*a_IB*a_EtOH - (k2/Keq2)*D*a_ETBE)/(a_IB + F*a_EtOH**2 + D*a_ETBE))


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
    _ , a_IB , _ , _, _, _, _ , a_water , a_EtOH , a_TBA , a_ETBE , a_di_IB , _ = activity(T, P, Q, arr)
    k, Ka = k_TBA(T), Ka_TBA(T)
    KwKTBA = 1.5
    K_TBA_K_IB = 7
    return k*(a_water*a_IB - Ka*a_TBA)/(a_IB + K_TBA_K_IB*a_TBA + a_water*(K_TBA_K_IB*KwKTBA))

############      Di-IB  (M.Honkela)       #################

def k_diB(T): ## T in K,  mol/s.kg_cat
    Fdib = 0.82 * (1000/3600) # mol/h.g_cat --> mol/s.kg_cat
    Edib = 30e3
    R = 8.314
    Tr = 373.15 
    return Fdib*np.exp((-Edib/R)*((T**-1) - (Tr**-1)))

def rate_diB_Honk(T, P, Q, arr):
    _ , a_IB , _ , _, _, _, _ , a_water , a_EtOH , a_TBA , a_ETBE , a_di_IB , _ = activity(T, P, Q, arr)
    k = k_diB(T)
    KwKTBA = 1.5
    K_TBA_K_IB = 7
    return (k*a_IB**2)/(a_IB + K_TBA_K_IB*a_TBA + a_water*(K_TBA_K_IB*KwKTBA))**2


############      Tri-IB  (M.Honkela)       #################
def k_TriiB(T): # T in K, returns mol/s.kg_cat
    Ftrib = 0.065 * 1000 / 3600
    Etrib = 1.8e6
    R = 8.314
    Tr = 373.15
    return Ftrib*np.exp((-Etrib/R)*((T**-1) - (Tr**-1)))

def rate_TriB_Honk(T, P, Q, arr): # takes F in mol/s, returns mol/s.kg_cat
    # arr = check_zero(arr)
    _ , a_IB , _ , _, _, _, _ , a_water , a_EtOH , a_TBA , a_ETBE , a_di_IB , _ = activity(T, P, Q, arr)
    k = k_TriiB(T) # mol/s.kg_cat
    K_TBA_K_IB = 7
    KwKTBA = 1.5
    return k*a_IB*a_di_IB/((a_IB + K_TBA_K_IB*a_TBA + a_water*(K_TBA_K_IB*KwKTBA))**3)

############ TBA + EtOH --> ETBE + H2O  (M. Umar)   #################
def k_TBA_ETBE(T): # T in K, returns mol/s.kg_cat
    "The paper reported kinetics in volumetric basis"
    "therefore, their bed density was used to convert to a catalyst mass basis"
    rho_b = 640.26 #kg/m3
    return np.exp(11.827 - 6429.6/T)*1000/rho_b

def Ka_TBA_ETBE(T): # activity-based chemical equilibrium constant (T in K)
    A = 1140
    B = 14580
    C = 232.9
    D = 1.087
    E = 1.114e-3
    F = 5.538e-7
    return np.exp(A - (B/T) + C*np.log(T) + D*T - E*T**2 + F*T**3)

def rate_TBA_Umar(T, P, Q, arr): ## takes F in mol/s, returns mol/s.kg_cat
    _ , a_IB , _ , _, _, _, _ , a_water , a_EtOH , a_TBA , a_ETBE , a_di_IB , _ = activity(T, P, Q, arr)
    k, Keq = k_TBA_ETBE(T), Ka_TBA_ETBE(T)
    # print(-k*(a_TBA*a_EtOH - (a_ETBE*a_water/Keq)))
    return -k*(a_TBA*a_EtOH - (a_ETBE*a_water/Keq))

# print(k_TBA_ETBE(350), k_TBA(350))
############# OVERALL RATE EQ ################

def RATE(T, P, Q, arr):
    " T: K" " P: kPa" " Q: m3/s" " arr: F (mol/s)"
    "T, P, Q, IB_ane', 'IB', '1B', 'B_diene', 'NB_ane', 'trans_B', 'cis_B', 'water', 'EtOH', 'TBA', 'ETBE', 'di_IB', 'tri_IB"
    "returns mass-based rate (mol/s.kg_cat)"
    
    r_ETBE_t = rate_ETBE_Thy_act(T, P, Q, arr)
    r_TBA_1 = rate_TBA_Honk(T, P, Q, arr)
    r_TBA_2 = rate_TBA_Umar(T, P, Q, arr)
    r_di_IB_t = rate_diB_Honk(T, P, Q, arr)
    r_tri_IB_t = rate_TriB_Honk(T, P, Q, arr)

    ### 7 reacting compounds ###
    r_IB_ane = 0
    r_IB = -r_ETBE_t - r_TBA_1 - 2*r_di_IB_t - r_tri_IB_t
    r_1B = 0
    r_B_diene = 0
    r_NB_ane = 0
    r_trans_B = 0
    r_cis_B = 0
    r_water = -r_TBA_1 - r_TBA_2
    r_EtOH = -r_ETBE_t + r_TBA_2 
    r_TBA = r_TBA_1 + r_TBA_2 
    r_ETBE = r_ETBE_t - r_TBA_2
    r_di_IB = r_di_IB_t
    r_tri_IB = r_tri_IB_t
    # print(r_TBA_1, r_TBA_2)

    return [r_IB_ane, r_IB, r_1B, r_B_diene, r_NB_ane, r_trans_B, r_cis_B, r_water, r_EtOH, r_TBA, r_ETBE, r_di_IB, r_tri_IB]

def EB_rates(T, P, Q, arr):
    r_ETBE_t = rate_ETBE_Thy_act(T, P, Q, arr)
    r_TBA_1 = rate_TBA_Honk(T, P, Q, arr)
    r_TBA_2 = rate_TBA_Umar(T, P, Q, arr)
    r_di_IB_t = rate_diB_Honk(T, P, Q, arr)
    r_tri_IB_t = rate_TriB_Honk(T, P, Q, arr)
    return [r_ETBE_t, r_TBA_1, r_di_IB_t, r_tri_IB_t,  r_TBA_2]