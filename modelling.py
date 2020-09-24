#%%
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Rates import rate_ETBE_Thy, rate_TBA_Honk, rate_diB_Honk, rate_TriB_Honk
from StreamData import C_F
tspan = np.linspace(0, 12600, 1000)
W = 2 #kg
print(C_F.Compound.tolist())
#%%

FIB, FEt, Fw, FTBA, FETBE, FDIB, FTRIB, F1B, FIBane = F0 = [F*(1000/3600) for F in  [52.255477, 58.885664, 21.650439, 7.936977, 14.89919, 0, 0 , 20.322345 , 6.7940117]] ## all in ]kmol/h
Q = 15 #m3/h
C_IB, C_Et, C_w, C_TBA, C_ETBE, C_DIB, C_TRIB, C_1B, C_IBane = C0 = [(F/Q) for F in F0] # all in kmol/m3 or mol/dm3
T0, P0, Q0 = [343, 1600, 15] # K, kPa, m3/h


#%%
def MB(t, Cls):
    rETBE = rate_ETBE_Thy(T0, P0, Q0, Cls)*W
    rTBA = rate_TBA_Honk(T0, P0, Q0, Cls)*W
    # rdiB = rate_diB_Honk(T0, P0, Q0, Cls)*W
    rdiB = 0
    return [-rETBE+rTBA-rdiB, -rETBE, -rTBA , +rTBA ,rETBE, rdiB , 0 , 0 , 0 ]
#%%
names = ['Isobutene' , ' Ethanol' , ' Water' , ' TBA' , ' ETBE' , ' Di-B' , ' Tri-B' , ' 1-butene' , ' isobutane']

# %%
def PBR(W, arr):
    "T, P, Q, Isobutene ,  Ethanol ,  Water ,  TBA ,  ETBE ,  Di-B ,  Tri-B ,  1-butene ,  isobutane"
    T, P, Q = arr[:3]
    Fls = arr[3:]
    Cls = [F/Q for F in Fls]
    rETBE = rate_ETBE_Thy(T, P, Q, Cls)
    rTBA = rate_TBA_Honk(T, P, Q, Fls)
    rdiB1 = rate_diB_Honk(T, P, Q, Fls)
    rtriB =rate_TriB_Honk(T, P, Q, Fls)
    # rdiB = 0


    rIB = -rETBE - 2*rdiB1 - rTBA - rtriB
    rdiB = rdiB1 - rtriB
    rEtOH = -rETBE
    # rTriB = 0
    r1B = 0
    riBane = 0
    rw = -rTBA
    

    dT = 0
    dP = 0
    dQ = 0
    rIB, rEtOH, rw , rTBA ,rETBE, rdiB , rtriB , r1B , riBane  = [r*rho_b for r in [rIB, rEtOH, rw , rTBA ,rETBE, rdiB , rtriB , r1B , riBane ]] # mol/s.m3rx
    return [dT,dP,dQ,rIB, rEtOH, rw , rTBA ,rETBE, rdiB , rtriB , r1B , riBane ]

rho_b = 610 # kg/m3 bed
Wtot = 36300
Wspan = np.linspace(0,Wtot/rho_b, 50)
ans = solve_ivp(PBR, [0, Wspan[-1]], [80+273.15, P0, Q0, *F0], dense_output=True).sol(Wspan)
for F, n in zip(ans[3:], names):
    plt.plot(Wspan, F, label = n)
plt.legend(loc = 'best')
plt.show()


# %%


# %%
