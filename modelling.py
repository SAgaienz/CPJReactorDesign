#%%
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Rates import rate_ETBE_Thy, rate_TBA_Honk, rate_diB_Honk

tspan = np.linspace(0, 12600, 1000)
W = 2 #kg
# print(np.format_float_scientific(W))
# intial conditions
FIB, FEt, Fw, FTBA, FETBE, FDIB, FTRIB, F1B, FIBane = F0 = [F*(1000/3600) for F in  [52.255477, 58.885664, 21.650439, 7.936977, 14.89919, 0, 0 , 20.322345 , 6.7940117]] ## all in ]kmol/h
Q = 15 #m3/h
C_IB, C_Et, C_w, C_TBA, C_ETBE, C_DIB, C_TRIB, C_1B, C_IBane = C0 = [(F/Q) for F in F0] # all in kmol/m3 or mol/dm3
T0, P0, Q0 = [343, 1600, 15] # K, kPa, m3/h
def MB(t, Cls):
    rETBE = rate_ETBE_Thy(T0, P0, Q0, Cls)*W
    rTBA = rate_TBA_Honk(T0, P0, Q0, Cls)*W
    rdiB = rate_diB_Honk(T0, P0, Q0, Cls)*W
    return [-rETBE+rTBA-rdiB, -rETBE, -rTBA , +rTBA ,rETBE, rdiB , 0 , 0 , 0 ]
#%%
# Cls = solve_ivp(MB, [0, 12600], C0, dense_output=True).sol(tspan)
names = ['Isobutene', 'Ethanol', 'Water', 'TBA', 'ETBE', 'Di-B', 'Tri-B', '1-butene', 'isobutane']
# #%%
# for C, n in zip(Cls, names):
#     plt.plot(tspan, C, label = n)
# plt.legend(loc = 'best')
# plt.show()

# %%
def PBR(W, arr):
    T, P, Q = arr[:3]
    Fls = arr[3:]
    Cls = [F/Q for F in Fls]
    rETBE = rate_ETBE_Thy(T, P, Q, Cls)
    rTBA = rate_TBA_Honk(T, P, Q, Fls)
    rdiB = rate_diB_Honk(T, P, Q, Fls)
    return [0,0,0,-rETBE-rTBA-2*rdiB, -rETBE, -rTBA , rTBA ,rETBE, rdiB , 0 , 0 , 0 ]
Wspan = np.linspace(0, 60000, 100)
ans = solve_ivp(PBR, [0, Wspan[-1]], [350, P0, Q0, *F0], dense_output=True).sol(Wspan)[3:]
for F, n in zip(ans, names):
    plt.plot(Wspan, F, label = n)
plt.legend(loc = 'best')
plt.show()


# %%


# %%
