#%%
from pandas import read_csv
import numpy as np
from thermo import Chemical
# data_dir = 'StreamData/7.5barStreamData.csv'
data_dir = 'StreamData/16barStreamData.csv'

comp = read_csv(data_dir, nrows=18, skiprows=1, names = ['Compound', 'unit', 'x'])
streamdf = read_csv(data_dir, nrows=23, skiprows=19, names=['Property', 'unit', 'value'])
# Q0 = streamdf
def x_to_F(xarr, Ftot):
    return [np.round(float(x*Ftot['value']), 7) for x in xarr['x']]

#%%
comp['F'] = x_to_F(comp, streamdf[1:2])
comp = comp[comp['F'] != 0]
comp.index = np.arange(0, len(comp))
comp['shortname'] = names = ['IB_ane', 'IB', '1B', 'B_diene', 'NB_ane', 'trans_B', 'cis_B', 'water', 'EtOH', 'TBA', 'ETBE']

# %%
C_F = comp[['Compound', 'shortname' ,'F']]
C_F = C_F.append({'Compound': "2,4,4-trimethyl-2-pentene", 'shortname': 'di_IB', 'F':0}, ignore_index = True)
C_F = C_F.append({'Compound': "4,4-dimethyl-2-neopentyl-1-pentene", 'shortname': 'tri_IB', 'F':0}, ignore_index = True)

# %%
def packer(df):
    full_name = df['Compound'].tolist()
    short_name = df['shortname'].tolist()
    Fls = df['F'].tolist()
    a = {}
    for fn, sn, F in zip(full_name, short_name, Fls):
        a[sn] = [fn, F]
    return a
CF_dict = packer(C_F)
# %%
sn_ls = []
fn_ls = []    
F0 = []
for sn, (fn, v) in CF_dict.items():
    F0.append(v)
    sn_ls.append(sn)
    fn_ls.append(fn)
# %%
mt0, Ft0, Q0, P0 = streamdf.loc[0], streamdf.loc[1], streamdf.loc[2], streamdf.loc[4]

def init_cond():
    return mt0['value'], Ft0['value'], Q0['value'], P0['value'] 

def reacted_filter(ls):
    return [ls[1], *ls[7:13]]

### determine Pmin ###
def phase_check(T, P, f_ls = fn_ls, s_ls = sn_ls):
    P = P*1e3
    P_ls = [Chemical(f, T, P).phase for f in fn_ls] 
    return P_ls
def DeterminPmin(Tmax = 393.15, Ps = 2836.325, n = 1):
    a = True
    Pmin = Ps
    while a:
        n_g = phase_check(Tmax, Pmin).count('g')
        if n_g == 0:
            a = False
        else:
            Pmin = Pmin+n
    return Pmin

P_min_120C = DeterminPmin()

fnr_ls = reacted_filter(fn_ls)
snr_ls = reacted_filter(sn_ls)