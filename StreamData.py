#%%
from pandas import read_csv
import numpy as np
comp = read_csv('InletStreamData.csv', nrows=18, skiprows=1, names = ['Compound', 'unit', 'x'])
streamdf = read_csv('InletStreamData.csv', nrows=23, skiprows=18, names=['Property', 'unit', 'value'])
Q0 = 
def x_to_F(xarr, Ftot):
    return [np.round(float(x*Ftot['value']), 5) for x in xarr['x']]

#%%
comp['F'] = x_to_F(comp, streamdf[2:3])
comp = comp[comp['F'] != 0]
comp.index = np.arange(0, len(comp))
comp['shortname'] = ['IB_ane', 'IB', '1B', 'B_diene', 'NB_ane', 'trans_B', 'cis_B', 'water', 'EtOH', 'TBA', 'ETBE']



# %%
C_F = comp[['Compound', 'shortname' ,'F']]
C_F


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
CF_dict.keys()
# %%
