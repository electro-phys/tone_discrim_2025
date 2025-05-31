import matplotlib.pyplot as plt
import sys
import pyspike as spk
import pandas as pd
import numpy as np

import os 
import math 
import matplotlib.pyplot as plt
import seaborn as sns


ac_spks = pd.read_pickle('ic_spks_full.pkl')
freq_ls = [1000.,  1500.,  2000.,  3000.,  4000.,  6000.,  8000., 12000.,
    16000., 20000., 24000., 30000., 35000., 40000., 45000., 50000.,
    55000., 60000., 65000.]
db_ls = [0,10,20,30,40,50,60,70,80,90]
cf_df = pd.read_csv('ic_cf_df.csv')
thresh_df = pd.read_csv('ic_thresh_df.csv')


def get_distance_for_CF(CF_cell, current_cell, edges, sanity):
    # time_range in form (-20,100)

    #spike_train = spk.SpikeTrain(np.array([0.1, 0.3, 0.45, 0.6, 0.9], [0.0, 1.0]))
    spike_trains = []

    for train in CF_cell: # trains 1-30 are for the CF_cell
        current_spike_train = spk.SpikeTrain(spike_times=train, edges=edges)
        spike_trains.append(current_spike_train) # need a list of spike trains

    for train in current_cell: # trains 31-60 are for the non_cf_cell
        current_spike_train = spk.SpikeTrain(spike_times=train, edges=edges)
        spike_trains.append(current_spike_train) # need a list of spike trains

    ri_spike_dist = spk.spike_distance(spike_trains, RI=True)
    spike_distance_mat = spk.spike_distance_matrix(spike_trains)

    if sanity == 'yes':
        plt.figure(figsize=(5,5))
        plt.imshow(spike_distance_mat, interpolation='none')
        plt.title("SPIKE-distance")

    return ri_spike_dist, spike_distance_mat


def get_genotype(filename):
    if 'KO' in filename:
        return 'KO'
    elif 'WT' in filename:
        return 'WT'
    else:
        return None

def add_distance_col_and_df(raw_train_df, db_ls, freq_ls, edges, cf_df, thresh_df, sanity):
    form_spk_df = pd.DataFrame(columns=['spike_train','spike_distance','spike_synch','file','channel','genotype'])
    intensity = db_ls

    for i in raw_train_df['file'].unique(): # need to group by file and by channel
        current_file = raw_train_df.loc[raw_train_df['file'] == i]
        
        for j in current_file['channel'].unique():
            current_unit = current_file.loc[current_file['channel'] == j]
            current_geno = current_unit['Genotype'] # grab the current genotype

            current_unit = current_unit[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]

            for index, row in current_unit.iterrows():
                try:
                    current_int = current_unit.iloc[index]
                    current_db = db_ls[index] # get current intensity
                    
                    thresh = thresh_df.loc[(thresh_df['file'] == i) & (thresh_df['channel'] == j)]

                    thresh = thresh['threshold'].iloc[0]
                    
                    rel_db = current_db - thresh

                    CF = cf_df.loc[(cf_df['file'] == i) & (cf_df['channel'] == j)]

                    CF = CF['CF'].iloc[0]
                    
                    cf_ind = freq_ls.index(CF)

                    CF_cell = current_int[cf_ind]

                    for series_name, series in current_int.items(): # loop over kHz
                        current_freq = freq_ls[series_name]  # get Hz freq  
                        current_cell = np.array(current_int[series_name]) # get current cell

                        CF_octs = auto_tune.calc_octaves(CF, current_freq)

                        distance_cell = get_distance_for_CF(CF_cell=CF_cell, current_cell=current_cell, edges=edges, sanity=sanity)

                        form_spk_df = form_spk_df._append({
                            'spike_train': current_cell,
                            'freq': current_freq,
                            'cf_octaves': CF_octs,
                            'intensity': current_db,
                            'rel_intensity': rel_db,
                            'dist_matrix': distance_cell[1],
                            'ri_distance': distance_cell[0],
                            'file': i,
                            'channel': j,
                            'genotype': current_geno
                        }, ignore_index=True)
                except Exception as e:
                    print(f"Skipping channel {j} in file {i} due to error: {e}, row: {row}")

    return form_spk_df


acx_cf_dist_df = add_distance_col_and_df(ac_spks,db_ls,freq_ls,edges=(0.0,0.05),cf_df=cf_df,thresh_df=thresh_df,sanity='no')
acx_data = acx_cf_dist_df



acx_data['genotype'] = acx_data['file'].apply(lambda x: get_genotype(x))

acx_data_test = acx_data.loc[acx_data['rel_intensity'] == 40]
acx_data_test1 = acx_data_test.loc[acx_data_test['genotype'] == 'WT']
acx_data_test2 = acx_data_test.loc[acx_data_test['genotype'] == 'KO']

acx_data_test['cf_abs'] = abs(acx_data_test['cf_octaves'])

pivot_df = acx_data_test[acx_data_test['cf_abs'] == 0].pivot_table(
    index=['file', 'channel', 'genotype'], values='ri_distance'
).reset_index().rename(columns={'ri_distance': 'ri_norm'})

acx_data_test = acx_data_test.merge(pivot_df, on=['file', 'channel', 'genotype'],suffixes=('','_norm'))

acx_data_test['normalized_data'] = acx_data_test['ri_distance'] / acx_data_test['ri_norm']


acx_data_test_g = acx_data_test.loc[(acx_data_test['cf_abs'] >= 0.0) & (acx_data_test['cf_abs'] <= 0.99)]


acx_data_test_g = (acx_data_test_g).assign(Bin=lambda x: pd.cut(x.cf_abs, bins=4)).groupby(['Bin','file','channel','genotype']).agg({'ri_distance': 'mean','normalized_data':'mean'})
acx_data_test_g = acx_data_test_g.reset_index()

colors = ["#FF0B04" ,"#000000"]
sns.set_palette(sns.color_palette(colors))
fig1,ax1 = plt.subplots(figsize=(5,5))

sns.pointplot(data=acx_data_test_g,x='Bin',y='normalized_data',hue='genotype',dodge=True)

ax1.set_xticklabels(['0','1/3','2/3','1'])
ax1.set_ylabel('Spike Distance')
ax1.set_xlabel('Octaves from CF')
ax1.set_title('IC tuning dissimilarity')

#plt.legend([],[], frameon=False)

sns.despine()
plt.show()
