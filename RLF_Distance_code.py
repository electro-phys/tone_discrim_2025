

#%% file import pre procesing
import pandas as pd
import numpy as np
import io 
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator
from array import *


# frequency and intensity list for later
freq_ls = [1000.,  1500.,  2000.,  3000.,  4000.,  6000.,  8000., 12000.,
       16000., 20000., 24000., 30000., 35000., 40000., 45000., 50000.,
       55000., 60000., 65000.]



db_ls = [0,10,20,30,40,50,60,70,80,90]


ac_spks_1 = pd.read_pickle('ic_spks') # read in the data (can made using my unit extractor program link in README)
ac_spks_2 = pd.read_pickle('ic_spks_2')
ac_spks_3 = pd.read_pickle('ic_spks_3')
ac_spks_4 = pd.read_pickle('ic_last_files_spks5')
ac_spks_5 = pd.read_pickle('ic_missing_files_spks')



ic_tuning = pd.concat([ac_spks_1,ac_spks_2,ac_spks_3,ac_spks_4,ac_spks_5])


def count_range_in_list(ls, min, max): 
    # Initialize a counter 'ctr' to keep track of the count
    # spiks_in_range = len([ele for ele in ls if ele <= max and ele >= min])==1
    spiks_in_range = sum( min <= x <= max for x in ls)
    print(spiks_in_range)
    

    return spiks_in_range


def freq_rlf(tuning_fr,frequency,freq_ls,db_ls,time_low,time_hi,prestim_start):
    bulk_df = pd.DataFrame(columns = ['file','channel','genotype','dB','spike_count_abs','spike_count_prestim','bl_sub_spike_count','freq','trial#'])
    
    # gets the index value for our wanted frequency
    foi = freq_ls.index(frequency) #frequency just needs to be the number in Hz (i.e. 8000)

    for i in tuning_fr['file'].unique(): 
        current_file = tuning_fr.loc[tuning_fr['file'] == i] # get the current recording file 
        if 'WT' in i:
            geno_current = 'WT'
        if 'KO' in i:
            geno_current = 'KO'


        
        for chan in current_file['channel'].unique():
            current_file_chan = current_file.loc[current_file['channel'] == chan]
            spike_ls = current_file_chan. iloc[:, foi]# get only the frequency of interest from this current channel
                
            for x in range(len(db_ls)):
                spike_count_curr_dB = 0;
                prestim_cntr = 0;
                bl_cntr = 0;
                
                dict = {}
                
                all_trials = spike_ls.iloc[x] # this returns an array with 30 trials
                # array of arrays with each array being a trial
                
                print(all_trials)
                
                current_trial = 0
                for trial in all_trials:
                    current_trial = current_trial + 1
                    # trial is an array for PSTH 
                    curr_spike_sum = count_range_in_list(trial,time_low,time_hi) # counts spikes in given time range
                    prestim_spks = count_range_in_list(trial,prestim_start,time_low) # count spikes presti
                    bl_subbed = curr_spike_sum - prestim_spks # subtract baseline

                    spike_count_curr_dB = spike_count_curr_dB + curr_spike_sum # adding current spike count to total spike count for this intensity
                    prestim_count_curr_dB = prestim_cntr + prestim_spks
                    prestim_sec = (prestim_count_curr_dB/0.02)/30
                    bl_count_curr_dB = bl_cntr + bl_subbed
                    spk_sec_abs = (spike_count_curr_dB/0.05)/30
                    spk_sec_bl = (bl_count_curr_dB/0.05)/30
                    
                    
                    
                bulk_df = bulk_df._append({'file': i, 'channel': chan,
                                              'genotype':geno_current,
                                               'dB': db_ls[x],
                                               'spike_count_abs': spike_count_curr_dB,
                                               'spike_count_prestim':prestim_count_curr_dB,
                                               'bl_sub_spike_count' : bl_count_curr_dB,
                                               'spk_sec_abs':spk_sec_abs,
                                               'spk_sec_bl':spk_sec_bl,
                                               'prestim_sec':prestim_sec,
                                               'freq':frequency,
                                               'trial#': current_trial}, ignore_index=True)
                    
    
    return bulk_df


evoked_window_time = 0.05

new1k = freq_rlf(ic_tuning,1000,freq_ls,db_ls,0,evoked_window_time,-.05)
new15k = freq_rlf(ic_tuning,1500,freq_ls,db_ls,0,evoked_window_time,-.05)
new4k = freq_rlf(ic_tuning,4000,freq_ls,db_ls,0,evoked_window_time,-.05)
new8k = freq_rlf(ic_tuning,8000,freq_ls,db_ls,0,evoked_window_time,-.05)
new16k = freq_rlf(ic_tuning,16000,freq_ls,db_ls,0,evoked_window_time,-.05)
new24k = freq_rlf(ic_tuning,24000,freq_ls,db_ls,0,evoked_window_time,-.05)
new30k = freq_rlf(ic_tuning,30000,freq_ls,db_ls,0,evoked_window_time,-.05)
new40k = freq_rlf(ic_tuning,40000,freq_ls,db_ls,0,evoked_window_time,-.05)
new60k = freq_rlf(ic_tuning,60000,freq_ls,db_ls,0,evoked_window_time,-.05)


new4k_wt = new4k.loc[new4k['genotype'] == 'WT']
new8k_wt = new8k.loc[new8k['genotype'] == 'WT']
new16k_wt = new16k.loc[new16k['genotype'] == 'WT']
new24k_wt = new24k.loc[new24k['genotype'] == 'WT']

new30k_wt = new30k.loc[new30k['genotype'] == 'WT']
new40k_wt = new40k.loc[new40k['genotype'] == 'WT']

new60k_wt = new60k.loc[new60k['genotype'] == 'WT']
new_wt = pd.concat([new4k_wt,new8k_wt,new16k_wt,new24k_wt,new30k_wt,new40k_wt,new60k_wt])

new4k_ko = new4k.loc[new4k['genotype'] == 'KO']
new8k_ko = new8k.loc[new8k['genotype'] == 'KO']
new16k_ko = new16k.loc[new16k['genotype'] == 'KO']
new24k_ko = new24k.loc[new24k['genotype'] == 'KO']
new30k_ko = new30k.loc[new30k['genotype'] == 'KO']
new40k_ko = new40k.loc[new40k['genotype'] == 'KO']
new60k_ko = new60k.loc[new60k['genotype'] == 'KO']
new_ko = pd.concat([new4k_ko,new8k_ko,new16k_ko,new24k_ko,new30k_ko,new40k_ko,new60k_ko])

new_prestim = pd.concat([new_wt,new_ko])


thresh_df = pd.read_csv('ic_handscored_thresh_df_curated_final.csv')
cf_df = pd.read_csv('ic_handscored_cf_df_curated_final.csv')


def generate_full_io_dataframe(new_prestim, cf_df):
    # Initialize an empty list to store results
    results = []
    
    # Iterate over every unique combination of file, genotype, and channel in cf_df
    unique_combinations = cf_df[['file', 'genotype', 'channel']].drop_duplicates()
    
    for _, row in unique_combinations.iterrows():
        file = row['file']
        genotype = row['genotype']
        channel = row['channel']
        
        # Step 1: Get the CF for the given combination of file, channel, genotype
        cf_value = cf_df[(cf_df['file'] == file) & 
                         (cf_df['channel'] == channel) & 
                         (cf_df['genotype'] == genotype)]['CF'].values[0]
        
        # Step 2: Filter new_prestim_file based on the current combination
        filtered_data = new_prestim[(new_prestim['file'] == file) &
                                         (new_prestim['genotype'] == genotype) &
                                         (new_prestim['channel'] == channel)]
        
        # Step 3: Group by frequency and dB and calculate the mean spk_sec_abs
        grouped_data = filtered_data.groupby(['freq', 'dB']).agg({'spk_sec_abs': 'mean','prestim_sec':'mean','bl_subbed':'mean','spk_sec_bl':'mean'}).reset_index()
        
        # Add CF value to the dataframe
        grouped_data['file'] = file
        grouped_data['genotype'] = genotype
        grouped_data['channel'] = channel
        grouped_data['CF'] = cf_value
        
        # Append to the results list
        results.append(grouped_data)
    
    # Combine all results into a single dataframe
    full_io_df = pd.concat(results, ignore_index=True)
    
    # Select only the necessary columns (file, genotype, channel, dB, frequency, spk_sec_abs)
    full_io_df = full_io_df[['file', 'genotype', 'channel', 'dB', 'freq', 'spk_sec_abs','prestim_sec','bl_subbed','spk_sec_bl']]
    
    return full_io_df

# Example usage:
new_prestim['bl_subbed'] = new_prestim['spk_sec_abs']-new_prestim['prestim_sec']
full_io_df = generate_full_io_dataframe(new_prestim, cf_df)
full_io_df


# for indiivudal unit graphs

from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, brier_score_loss, r2_score
import numpy as np
import pandas as pd


# Define the sigmoid plus Gaussian model
def sigmoid_gaussian(x, a, b, c, d, e, f):
    return a + (d - a) / (1 + np.exp((b - x) / c)) + e * np.exp(-((x - f) ** 2) / (2 * (c ** 2)))

# Function to calculate RMSE
def calculate_rmse(y_data, y_fit):
    return np.sqrt(mean_squared_error(y_data, y_fit))

# Function to calculate log-likelihood using y_fit
def calculate_log_likelihood(y_data, y_fit):
    epsilon = 1e-10  # Avoid log(0) by adding a small constant
    log_likelihood_value = np.sum(y_data * np.log(y_fit + epsilon) + (1 - y_data) * np.log(1 - y_fit + epsilon))
    return log_likelihood_value

# Define a function to calculate the slope and threshold at 20% of the range
def calculate_slope_and_threshold(params, x_data):
    a, b, c, d, e, f = params
    # Generate a dense set of x values for more accurate min/max detection
    x_fit = np.linspace(min(x_data), max(x_data), 70)
    y_fit = sigmoid_gaussian(x_fit, *params)
    
    y_min = np.min(y_fit)
    y_max = np.max(y_fit)
    y_range = y_max - y_min
    y_threshold = y_min + 0.2 * y_range
    
    # Find the corresponding x value (intensity) where y_threshold is reached
    x_threshold = x_fit[np.argmin(np.abs(y_fit - y_threshold))]
    
    # Slope is the derivative of the sigmoid part at the threshold
    # sigmoid_slope = (d - a) * np.exp((b - x_threshold) / c) / (c * (1 + np.exp((b - x_threshold) / c)) ** 2)

    y_asymp = y_min + 0.8 * y_range
    x_asymp = x_fit[np.argmin(np.abs(y_fit - y_asymp))]

    # calculate slope based on 20 and 80% values of fitted line
    rise = y_asymp - y_threshold
    run = x_asymp - x_threshold
    sigmoid_slope = rise/run

    

    
    return sigmoid_slope, x_threshold, y_min, y_max, x_asymp, y_fit

# Example data
data = new_prestim
data = data[data['file'].str.contains('BA180409C') == False]
data = data[data['file'].str.contains('BA180409B') == False]

# Fit the model for each file, genotype, and channel
files = data['file'].unique()
results = []

for file in files:
    subset_file = data[data['file'] == file]
    genotypes = subset_file['genotype'].unique()
    
    for genotype in genotypes:
        subset_genotype = subset_file[subset_file['genotype'] == genotype]
        channels = subset_genotype['channel'].unique()
        
        for channel in channels:
            subset = subset_genotype[subset_genotype['channel'] == channel]
            x_data = subset['dB'].values
            y_data = subset['spk_sec_abs'].values
            
            # Initial guesses for the parameters
            initial_guess = [min(y_data), np.mean(x_data), np.std(x_data), max(y_data), max(y_data) - min(y_data), np.mean(x_data)]
            
            try:
                popt, pcov = curve_fit(sigmoid_gaussian, x_data, y_data, p0=initial_guess)
                slope, threshold, y_min, y_max, x_asymp, y_fit = calculate_slope_and_threshold(popt, x_data)

                r2 = r2_score(y_data,y_fit)

                
                result = {
                    'file': file,
                    'genotype': genotype,
                    'channel': channel,
                    'params': popt,
                    'slope': slope,
                    'threshold': threshold,
                    'y_min': y_min,
                    'y_max': y_max,
                    'asymptote': x_asymp,
                    'R2': r2
                }
                results.append(result)
                results = results.loc[results['R2'] >= -2] #RSquared cutoff
                results = results.loc[results['slope'] > 0] # need slope to not be flat as no responsive units have this
                results = results.loc[results['y_min'] >= 0] # bad fit would also mean y_min is less than 0 (not possible)
                
            except RuntimeError:
                print(f"Fit could not be performed for file {file}, genotype {genotype}, channel {channel}")

results_df = pd.DataFrame(results)

# Print the results
for result in results:
    print(f"File: {result['file']}, Genotype: {result['genotype']}, Channel: {result['channel']}")
    print(f"  Parameters: {result['params']}")
    print(f"  Slope: {result['slope']}")
    print(f"  Threshold: {result['threshold']}")
    print(f"  Min Y: {result['y_min']}")
    print(f"  Max Y: {result['y_max']}")
    print()



sns.set_context("talk")
sns.set_style("white")
colors = ["#808080","#FF0B04" ]
sns.set_palette(sns.color_palette(colors))
fig1,ax1 = plt.subplots(figsize=(3,5))
sns.boxplot(data=test,x='genotype',y='y_max')
colors = ["#000000","#FF6666" ]
sns.set_palette(sns.color_palette(colors))
sns.stripplot(data=results_df,x='genotype',y='y_max',size=2,palette=colors)
sns.despine()
plt.tight_layout()

sns.set_context("talk")
sns.set_style("white")
colors = ["#808080","#FF0B04" ]
sns.set_palette(sns.color_palette(colors))
fig1,ax1 = plt.subplots(figsize=(3,5))
sns.boxplot(data=test,x='genotype',y='y_min')
colors = ["#000000","#FF6666" ]
sns.set_palette(sns.color_palette(colors))
sns.stripplot(data=results_df,x='genotype',y='y_min',size=2,palette=colors)
sns.despine()
plt.tight_layout()



sns.set_context("talk")
sns.set_style("white")
colors = ["#808080","#FF0B04" ]
sns.set_palette(sns.color_palette(colors))
fig1,ax1 = plt.subplots(figsize=(3,5))
test = results_df.loc[results_df['R2'] >= -2]
sns.boxplot(data=test,x='genotype',y='threshold')
colors = ["#000000","#FF6666" ]
sns.set_palette(sns.color_palette(colors))
sns.stripplot(data=results_df,x='genotype',y='threshold',size=2,palette=colors)
sns.despine()
plt.tight_layout()



sns.set_context("talk")
sns.set_style("white")
colors = ["#808080","#FF0B04" ]
sns.set_palette(sns.color_palette(colors))
fig1,ax1 = plt.subplots(figsize=(3,5))
sns.boxplot(data=results_df,x='genotype',y='slope')
colors = ["#000000","#FF6666" ]
sns.set_palette(sns.color_palette(colors))
sns.stripplot(data=results_df,x='genotype',y='slope',size=2,palette=colors)
sns.despine()
plt.tight_layout()




