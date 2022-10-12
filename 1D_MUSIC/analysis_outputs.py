import glob, json, h5py, math, time, os, argparse
import numpy as np
import matplotlib.pyplot as plt

from NPLM.PLOTutils import *
from NPLM.ANALYSISutils import *

##########################################

parser = argparse.ArgumentParser()
parser.add_argument('-jobs', '--jobs', type = str, help = "folder where the t values .txt files are stored." , required=True )
parser.add_argument('-out' , '--out' , type = str, help = "chose a folder to store the outputs."             , required=True )
args = parser.parse_args()

jobs_folder = args.jobs
out_folder  = args.out

json_file = '%sconfig.json'%(jobs_folder)
with open(json_file, 'r') as jsonfile:
    config_json = json.load(jsonfile)
    
TAU_architecture    = config_json['BSMarchitecture']
TAU_df              = compute_df(input_size=TAU_architecture[0], hidden_layers=TAU_architecture[1:-1])
TAU_wc              = config_json['BSMweight_clipping']
TAU_patience        = config_json['patience']

N_Bkg               = 6076    #1Ele_1Muon_1MET 2017UL 6076  2016 4444  |  

##########################################
#### Collecting jobs to summary files ####
##########################################

# creating 3 arrays: for  t values, labels and seeds 
values, files_id_tau, seeds_tau = collect_txt(DIR_IN = jobs_folder, suffix = 'TAU', files_prefix = [],  verbose = False) 
# saving the arrays in a TAU_final.h5 file
save_txt_to_h5(values, files_id_tau, seeds_tau, suffix = 'final', DIR_OUT = out_folder, FILE_NAME = 'TAU') 

keys = ['loss'] #'norm_0', 'shape_0']
for key in keys:
    # creating a loss array with shape (nr. toys, nr. check points). i.e. for each toy experiment we collect t = -2 * loss at specific checkpoints
    key_history = collect_history(files_id = files_id_tau, DIR_IN = jobs_folder, suffix = 'TAU', key = key, verbose = False) 
    # saving the array in a TAU_history.h5 file
    save_history_to_h5(suffix = 'history', patience = TAU_patience, tvalues_check = key_history, DIR_OUT = out_folder, FILE_NAME = 'TAU', seeds=seeds_tau) 


##########################################
##### Reading/analysing summary files ####
##########################################

label            = 'B=%i'%(N_Bkg)
# reading the file TAU_final.h5 and returning 2 arrays inside
tau, tau_seeds   = Read_final_from_h5(DIR_IN = out_folder, FILE_NAME = 'TAU', suffix='_final')      
# reading the file TAU_history.h5 and returning the loss array inside
tau_history      = Read_history_from_h5(DIR_IN = out_folder, FILE_NAME = 'TAU', suffix='_history')  

##### Plotting empirical TAU distribution 
#plot_1distribution(tau, df=TAU_df, xmin=-50, xmax=150, nbins=60, label=r'$\tau(D)$, '+label, save=True, save_path=out_folder, file_name='TAU')

##### Plotting TAU distribution's evolution during training time  
#Plot_Percentiles_ref(tau_history, df=TAU_df, patience=TAU_patience,  wc=str(TAU_wc), ymax=140, ymin=-100, save=True, save_path=out_folder, file_name='PercentilesRef')

##### Plotting TAU and TAU - DELTA(=0) empirical distributions

#plot_2distribution(tau, tau, df=TAU_df, xmin=0, xmax=150, nbins=60, 
#                   label1=r'$\tau(D,\,A), $'+label, label2=r'$\tau(D,\,A)-\Delta(D,\,A)$, '+label,
#                   save=True, save_path=out_folder, file_name='TAU_DELTA')


pval = KS_test(tau, dof=TAU_df, Ntoys=50)   # checking asymtotic behaviour with CHI^2 function. Don't forget to change Ntoys as necessary !! 
print('') 
print('------ Kolmogorov-Smirnov test ------')
print('') 


##### Probability to discover New Physics as function of Z score 
#plot_alpha_scores(t1=tau, t2=tau, df=TAU_df, label1=r'$\tau(D,\,A)$', label2=r'$\tau(D,\,A)-\Delta(D,\,A)$', 
#                  Zscore_star_list=[2, 3, 4, 5, 6, 7], save=True, save_path=out_folder, file_name='AlphaScores')

