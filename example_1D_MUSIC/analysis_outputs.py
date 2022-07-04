import glob, json, h5py, math, time, os
import numpy as np
import matplotlib.pyplot as plt

from NPLM.PLOTutils import *
from NPLM.ANALYSISutils import *

###########################################################

### PUT A SLASH AT THE END OF THE PATH !!! ### 
jobs_folder = '/eos/user/a/aalvesan/ml_test/Nbkg4693_patience1000_epochs4000_arc1_4_1_wclip14/'
out_folder  = '/eos/user/a/aalvesan/ml_test/analysis_outputs/'

json_file = '%sconfig.json'%(out_folder)
with open(json_file, 'r') as jsonfile:
    config_json = json.load(jsonfile)
    
TAU_architecture    = config_json['BSMarchitecture']
TAU_df              = compute_df(input_size=TAU_architecture[0], hidden_layers=TAU_architecture[1:-1])
TAU_wc              = config_json['BSMweight_clipping']
TAU_patience        = config_json['patience']
N_Bkg               = config_json['N_Bkg']

##########################################
#### Collecting jobs to summary files ####
##########################################

values, files_id_tau, seeds_tau = collect_txt(DIR_IN = jobs_folder, suffix = 'TAU', files_prefix = [],  verbose = False) # this function returns 3 arrays: for  t values, labels and seeds 

#print ('')
#print ('tau values after Read_final_from_h5 is : \n')
#print (values)

save_txt_to_h5(values, files_id_tau, seeds_tau, suffix = 'final', DIR_OUT = out_folder, FILE_NAME = 'TAU') # this function saves the 3 arrays above as datasets in a .h5 file 

print '\n values are : \n' 
print values
print '\n files_id are : \n' 
print files_id_tau 
print '\n seeds are : \n' 
print seeds_tau

keys = ['loss'] #'norm_0', 'shape_0']
for key in keys:
    key_history = collect_history(files_id = files_id_tau, DIR_IN = jobs_folder, suffix = 'TAU', key = key, verbose = False) # this function returns a 2D array of the loss history  with shape (nr. toys, nr. check points). i.e. for each toy experiment we collect the t = -2 * loss at specific checkpoints
    print ('key_history.shape is : ' + str(key_history.shape))
    save_history_to_h5(suffix = 'history', patience = TAU_patience, tvalues_check = key_history, DIR_OUT = out_folder, FILE_NAME = 'TAU', seeds=seeds_tau) # this function saves the 2D array above into a .h5 log_file 


##########################################
##### Reading/analysing summary files ####
##########################################

tau, tau_seeds   = Read_final_from_h5(DIR_IN = out_folder, FILE_NAME = 'tau', suffix='_final')      # this function returns 2 arrays: the t values and the seed values 

print ('tau after Read_final_from_h5 is : \n')
print (tau)

tau_history      = Read_history_from_h5(DIR_IN = out_folder, FILE_NAME = 'tau', suffix='_history')  # this function opens the .h5 history log_file from above and creates a 2D array

##### Plotting empirical TAU distribution 

label            = 'B=%i'%(N_Bkg)
plot_1distribution(tau, df=TAU_df, xmin=0, xmax=120, nbins=12, label=r'$\tau(D)$, '+label, save=False, save_path='', file_name='')

##### Plotting TAU distribution's evolution during training time  

Plot_Percentiles_ref(tau_history, df=TAU_df, patience=TAU_patience,  wc=str(TAU_wc), ymax=140, ymin=-100, save=False, save_path='', file_name='')

##### Plotting TAU and TAU - DELTA empirical distributions

label = 'B=%i'%(N_Bkg)

plot_2distribution(tau, tau-delta, df=TAU_df, xmin=0, xmax=120, nbins=16, 
                   label1=r'$\tau(D,\,A), $'+label, label2=r'$\tau(D,\,A)-\Delta(D,\,A)$, '+label,
                   save=False, save_path='', file_name='')

pval = KS_test(tau-delta, dof=TAU_df, Ntoys=100)   # checking asymtotic behaviour
print('')
print('Kolmogorov-Smirnov test | p-value: %f'%(pval))

##### Plotting TAU - DELTA distribution's evolution during training time  

Plot_Percentiles_ref(tau_history-delta_history[:, -1:], df=TAU_df, patience=TAU_patience,  wc=str(TAU_wc), ymax=140, ymin=-100, save=False, save_path='', file_name='')

##### Probability to discover New Physics as function of Z score 

plot_alpha_scores(t1=tau, t2=tau-delta, df=TAU_df, label1=r'$\tau(D,\,A)$', label2=r'$\tau(D,\,A)-\Delta(D,\,A)$', 
                  Zscore_star_list=[2, 3, 4, 5, 6, 7], save=False, save_path='', file_name='') 








