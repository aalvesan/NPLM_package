import sys, os, time, datetime, h5py, json, argparse
import numpy as np
from scipy.stats import norm, expon, chi2, uniform, chisquare

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import metrics, losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow import Variable

from NPLM.NNutils import *
from NPLM.PLOTutils import *
from NPLM.DATAutils import *

parser = argparse.ArgumentParser()    
parser.add_argument('-j', '--jsonfile'  , type=str, help="json file", required=True)
args = parser.parse_args()

#########################################                           
######## Setting up parameters ########## 
######################################### 

with open(args.jsonfile, 'r') as jsonfile:
    config_json = json.load(jsonfile)
    
columns_training = config_json["features"]

#### nuisance parameters configuration   
correction= config_json["correction"]
NU_S, NUR_S, NU0_S, SIGMA_S = [], [], [], []
NU_N, NUR_N, NU0_N, SIGMA_N = 0, 0, 0, 0
shape_dictionary_list = []

#### training time                
total_epochs_tau   = config_json["epochs"]
patience_tau       = config_json["patience"]

#### architecture                
BSMweight_clipping = config_json["BSMweight_clipping"]
BSMarchitecture    = config_json["BSMarchitecture"]
inputsize          = BSMarchitecture[0]
BSMdf              = compute_df(input_size=BSMarchitecture[0], hidden_layers=BSMarchitecture[1:-1])

##### define output path ######################
OUTPUT_PATH       = config_json["output_directory"]

for classname in ML_Classes:
    OUTPUT_FILE_ID    = '/Data_'+classname

# do not run the job if the toy label is already in the folder                                      
if os.path.isfile("%s/%s_t.txt" %(OUTPUT_PATH, OUTPUT_FILE_ID)):
        exit()

#########################################                           
######## Reading input H5 Files ######### 
######################################### 
                                                                                                                                                                   
INPUT_MC   = '/eos/user/a/aalvesan/ml_test/H5_MC_input/'                              # path to the MC h5 file to serve as input            
INPUT_DATA = '/eos/user/a/aalvesan/ml_test/H5_DATA_input/'                              # path to the MC h5 file to serve as input            

feature_dict    = { 'weight_REF': np.array([]),
                    'weight_DATA': np.array([])}

for key in columns_training:
        feature_dict[key] = np.array([])

for classname in ML_Classes:                                                                # ML_Classes is defined in DATAutils.py
    
    f = h5py.File(INPUT_MC+'H5_'+classname+'.h5', 'r')
    w = np.array(f.get('NewWeights'))                                                   # this is the NewWeights dataset in the MC H5 file                                                               
    feature_dict['weight_REF']  = np.append(feature_dict['weight_REF'],  w)
        
    for key in columns_training:
        feature_dict[key] = np.append(feature_dict[key], np.array(f.get(key)))      # reading SumPt amd creating a feature             

    f.close()        
    print('\n%s |  nr. of simulations: %i |  yield: %f'%(classname,w.shape[0], np.sum(w)))

    f      = h5py.File(INPUT_DATA+'DataH5_'+classname+'.h5', 'r')
    w_data = np.array(f.get('weights'))                                                   # this is the NewWeights dataset in the MC H5 file                
    feature_dict['weight_DATA'] = np.append(feature_dict['weight_DATA'], w_data)

    for key in columns_training:
        DATA = np.array(f.get(key))[:,np.newaxis]      # reading SumPt amd creating a feature             
    f.close()

    print ('Check that all weights are 1 for DATA : \n')
    print (w_data)

W_REF   = feature_dict['weight_REF']                                                  # this is just the NewWeights dataset from H5 file          
W_DATA  = feature_dict['weight_DATA']                                                 # this is just the NewWeights dataset from H5 file          

print ('W_data.shape is : \n')
print (W_DATA.shape)
                                                                                     
for key in columns_training:
    REF  = np.stack([feature_dict[key] for key in list(columns_training)], axis=1)     # REF gives [[SumPt1, InvMass1, MET1]                     

print ('Check if REF and DATA models are consistent : ')
print ('\n REF is : ')
print (REF)
print ('\n DATA is : ')
print (DATA)

if np.sum(W_DATA == np.ones(W_DATA.shape[0])) != W_DATA.shape[0]:
    print ('--> THERE IS SOME ERROR WITH THE DATA WEIGHTS !! <--')
    exit()

print ('\n New feacture_dict is : \n')
print (feature_dict)

N_DATA = DATA.shape[0]

print('Data events in H5 file : '+str(N_DATA))                                                         # actual number of data events                                                  

#########################################                        
### Preparing feature/target datasets ###                  
#########################################                                                                      

feature = np.concatenate((REF, DATA), axis=0)                                         # we concatenate the SumPt REF dataset and the SumPt_Data                                      
weights = np.concatenate((W_REF, W_DATA), axis=0)                                     # we concatenate the weights_REF with W_DATA, which is just an array of ones [1,1,1...]
target  = np.concatenate((np.zeros(REF.shape[0]), np.ones(DATA.shape[0])), axis=0)    # arrays filled with zeros/ones to define reference label = 0 and data label = 1                
target  = np.stack((target, weights), axis=1)                                         # returns [[0,w][0,w] ... [1,1] [1,1]] i.e. each weight has a label                     

for j in range(feature.shape[1]):          # i.e. j in range(1) because we are using 1 kinematic variables in REF: SumPt

    vec  = feature[:, j]                   # choosing either SumPt (j=0), InvMass (j=1) or MET (j=2)                                                                        
    mean = np.mean(vec)                    # just the arithmetic mean
    std  = np.std(vec)                     # standard deviation of the dataset

    if np.min(vec) < 0:                    # ideally this will never happen (neither SumPt, InvMass nor MET are < 0 )                                                
        vec = vec-mean
        vec = vec*1./ std

    elif np.max(vec) > 1.0:                # Assume data is exponential -- just set mean to 1.                                              
        vec = vec *1./ mean
    feature[:, j] = vec


#########################################                           
############## Training TAU ############# 
######################################### 

print('\n' +'-->  Training TAU ... \n ')
batch_size = feature.shape[0]
input_shape = feature.shape[1]

tau = imperfect_model(input_shape=(None, inputsize),
                      NU_S=NU_S, NUR_S=NUR_S, NU0_S=NU0_S, SIGMA_S=SIGMA_S, 
                      NU_N=NU_N, NUR_N=NUR_N, NU0_N=NU0_N, SIGMA_N=SIGMA_N,
                      correction=correction, shape_dictionary_list=shape_dictionary_list,
                      BSMarchitecture=BSMarchitecture, BSMweight_clipping=BSMweight_clipping, train_f=True, train_nu=True)
print(tau.summary())
tau.compile(loss=imperfect_loss,  optimizer='adam')

t0=time.time()
hist_tau = tau.fit(feature, target, batch_size=batch_size, epochs=total_epochs_tau, verbose=False)
t1=time.time()

print('\n'+'--> End of TAU taining | Training time in seconds:'+str(t1-t0)+'\n')

#########################################
################ OUTPUT #################                     
#########################################

#calculating test statistic tau 
loss_tau  = np.array(hist_tau.history['loss'])                         
final_loss = loss_tau[-1]
tau_OBS    = -2*final_loss
print('--> TAU: %f \n'%(tau_OBS))

# save t                                                                                                               
log_t = OUTPUT_PATH+OUTPUT_FILE_ID+'_TAU.txt'
out   = open(log_t,'w')
out.write("%f\n" %(tau_OBS))
out.close()

# save the training history                                       
log_history = OUTPUT_PATH+OUTPUT_FILE_ID+'_TAU_history.h5'
f           = h5py.File(log_history,"w")
epoch       = np.array(range(total_epochs_tau))
keepEpoch   = epoch % patience_tau == 0
f.create_dataset('epoch', data=epoch[keepEpoch], compression='gzip')
for key in list(hist_tau.history.keys()):
    monitored = np.array(hist_tau.history[key])
    print('%s: %f'%(key, monitored[-1]))
    f.create_dataset(key, data=monitored[keepEpoch],   compression='gzip')
f.close()

# save the model    
log_weights = OUTPUT_PATH+OUTPUT_FILE_ID+'_TAU_weights.h5'
tau.save_weights(log_weights)

#Plot_variable(loss_tau, 'Loss')
