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
                                                                                                                                                                   
INPUT  = '/eos/user/a/aalvesan/ml_test/2017UL_H5/'                              # path to the MC h5 file to serve as input            

feature_dict    = { 'weight_REF': np.array([]),
                    'weight_DATA': np.array([])}

for key in columns_training:
    feature_dict[key] = np.array([])

DATA = np.array([])

# ML_Classes is defined in DATAutils.py
for classname in ML_Classes:                                                                
    
    # Opening the MC classanme.h5 file
    f = h5py.File(INPUT+'H5_'+classname+'.h5', 'r')
    w = np.array(f.get('NewWeights'))                     
    feature_dict['weight_REF']  = np.append(feature_dict['weight_REF'],  w)

    for key in columns_training:
        feature_dict[key] = np.append(feature_dict[key], np.array(f.get(key)))      # e.g. reading SumPt
        print ('')
        print ('Min and Max MC variable values are: '+ str(min(feature_dict[key]))+' | '+ str(max(feature_dict[key])))
        #feature_dict[key] = Normalize(feature_dict[key])

    f.close()

    # Opening the DATA_classname.h5 file
    f      = h5py.File(INPUT+'DataH5_'+classname+'.h5', 'r')
    w_data = np.array(f.get('weights'))                          
    feature_dict['weight_DATA'] = np.append(feature_dict['weight_DATA'], w_data)

    for i, key in enumerate(columns_training):

        if i==0:
            DATA = np.array(f.get(key))
        else:
            DATA = np.c_[DATA,np.array(f.get(key))]
        print ('Min and Max DATA variable values are: '+ str(min(DATA))+' | '+ str(max(DATA))+'\n')

    f.close()
    
    print('-------------------------------------------------------------------------------------------')        
    print('%s |  Unweighted MC events: %i |  yield: %f'%(classname,w.shape[0], np.sum(w)))
    print('-------------------------------------------------------------------------------------------')
    print('')

weight_sum_R  = np.sum(feature_dict['weight_REF'])

W_REF   = feature_dict['weight_REF']                                                  # this is just the NewWeights dataset from H5 file         
W_DATA  = feature_dict['weight_DATA']                                                 # this is just the NewWeights dataset from H5 file

for key in columns_training:
    REF = np.stack([feature_dict[key] for key in list(columns_training)], axis=1)     # REF gives [[SumPt1, InvMass1, MET1]


N_Bkg   = np.sum(W_REF)

print ('Before cleaning : REF.shape = ' + str(REF.shape) + '  | W_REF.shape[0] = ' + str(W_REF.shape[0])+ ' | sum(W_REF) = ' + str(N_Bkg) + '\n')
print ('W_REF min = ' + str(np.min(W_REF)))
print ('W_REF max = ' + str(np.max(W_REF)) + '\n')

index     = np.arange(W_REF.shape[0])
event_idx = np.array([], dtype = int)

negative_w    = 0
large_w       = 0
large_SumPt   = 0

for i in index:
    if W_REF[i] < 0 :
        negative_w+=1
        event_idx = np.append(event_idx, i)
    if W_REF[i] > 0.2 :
        large_w +=1
        event_idx = np.append(event_idx, i)
    if REF[i] >= 1000:
        large_SumPt+=1
        event_idx = np.append(event_idx, i)

print ('Large SumPt events : ' + str(large_SumPt))
print ('Weights < 0        : ' + str(negative_w))
print ('Weights > 0.2      : ' + str(large_w)+ '\n')

# we need to reweight the remaining events so that the luminosity is conserved                                                                                
W_REF = np.delete(W_REF, event_idx, 0)         # deleting weights outside the [0,0.2] intreval                                                                
REF   = np.delete(REF  , event_idx, 0)         # deleting the corresponding SumPt of those weights                                                            
W_REF = W_REF * weight_sum_R/np.sum(W_REF)     # increasing the events weights to conserve sum(NewWeights)                                                    

print ('\nAfter cleaning  : REF.shape = ' + str(REF.shape) + '  | W_REF.shape[0] = ' + str(W_REF.shape[0])+ ' | sum(W_REF) = ' + str(np.sum(W_REF)) + '\n')
print ('W_REF min = ' + str(np.min(W_REF)))
print ('W_REF max = ' + str(np.max(W_REF)) + '\n')

print ('W_Data.shape is : ' + str(W_DATA.shape)+'\n')
print (W_DATA)
print ('')


DATA = DATA[:,np.newaxis]
print ('REF shape is  : ' + str(REF.shape))
print ('DATA shape is : ' + str(DATA.shape)+'\n')

if np.sum(W_DATA == np.ones(W_DATA.shape[0])) != W_DATA.shape[0]:
       print ('--> THERE IS SOME ERROR WITH THE DATA WEIGHTS !! <--')
       exit()

#########################################                        
### Preparing feature/target datasets ###                  
#########################################                                                                      

# we concatenate the SumPt REF dataset and the SumPt_Data. We also concatenate the MC weights and DATA weights (=1) into a single dataset.
feature = np.concatenate((REF, DATA), axis=0)                                         
weights = np.concatenate((W_REF, W_DATA), axis=0)                                     

print ('Before normalization, the feature dataset is : ')
print (feature)
print ('')
# arrays filled with zeros/ones to define reference label = 0 and data label = 1                
target  = np.concatenate((np.zeros(REF.shape[0]), np.ones(DATA.shape[0])), axis=0) 
target  = np.stack((target, weights), axis=1)                               
# target returns [[0,w][0,w] ... [1,1] [1,1]] i.e. each weight has a label                               
'''
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
'''

x_max = np.max(feature)
x_min = np.min(feature)

for j in range(feature.shape[1]):
    vec = feature[:, j]
    vec = (vec - x_min) / (x_max - x_min)
    feature[:,j] = vec


print ('Target shape  : ' + str(target.shape)+'\n')
print (target)
print ('')
print ('After normalization, the feature dataset is : ')
print ('Feature shape : ' + str(feature.shape)+'\n')
print (feature)

print('\nfeature.min   : ' + str(np.min(feature)))
print('feature.max   : ' + str(np.max(feature)))


bins_code      = {'SumPt': np.arange(0.75, 0.01)}
ymax_code      = {'SumPt': 1}
xlabel_code    = {'SumPt': r'$SumPt$',}
feature_labels = list(bins_code.keys())

batch_size     = feature.shape[0]
inputsize      = feature.shape[1]

REF            = feature[target[:, 0]==0]
DATA           = feature[target[:, 0]==1]
weight         = target[:, 1]
weight_REF     = weight[target[:, 0]==0]
weight_DATA    = weight[target[:, 0]==1]

plot_training_data(data=DATA, weight_data=weight_DATA, ref=REF, weight_ref=weight_REF,
                   feature_labels=feature_labels, bins_code=bins_code, xlabel_code=xlabel_code,
                   ymax_code=ymax_code,save=True, save_path='/eos/user/a/aalvesan/ml_test/OUTPUTS/2017UL_1Ele_1Muon_1MET/analysis_outputs/', file_name='Training_REF_DATA_CMS')


fig = plt.figure(figsize=(9,6))
fig.patch.set_facecolor('white')

bins = np.arange(0,1,0.01)

plt.hist(DATA[:,0], weights = weight_DATA , bins = bins, label = 'DATA', color='black'  , lw = 1.5, histtype='step', zorder=4)
plt.hist(REF[:, 0], weights = weight_REF  , bins = bins, label = 'REF' , color='#a6cee3', lw = 1  , ec='#1f78b4')
plt.yscale('log')

font=font_manager.FontProperties(family='serif', size=18)
plt.legend(prop=font)
plt.ylabel('Events', fontsize=18, fontname='serif')
plt.xlabel('SumPt' , fontsize=18, fontname='serif')
#plt.xlim(0,0.2)                                                                                                                                              
plt.xticks(fontsize=16, fontname='serif')
plt.yticks(fontsize=16, fontname='serif')
plt.grid()
plt.savefig('Training_REF_DATA_CMS.pdf')
plt.close()

#########################################                           
############## Training TAU ############# 
######################################### 
print('')
print('-------------------------------------------------------------------------------------------')
print('')
print('-->  Training TAU ... \n ')
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

print('')
print('--> End of TAU taining | Training time in seconds:'+str(t1-t0)+'\n')

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
print('-------------------------------------------------------------------------------------------\n')

fig = plt.figure(figsize=(9,6))                                                                                                                              \

fig.patch.set_facecolor('white')
plt.plot(loss_tau, label='tau')
font=font_manager.FontProperties(family='serif', size=18)
plt.legend(prop=font)
plt.ylabel('Loss', fontsize=18, fontname='serif')
plt.xlabel('Epochs', fontsize=18, fontname='serif')
plt.xticks(fontsize=16, fontname='serif')
plt.yticks(fontsize=16, fontname='serif')
plt.grid()
plt.savefig('Loss_DATA_CMS.pdf')
plt.close()
