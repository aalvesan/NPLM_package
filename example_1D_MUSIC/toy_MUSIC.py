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
    
seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
np.random.seed(seed)
print('\n Random seed: '+str(seed))

columns_training = config_json["features"]

#### statistics                                                                                                            
N_Bkg      = config_json["N_Bkg"]
N_D        = N_Bkg

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
OUTPUT_PATH    = config_json["output_directory"]
OUTPUT_FILE_ID = '/seed'+str(seed)

# do not run the job if the toy label is already in the folder                                      
if os.path.isfile("%s/%s_t.txt" %(OUTPUT_PATH, OUTPUT_FILE_ID)):
        exit()


#########################################                           
######## Reading input H5 Files ######### 
######################################### 
                                                                                                                                                                   
INPUT_PATH_REF  = '/eos/user/a/aalvesan/ml_test/H5_MC_input/'                              # path to the MC h5 file to serve as input            

feature_dict    = { 'weight_REF': np.array([]),
                    'weight_DATA': np.array([])}

for key in columns_training:
        feature_dict[key] = np.array([])

for classname in ML_Classes:                                                                # ML_Classes is defined in DATAutils.py                                                                          
        f = h5py.File(INPUT_PATH_REF+'H5_'+classname+'.h5', 'r')
        w = np.array(f.get('NewWeights'))                                                   # this is the NewWeights dataset in the MC H5 file                                                               
        feature_dict['weight_REF']  = np.append(feature_dict['weight_REF'],  w)
        feature_dict['weight_DATA'] = np.append(feature_dict['weight_DATA'], w)

        for key in columns_training:
                feature_dict[key] = np.append(feature_dict[key], np.array(f.get(key)))      # reading SumPt amd creating a feature                                                                           
        f.close()
        print('\n%s |  nr. of simulations: %i |  yield: %f'%(classname,w.shape[0], np.sum(w)))

weight_sum_R  = np.sum(feature_dict['weight_REF'])                                          # = sum(NewWeights)                                                          
weight_sum_D  = np.sum(feature_dict['weight_DATA'])                                         # = sum(NewWeights) same as above                            

W_REF         = feature_dict['weight_REF']                                                  # this is just the NewWeights dataset from H5 file          
REF           = np.stack([feature_dict[key] for key in list(columns_training)], axis=1)     # REF gives [[SumPt1, InvMass1, MET1]                     
                                                                                            #            [SumPt2, InvMass2, MET2] ... ] 
                                                                                            # in this case is only [[SumPt1],[SumPt2], ... ]         
print ('\n feacture_dict is : \n')
print (feature_dict)

#####################################################################                                    
####### Hit or Miss Method - Building Pseudo datasets  ##############                                   
#####################################################################                                                                                                                                        

N_REF    = REF.shape[0]                 # number of unweighted MC events i.e. number of 3-element arrays [SumPt, InvMass, MET]              
weight   = feature_dict['weight_DATA']  # same as W_REF, is just the NewWeights dataset                                                          
f_MAX    = np.max(weight)               # selecting the maximum weight                                                       

indices  = np.arange(weight.shape[0])   # returns a list of integeres indices in the range [0, 1, ... , #nr of unweighted events -1]                        
np.random.shuffle(indices)              # everyday I am shuffeling                                                                         
DATA     = np.array([])                 # this is our pseudo dataset. It is just an empty list now. I will be filled later                    
DATA_idx = np.array([],dtype=int)       # DATA_inx will be a list filled with the shuffled indices of the picked up events. e.g. [3, 27, 89, 2, ... ]      

# We want a different nr of events (N_DATA) for each pseudo dataset. We use a Poisson distribution (N_Bkg as expected events) to pick a different  N_DATA each time   
N_DATA   = np.random.poisson(lam=N_Bkg*weight_sum_D/weight_sum_R, size=1)[0]

print('\n'+'cross section effect on N_D: %f'%(weight_sum_D/weight_sum_R))            # this is just = 1                                                                       
print('N_Bkg: '+str(N_Bkg))                                                     # actual number of data events                                                  
print('N_Bkg_Pois: '+str(N_DATA)+'\n')                                               # number of events for this pseudo toy dataset       

if N_REF<N_DATA: # ifnp.delete(REF, DATA_idx, 0)     the reference sample is smaller than the desired pseudo dataset then it is not possible to apply the Hit or Miss method
        print('Cannot produce %i events; only %i available'%(N_DATA, N_REF))
        exit()

counter  = 0                 # the counter will count how many times we apply the Hit or Miss Method to select an event      
counter2 = 0

while DATA.shape[0]<N_DATA:  # filling the pseudo datasets until we have the desired number of pseudo events N_DATA                                                                                          

    if counter>=int(indices.shape[0]) and counter2 < 4 :
        counter = counter - indices.shape
        counter2 +=1

    i = indices[counter] # counter starts at 0 and increases at each loop. i = indices[0]. If indices = [3, 27, 89, 2, ... ], then indices[0]=3                                                          
        
    x = REF[i:i+1, :]    # selects the information from one event e.g. REF[0:1,:] = [[SumPt3, InvMass3, MET3]]                                                                                           
    f = weight[i]        # NewWeight of this specific event weight[3]                                                                                                                                    

    if f<0:                                     # we neglect negative weighted events                                                                                                                    
        DATA_idx = np.append(DATA_idx, i)
        counter+=1                          # go to the next event. i = indices[1] = 27                                                                                                              
        continue

    r = f/f_MAX       # defining the weight ratio between the event weight and the maximum weight in NewWeights                                                                                          

    if r>=1:          # we accept events fulfilling this condition, i.e. this should only happen at the event with w_max                                                                                 
        if DATA.shape[0]==0:
            DATA = x                                  # for the first event we set DATA = x = [SumPt3, InvMass3, MET3]                                                                           
            DATA_idx = np.append(DATA_idx, i)
        else:
            DATA = np.concatenate((DATA, x), axis=0)  # if there are already some events in the pseudo dataset DATA, we just concatenate x                                                       
            DATA_idx = np.append(DATA_idx, i)

    else: # if r < 1                                                                                                                                                                                     
        u = np.random.uniform(size=1) # generating a random number between [0,1[ to be compared with the weight ratio                                                                                

        if u<= r:                     # we only accept events fulfilling this condition, if u>r then the event is rejected                                                                           
            if DATA.shape[0]==0:
                DATA = x
                DATA_idx = np.append(DATA_idx, i)
            else:
                DATA = np.concatenate((DATA, x), axis=0)
                DATA_idx = np.append(DATA_idx, i)
    counter+=1
        
    if counter>=REF.shape[0]:          # when counter = total number of unweighted events, we have looped over the entire dataset once                                                                   
        #print('--> End of file \n')
        N_DATA = DATA.shape[0]     # the final shape of the pseudo dataset is exactly N_DATA that we have choosen above                                                                              
        break

weight  = np.delete(W_REF, DATA_idx, 0)    # we now remove the event weights we picked up above from the full NewWeights dataset                                                                             
REF     = np.delete(REF, DATA_idx, 0)      # we also delete the feature information of these events, i.e. we are deleting the corresponding [[SumPt3, InvMass3, MET3], [SumPt27,InvMass27, MET27, etc]] from the REF dataset                                       

# we now need the reweight the events that were not chosen by the Hit or Miss Method, i.e. they must be increased to compensate for the events that were removed above                                  
weight  = weight * weight_sum_D / np.sum(weight)    #  note that  sum(NewWeights) / sum(weights) > 1                                                                     

#########################################                        
### Preparing feature/target datasets ###                  
#########################################                                                                      

feature = np.concatenate((REF, DATA), axis=0)                                         # we concatenate the redefined REF dataset and the pseudo dataset                                      
weights = np.concatenate((weight, np.ones(DATA.shape[0])), axis=0)                    # we concatenate the reweighted weights with an array of ones [1,1,1...] because the real data also has weight = 1, so our array of ones has shape[0]=lenght of the pseudo dataset                           
target  = np.concatenate((np.zeros(REF.shape[0]), np.ones(DATA.shape[0])), axis=0)    # arrays filled with zeros/ones to define reference label = 0 and psedo data label = 1                
target  = np.stack((target, weights), axis=1)                                         # returns [[0,w][0,w] ... [1,1] [1,1]] i.e. each weight has a label                     

for j in range(feature.shape[1]):          # i.e. j in range(3) because we are using 3 kinematic variables in REF: SumPt, InvMass and MET                                                                    
    vec  = feature[:, j]                   # choosing either SumPt (j=0), InvMass (j=1) or MET (j=2)                                                                        
    #mean = mean_bkg[columns_training[j]]  # columns_training[0]='SumPt' so we will get the value associated with 'SumPt' in mean_bkg in DATAUtils.py    
    #std  = std_bkg[columns_training[j]]   # same as above for std_bkg in DATAUtils.py

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
