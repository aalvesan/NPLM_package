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
#print('Random seed: '+str(seed))

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
OUTPUT_PATH    = config_json["output_directory"]
OUTPUT_FILE_ID = '/seed'+str(seed)

# do not run the job if the toy label is already in the folder                                      
if os.path.isfile("%s/%s_t.txt" %(OUTPUT_PATH, OUTPUT_FILE_ID)):
        exit()

#########################################                           
######## Reading input H5 Files ######### 
######################################### 

INPUT = '/eos/user/a/aalvesan/ml_test/2016_H5/'                     # path to the h5 files directory        

feature_dict    = { 'weight_REF': np.array([]),
                    'weight_DATA': np.array([])}

print ('\nReading h5 MC and DATA files from' + str(INPUT) + '\n')

for key in columns_training:
        feature_dict[key] = np.array([])

# ML_Classes is defined in DATAutils.py                                                                          
for classname in ML_Classes:             

    f = h5py.File(INPUT+'H5_'+classname+'.h5',  'r')
    w = np.array(f.get('NewWeights')) 
    feature_dict['weight_REF']  = np.append(feature_dict['weight_REF'],  w)
    feature_dict['weight_DATA'] = np.append(feature_dict['weight_DATA'], w)

    for key in columns_training:
        feature_dict[key] = np.append(feature_dict[key], np.array(f.get(key)))      # reading SumPt
        f.close()

    '''
    f = h5py.File(INPUT+'DataH5_'+classname+'.h5',  'r')
    w = np.array(f.get('weights')) 
    N_Bkg_data    = w.shape[0]
    f.close()
    '''

    print('----------------------------------------------------------------------------------')
    print('%s |  nr. of simulations: %i |  yield: %f'%(classname,w.shape[0], np.sum(w)))
    print('----------------------------------------------------------------------------------')
    print('')

weight_sum_R  = np.sum(feature_dict['weight_REF'])                                          
weight_sum_D  = np.sum(feature_dict['weight_REF'])                                         

W_REF         = feature_dict['weight_REF']                                                  # this is just the NewWeights dataset from H5 file
REF           = np.stack([feature_dict[key] for key in list(columns_training)], axis=1)     # REF gives [[SumPt1, InvMass1, MET1]                                                                                                                                                                                                        #            [SumPt2, InvMass2, MET2] ... ] 

N_Bkg         = np.sum(W_REF)

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
    if W_REF[i] > 0.7 :
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


#####################################################################                                    
####### Hit or Miss Method - Building Pseudo datasets  ##############                                   
#####################################################################

print('******** Hit or Miss Sampling Procedure ********\n')

weight   = W_REF                                
indices  = np.arange(weight.shape[0])           # attribute an index to each weight
np.random.shuffle(indices)                      # every day I am shuffeling 

DATA     = np.array([])                         # this will be our pseudo dataset with SumPt information from the chosen events
DATA_idx = np.array([], dtype = int)            # to be filled with the indices of chosen events 

N_REF    = REF.shape[0]                         # number of reference events
f_ref    = 0.9                                  # reference weight; should be larger than f_max
                                                                          
print('Chosen f_ref is : %f'%(f_ref))

# We want a different nr of events (N_DATA) for each pseudo dataset. We use a Poisson distribution (N_Bkg as expected events) to pick a different  N_DATA each time   
N_DATA   = np.random.poisson(lam=N_Bkg*weight_sum_D/weight_sum_R, size=1)[0]

print('cross section effect on N_D: %f'%(weight_sum_D/weight_sum_R))                 # this is just = 1             
print('N_Bkg: '+str(N_Bkg))                                                          # actual number of data events     
print('N_Bkg_Pois: '+str(N_DATA)+'\n')                                               # number of events for this pseudo toy dataset       

if N_REF<N_DATA: # if the reference sample is smaller than the desired pseudo dataset then it is not possible to apply the Hit or Miss method
        print('Cannot produce %i events; only %i available'%(N_DATA, N_REF))
        exit()

counter  = 0                 # the counter will count how many times we apply the Hit or Miss Method to select an event      
#counter2 = 0
rejected = 0

while DATA.shape[0]<N_DATA:  # filling the pseudo datasets until we have the desired number of pseudo events N_DATA                               

    #if counter>=int(indices.shape[0]) and counter2 < 1 :  # we allow to loop over the MC dataset twice 
     #   counter = counter - indices.shape[0]
      #  counter2 +=1

    i = indices[counter]     # counter starts at 0 and increases at each loop. If indices = [3, 27, 89, 2, ... ] then i = indices[0] = 3, etc ...
    x = REF[i:i+1, :]        # selects the information from one event e.g. REF[0:1,:] = [[SumPt3, InvMass3, MET3]]
    f = weight[i]

    if f<0 or f>0.7 or x>=1000 :                # events with f>1 are only 0.05% so it is safe to reject them. 
        DATA_idx = np.append(DATA_idx, i)
        counter+=1                                    # go to the next event. i = indices[1] = 27    
        rejected+=1
        continue 

    r = f/f_ref                                       # defining the weight ratio between the event weight and the maximum weight in NewWeights 

    if r>=1:                                          # we accept events fulfilling this condition, i.e. this should only happen at the event with w_max      
        if DATA.shape[0]==0:
            DATA = x                                  # if DATA is still empty, we simply set DATA = x (x is then the first accepted event) 
            DATA_idx = np.append(DATA_idx, i)
        else:
            DATA = np.concatenate((DATA, x), axis=0)  # if there are already some events in DATA, we just concatenate x                     
            DATA_idx = np.append(DATA_idx, i)

    else: 
        u = np.random.uniform(size=1)                 # generating a random number between [0,f_MAX[ to be compared with the weight ratio      
        if u<= r:                                     # we only accept events fulfilling this condition
            if DATA.shape[0]==0:
                DATA = x
                DATA_idx = np.append(DATA_idx, i)
                #print ('Event was accepted ! ')
            else:
                DATA = np.concatenate((DATA, x), axis=0)
                DATA_idx = np.append(DATA_idx, i)
                #print ('Event was accepted ! ')

    #print ('counter : ' + str(counter) + ' | i index : ' + str(i) + ' | Number of pseudo events, N_DATA : ' + str(DATA.shape[0]) +'\n')
    counter+=1
    
    if counter>=REF.shape[0]:          # when counter = total number of unweighted events, we have looped over the entire dataset once  
        print('\n--> End of file ! We have looped over all MC events !')
        #N_DATA = DATA.shape[0]         # the final shape of the pseudo dataset is exactly N_DATA that we have choosen above                 
        break

weight  = np.delete(W_REF, DATA_idx, 0)    # deleting the weights of the events we used for the pseudodataset
REF     = np.delete(REF, DATA_idx, 0)      # deleting the feature information of those events, i.e. we are deleting the corresponding [[SumPt3, InvMass3, MET3], ... ] from the REF dataset         

# reweighting the left over events to conserve luminosity
weight  = weight * weight_sum_R / np.sum(weight)

print ('After the HitMiss method we have that : \n' )

print('Events in the toy dataset : '+str(DATA.shape[0]))
print('Rejected events  : '+ str(rejected))
print('Deleted events from the REF dataset : ' + str(DATA_idx.shape[0])+'\n')

print('REF.shape  : ' + str(REF.shape))
print('REF.min    : ' + str(np.min(REF)))
print('REF.max    : ' + str(np.max(REF))+'\n')

print('DATA.shape : ' + str(DATA.shape))
print('DATA.min   : ' + str(np.min(DATA)))
print('DATA.max   : ' + str(np.max(DATA)))

#########################################                        
### Preparing feature/target datasets ###                  
#########################################                                 

# we concatenate the redefined REF dataset and the pseudo dataset. We also concatenate the MC weights and the data weights(=1) into a single dataset        

feature = np.concatenate((REF, DATA), axis=0)                                         
weights = np.concatenate((weight, np.ones(DATA.shape[0])), axis=0)                    

# we concatenate the reweighted weights with an array of ones [1,1,1...] because the real data also has weight = 1, so our array of ones has shape[0]=lenght of the pseudo dataset

target  = np.concatenate((np.zeros(REF.shape[0]), np.ones(DATA.shape[0])), axis=0)    # arrays filled with zeros/ones to define reference label = 0 and psedo data label = 1
target  = np.stack((target, weights), axis=1)                                         # returns [[0,w][0,w] ... [1,1] [1,1]] i.e. each weight has a label  

'''
for j in range(feature.shape[1]):          # i.e. j can be in range(3) if we use all 3 kinematic variables in columns_training in the config.json file. Here we only use SumPt so automatically j=0.

    vec  = feature[:, j]                   # SumPt (j=0), InvMass (j=1) or MET (j=2)     
    mean = np.mean(vec)                    # just the arithmetic mean
    std  = np.std(vec)                     # standard deviation of the dataset
    print('\nNormalizing feature dataset with mean = ' +str(mean) + ' and standard dev. = ' + str(std)+'\n')

    if np.min(vec) < 0:                    # ideally this will never happen (neither SumPt, InvMass nor MET are < 0 )
        vec = vec-mean
        vec = vec*1./ std

    elif np.max(vec) > 1.0:                # Assume data is exponential -- just set mean to 1.                                              
        vec = vec *1./ mean
    
    feature[:, j] = vec
'''

# Normalizind the feature dataset to [0,1] intreval 
x_max = np.max(feature)
x_min = np.min(feature)

for j in range(feature.shape[1]):
    vec = feature[:, j]
    vec = (vec - x_min) / (x_max - x_min)
    feature[:,j] = vec


print('\nfeature.shape : ' + str(feature.shape))
print('target.shape  : ' + str(target.shape)+'\n')
print(target)
print ('')

print('\nfeature.min   : ' + str(np.min(feature)))
print('feature.max   : ' + str(np.max(feature)))


bins_code      = {'SumPt': np.arange(0,1, 0.05)}  
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
                   ymax_code=ymax_code,save=True, save_path='/eos/user/a/aalvesan/ml_test/', file_name='Training_REF_DATA_NPLM')


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
plt.savefig('Training_REF_DATA.pdf')
plt.close()

#########################################                           
############## Training TAU ############# 
######################################### 

print('')
print('----------------------------------------------------------------------------------')
print('-->  Training TAU ... \n ')

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

print('----------------------------------------------------------------------------------')
print('')

fig = plt.figure(figsize=(9,6))                                                                                                                                            
fig.patch.set_facecolor('white') 
plt.plot(loss_tau, label='tau')
font=font_manager.FontProperties(family='serif', size=18)
plt.legend(prop=font)
plt.ylabel('Loss', fontsize=18, fontname='serif')
plt.xlabel('Epochs', fontsize=18, fontname='serif')
plt.xticks(fontsize=16, fontname='serif')
plt.yticks(fontsize=16, fontname='serif')
plt.grid()
plt.savefig('Loss_toy.pdf')
plt.close()
