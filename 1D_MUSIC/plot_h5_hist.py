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


#### ORGANIZING EVENTS BY PROCESS GROUP ####

process_group_patterns = {}

process_group_patterns['DrellYan'] = ['ZTo*', 'DrellYan*' , 'DY*']
process_group_patterns['ZToInvisible'] = ['ZJetsToNuNu*']
process_group_patterns['W'] = ['WTo*', 'WJets*']
process_group_patterns['Gamma'] = ['GJets*']

process_group_patterns['WZ'] = ['WZTo*']
process_group_patterns['WG'] = ['WGTo*', 'WGJets*']
process_group_patterns['ZG'] = ['ZGTo*']
process_group_patterns['ZZ'] = ['ZZTo*']
process_group_patterns['WW'] = ['WWTo*']
process_group_patterns['GG'] = ['GG_*', 'GGJets_*', 'DiPhotonJets*']

process_group_patterns['ZZZ'] = ['ZZZ_*']
process_group_patterns['WWW'] = ['WWW_*']
process_group_patterns['WWG'] = ['WWG_*']
process_group_patterns['WWZ'] = ['WWZ_*']
process_group_patterns['WGG'] = ['WGG_*']
process_group_patterns['WZG'] = ['WZG_*']
process_group_patterns['WZZ'] = ['WZZ_*']

process_group_patterns['TTbar'] = ['TT_*', 'TTTo*']
process_group_patterns['TTW'] = ['TTWJets*']
process_group_patterns['TTG'] = ['TTGJets*']
process_group_patterns['TTZ'] = ['TTZTo*']
process_group_patterns['TTGG'] = ['TTGG_0Jets*']

process_group_patterns['tG'] = ['TGJets*']
process_group_patterns['tZQ'] = ['tZq_*']
process_group_patterns['Top'] = ['ST_*-channel*', 'ST_tW_*']
process_group_patterns['TTbarTTbar'] = ['TTTT_*']

process_group_patterns['HIG'] = ['GluGluHTo*', 'VBFHTo*', 'VBF_HTo*', 'VHTo*', 'WplusH_HTo*', 'WminusH_HTo*', 'ZH_HTo*', 'ggZH_HTo*', 'ttHTo*']

process_group_patterns['QCD'] = ['QCD_*']


#########################################                           
######## Reading input H5 Files ######### 
######################################### 


INPUT       = '/eos/user/a/aalvesan/NPLM/Input_2017UL/'  # 2016 or 2017UL !!  path to the h5 files directory        
OUTPUT_PATH = '/afs/cern.ch/user/a/aalvesan/private/NPLM_package/1D_MUSIC/music_test2017/plots/'  

columns_training = ['SumPt']

feature_dict     = { 'weight_REF': np.array([]),
                    'processName_REF': np.array([]),
                    'weight_DATA': np.array([])}

print ('\nReading h5 MC from' + str(INPUT) + '\n')

for key in columns_training:
        feature_dict[key]         = np.array([])
        feature_dict[key+'_DATA'] = np.array([])

# ML_Classes is defined in DATAutils.py                                                                          
for classname in ML_Classes:             

    f = h5py.File(INPUT+'H5_'+classname+'.h5',  'r')
    w = np.array(f.get('NewWeights')) 
    p = np.array(f.get('ProcessName')) 
    feature_dict['weight_REF']       = np.append(feature_dict['weight_REF'],  w)
    feature_dict['processName_REF']  = np.append(feature_dict['processName_REF'],  p)
    for key in columns_training:
        feature_dict[key] = np.append(feature_dict[key], np.array(f.get(key)))      # reading SumPt
        f.close()
    
    f = h5py.File(INPUT+'DataH5_'+classname+'.h5',  'r')
    w_d = np.array(f.get('weights')) 
    feature_dict['weight_DATA'] = np.append(feature_dict['weight_DATA'], w_d)
    N_Bkg_data    = w_d.shape[0]

    for key in columns_training:
        feature_dict[key + '_DATA'] = np.append(feature_dict[key + '_DATA'], np.array(f.get(key)))      # reading SumPt
        f.close()


weight_sum_R  = np.sum(feature_dict['weight_REF'])                                          
weight_sum_D  = np.sum(feature_dict['weight_REF'])                                         

W_REF         = feature_dict['weight_REF']                                                
proc_Name_REF = feature_dict['processName_REF']

N_Bkg         = np.sum(W_REF)

REF           = np.stack([feature_dict[key] for key in list(columns_training)], axis=1)
DATA          = np.stack([feature_dict[key + '_DATA'] for key in list(columns_training)], axis=1)

print ('Before cleaning : REF.shape = ' + str(REF.shape) + '  | W_REF.shape[0] = ' + str(W_REF.shape[0])+ ' | sum(W_REF) = ' + str(N_Bkg) + '\n')
print ('W_REF min = ' + str(np.min(W_REF)))
print ('W_REF max = ' + str(np.max(W_REF)) + '\n')

index     = np.arange(W_REF.shape[0])
event_idx = np.array([], dtype = int) 

max_x         = 1350    
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
    if REF[i] >= max_x:
        large_SumPt+=1
        event_idx = np.append(event_idx, i)

print ('Total # of events  : ' + str(w.shape[0]))
print ('Large SumPt events : ' + str(large_SumPt) + ' | percentage :' + str(large_SumPt/w.shape[0]))  
print ('Weights < 0        : ' + str(negative_w)  + ' | percentage : ' + str(negative_w/w.shape[0]))  
print ('Weights > 0.7      : ' + str(large_w)     + ' | percentage :' + str(large_w/w.shape[0]) + '\n')   

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
plt.xticks(fontsize=16, fontname='serif')
plt.yticks(fontsize=16, fontname='serif')
plt.grid()
plt.savefig(OUTPUT_PATH + '/hist_REF_DATA.pdf')
plt.close()



W_REF          = np.delete(W_REF, event_idx, 0)                  # deleting event weights outside the [0,0.7] intreval 
REF            = np.delete(REF  , event_idx, 0)                  # deleting the corresponding events' SumPt 
proc_Name_REF  = np.delete(proc_Name_REF , event_idx, 0)         # deleting the corresponding events' Process Name 

print('process names are')
print(proc_Name_REF)

for i in range(len(proc_Name_REF)):
    for key in process_group_patterns:
        for proc_string in process_group_patterns[key]:
            if proc_string in proc_Name_REF[i]:
                proc_Name_REF[i] = key 

print('\nProcess names turned into groups are')
print(proc_Name_REF)

# we need to reweight the remaining events so that the luminosity is conserved
W_REF = W_REF * weight_sum_R/np.sum(W_REF)     

print ('\nAfter the weight filter  : REF.shape = ' + str(REF.shape) + '  | W_REF.shape[0] = ' + str(W_REF.shape[0])+ ' | sum(W_REF) = ' + str(np.sum(W_REF)) + '\n')
print ('W_REF min = ' + str(np.min(W_REF)))
print ('W_REF max = ' + str(np.max(W_REF)) + '\n')


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
plt.xticks(fontsize=16, fontname='serif')
plt.yticks(fontsize=16, fontname='serif')
plt.grid()
plt.savefig(OUTPUT_PATH + '/hist_REF_TOY.pdf')
plt.close()


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

    if f<0 or f>0.7 or x>=max_x :                # events with f>1 are only 0.05% so it is safe to reject them. 
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

w_sum   = np.sum(weight)

print('»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»')
print('weighted min   : ' + str(np.min(weight)))
print('weighted max   : ' + str(np.max(weight)))
print('')

weight  = np.delete(W_REF, DATA_idx, 0)    # deleting the weights of the events we used for the pseudodataset
REF     = np.delete(REF, DATA_idx, 0)      # deleting the feature information of those events, i.e. we are deleting the corresponding [[SumPt3, InvMass3, MET3], ... ] from the REF dataset         
w_sum_reweighted = np.sum(weight)
# reweighting the left over events to conserve luminosity
weight  = weight * weight_sum_R / w_sum_reweighted

print('weight_sum_R     : ' + str(weight_sum_R))
print('w_sum            : ' + str(w_sum))
print('')
print('w_sum_reweighted : ' + str(w_sum_reweighted))
print('reweighted min   : ' + str(np.min(weight)))
print('reweighted max   : ' + str(np.max(weight)))
print('»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»')


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


bins_code      = {'SumPt': np.arange(0,1, 0.01)}  
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
                  ymax_code=ymax_code,save=True, save_path=OUTPUT_PATH, file_name='/plot_REF_DATA')


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
plt.savefig(OUTPUT_PATH + '/hist_REF_TOY.pdf')
plt.close()


