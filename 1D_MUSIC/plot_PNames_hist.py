import numpy as np
import h5py, glob
import math
import uproot
import os
import ROOT
import argparse
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

#data_labels=['SingleMuon','SingleElectron','SinglePhoton','DoubleMuon','DoubleEG']

#### ORGANIZING EVENTS BY PROCESS GROUP ####

process_group_patterns = {}

process_group_patterns['DrellYan'] = ['ZTo', 'DrellYan' , 'DY']
process_group_patterns['ZToInvisible'] = ['ZJetsToNuNu']
process_group_patterns['W'] = ['WTo', 'WJets']
process_group_patterns['Gamma'] = ['GJets']

process_group_patterns['WZ'] = ['WZTo']
process_group_patterns['WG'] = ['WGTo', 'WGJets']
process_group_patterns['ZG'] = ['ZGTo']
process_group_patterns['ZZ'] = ['ZZTo']
process_group_patterns['WW'] = ['WWTo']
process_group_patterns['GG'] = ['GG_', 'GGJets_', 'DiPhotonJets']

process_group_patterns['ZZZ'] = ['ZZZ_']
process_group_patterns['WWW'] = ['WWW_']
process_group_patterns['WWG'] = ['WWG_']
process_group_patterns['WWZ'] = ['WWZ_']
process_group_patterns['WGG'] = ['WGG_']
process_group_patterns['WZG'] = ['WZG_']
process_group_patterns['WZZ'] = ['WZZ_']

process_group_patterns['TTbar'] = ['TT_', 'TTTo']
process_group_patterns['TTW'] = ['TTWJets']
process_group_patterns['TTG'] = ['TTGJets']
process_group_patterns['TTZ'] = ['TTZTo']
process_group_patterns['TTGG'] = ['TTGG_0Jets']

process_group_patterns['tG'] = ['TGJets']
process_group_patterns['tZQ'] = ['tZq_']
process_group_patterns['Top'] = ['ST_', 'ST_tW_']
process_group_patterns['TTbarTTbar'] = ['TTTT_']

process_group_patterns['HIG'] = ['GluGluHTo', 'VBFHTo', 'VBF_HTo', 'VHTo', 'WplusH_HTo', 'WminusH_HTo', 'ZH_HTo', 'ggZH_HTo', 'ttHTo']

process_group_patterns['QCD'] = ['QCD_']

if __name__ == '__main__':

    parser    = argparse.ArgumentParser()

    parser.add_argument('-in','--inputDir', type=str, help='Directory of h5 file to be plotted.', required=True)
    parser.add_argument('-dsetName','--dsetName', type=str, help='Data set to be plotted. Choose SumPt, InvMass or MET.', required=True)

    args       = parser.parse_args()
    fileH5     = args.inputDir
    dset       = args.dsetName
    eventClass = fileH5.replace('/eos/user/a/aalvesan/H5_files/2016_MC_v9/H5_','')
    evClass    = eventClass.replace('.h5','')

    x          = []
    pName      = []

    with h5py.File(fileH5, 'r') as f:
        x      = list(f[dset])
        pName  = list(f['ProcessName'])
        f.close()
    fulldataset=list(zip(x,pName))

    print('pName entries are:')
    print(pName)

    pGroupName = []
    
    for string in pName:
        for key in process_group_patterns:
            pStrings = process_group_patterns[key]
            for pstring in pStrings:
                if str(pstring) in str(string):
                    pGroupName = np.append(pGroupName,key)
                else:
                    print('Group ' + str(string) + ' could not be found!')


    print('pGroupName entries are:')
    print(pGroupName)

'''
    hist, bin_edges = np.histogram(x, bins=30,density=False)
    bin_center      = (bin_edges[:-1] + bin_edges[1:])/2
    plt.hist(x, bins=30, density=False,fc='None')
    plt.errorbar(bin_center, hist, xerr=len(bin_edges)/2, fmt='.')
    plt.xlabel(dset+' [GeV]')
    plt.ylabel('Number of Events')
    plt.title(evClass)
    plt.show('plot_'+evClass+'_'+dset+'.png')
'''
