import os
import json
import argparse
import numpy as np
import glob
import os.path
import time
from NPLM.DATAutils import *
from NPLM.NNutils import *

#############################################################
############## Creating the  config.json file ###############
#############################################################
                                                            
OUTPUT_DIRECTORY = '/eos/user/a/aalvesan/ml_test/'

def create_config_file(config_table, OUTPUT_DIRECTORY):
    with open('%s/config.json'%(OUTPUT_DIRECTORY), 'w') as outfile:
        json.dump(config_table, outfile, indent=4)
    return '%s/config.json'%(OUTPUT_DIRECTORY)

config_json      = {
    "features"                   : ['SumPt'],
    "N_Bkg"                      : 4693,
    "output_directory"           : OUTPUT_DIRECTORY,

    "epochs"                     : 4000,
    "patience"                   : 1000,

    "BSMarchitecture"            : [1,4,1],
    "BSMweight_clipping"         : 14, 
    "correction"                 : "",
}

ID ='Nbkg'+str(config_json["N_Bkg"])
ID+='_patience'+str(config_json["patience"])+'_epochs'+str(config_json["epochs"])
ID+='_arc'+str(config_json["BSMarchitecture"]).replace(', ', '_').replace('[', '').replace(']', '')+'_wclip'+str(config_json["BSMweight_clipping"])

#############################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pyscript',    type=str, help = "name of python script to execute", required=True) 
    parser.add_argument('-l','--local',       type=int, help = 'if to be run locally', required=False, default=0)
    parser.add_argument('-t', '--toys',       type=int, default = "100", help = "number of toys to be processed")
    #parser.add_argument('-j', '--json',      type=str, default = "100", help = "configuration file")         

    args     = parser.parse_args()
    pyscript = args.pyscript
    ntoys    = args.toys

    config_json['pyscript'] = pyscript
    
    pyscript_str = pyscript.replace('.py', '')
    pyscript_str = pyscript_str.replace('_', '/')
    
    config_json["output_directory"] = OUTPUT_DIRECTORY+ID

    if not os.path.exists(config_json["output_directory"]):
        os.makedirs(config_json["output_directory"])
    
    json_path = create_config_file(config_json, config_json["output_directory"])

    if args.local:
        print('!!! Be sure you sourced /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc8-opt/setup.sh !!!')
        print('!!! or activate your personal environment before                                             !!!')
        os.system("python %s/%s -j %s" %(os.getcwd(), pyscript, json_path))
    else:
        label = "Jobs_Outputs"
        os.system("mkdir %s" %label)
        for i in range(ntoys):        
            # creating the src file
            script_src = open("%s/%i.src" %(label, i) , 'w')
            script_src.write("#!/bin/bash\n")
            script_src.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc10-opt/setup.sh\n")
            script_src.write("python %s/%s -j %s" %(os.getcwd(), args.pyscript, json_path))
            script_src.close()
            os.system("chmod a+x %s/%i.src" %(label, i))
            # creating the condor file
            script_condor = open("%s/%i.condor" %(label, i) , 'w')
            script_condor.write("executable = %s/%i.src\n" %(label, i))
            script_condor.write("universe = vanilla\n")
            script_condor.write("output = %s/%i.out\n" %(label, i))
            script_condor.write("error =  %s/%i.err\n" %(label, i))
            script_condor.write("log = %s/%i.log\n" %(label, i))
            script_condor.write("+MaxRuntime = 500000\n")
            script_condor.write('requirements = (OpSysAndVer =?= "CentOS7")\n')
            script_condor.write("queue\n")
            script_condor.close()
            # creating the condor file submission
            os.system("condor_submit %s/%i.condor" %(label,i))
