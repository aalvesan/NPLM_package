import os
import json
import argparse
import numpy as np
import glob
import os.path
import time
from config_utils import parNN_list

OUTPUT_DIRECTORY = '/eos/user/a/aalvesan/ML_test/'

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
ID+='_arc'+str(config_json["BSMarchitecture"]).replace(', ', '_').replace('[', '').replace(']', '')+'\
_wclip'+str(config_json["BSMweight_clipping"])


#### launch python script ###########################
if __name__ == '__main__':
    parser   = argparse.ArgumentParser()
    parser.add_argument('-p','--pyscript',  type=str, help="name of python script to execute", required=True)

    args     = parser.parse_args()
    pyscript = args.pyscript
    config_json['pyscript'] = pyscript
    
    pyscript_str = pyscript.replace('.py', '')
    pyscript_str = pyscript_str.replace('_', '/')

    config_json["output_directory"] = OUTPUT_DIRECTORY+'/'+ID

    if not os.path.exists(config_json["output_directory"]):
        os.makedirs(config_json["output_directory"])
        
    json_path = create_config_file(config_json, config_json["output_directory"])
        
    os.system("python %s/%s -j %s" %(os.getcwd(), args.pyscript, json_path))
   
