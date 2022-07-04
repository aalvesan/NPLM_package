import numpy as np

ML_Classes = ['1Ele_1Muon_1MET']

bkg_list = ['AllData_ZX_redTree_2018',
            'ggTo2e2tau_Contin_MCFM701_redTree_2018',   
            'ggTo4mu_Contin_MCFM701_redTree_2018',
            'ggTo2mu2tau_Contin_MCFM701_redTree_2018',  
            'ggTo4tau_Contin_MCFM701_redTree_2018',
            'ggTo2e2mu_Contin_MCFM701_redTree_2018',  
            'ggTo4e_Contin_MCFM701_redTree_2018',
            'ZZTo4lext_redTree_2018_0',
            'ZZTo4lext_redTree_2018_1',
            'ZZTo4lext_redTree_2018_2',
            'ZZTo4lext_redTree_2018_3',
        ]

sig_list = [ 
    'ZH125_redTree_2018',
    'VBFH125_redTree_2018',
    'bbH125_redTree_2018', 
    'ttH125_redTree_2018',
    'ggH125_redTree_2018',
    'WminusH125_redTree_2018',     
    'WplusH125_redTree_2018', 
]

#### std variables
mean_bkg = {
     'Z1Eta': -0.004484998969496296,
     'Z1Mass': 85.8519627954568,
     'Z1Phi': 0.0004336799502239779,
     'Z1Pt': 63.1005141254963,
     'Z1Z2DeltaPhi': 0.00038650320289238335,
     'Z2Eta': -0.003583167486280177,
     'Z2Mass': 68.92534143438355,
     'Z2Phi': 0.0028061950815497457,
     'Z2Pt': 56.62463498724115,
     'ZZEta': -0.006722796,
     'ZZMass': 229.74287,
     'ZZPhi': -0.0023281502,
     'ZZPt': 43.72643,
     'costhetastar': -0.0005796023,
     'helcosthetaZ1': 0.0011582823,
     'helcosthetaZ2': 0.0003175177,
     'helphi': 0.0022883567,
     'l1Eta': -0.0021192406,
     'l1Id': -12.175942157731603,
     'l1Phi': 0.00010763579,
     'l1Pt': 50.434803,
     'l2Eta': -0.0029567885,
     'l2Id': 12.175942157731603,
     'l2Phi': -0.0012176564,
     'l2Pt': 50.634857,
     'l3Eta': -0.0021477398,
     'l3Id': -11.457448950919266,
     'l3Phi': -0.0005667373,
     'l3Pt': 42.978096,
     'l4Eta': -0.0015593123,
     'l4Id': 11.365803502939631,
     'l4Phi': 0.0026263122,
     'l4Pt': 42.139683,
     'phistarZ1': -0.0007681197
}
std_bkg = {
         'Z1Eta': 1.725167203224534,
         'Z1Mass': 12.820934009611815,
         'Z1Phi': 1.8147887617314518,
         'Z1Pt': 58.35893101257469,
         'Z1Z2DeltaPhi': 2.5521870103436193,
         'Z2Eta': 1.6425869375105375,
         'Z2Mass': 30.819486094834815,
         'Z2Phi': 1.8153657483615246,
         'Z2Pt': 56.57528084742795,
         'ZZEta': 2.816692,
         'ZZMass': 113.03855,
         'ZZPhi': 1.8155365,
         'ZZPt': 59.55866,
         'costhetastar': 0.64380634,
         'helcosthetaZ1': 0.60344523,
         'helcosthetaZ2': 0.58041203,
         'helphi': 1.8413788,
         'l1Eta': 1.1438663,
         'l1Id': 0.9844005064671354,
         'l1Phi': 1.8157566,
         'l1Pt': 36.452908,
         'l2Eta': 1.1554388,
         'l2Id': 0.9844005064671354,
         'l2Phi': 1.8143846,
         'l2Pt': 36.582676,
         'l3Eta': 1.1718746,
         'l3Id': 4.116473487761316,
         'l3Phi': 1.8146842,
         'l3Pt': 39.632454,
         'l4Eta': 1.1804229,
         'l4Id': 4.363141204559856,
         'l4Phi': 1.8156768,
         'l4Pt': 36.505836,
         'phistarZ1': 1.7840887
}
