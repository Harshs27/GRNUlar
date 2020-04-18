import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import simulator.sim_expt3.data.gen_data as gen_data
import sys, copy
import pickle

parser = argparse.ArgumentParser(description='Save the data generated from SERGIO')
#****************** general parameters
parser.add_argument('--K_train', type=int, default=5, #2, #1000,
                    help='Num of training examples for a fixed D')
parser.add_argument('--K_valid', type=int, default=5, #5, #2, #1000,
                    help='Number of valid examples for a fixed D ')
parser.add_argument('--K_test', type=int, default=100, #100, #10,
                    help='Number of testing examples for a fixed D')
parser.add_argument('--D', type=int, default=100, #1000,
                    help='Number of genes ')
parser.add_argument('--C', type=int, default=9,
                    help='different cell types, target variable')
parser.add_argument('--sparsity', type=float, default=0.2, #0.2,
                    help='sparsity of erdos-renyi graph')
parser.add_argument('--DATA_TYPE', type=str,  default='clean', #'case2', 
                    help='expt details in draft: sim_exp1 clean/noisy')
# SERGIO simulator parameters
parser.add_argument('--DATA_METHOD', type=str,  default='sim_expt3', #'syn_expt2', #'sim_expt1', 
                    help='expt details in draft: random/syn_same_precision, sim_expt1=DS1 and sim_expt2=Custom, sim_expt3=GRN')
parser.add_argument('--DATA_NAME', type=str,  default='CUSTOM', #'DS1', 
                    help='expt details in draft: DS1, DS2, DS3, CUSTOM')
parser.add_argument('--gen_DS', type=str,  default='no',#'no',
                    help='expt details in draft: DS1, DS2, DS3- simulates data using actual DS graphs')
parser.add_argument('--POINTS_PER_CLASS', type=int, default=2000,# NOTE: try 2000
                    help='cells per class type')
#parser.add_argument('--DS1_POINTS', type=int, default=300,# NOTE: 300
#                    help='cells per class type for DS1 data')
#parser.add_argument('--TOTAL_SIMULATIONS', type=int, default=1,
#                    help='just run on some set of simulation')
parser.add_argument('--SAMPLING_STATE', type=int, default=15, #1,
                    help='num of simulations')
parser.add_argument('--NOISE_PARAMS', type=float, default=0.1, #1,
                    help='num of noise params')
parser.add_argument('--DECAYS', type=float, default=5, #0.8,#0.8,
                    help='decay params')
parser.add_argument('--NOISE_TYPE', type=str,  default='dpd', #'dpd', 
                    help='different noise types: "dpd", “sp”, “spd”')
parser.add_argument('--SHARED_COOP_STATE', type=int, default=2, #1,
                    help='shared coop state')

parser.add_argument('--ratio_MR', type=float, default=0.1, #1,
                    help='number of master regulators ~ ratio_MR * D')
parser.add_argument('--connect_TF_prob', type=float, default=0.2, #1,
                    help='probability of connecting master regulators')

parser.add_argument('--pcr_low_min', type=float, default=0.2, #1,
                    help='production cell rate: low expression range')
parser.add_argument('--pcr_low_max', type=float, default=0.5, #1,
                    help='production cell rate: low expression range')
parser.add_argument('--pcr_high_min', type=float, default=0.7, #1,
                    help='production cell rate: high expression range')
parser.add_argument('--pcr_high_max', type=float, default=1.0, #1,
                    help='production cell rate: high expression range')
parser.add_argument('--Kij_min', type=float, default=1.0, #1,
                    help='Interaction strengths Kij min')
parser.add_argument('--Kij_max', type=float, default=5.0, #1,
                    help='Interaction strengths Kij max')

args = parser.parse_args()

def get_filepath(args):
    dict1 = vars(args)
    filename = ''
    abbrv_dict = {'K_train': 'KTr', 'K_valid': 'KVa', 'K_test': 'KTe', 'D': 'D', 'C':'C',
                  'sparsity': 'Sp', 'DATA_TYPE':'Dt', 'POINTS_PER_CLASS': 'ppc',
                  'SAMPLING_STATE': 'SS', 'NOISE_PARAMS': 'NP', 'DECAYS': 'De', 
                  'NOISE_TYPE': 'NT', 'SHARED_COOP_STATE': 'SCS', 'pcr_low_min': 'pcrln',
                  'pcr_low_max': 'pcrlx', 'pcr_high_min': 'pcrhn', 'pcr_high_max': 'pcrhx',
                  'Kij_min': 'kmin', 'Kij_max': 'kmax', 'ratio_MR': 'rMR', 
                  'connect_TF_prob': 'TFp'}
    for k in abbrv_dict.keys():
        v = dict1[k]
        filename = filename+str(abbrv_dict[k])+str(v)+'_'
    

    SAVEPATH = 'simulator/'+args.DATA_METHOD+'/data/saved_data/'
    if args.gen_DS in ['DS1', 'DS2', 'DS3']:
        FILEPATH = SAVEPATH + filename + args.gen_DS + '.pickle'        
    else:
        FILEPATH = SAVEPATH + filename + '.pickle'
    print('Filepath: ', FILEPATH)
    return FILEPATH


def main():
    print(args)
    print('Creating the data files and saving for expts')
    train_data, valid_data, test_data = gen_data.create_GRN_data(args)
    all_data = [train_data, valid_data, test_data]
    #print('before: ', all_data[0][0])
    # saving the data
    FILEPATH = get_filepath(args)
    with open(FILEPATH, 'wb') as handle:
        pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(FILEPATH, 'rb') as handle:
        load_all_data = pickle.load(handle)
    #print('after: ', load_all_data[0][0])
    print('Files saved')    

if __name__=="__main__":
    main()
