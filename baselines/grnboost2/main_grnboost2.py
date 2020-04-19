import argparse
import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx
from numpy import *
import grnular.data.gen_data as gen_data
from grnular.utils.metrics import report_metrics
import sys, copy, pickle
from arboreto.algo import grnboost2
from arboreto.utils import load_tf_names
import pandas as pd
import sklearn 
TRAIN=True

parser = argparse.ArgumentParser(description='Classification of different cell types as well as recovering the gene regulatory network of RNA seq data: using SERGIO simulator for training')
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
parser.add_argument('--sparsity', type=float, default=0.3, #0.2,
                    help='sparsity of erdos-renyi graph')
parser.add_argument('--DATA_METHOD', type=str,  default='sim_expt', #'syn_expt2', #'sim_expt1', 
                    help='expt details in draft: random/syn_same_precision, sim_expt1=DS1 and sim_expt2=Custom, sim_expt3=GRN')
parser.add_argument('--DATA_TYPE', type=str,  default='clean', #'case2', 
                    help='expt details in draft: sim_exp1 clean/noisy')

parser.add_argument('--USE_TF_NAMES', type=str,  default='no',# 'yes' 
                    help='use transcription factors for grnboost2: will use in general')

# SERGIO simulator parameters
parser.add_argument('--DATA_NAME', type=str,  default='CUSTOM', #'DS1', 
                    help='expt details in draft: DS1, DS2, DS3, CUSTOM')
parser.add_argument('--POINTS_PER_CLASS', type=int, default=2000,# NOTE: try 2000
                    help='cells per class type')
parser.add_argument('--SAMPLING_STATE', type=int, default=1, #1,
                    help='num of simulations')
parser.add_argument('--NOISE_PARAMS', type=float, default=0.1, #1,
                    help='num of noise params')
parser.add_argument('--DECAYS', type=float, default=5, #0.8,#0.8,
                    help='decay params')
parser.add_argument('--NOISE_TYPE', type=str,  default='dpd', #'dpd', 
                    help='different noise types: "dpd", “sp”, “spd”')
parser.add_argument('--SHARED_COOP_STATE', type=int, default=2, #1,
                    help='shared coop state')
parser.add_argument('--pcr_low_min', type=float, default=0.2, #1,
                    help='production cell rate: low expression range')
parser.add_argument('--pcr_low_max', type=float, default=0.5, #1,
                    help='production cell rate: low expression range')
parser.add_argument('--pcr_high_min', type=float, default=0.7, #1,
                    help='production cell rate: high expression range')
parser.add_argument('--pcr_high_max', type=float, default=1, #1,
                    help='production cell rate: high expression range')
parser.add_argument('--Kij_min', type=float, default=1, #1,
                    help='Interaction strengths Kij min')
parser.add_argument('--Kij_max', type=float, default=5, #1,
                    help='Interaction strengths Kij max')
parser.add_argument('--ratio_MR', type=float, default=0.1, #1,
                    help='number of master regulators ~ ratio_MR * D')
parser.add_argument('--connect_TF_prob', type=float, default=0.2, #1,
                    help='probability of connecting master regulators')
# SERGIO technical noise parameters
parser.add_argument('--ADD_TECHNICAL_NOISE', type=str,  default='yes',# 'no' 
                    help='add technical noise on the saved clean data')
parser.add_argument('--dropout_shape', type=float, default=6.5, #1,
                    help='SERGIO dropout param: shape, higher -> less dropout')
parser.add_argument('--dropout_percentile', type=float, default=82, #1,
                    help='SERGIO dropout param: percentile, lower -> less dropout')

args = parser.parse_args()

def get_args_str(dict1):
    args_str = ''
    for i, (k, v) in enumerate(dict1.items()):
#        print(k , item)
        if k in ['C', 'GLAD_LOSS', 'MODEL_SELECT', 'SUB_METHOD']:
            args_str = args_str+str(k)+str(v)+'_'
    return args_str

args_str = get_args_str(vars(args))

def get_res_filepath(name):
    FILE_NUM = str(np.random.randint(10000))
    savepath = 'simulator/BEELINE-data/my_pred_networks/'
    filepath = savepath +name+'_beeline_pred_tag'+str(FILE_NUM)+'.pickle'
    return filepath


def fit_grnboost2(data, PREDICT_TF=False, BEELINE=False):
    EARLY_BREAK = 9 
    print('FITTING GRNBOOST2')
    # #############################################################################
    res = []
    typeS = 'mean'
    print('Using ', typeS, ' scaling')
    for i, d in enumerate(data):
        X, y, theta_true, master_regulators = d
        Xc = normalizing_data(X, typeS)
        print('\n grnboost2: TRAIN data batch : ', i, ' total points = ', X.shape[0])
        if args.USE_TF_NAMES=='yes' and PREDICT_TF:
            res.append(helper_grnboost2(Xc, theta_true, tf_names = master_regulators))

        else:
            # NOTE: breaking early as tf=None takes lot of time
            if i > EARLY_BREAK:
                print('Breaking at i = ', i, ' as tf=None case takes a lot of time')
                break
            res.append(helper_grnboost2(Xc, theta_true))


    res_mean = np.mean(np.array(res).astype(np.float64), 0)
    res_std  = np.std(np.array(res).astype(np.float64), 0)
    res_mean = ["%.3f" %x for x in res_mean]
    res_std  = ["%.3f" %x for x in res_std]
    res_dict = {} # name: [mean, std]
    for i, _name in enumerate(['FDR', 'TPR', 'FPR', 'SHD', 'nnz_true', 'nnz_pred', 'precision', 'recall', 'Fb', 'aupr', 'auc']): # dictionary
        res_dict[_name]= [res_mean[i], res_std[i]]#mean std
    if PREDICT_TF:
        print('\nAvg GRNBOOST2-TF: FDR, ,TPR, ,FPR, ,SHD, ,nnz_true, ,nnz_pred, ,precision, ,recall, ,Fb, ,aupr, ,auc, ')
    else:
        print('\nAvg GRNBOOST2: FDR, ,TPR, ,FPR, ,SHD, ,nnz_true, ,nnz_pred, ,precision, ,recall, ,Fb, ,aupr, ,auc, ')
    mean_std = [[rm, rs] for rm, rs in zip(res_mean, res_std)]
    flat_list = [item for ms in mean_std for item in ms]
    print('%s' % ', '.join(map(str, flat_list))) 
    return


def normalizing_data(X, typeS='log'):
    if typeS == 'mean':
        #print('Centering and scaling the input data...')
        scaledX = X - X.mean(axis=0)
        scaledX = scaledX/X.std(axis=0)
        # NOTE: replacing all nan's by 0, as sometimes in dropout the complete column
        # goes to zero
        scaledX = convert_nans_to_zeros(scaledX)
    elif typeS == 'log':
        scaledX = np.log(X+1)
    else:
        print('Check the valid scaling')
    return scaledX


def convert_nans_to_zeros(X):
    where_are_nans = isnan(X)
    X[where_are_nans] = 0
    return X


def helper_grnboost2(X, theta_true, tf_names=[], BEELINE=False):#_string
    print('Running GRNBoost2 method', X.shape)
    theta_true = theta_true.real
    ex_matrix = pd.DataFrame(X)
    if args.USE_TF_NAMES == 'yes' and len(tf_names)!=0:
        tf_names = ['G'+str(n) for n in tf_names]
    else:
        tf_names = None
        
    gene_names = ['G'+str(c) for c in ex_matrix.columns]
    ex_matrix.columns = gene_names
    network = grnboost2(expression_data=ex_matrix, gene_names=gene_names, tf_names=tf_names)#, verbose=True)
    pred_edges = np.array(network[['TF', 'target', 'importance']])
    G_pred = nx.Graph()
#    G_pred.add_nodes_from(['G'+str(n) for n in range(args.D)])
    G_pred.add_nodes_from(['G'+str(n) for n in range(len(gene_names))])
    G_pred.add_weighted_edges_from(pred_edges)
#    pred_theta = nx.adj_matrix(G_pred).todense() + np.eye(args.D)
    pred_theta = nx.adj_matrix(G_pred).todense() + np.eye(len(gene_names))
    recovery_metrics = report_metrics(np.array(theta_true), np.array(pred_theta))
    print('GRNBOOST2: FDR, TPR, FPR, SHD, nnz_true, nnz_pred, precision, recall, Fb, aupr, auc')
    print('GRNBOOST2: Recovery of true theta: ', *np.around(recovery_metrics, 3))
    
    res = list(recovery_metrics)
    return res


def get_filepath():
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

    SAVEPATH = 'grnular/data/saved_data/'
    FILEPATH = SAVEPATH + filename + '.pickle'
    print('Filepath: ', FILEPATH)
    return FILEPATH


def load_saved_data():
    FILEPATH = get_filepath()
    with open(FILEPATH, 'rb') as handle:
        data = pickle.load(handle)
    return data


def main():
    print(args)
    print('\nReading the input data: Single cell RNA: M(samples) x D(genes) & corresponding C(targets)')
    train_data, valid_data, test_data = load_saved_data()
    if args.ADD_TECHNICAL_NOISE == 'yes':
        print('adding technical noise')
#        train_data = gen_data.add_technical_noise(args, train_data)
#        valid_data = gen_data.add_technical_noise(args, valid_data)
        test_data = gen_data.add_technical_noise(args, test_data)

    # Fitting a grnboost2 
    fit_grnboost2(test_data)
    print('Using TF NAMES')
    fit_grnboost2(test_data, PREDICT_TF=True)
    print('\nExpt Done')
    return 

if __name__=="__main__":
    main()
