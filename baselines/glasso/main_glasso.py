import argparse
import numpy as np 
from numpy import *
import matplotlib.pyplot as plt
import networkx as nx
import grnular.data.gen_data as gen_data
from grnular.utils.metrics import report_metrics
import sys, copy, pickle
from sklearn.covariance import GraphicalLassoCV
from sklearn.covariance import GraphicalLasso
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))
#np.set_printoptions(threshold=sys.maxsize)
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
parser.add_argument('--DATA_METHOD', type=str,  default='sim_expt', # 
                    help='expt details in draft: random/syn_same_precision, sim_expt1=DS1 and sim_expt2=Custom, sim_expt3=GRN')
parser.add_argument('--DATA_TYPE', type=str,  default='clean', #'case2', 
                    help='expt details in draft: sim_exp1 clean/noisy')
parser.add_argument('--alpha_l1', type=float, default=0.1, #0.2,
                    help='L1 penalty for graphical lasso')
parser.add_argument('--mode', type=str,  default='cd', #'lars'
                    help='glasso mode: cd or lars or cv')


# SERGIO simulator parameters
parser.add_argument('--DATA_NAME', type=str,  default='CUSTOM', #'DS1', 
                    help='expt details in draft: DS1, DS2, DS3, CUSTOM')
parser.add_argument('--POINTS_PER_CLASS', type=int, default=2000,# NOTE: try 2000
                    help='cells per class type')
#parser.add_argument('--DS1_POINTS', type=int, default=300,# NOTE: 300
#                    help='cells per class type for DS1 data')
#parser.add_argument('--TOTAL_SIMULATIONS', type=int, default=1,
#                    help='just run on some set of simulation')
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
parser.add_argument('--USE_TF_NAMES', type=str,  default='no',# 'yes' 
                    help='use transcription factors to improve prediction: will use in general')
# SERGIO technical noise parameters
parser.add_argument('--ADD_TECHNICAL_NOISE', type=str,  default='yes',# 'no' 
                    help='add technical noise on the saved clean data')
parser.add_argument('--dropout_shape', type=float, default=6.5, #1,
                    help='SERGIO dropout param: shape, higher -> less dropout')
parser.add_argument('--dropout_percentile', type=float, default=82, #1,
                    help='SERGIO dropout param: percentile, lower -> less dropout')

args = parser.parse_args()
print(args)
print('\n')

def get_args_str(dict1):
    args_str = ''
    for i, (k, v) in enumerate(dict1.items()):
#        print(k , item)
        if k in ['C', 'GLAD_LOSS', 'MODEL_SELECT', 'SUB_METHOD']:
            args_str = args_str+str(k)+str(v)+'_'
    return args_str

args_str = get_args_str(vars(args))


def fit_graphical_lasso(data, PREDICT_TF=False, BEELINE=False): 
    print('FITTING a graphical lasso')
    # #############################################################################
    res = []
    typeS = 'mean'
    print('Using ', typeS, ' scaling')
    for i, d in enumerate(data):
        X, y, theta_true, master_regulators = d
        Xc = normalizing_data(X, typeS)
        print('\nGRAPHICAL Lasso: TRAIN data batch : ', i, ' total points = ', X.shape[0])
        if args.USE_TF_NAMES=='yes' and PREDICT_TF:
            res.append(helper_graphical_lasso(Xc, theta_true, tf_names=master_regulators))
        else:
            res.append(helper_graphical_lasso(Xc, theta_true))
    res_mean = np.mean(np.array(res), 0)
    res_std  = np.std(np.array(res).astype(np.float64), 0)
    res_mean = ["%.3f" %x for x in res_mean]
    res_std  = ["%.3f" %x for x in res_std]
    res_dict = {} # name: [mean, std]
    for i, _name in enumerate(['FDR', 'TPR', 'FPR', 'SHD', 'nnz_true', 'nnz_pred', 'precision', 'recall', 'Fb', 'aupr', 'auc']): # dictionary
        res_dict[_name]= [res_mean[i], res_std[i]]#mean std
    if PREDICT_TF:
        print('\nAvg GLASSO-TF: FDR, ,TPR, ,FPR, ,SHD, ,nnz_true, ,nnz_pred, ,precision, ,recall, ,Fb, ,aupr, ,auc, ')
    else:
        print('\nAvg GLASSO: FDR, ,TPR, ,FPR, ,SHD, ,nnz_true, ,nnz_pred, ,precision, ,recall, ,Fb, ,aupr, ,auc, ')
    mean_std = [[rm, rs] for rm, rs in zip(res_mean, res_std)]
    flat_list = [item for ms in mean_std for item in ms]
    print('%s' % ', '.join(map(str, flat_list)))
    #print(*sum(list(map(list, zip(res_mean, res_std))), []), sep=', ')
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



def get_PSD_matrix(A, u=1):
    smallest_eigval = np.min(np.linalg.eigvals(A))
    # Just in case : to avoid numerical error in case a epsilon complex component present
    smallest_eigval = smallest_eigval.real
    # making the min eigenvalue as 1
    target_precision_mat = A + np.eye(A.shape[-1])*(u - smallest_eigval)
    #print('CHEKKK: smallest eigen? = ', np.min(np.linalg.eigvals(target_precision_mat)))
    return target_precision_mat

def postprocess_tf(prec, tf_names):
#    print('Postprocesing for TF NAMES')
    # remove all the edges whose at least one of the vertices is not in tf_names
    # zeroing the diagonal to get adj matrix
#    print('tf names: ', tf_names)
    tf_names = ['G'+str(n) for n in tf_names]
    np.fill_diagonal(prec, 0)
    G_pred = nx.from_numpy_matrix(prec)
    #mapping = {n: 'G'+str(n) for n in range(args.D)}
    mapping = {n: 'G'+str(n) for n in range(prec.shape[-1])}
    G_pred = nx.relabel_nodes(G_pred, mapping)
    set_tf_names = tf_names
    edges = copy.deepcopy(G_pred.edges)
    for e in edges:
        if not (set(e) & set(tf_names)):
            # remove the edge
            G_pred.remove_edge(*e[:2])
    prec = get_PSD_matrix(nx.adj_matrix(G_pred).todense())
    return np.array(prec)


def helper_graphical_lasso(X, theta_true, tf_names=[]):
    # Estimate the covariance
    if args.mode == 'cv':
        model = GraphicalLassoCV()
    else:
        model = GraphicalLasso(alpha=args.alpha_l1, mode=args.mode, tol=1e-7, enet_tol=1e-6, 
                                max_iter=100, verbose=False, assume_centered=False)
    #model = GraphicalLasso(alpha=args.alpha, mode='lars', tol=1e-7, enet_tol=1e-6, 
    #           max_iter=args.100, verbose=True, assume_centered=False)
    model.fit(X)
#    cov_ = model.covariance_
    prec_ = model.precision_
    if args.USE_TF_NAMES=='yes' and len(tf_names) != 0:
        prec_ = postprocess_tf(prec_, tf_names)
    recovery_metrics = report_metrics(np.array(theta_true), prec_)
    print('GLASSO: FDR, TPR, FPR, SHD, nnz_true, nnz_pred, precision, recall, Fb, aupr, auc')
    print('GLASSO: TEST: Recovery of true theta: ', *np.around(recovery_metrics, 3))
    return list(recovery_metrics)

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
    print('Reading the input data: Single cell RNA: M(samples) x D(genes) & corresponding C(targets)')
    train_data, valid_data, test_data = load_saved_data()
        
    if args.ADD_TECHNICAL_NOISE == 'yes':
        print('adding technical noise')
#        train_data = gen_data.add_technical_noise(args, train_data)
#        valid_data = gen_data.add_technical_noise(args, valid_data)
        test_data = gen_data.add_technical_noise(args, test_data)

    # Fitting a graphical lasso 
    fit_graphical_lasso(test_data)
    print('Using TF NAMES')
    fit_graphical_lasso(test_data, PREDICT_TF=True)
    print('\nExpt Done')
    return 

if __name__=="__main__":
    main()
