#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
import numpy as np 
from numpy import *
import matplotlib.pyplot as plt
import networkx as nx
import grnular.data.gen_data as gen_data
import grnular.source.grnular as grnular
from grnular.source.grnular_model import glad_model
from grnular.source.grnular_model import dnn_model
from grnular.utils.metrics import report_metrics
import sys, copy, pickle, time
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))
#np.set_printoptions(threshold=sys.maxsize)
TRAIN=True

parser = argparse.ArgumentParser(description='GRNUlar: unrolled algo for recovery of directed GRN')
#****************** general parameters
parser.add_argument('--K_train', type=int, default=20, #2, #1000,
                    help='Num of training examples for a fixed D')
parser.add_argument('--K_valid', type=int, default=20, #5, #2, #1000,
                    help='Number of valid examples for a fixed D ')
parser.add_argument('--K_test', type=int, default=50, #100, #10,
                    help='Number of testing examples for a fixed D')
parser.add_argument('--USE_CUDA_FLAG', type=int, default=1,
                    help='USE GPU if = 1')
parser.add_argument('--SAVE_GRAPHS', type=str,  default='yes', #'no',
                    help='save input and predicted graphs: yes/no')
parser.add_argument('--D', type=int, default=100, #1000,
                    help='Number of genes ')
parser.add_argument('--C', type=int, default=9,
                    help='different cell types, target variable')
parser.add_argument('--sparsity', type=float, default=0.3, #0.2,
                    help='sparsity of erdos-renyi graph')
parser.add_argument('--DATA_METHOD', type=str,  default='sim_expt', 
                    help='expt details in draft: random/syn_same_precision, sim_expt=GRN')
parser.add_argument('--DATA_TYPE', type=str,  default='clean', #'case2', 
                    help='expt details in draft: sim_exp1 clean/noisy')
parser.add_argument('--EPOCHS', type=int, default=150,
                    help='Number of epochs for training GLAD')
parser.add_argument('--PRINT_EPOCH', type=int, default=20,
                    help='Print every X epochs while training GLAD')
parser.add_argument('--MODEL_SELECT', type=str,  default='Fb', #'current', #'Fb', 
                    help='select model based on best results: {graph recovery}Fb=auc=aupr/shd/current ')


# ***************GLAD related parameters
parser.add_argument('--nF', type=int, default=3,
                    help='number of input features for NN rho')
parser.add_argument('--H', type=int, default=3,
                    help='Hidden layer size of NN rho')
parser.add_argument('--lambda_init', type=float, default=1, #1,
                    help='initial value of lambda ')
parser.add_argument('--theta_init_offset', type=float, default=1, #0.01, #0.03, #075,
                    help='offset for setting the diagonal init theta approximation')
parser.add_argument('--L', type=int, default= 10,
                    help='Unroll the network L times')
parser.add_argument('--INIT_DIAG', type=int, default=0,
                    help='1 : initialize the theta0 diagonally')
parser.add_argument('--GLAD_LOSS', type=str,  default='mse_fb', #'F_beta', #'mse',
                    help='intermediate loss for graph recovery: mse/le/F_beta/no')
parser.add_argument('--beta', type=float,  default=1.0,# 1.0
                    help='differentiable F-beta loss')
parser.add_argument('--use_optimizer', type=str,  default='adam',
                    help='can use either: adam, adadelta, rms, sgd')
parser.add_argument('--lr_glad', type=float, default=0.01, #0.01,
                    help='learning rate for the GLAD model')
parser.add_argument('--USE_TF_NAMES', type=str,  default='yes',# 'yes' 
                    help='use transcription factors to improve prediction: will use in general')
# ****************************DNN 
parser.add_argument('--Hd', type=int, default=20,
                    help='Hidden layer size of DNN')
parser.add_argument('--DNN_EPOCHS', type=int, default=40,
                    help='Num of epoch for optimizing DNN')
parser.add_argument('--lrDNN', type=float, default=0.01,
                    help='learning rate for the DNN model')
parser.add_argument('--P', type=int, default= 5,
                    help='Unroll the DNN network P times')


# SERGIO simulator parameters
parser.add_argument('--DATA_NAME', type=str,  default='DS1', #'DS1', 
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
# SERGIO technical noise parameters
parser.add_argument('--ADD_TECHNICAL_NOISE', type=str,  default='yes',# 'no' 
                    help='add technical noise on the saved clean data')
parser.add_argument('--dropout_shape', type=float, default=6.5, #1,
                    help='SERGIO dropout param K: shape, higher -> less dropout')
parser.add_argument('--dropout_percentile', type=float, default=82, #1,
                    help='SERGIO dropout param P: percentile, lower -> less dropout')


args = parser.parse_args()
print(args)
print('\n')
#print(str(vars(args)))
#args_str = str(vars(args))

def get_args_str(dict1):
    args_str = ''
    for i, (k, v) in enumerate(dict1.items()):
#        print(k , item)
        if k in ['C', 'GLAD_LOSS', 'MODEL_SELECT', 'SUB_METHOD']:
            args_str = args_str+str(k)+str(v)+'_'
    return args_str

args_str = get_args_str(vars(args))

# Global Variables
USE_CUDA = False
if args.USE_CUDA_FLAG == 1:
    USE_CUDA = True


def _npy(a):# convert from torch gpu to numpy
    return a.detach().cpu().numpy()

def convert_to_torch(data, TESTING_FLAG=False, USE_CUDA=USE_CUDA):# convert from numpy to torch variable 
    if USE_CUDA == False:
        data = torch.from_numpy(data.astype(np.float, copy=False)).type(torch.FloatTensor)
        if TESTING_FLAG == True:
            data.requires_grad = False
    else: # On GPU
        if TESTING_FLAG == False:
            data = torch.from_numpy(data.astype(np.float, copy=False)).type(torch.FloatTensor).cuda()
        else: # testing phase, no need to store the data on the GPU
            data = torch.from_numpy(data.astype(np.float, copy=False)).type(torch.FloatTensor).cuda()
            data.requires_grad = False
    return data


def print_params(model, name=None):
    if name:
        print('Parameters of model: ', name)
    for n, p in model.named_parameters():
        print(n, p, p.grad)
    return

def compare_theta(theta_true, theta_pred):
    return report_metrics(np.array(theta_true), theta_pred.detach().cpu().numpy())


def get_optimizers(model_glad):
    lrG = args.lr_glad
    if args.use_optimizer == 'adadelta':
        optimizer_glad = torch.optim.Adadelta(model_glad.parameters(), lr=lrG, rho=0.9, eps=1e-06, weight_decay=0)# LR range = 5 -> 
    elif args.use_optimizer == 'rms':
        optimizer_glad = torch.optim.RMSprop(model_glad.parameters(), lr=lrG, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.25, centered=False)
    elif args.use_optimizer == 'sgd':
        optimizer_glad = torch.optim.SGD(model_glad.parameters(), lr=lrG, momentum=0.9,  dampening=0, weight_decay=0, nesterov=False)
    elif args.use_optimizer == 'adam':
        optimizer_glad = torch.optim.Adam(model_glad.parameters(), lr=lrG, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        print('Optimizer not found!')
    return optimizer_glad 


def get_glad_criterion():
    if args.GLAD_LOSS == 'mse': 
#        print('Using MSE loss for the glad loss')
        criterion_graph = nn.MSELoss()# reduction = "mean" by default
    elif args.GLAD_LOSS == 'l1':
#        print('Using L1 loss for the glad loss')
        criterion_graph = nn.L1Loss()# reduction = "mean" by default
    elif args.GLAD_LOSS == 'F_beta':
        criterion_graph = f_beta_loss# differentiable F_beta loss
    elif args.GLAD_LOSS == 'mse_fb':
        criterion_graph = mse_f_beta# MSE + F_beta loss
    elif args.GLAD_LOSS == 'no':
        print('Not including the intermediate glad loss in final loss')
        criterion_graph = nn.MSELoss()
    else:
        print('CHECK the glad loss input!!!')
    return criterion_graph


def format_torch(data):# convert the data to pytorch
    data_torch = []
    typeS = 'mean'
    print('Using ', typeS, ' scaling')
    for i, d in enumerate(data):
        X, y, theta, TF = d
        if i==0:
            print('TF',  TF)
        X = normalizing_data(X, typeS)
        theta = theta.real
        X = convert_to_torch(X, TESTING_FLAG=True)    
        theta_true = convert_to_torch(theta, TESTING_FLAG=True)
        dtype = torch.FloatTensor
        if USE_CUDA == True:
            dtype = torch.cuda.FloatTensor
            theta_true = theta_true.type(dtype)
        data_torch.append([X, y, theta_true, list(TF)])
    return data_torch


def glad_train_batch(train_data, valid_data=None):
    print('Training phase of grnular: batch')
    # Initialize all the models
    model_glad = glad_model(args.L, args.theta_init_offset, args.nF, args.H,  USE_CUDA=USE_CUDA) 
    print('GLAD model check: ', model_glad.rho_l1)
    if USE_CUDA == True:
        model_glad = model_glad.cuda()

    criterion_graph = get_glad_criterion()
    optimizer_glad = get_optimizers(model_glad)
    #scheduler = MultiStepLR(optimizer, milestones=[1000], gamma=0.5)
    milestones = [i for i in range(args.EPOCHS) if i%int(args.EPOCHS/2)==0 and i>0]
    print('Scheduler milestones: ', milestones, ' gamma = 0.75' )
    scheduler = MultiStepLR(optimizer_glad, milestones=milestones, gamma=0.75)
    best_valid_shd, best_valid_Fb = np.inf, -1*np.inf

    # shift the complete train_data & valid data to GPU
#    train_data = format_torch(train_data)
#    if len(valid_data)>0:
#        valid_data = format_torch(valid_data)

#    rd = np.random.choice(len(train_data), size=len(train_data), replace=False) # get a rand number
    
    rd = np.random.choice(len(train_data), size=5, replace=False) # get a rand number
    print('selecting random points for training: ', rd, len(rd))
    #for i, d in enumerate(train_data[rd]):
    for i, ri in enumerate(rd):
        X, y, theta_true, TF = train_data[ri]
        #X, y, theta_true, TF = d
#        print('che', i, d)
#        br
        do_PRINT = True # print every epoch for the 1st batch

        for epoch in range(args.EPOCHS):# in each epoch, go through the complete batch
#        for i, d in enumerate(train_data):
            #X, y, theta, TF = d
#            theta = theta.real
#            X = convert_to_torch(X, TESTING_FLAG=True)    
#            theta_true = convert_to_torch(theta, TESTING_FLAG=True)
#            dtype = torch.FloatTensor
#            if USE_CUDA == True:
#                dtype = torch.cuda.FloatTensor
#                theta_true = theta_true.type(dtype)

            batch_num = i
            if batch_num==0 and epoch==0: # checking dnn model
                model_dnn_check = dnn_model(T=len(TF), O=X.shape[1], H=args.Hd, USE_CUDA=(args.USE_CUDA_FLAG==1))
                print('DNN model check:', model_dnn_check.getDNN())
                print('#################### total points = ', X.shape[0])
            # a good init for every X
            #model_dnn, optimizer_dnn = grnular.goodINIT(X, TF, args)
            #if i==0:
            #    print('DNN model check:', model_dnn.getDNN())

            t1 = time.time()
#        for epoch in range(args.EPOCHS):# optimize the complete unrolled algo
            optimizer_glad.zero_grad()
#            theta_s, loss_glad = glad(S, theta_true, model_glad, args, criterion_graph)
#            theta_s, loss_glad = grnular.grnular(X, theta_true, TF, [model_glad, model_dnn, optimizer_dnn], args, criterion_graph, do_PRINT) # output = theta_s, loss_glad
            theta_s, loss_glad = grnular.grnular(X, theta_true, TF, model_glad, args, criterion_graph, do_PRINT) # output = theta_s, loss_glad
            theta_s = torch.squeeze(theta_s)

            if epoch % args.PRINT_EPOCH == 0:
                print('\n Epoch = ', epoch, 'Batch = ',batch_num, ' loss glad = ', loss_glad.detach().cpu().numpy()[0])
                print('Recovery :FDR, TPR, FPR, SHD, nnz_true, nnz_pred, precision, recall, Fb, aupr, auc ', *np.around(compare_theta(_npy(theta_true), theta_s), 3))
            t2 = time.time()

            loss_glad.backward() # calculate the gradients
            optimizer_glad.step() # update the weights
            scheduler.step()
            
            t3 = time.time() # running a batch

            # Choose the best model based on validation result: only update 3 times
            #if len(valid_data)>0 and epoch%int(args.EPOCHS/3)==0:# and args.MODEL_SELECT!='current':
            #if len(valid_data)>0 and (i>0) and (i % int(len(train_data)/2) ==0 or i==len(train_data)-1): #Update 2 times per epoch : and args.MODEL_SELECT!='current':
            #if len(valid_data)>0 and (i>0) and (i==len(train_data)-1) and (epoch%int(args.EPOCHS/4)==0 or epoch==args.EPOCHS-1): #Update once every 3 epochs : and args.MODEL_SELECT!='current':
            if len(valid_data)>0 and epoch>0 and (epoch%int(args.EPOCHS/2)==0 or epoch==args.EPOCHS-1): #Update once every 3 epochs : and args.MODEL_SELECT!='current':
                # best best Fb model
                curr_valid_shd, curr_valid_Fb = glad_predict_batch(model_glad, valid_data, PRINT=False)
                print('valid: shd Fb accuracy : ',  curr_valid_shd, curr_valid_Fb)
                # NOTE: only updating best glad model with 
                if curr_valid_Fb >= best_valid_Fb:
                    print('epoch = ', epoch, ' Updating the best Fb model with valid Fb = ', curr_valid_Fb)
                    best_Fb_model = copy.deepcopy(model_glad)
                    best_valid_Fb = curr_valid_Fb

                if curr_valid_shd <= best_valid_shd:
                    print('epoch = ', epoch, ' Updating the best shd model with valid shd = ', curr_valid_shd)
                    best_shd_model = copy.deepcopy(model_glad)
                    best_valid_shd = curr_valid_shd
                model_glad.train()
                t4 = time.time() # running on validation data
                print('time req for grnular forward call (secs): ', t2-t1)
                print('time req for loss backward call & update (secs): ', t3-t2)
                print('time req for validation model update (secs): ', t4-t3)
            do_PRINT = False # only print at the start of the batch and 0th epoch

    print('CHECK RHO & theta Learned, may not correspond to the best metric model: ', model_glad.rho_l1[0].weight, model_glad.theta_init_offset)
    if args.MODEL_SELECT in ['auc', 'aupr', 'Fb']: # best graph recovery metric 
        return best_Fb_model 
    elif args.MODEL_SELECT == 'shd':
        return best_shd_model 
    elif args.MODEL_SELECT == 'current': # #best GLAD auc and best CoNN acc
        return model_glad
    else:
        return 'Check model select'

def get_res_filepath():
    FILE_NUM = str(np.random.randint(10000))
    savepath = 'simulator/BEELINE-data/my_pred_networks/'
    filepath = savepath +'grnular_beeline_pred_tag'+str(FILE_NUM)+'.pickle'
    return filepath

def glad_predict_batch(model, data, PRINT=True, PREDICT_TF=False, BEELINE=False):
    print('Check Hd: ', args.Hd)
    with torch.no_grad():
        # return the mean and std_dev of the data pairs
        # putting the models in eval mode
        model.eval()
        # criterion for conn loss
        criterion_graph = get_glad_criterion()
        model_auxiliary = criterion_graph #[]
        res = [] # add res_tf
        for i, d in enumerate(data):
            X, y, theta_true, TF = d
            if BEELINE:
                print('Data set : ', i, y[:2]) 
                res.append(glad_predict_single(model, model_auxiliary, theta_true, X, TF, PRINT=True, pair_num=i, BEELINE=BEELINE))
                
            else:
#            theta_true = theta_true.real
        #        if args.USE_TF_NAMES=='yes' and PREDICT_TF:
                res.append(glad_predict_single(model, model_auxiliary, theta_true, X, TF, PRINT=False, pair_num=i))
        #    print('Check res: ', res)
        if BEELINE:
            save_res = []
            for i, r1 in enumerate(res):# r1 = [res, [theta_pred]]
                r = r1[0]
                r = ["%.3f" %x for x in r]
                print(data[i][1][:2], r)
                data_numpy = [_npy(data[i][0]), data[i][1], _npy(data[i][2]), data[i][3]]
                save_res.append([data_numpy, r1])
                #save_res.append([data[i], r1])
            print('Saving the results, [dataname, mapping, theta_pred, theta_true]')
            FILEPATH = get_res_filepath()
            print(FILEPATH)
            with open(FILEPATH, 'wb') as handle:
                pickle.dump(save_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            return
        res_mean = np.mean(np.array(res).astype(np.float64), 0)
        res_std  = np.std(np.array(res).astype(np.float64), 0)
        res_mean = ["%.3f" %x for x in res_mean]
        res_std  = ["%.3f" %x for x in res_std]
        res_dict = {} # name: [mean, std]
        for i, _name in enumerate(['FDR', 'TPR', 'FPR', 'SHD', 'nnz_true', 'nnz_pred', 'precision', 'recall', 'Fb', 'aupr', 'auc']): # dictionary
            res_dict[_name]= [res_mean[i], res_std[i]]#mean std
        if PRINT:
            #res_mean = np.array(res_mean).astype(np.float64)
            #res_std = np.array(res_std).astype(np.float64)
            mean_std = [[rm, rs] for rm, rs in zip(res_mean, res_std)]
            flat_list = [item for ms in mean_std for item in ms]
            print('%s' % ', '.join(map(str, flat_list)))
            #print(*sum(list(map(list, zip(res_mean, res_std))), []), sep=', ')
        #[SHD, Fb, accuracy]
        return float(res_dict['SHD'][0]), float(res_dict[args.MODEL_SELECT][0])
        #return float(res_dict['SHD'][0]), float(res_dict['auc'][0])


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


def get_tf_mask(tf_names, dim):
    mask = torch.zeros(dim, dim) + torch.eye(dim)
    for n in tf_names:
        mask[n, :] = 1
        mask[:, n] = 1
#    print(mask)
    return mask


def matrix_inner_product(A, B):
    return torch.sum(A * B)

def f_beta_loss(out, target, beta=args.beta, GRNUlar=True, PRINT=False):
    # NOTE: Only for binary target
    # masking the diagonals with zeros
    if GRNUlar:
        dim0, dim1 = out.shape # T, D
        maskD = torch.ones(dim0, dim1) - torch.eye(dim0, dim1)
    else: # GLAD
        dim = target.shape[-1]
        maskD = torch.ones(dim, dim) - torch.eye(dim, dim)

    if USE_CUDA == True:
        maskD = maskD.cuda()
    masked_out = maskD * out
    masked_target = maskD * target
    # re-scale the masked_out between [0, 1]
    pred = torch.tanh(torch.abs(masked_out))
    TP = matrix_inner_product(pred, masked_target)
    FP = matrix_inner_product(pred, 1-masked_target)
    FN = matrix_inner_product(1-pred, masked_target)
    #TN = matrix_inner_product(1-out_f1, 1-masked_target)
    num = (1+beta**2)*TP
    den = ((1+beta**2)*TP + beta**2 * FN + FP)
    loss_F_beta = 1.0-1.0 * num/den # (1-F_beta) negative, as we want to minimize loss
    return loss_F_beta


def mse_f_beta(masked_out, masked_target, PRINT):
    mse_criterion = nn.MSELoss()
    loss_mse = mse_criterion(masked_out, masked_target)
    loss_fb = f_beta_loss(masked_out, masked_target)
    # Getting the ratio to scale the losses to roughly the same values.
    r = loss_fb.detach()/loss_mse.detach()
    if PRINT:
        print('Different loss: mse ', loss_mse, ' fb', loss_fb, ' Balancing r = fb/mse', r)
    return r*loss_mse + loss_fb

def get_graph_from_theta(theta):
    # convert to numpy 
    theta = theta.detach().cpu().numpy()
    dim = theta.shape[0]
    mask = np.ones((dim, dim)) - np.eye(dim)
    G = nx.from_numpy_matrix(np.multiply(theta, mask))
    return G



def glad_predict_single(model, model_auxiliary, theta_true, X, TF, PRINT=True, pair_num=-1, BEELINE=False):
    model_glad = model
    criterion_graph =  model_auxiliary
    # preparing the data
#    X = convert_to_torch(X, TESTING_FLAG=True) 
#    theta_true = convert_to_torch(theta, TESTING_FLAG=True)

#    dtype = torch.FloatTensor
#    if USE_CUDA == True:
#        dtype = torch.cuda.FloatTensor
#        theta_true = theta_true.type(dtype)

    # good INIT
#    model_dnn, optimizer_dnn = grnular.goodINIT(X, TF, args, PRINT=PRINT, PREDICT=True)
    # run unrolled model
#    theta_s, loss_glad = grnular.grnular(X, theta_true, TF, [model_glad, model_dnn, optimizer_dnn], args, criterion_graph, PRINT=PRINT, PREDICT=True) # output = theta_s, loss_glad
    theta_s, loss_glad = grnular.grnular(X, theta_true, TF, model_glad, args, criterion_graph, PRINT=PRINT, PREDICT=True) # output = theta_s, loss_glad
    theta_s = torch.squeeze(theta_s)
    
    #recovery_metrics = compare_theta(theta, theta_s)
    recovery_metrics = compare_theta(_npy(theta_true), theta_s)
    itr_details = [loss_glad.detach().cpu().numpy()[0]]
    if PRINT:
        print('FDR, TPR, FPR, SHD, nnz_true, nnz_pred, precision, recall, Fb, aupr, auc')
        print('TEST: Recovery of true theta: ', *np.around(recovery_metrics, 3))
        print('TEST: glad loss', *np.around(itr_details, 3))
    

#    # POSTPROCESSING code for TF
#    if args.USE_TF_NAMES=='yes' and len(tf_names) != 0:
#        prec_tf = postprocess_tf(theta_s.detach().cpu().numpy(), tf_names)
#        recovery_metrics = report_metrics(np.array(theta), prec_tf)

    res = list(recovery_metrics)# + itr_details
    if BEELINE:
        res = [list(recovery_metrics), theta_s.detach().cpu().numpy()]
    return res 
    #return list(recovery_metrics) + itr_details # concatenate results


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
    if args.DATA_METHOD == 'sim_expt':
        print('This should work')
        train_data, valid_data, test_data = load_saved_data()
        print('Data loaded')

    if args.ADD_TECHNICAL_NOISE == 'yes':
        print('adding technical noise')
        train_data = gen_data.add_technical_noise(args, train_data)
        valid_data = gen_data.add_technical_noise(args, valid_data)
        test_data = gen_data.add_technical_noise(args, test_data)
            
    if args.DATA_METHOD in ['sim_expt']:
        # convert the data to torch format
        train_data = format_torch(train_data)
        valid_data = format_torch(valid_data)
        test_data = format_torch(test_data)
        original_Hd = args.Hd

        if TRAIN:
            print('Training the GLAD model')
            model = glad_train_batch(train_data, valid_data)
        print('*****************************************************************************')
        print('GLAD batch predict results: Number of data pairs Train/valid/test ', len(train_data), len(valid_data), len(test_data))
        #print('FDR, TPR, FPR, SHD, nnz_true, nnz_pred, precision, recall, Fb, aupr, auc, total loss, glad loss, conn loss, accuracy, ARI')
        print('FDR, ,TPR, ,FPR, ,SHD, ,nnz_true, ,nnz_pred, ,precision, ,recall, ,Fb, ,aupr, ,auc, ')
#        print('Final results on Training data')
        glad_predict_batch(model, train_data, PRINT=True)
#        print('Final results on Valid data')
        glad_predict_batch(model, valid_data, PRINT=True)
#        print('Model trained, now predicting on test data')
        glad_predict_batch(model, test_data, PRINT=True)
#        print('\n With TF_NAMES: FDR, ,TPR, ,FPR, ,SHD, ,nnz_true, ,nnz_pred, ,precision, ,recall, ,Fb, ,aupr, ,auc, ')
#        print('Final results on Training data')
#        glad_predict_batch(model, train_data, PRINT=True, PREDICT_TF=True)
#        print('Final results on Valid data')
#        glad_predict_batch(model, valid_data, PRINT=True, PREDICT_TF=True)
#        print('Model trained, now predicting on test data')
#        glad_predict_batch(model, test_data, PRINT=True, PREDICT_TF=True)

    return 

if __name__=="__main__":
   main()
