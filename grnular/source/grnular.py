import torch
import torch.nn as nn
import numpy as np
from grnular.utils.torch_sqrtm import MatrixSquareRoot
from grnular.source.grnular_model import dnn_model

torch_sqrtm = MatrixSquareRoot.apply
def batch_matrix_sqrt(A):
    # A should be PSD
    # if shape of A is 2D, i.e. a single matrix
    if len(A.shape)==2:
        return torch_sqrtm(A)
    else:
        n = A.shape[0]
        sqrtm_torch = torch.zeros(A.shape).type_as(A)
        for i in range(n):
            sqrtm_torch[i] = torch_sqrtm(A[i])
        return sqrtm_torch


def get_frobenius_norm(A, single=False):
    if single:
        return torch.sum(A**2)
    return torch.mean(torch.sum(A**2, (1,2)))


def normalizing_data(X):
    #print('Centering and scaling the input data...')
    scaledX = X - X.mean(0)
    scaledX = scaledX/X.std(0)
    # NOTE: replacing all nan's by 0, as sometimes in dropout the complete column
    # goes to zero
    scaledX = convert_nans_to_zeros(scaledX)
    return scaledX

def convert_nans_to_zeros(X):
    X[X != X] = 0
    return X


def get_OT_submatrix(A, TF):
    # Choose the lower diagonal part
    # ignore diagonal as well
    # select the columns with TF 
    return A.triu(1)[TF, :] # TxD
    #return A.tril(-1)[:, TF] # DxT


def covTF(X, TF):
    Xc = normalizing_data(X)
    S = torch.matmul(Xc.t(), Xc)/Xc.shape[0]
    St = get_OT_submatrix(S, TF)
    return St

def get_prod_weights(model_dnn):
    for i, (n, p) in enumerate(model_dnn.DNN.named_parameters()):
        if i==0:
            if 'weight' in n:
                W = torch.abs(p).t() # TxH
#                print('i W ', i, W.shape)
        else:# i > 0
            if 'weight' in n:
                W = torch.matmul(W, torch.abs(p).t())
#                print('i W ', i, W.shape)
    #W = W.t()
#    print('prod shape: ', W.shape) # T x D 
    return W


def goodINIT(X, TF, args, PRINT=True, PREDICT=False):
    torch.set_grad_enabled(True)
    Xt = X[:, TF]
    model_dnn = dnn_model(T=len(TF), O=X.shape[1], H=args.Hd, USE_CUDA= (args.USE_CUDA_FLAG==1))
    optimizer_dnn = torch.optim.Adam(model_dnn.parameters(), lr=args.lrDNN, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    criterion1 = nn.MSELoss()
    for e in range(args.DNN_EPOCHS):
        optimizer_dnn.zero_grad()
        Xp = model_dnn.DNN(Xt)
        loss_dnn = criterion1(Xp, X)
        loss_dnn.backward() # calculate the gradients
        optimizer_dnn.step() # update the weights
        if e%int(args.DNN_EPOCHS/10)==0 and PRINT:
            print('INIT Dnn epoch = ', e, ' Loss DNN: ', loss_dnn)
    if PREDICT:
        torch.set_grad_enabled(False)
    return [model_dnn, optimizer_dnn]


def optDNN(model_details, X, Z, TF, lambda_k, args, PRINT, PREDICT):
    torch.set_grad_enabled(True)
    model_dnn, optimizer_dnn = model_details
    Xt = X[:, TF]
#    print('T D: ', Z.shape, Xt.shape, args.Hd)
    # fit the regression using NN
    # init the NN model
    # NOTE: original model implemented
#    model_dnn = dnn_model(T=Z.shape[0], O=Z.shape[1], H=args.Hd, USE_CUDA= (args.USE_CUDA_FLAG==1))
#    optimizer_dnn = torch.optim.Adam(model_dnn.parameters(), lr=args.lrDNN, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    #for p in range(args.DNN_EPOCHS):
    for p in range(args.P):
        #print('inner unroll p = ', p)
        optimizer_dnn.zero_grad()
        Xp = model_dnn.DNN(Xt)
#        print('Xt xp X ', Xt.shape, Xp.shape, X.shape)
        loss1 = criterion1(Xp, X)
        prod_W = get_prod_weights(model_dnn)
        # NOTE: try retain graph = true and remove the detach
        loss2 = lambda_k * criterion2(prod_W, Z)
        #loss2 = criterion2(prod_W, Z)
        #loss2 = criterion2(prod_W, Z)
        # total loss 
        loss_dnn = loss1 + loss2
        loss_dnn.backward() # calculate the gradients
        #loss_dnn.backward(retain_graph=True) # calculate the gradients
        optimizer_dnn.step() # update the weights
        #if p%int(args.DNN_EPOCHS/10)==0 and PRINT:
        if PRINT and p%10==0:
            print('Dnn epoch = ', p, ' Loss DNN: ', loss_dnn)
    prod_W = get_prod_weights(model_dnn)#.detach()
    if PREDICT:
        torch.set_grad_enabled(False)
#    prod_W.requires_grad = False
#    del model_dnn
#    print('prod W: ', prod_W)
    model_details = [model_dnn, optimizer_dnn]
    return prod_W, model_details #'model_details'

def old_reconstruct_adj(Z):# Z = TxD
    T, D = Z.shape
    # reconstruct the adj matrix
    theta_s = torch.zeros(D, D).type(Z.type())
    theta_st = torch.zeros(D, D).type(Z.type())
    theta_s[:T, :D] = Z
    theta_st[:D, :T] = Z.t()
    theta_s = theta_s + theta_st # Note: only symmetric
    theta_s[:T, :T] = theta_s[:T, :T]/2.0
#    print('CHECK symmetric: ', check_symmetric(theta_s.detach().cpu().numpy()))
#    brr
    return theta_s



def reconstruct_adj(Z, TF):# Z = TxD
    T, D = Z.shape
    # reconstruct the adj matrix
    theta_s = torch.zeros(D, D).type(Z.type())
    theta_st = torch.zeros(D, D).type(Z.type())
    theta_s[TF, :D] = Z
    theta_st[:D, TF] = Z.t()
    theta_s = theta_s + theta_st # Note: only symmetric
    theta_s = theta_s/2.0
#    print('CHECK symmetric: ', check_symmetric(theta_s.detach().cpu().numpy()))
#    brr
    return theta_s



    
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def nonL(Z):
#    return torch.sigmoid(Z)
#    return torch.relu(Z)
    return Z

def grnular(X, theta_true, TF, models, args, criterion_graph, PRINT=True, PREDICT=False):
    #model_glad, model_dnn, optimizer_dnn = models
    model_glad = models
    do_PRINT = False
    zero = torch.Tensor([0])
    dtype = torch.FloatTensor
    if args.USE_CUDA_FLAG == 1:
        zero = zero.cuda()
        dtype = torch.cuda.FloatTensor

    # good INIT
    #model_dnn, optimizer_dnn = goodINIT(X, TF, args, PRINT=PRINT, PREDICT=PREDICT)
    # for fast
    model_details = goodINIT(X, TF, args, PRINT=PRINT, PREDICT=PREDICT)
    #model_details = 'original'    


    lambda_k = model_glad.lambda_forward(zero + args.lambda_init, zero,  k=0)
    loss_glad = torch.Tensor([0]).type(dtype)

    St = covTF(X, TF)# T x D 
    Z = torch.zeros(St.shape).type(dtype).detach() # T x D
    St.requires_grad = False
    theta_true = get_OT_submatrix(theta_true, TF)
#    print('St check: ', St.shape, TF)
#    print('CHECK grad: ', Z, St, theta_true, X)
    for k in range(args.L):
#        print('k = ', k)
        #theta_k = optDNN(X, Z, TF, lambda_k, args)# DxT
        if PRINT and k==args.L-1: # only print in last iteration of K
            do_PRINT = True
        theta_k, model_details = optDNN(model_details, X, Z.detach(), TF, lambda_k.detach(), args, do_PRINT, PREDICT)# DxT
        #theta_k = optDNN([model_dnn, optimizer_dnn], X, Z.detach(), TF, lambda_k.detach(), args, do_PRINT, PREDICT)# DxT
        #theta_k = optDNN([model_dnn, optimizer_dnn], X, Z, TF, lambda_k, args, do_PRINT, PREDICT)# DxT
        Z = model_glad.eta_forward(theta_k, St, k, Z)#.detach() 
        # update the lambda
        lambda_k = model_glad.lambda_forward(torch.Tensor([get_frobenius_norm(Z-theta_k, single=True)]).type(dtype), lambda_k, k)
        # the Z values should be between [0, 1] for optimization
#        print('Z before:', Z, Z[0])
#        Z = torch.sigmoid(Z) # use abs or not?, torch.tanh(torch.abs(Z))
#        print('Z after sigmoid', Z)
        # if we use abs then sigmoid will only go to min 0.5, should use tanh?
        #loss_glad += criterion_graph(torch.sigmoid(Z), theta_true)/args.L
        loss_glad += criterion_graph(nonL(Z), theta_true, PRINT=do_PRINT)/args.L
#        loss_glad += criterion_graph(Z, theta_true)/args.L
#        print('loss grnular: ', loss_glad)
#        brr
    # get the theta_s 
#    theta_s = reconstruct_adj(Z) # TxD -> DxD 
    theta_s = reconstruct_adj(nonL(Z), TF) # TxD -> DxD 
    return theta_s, loss_glad
