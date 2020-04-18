import torch
from simulator.sim_expt3.source.torch_sqrtm import MatrixSquareRoot

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


def glad(Sb, theta_true, model, args, criterion_graph, mask_tf=[]):
    USE_CUDA = False
    if args.USE_CUDA_FLAG == 1:
        USE_CUDA = True

#    print('Checking the glad function: Sb ', Sb, args.D)
    # if batch is 1, then reshaping Sb
    if len(Sb.shape)==2:
        Sb = Sb.reshape(1, Sb.shape[0], Sb.shape[1])
    # Initializing the theta
    if args.INIT_DIAG == 1:
        #print(' extract batchwise diagonals, add offset and take inverse')
        batch_diags = 1/(torch.diagonal(Sb, offset=0, dim1=-2, dim2=-1) + model.theta_init_offset)
        theta_init = torch.diag_embed(batch_diags)
    else:
        #print('***************** (S+theta_offset*I)^-1 is used')
        #theta_init = torch.inverse(Sb+model.theta_init_offset * torch.eye(args.D).expand_as(Sb).type_as(Sb))
        theta_init = torch.inverse(Sb+model.theta_init_offset * torch.eye(Sb.shape[-1]).expand_as(Sb).type_as(Sb))

    theta_pred = theta_init#[ridx]
    if len(mask_tf) != 0:
#        print('Masking theta_pred')
        theta_pred = theta_pred * mask_tf
    identity_mat = torch.eye(Sb.shape[-1]).expand_as(Sb)
    # diagonal mask
#    mask = torch.eye(Sb.shape[-1], Sb.shape[-1]).byte()
    dim = Sb.shape[-1]
#    mask1 = torch.ones(dim, dim) - torch.eye(dim, dim)
    if USE_CUDA == True:
        identity_mat = identity_mat.cuda()
#        mask = mask.cuda()
#        mask1 = mask1.cuda()
    #print('ERRR check: ', theta_pred.shape, get_frobenius_norm(theta_pred), get_frobenius_norm(theta_pred).shape)
    zero = torch.Tensor([0])
    dtype = torch.FloatTensor
    if USE_CUDA == True:
        zero = zero.cuda()
        dtype = torch.cuda.FloatTensor

    lambda_k = model.lambda_forward(zero + args.lambda_init, zero,  k=0)
    loss_glad = torch.Tensor([0]).type(dtype)
    for k in range(args.L):
#        print('inner loop of glad start = ', k, theta_pred)#, theta_true[ridx])
        # GLAD CELL
        b = 1.0/lambda_k * Sb - theta_pred
        b2_4ac = torch.matmul(b.transpose(-1, -2), b) + 4.0/lambda_k * identity_mat
        sqrt_term = batch_matrix_sqrt(b2_4ac)
        theta_k1 = 1.0/2*(-1*b+sqrt_term)      
        #print('inner loop of glad theta_k1= ', k, theta_k1)#, theta_true[ridx])

        theta_pred = model.eta_forward(theta_k1, Sb, k, theta_pred) 
        if len(mask_tf) != 0:
            theta_pred = theta_pred * mask_tf
#        print('inner loop of glad end = ', k, theta_pred)#, theta_true[ridx])
        # update the lambda
        lambda_k = model.lambda_forward(torch.Tensor([get_frobenius_norm(theta_pred-theta_k1)]).type(dtype), lambda_k, k)
        
        # Since, we do not want to optimize for the diagonal match, we set the theta_true diagonal same as theta_pred
#        print('diag check: ', theta_true.shape[-1], theta_pred, theta_pred.shape)
#        print(theta_pred[0].diag(), theta_pred[0].diag().shape)
#        theta_true[torch.eye(theta_true.shape[-1]).byte()] = theta_pred[0].diag()
        #if criterion_graph!= None:
#        print('MASKING the diagonal for glad loss')
#        loss_glad += criterion_graph(theta_pred*mask1, theta_true*mask1)/args.L
        #loss_glad += criterion_graph(theta_pred.masked_fill_(mask, 0), theta_true.masked_fill_(mask, 0))/args.L
#        print('NOT MASKING the diagonal for glad loss')
        loss_glad += criterion_graph(theta_pred, theta_true)/args.L

    return theta_pred, loss_glad
