import torch
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np
import scipy.linalg

class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        itr_TH = 10 # number of iterations threshold 
        dim = input.shape[0]
        norm = torch.norm(input)#.double())
        #Y = input.double()/norm
        Y = input/norm
        I = torch.eye(dim,dim,device=input.device)#.double()
        Z = torch.eye(dim,dim,device=input.device)#.double()
        #print('Check: ', Y.type(), I.type(), Z.type())
        for i in range(itr_TH):
            T = 0.5*(3.0*I - Z.mm(Y))
            Y = Y.mm(T)
            Z = T.mm(Z)
        sqrtm = Y*torch.sqrt(norm)
        ctx.mark_dirty(Y,I,Z)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        itr_TH = 10 # number of iterations threshold 
        grad_input = None
        sqrtm, = ctx.saved_tensors
        dim = sqrtm.shape[0]
        norm = torch.norm(sqrtm)
        A = sqrtm/norm
        I = torch.eye(dim, dim, device=sqrtm.device)#.double()
        #Q = grad_output.double()/norm
        Q = grad_output/norm
        for i in range(itr_TH):
            Q = 0.5*(Q.mm(3.0*I-A.mm(A))-A.t().mm(A.t().mm(Q)-Q.mm(A)))
            A = 0.5*A.mm(3.0*I-A.mm(A))
        grad_input = 0.5*Q
        return grad_input
sqrtm = MatrixSquareRoot.apply


#class MatrixSquareRoot(Function):
#    @staticmethod
#    def forward(ctx, input):
#        #m = input.numpy().astype(np.float_)
#        m = input.detach().cpu().numpy().astype(np.float_)
#        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real)#.type_as(input)
#        ctx.save_for_backward(sqrtm) # save in cpu
#        sqrtm = sqrtm.type_as(input)
#        return sqrtm
#
#    @staticmethod
#    def backward(ctx, grad_output):
#        grad_input = None
#        if ctx.needs_input_grad[0]:
#            #sqrtm, = ctx.saved_variables
#            sqrtm, = ctx.saved_tensors
#            #sqrtm = sqrtm.data.numpy().astype(np.float_)
#            sqrtm = sqrtm.data.numpy().astype(np.float_)
#            #gm = grad_output.data.numpy().astype(np.float_)
#            gm = grad_output.data.cpu().numpy().astype(np.float_)
##            gm = np.eye(grad_output.shape[-1])
#            # Given a positive semi-definite matrix X,
#            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
#            # matrix square root dX^{1/2} by solving the Sylvester equation:
#            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
#            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)
#            grad_input = torch.from_numpy(grad_sqrtm).type_as(grad_output.data)
#        return Variable(grad_input)


#sqrtm = MatrixSquareRoot.apply

def original_main():
    from torch.autograd import gradcheck
    k = torch.randn(20, 10).double()
    # Create a positive definite matrix
    pd_mat = k.t().matmul(k)
    pd_mat = Variable(pd_mat, requires_grad=True)
    test = gradcheck(MatrixSquareRoot.apply, (pd_mat,))
    print(test)

def single_main():
    from torch.autograd import gradcheck 
    n = 1
    A = torch.randn( 20, 10).double()
    # Create a positive definite matrix
    pd_mat = A.t().matmul(A)
    pd_mat = Variable(pd_mat, requires_grad=True)
    test = gradcheck(MatrixSquareRoot.apply, (pd_mat,))
    print(test)

    #sqrtm_scipy = np.zeros_like(A)
    print('err: ', pd_mat)
    sqrtm_scipy = scipy.linalg.sqrtm(pd_mat.detach().numpy().astype(np.float_))
#    for i in range(n):
#        sqrtm_scipy[i] = sqrtm(pd_mat[i].detach().numpy())
    sqrtm_torch = sqrtm(pd_mat)
    print('sqrtm torch: ', sqrtm_torch)
    print('scipy', sqrtm_scipy)
    print('Difference: ', np.linalg.norm(sqrtm_scipy - sqrtm_torch.detach().numpy()))

def main():# batch
    from torch.autograd import gradcheck 
    n = 2
    A = torch.randn(n, 4, 5).double()
    A.requires_grad = True
    # Create a positive definite matrix
    #pd_mat = A.t().matmul(A)
    pd_mat = torch.matmul(A.transpose(-1, -2), A)
    pd_mat = Variable(pd_mat, requires_grad=True)
    print('err: ', pd_mat.shape)
    #test = gradcheck(MatrixSquareRoot.apply, (pd_mat,))
    #print(test)

    sqrtm_scipy = np.zeros_like(pd_mat.detach().numpy())
    #sqrtm_scipy = scipy.linalg.sqrtm(pd_mat.detach().numpy().astype(np.float_))
    for i in range(n):
        sqrtm_scipy[i] = scipy.linalg.sqrtm(pd_mat[i].detach().numpy())
    # batch implementation
    sqrtm_torch = torch.zeros(pd_mat.shape)
    for i in range(n):
        sqrtm_torch[i] = sqrtm(pd_mat[i])
    #sqrtm_torch = sqrtm(pd_mat)
    print('sqrtm torch: ', sqrtm_torch)
    print('scipy', sqrtm_scipy)
    print('Difference: ', np.linalg.norm(sqrtm_scipy - sqrtm_torch.detach().numpy()))

if __name__ == '__main__':
    main()
