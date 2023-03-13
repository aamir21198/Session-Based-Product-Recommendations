import theano
from theano import tensor as T

def gpu_diag_wide(X):
    E = T.eye(*X.shape)
    return T.sum(X*E, axis=1)

def gpu_diag_tall(X):
    E = T.eye(*X.shape)
    return T.sum(X*E, axis=0)