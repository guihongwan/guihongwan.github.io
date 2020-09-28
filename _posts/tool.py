
import numpy as np
import scipy

def chToDir(newpath):
    import os
    os.chdir(newpath)

def getRandomMatrix(m, k, mean=0, std=1, seed=-1):
    '''
    Generate randomly(Gaussian) a matrix; the shape is m by k.
    '''
    if seed > 0:
        np.random.seed(seed)
    return np.random.normal(mean, std, m*k).reshape((m,k))

def getOrthogonalMatrix(m, k, mean=0, std=1, seed = -1):
    '''
    Generate randomly(Gaussian) a matrix; the shape is m by k.
    And then QR s.t. the columns of produced matrix are orthogonal.
    Q.T@Q=I
    '''
    if seed > 0:
        np.random.seed(seed)
    H = getRandomMatrix(m, k, mean, std, seed)
    Q, R = np.linalg.qr(H)
    return Q

def getOrthogonalMatrixUniform(m, k, seed = -1):
    '''
    Generate randomly(Uniform) a matrix;
    The columns of produced matrix are orthogonal
    Q.T@Q=I
    '''
    if seed > 0:
        np.random.seed(seed)
        
    H = np.random.rand(m,k)
    Q, R = np.linalg.qr(H)
    return Q
    

def UncenterEigen(At):
    '''
    Uncenter PCA: all of eigen values and eigenvectors of At@At.T
    ''' 
    s, v =scipy.linalg.eigh(At.dot(At.T))

    sorted_idxes = np.argsort(-s)
    s = s[sorted_idxes]
    v = v[:, sorted_idxes]
        
    return s, v # sorted

def centered_PCA(X):
    mean = np.mean(X, axis=1)
    mean = mean.reshape((mean.shape[0], 1))
    X_red = X - mean
    e, V = scipy.linalg.eigh(X_red@X_red.T)

    sorted_idxes = np.argsort(-e)
    e = e[sorted_idxes]
    V = V[:, sorted_idxes]

    return (mean, e, V) # sorted

def EVD(B,k=-1):
    '''
    The full EVD of B or R.
    @input: B or R
    B = AA.T
    R = (1/n)AA.T
    '''
    e, V = scipy.linalg.eigh(B)

    sorted_idxes = np.argsort(-e)
    e = e[sorted_idxes]
    V = V[:, sorted_idxes]

    return (e, V)

def EVDSparse(B,k):
    '''
    The full EVD of B or R.
    @input: B or R
    B = AA.T
    R = (1/n)AA.T
    '''
    e, V = scipy.sparse.linalg.eigsh(B,k)

    sorted_idxes = np.argsort(-e)
    e = e[sorted_idxes]
    V = V[:, sorted_idxes]

    return (e, V)

def EVDnotPSD(B):
    '''
    The full EVD of B or R.
    @input: B or R
    B = AA.T
    R = (1/n)AA.T
    B and R are not PSD.
    '''
    e, V = scipy.linalg.eig(B)
    # print('D:', e)
    # print('B-B.T:\n', B-B.T)
    # print('B:\n', B)

    sorted_idxes = np.argsort(-e)
    e = e[sorted_idxes]
    V = V[:, sorted_idxes]

    return (e.real, V.real)

def isPSD(B):
    '''B is a sqaure matrix'''
    e, V = scipy.linalg.eig(B)
    e = e.real
    for i in range(len(e)):
        if e[i] <= 0:
            print(e[i])
            return False
    return True
def randomSign():

    v = np.random.randint(0, 2)
    if v == 0:
        return 1
    if v ==1:
        return -1

def randomInt(low, high, size):
    '''
    return a list 
    with all values are integers, no repeating, 
    and with len equals to size.
    '''

    ret = []
    while(len(ret) < size):
        v = np.random.randint(low, high)
        if v not in ret:
            ret.append(v)
    return ret

def normalizeP(P):
    '''
    P is a list
    '''

    Z = sum(P)
    if Z != 0:
        P = P/Z
    return P

def normalizeDict(P):
    if sum(P.values()) == 0:
        factor = 0
    else:
        factor=1.0/sum(P.values())
    normalised_P = {k: v*factor for k, v in P.items()}
    return normalised_P

def argmaxDict(P):
    import operator
    return max(P.items(), key=operator.itemgetter(1))[0]

def sortDictbyKey(d):
    import collections
    return collections.OrderedDict(sorted(d.items()))

def printred(args):
    CRED = '\033[91m'
    CEND = '\033[0m'
    print(CRED+args+CEND)

def trueUncenterEVD(At, k):

    m,n = At.shape
    C = (At@At.T)/n
    s, v =scipy.linalg.eigh(C)

    sorted_idxes = np.argsort(-s)
    s = s[sorted_idxes]
    v = v[:, sorted_idxes]
    tD = s[0:k]
    tV = v[:,0:k]

    return tD,tV

def trueCenterEVD(At, k):
    m,n = At.shape
    mean = np.mean(At, axis=1)
    mean = mean.reshape((mean.shape[0], 1))
    At_c = At - mean
    C = (At_c@At_c.T)/n
    s, v =scipy.linalg.eigh(C)

    sorted_idxes = np.argsort(-s)
    s = s[sorted_idxes]
    v = v[:, sorted_idxes]
    tD = s[0:k]
    tV = v[:,0:k]

    return tD,tV

def coverToList(data, method='integer'):
    '''
    Convert string list to float or integer list
    input should be something like [30, 31, 32] or [30.1, 31.3, 32.0]
    method: float, or integer(default)
    '''
    data = data.split("[")[1]
    data = data.split("]")[0]
    data = data.split(",")
    data = list(data)
    ret =[]
    for i, item in enumerate(data):
        item = item.split(' ')
        item = item[len(item)-1]
        if method == 'float':
            ret.append(float(item))
        if method == 'integer':
            ret.append(int(item))
    return ret






