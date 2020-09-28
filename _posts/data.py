
import numpy as np

# Data: For A,  rows are points
def getSimulationData(n_number, n_number_outlier, NOOUTLIER = True, PLOT = False):
    '''produce n_number*2 matrix'''
    import numpy as np
    import matplotlib.pyplot as plt
    # np.random.seed(42)
    
    x1 = np.linspace(60, 70, n_number).reshape((n_number,-1))
    noise = np.random.normal(10, 10, n_number).reshape((n_number,-1))
    print(noise)
    x1_o = np.linspace(10, 20, n_number_outlier).reshape((n_number_outlier,-1))
    x2_o = np.random.normal(10, 10, n_number_outlier).reshape((n_number_outlier,-1))

    A_out = np.append(x1_o, x2_o, axis=1)
    # print(A_out)
    A_in = np.append(x1, -2*x1+noise, axis=1)
    A = np.append(A_in, A_out).reshape((n_number+n_number_outlier,-1))
    if(NOOUTLIER):
        A =A_in
    
    if(PLOT):
        plt.figure(figsize = (7,7))
        plt.axis('equal')
        plt.title("Data", fontsize=14)
        
        plt.scatter(A_in[:,0],A_in[:,1], alpha=0.5)
        plt.scatter(A_out[:,0],A_out[:,1], alpha=0.5)

        print(A[0:5,])
        print()
        print('Insider shape:', A_in.shape)
        print('Outlier shape:', A_out.shape)
        print('shape:', A.shape)
        plt.show()
    
    return A, A_in, A_out

# real data

# csv format: feature_0, feature_1, .., feature_{n-1}, label
# n features followed by one label
# features are real, label is int. Label values: 0..NUM_CATEGORIES-1

def read_csv(filename, shuffle=False):
    import numpy as np
    tmp_matrix = np.genfromtxt(filename, delimiter=",")
 
    if shuffle:
        np.random.shuffle(tmp_matrix)
    X = tmp_matrix[:,0:-1]
    y = tmp_matrix[:,-1]
    y = np.array(y).astype("int32")
    return(X, y)

def readDataC(filename, shuffle=False):
    print('Read data from '+filename)
    (X_train, y_train) = read_csv(filename, shuffle)

    NUM_CATEGORIES = np.max(y_train)+1
    if (NUM_CATEGORIES != 2):
        print('!!!error?')

    n, m = X_train.shape

    idx = ( np.where(y_train==1))[0]
    X1 = (np.delete(X_train, idx, axis=0)).reshape((-1,m))

    idx = ( np.where(y_train==0))[0]
    X2 = (np.delete(X_train, idx, axis=0)).reshape((-1,m))
    return X1, X2

def getRealData(filename, shuffle=False):
    print('Read data from '+filename)
    (X_train, y_train) = read_csv(filename, shuffle)
    A = X_train
    return A

# Data: For A,  rows are points
def getSimpleSimulationData(n_points, n_outliers):
    '''produce n_number*2 matrix'''
    import numpy as np
    np.random.seed(42)
    
    x1 = np.linspace(10, 50, n_points).reshape((n_points,-1))

    x1_o = np.linspace(70, 1000, n_outliers).reshape((n_outliers,-1))
    x2_o = np.random.normal(40, 5, n_outliers).reshape((n_outliers,-1))

    A_in = np.append(x1, 2*x1, axis=1)
    A_out = np.append(x1_o, x2_o, axis=1)
    
    A = np.append(A_in, A_out).reshape((n_points+n_outliers,-1))
    
    return A

    # Data: For A,  rows are points
def getSimpleSimulationDataCross(n_points, n_outliers):
    '''produce n_number*2 matrix'''
    import numpy as np
    np.random.seed(42)
    
    x1 = np.linspace(10, 50, n_points).reshape((n_points,-1))

    x1_o = np.linspace(-15, 22, n_outliers).reshape((n_outliers,-1))
    x2_o = np.random.normal(40, 5, n_outliers).reshape((n_outliers,-1))

    A_in = np.append(x1, 2*x1, axis=1)
    A_out = np.append(x1_o, x2_o, axis=1)
    
    A = np.append(A_in, A_out).reshape((n_points+n_outliers,-1))
    
    return A

    # Data: For A,  rows are points
def getSimpleSimulationData3Dline(n_points, n_outliers):
    '''produce n_number*2 matrix'''
    import numpy as np
    np.random.seed(42)
    
    x = np.linspace(10, 50, n_points).reshape((n_points,-1))
    y = np.linspace(-50, 200, n_points).reshape((n_points,-1))
    z = -2*x + 5*y

    x_o = np.linspace(-50, 50, n_outliers).reshape((n_outliers,-1))
    y_o = np.random.normal(40, 40, n_outliers).reshape((n_outliers,-1))
    z_o = np.random.normal(100, 100, n_outliers).reshape((n_outliers,-1))

    A_in = np.append(x, y, axis=1)
    A_in = np.append(A_in, z, axis=1)

    A_out = np.append(x_o, y_o, axis=1)
    A_out = np.append(A_out, z_o, axis=1)

    A = np.append(A_in, A_out).reshape((n_points+n_outliers,-1))
    
    return A

    # Data: For A,  rows are points
def getSimpleSimulationData3Dplane(n_points, n_outliers):
    '''produce n_number*2 matrix'''
    import numpy as np
    np.random.seed(42)
    
    x = np.linspace(-100, 100, n_points).reshape((n_points,-1))
    y = np.random.normal(0, 300, n_points).reshape((n_points,-1))
    z = 5*x + 5*y

    x_o = np.random.normal(0, 200, n_outliers).reshape((n_outliers,-1))
    y_o = np.random.normal(0, 100, n_outliers).reshape((n_outliers,-1))
    z_o = np.random.normal(200, 2000, n_outliers).reshape((n_outliers,-1))

    A_in = np.append(x, y, axis=1)
    A_in = np.append(A_in, z, axis=1)

    A_out = np.append(x_o, y_o, axis=1)
    A_out = np.append(A_out, z_o, axis=1)

    A = np.append(A_in, A_out).reshape((n_points+n_outliers,-1))
    
    return A,A_in,A_out

def getSimpleSimulationData6D(n_points, n_outliers):
    '''produce n_number*2 matrix'''
    import numpy as np
    np.random.seed(42)
    
    x1 = np.linspace(-100, 100, n_points).reshape((n_points,-1))
    x2 = np.linspace(-100, 100, n_points).reshape((n_points,-1))
    x3 = np.random.normal(100, 200, n_points).reshape((n_points,-1))
    x4 = np.random.normal(200, 200, n_points).reshape((n_points,-1))
    x5 = np.linspace(-100, 100, n_points).reshape((n_points,-1))
    y = -2*x1 + 5*x2 + 6*x3 + 7*x4 + x5

    x1_o = np.random.normal(0, 200, n_outliers).reshape((n_outliers,-1))
    x2_o = np.random.normal(40, 40, n_outliers).reshape((n_outliers,-1))
    x3_o = np.random.normal(100, 300, n_outliers).reshape((n_outliers,-1))
    x4_o = np.random.normal(-100, 100, n_outliers).reshape((n_outliers,-1))
    x5_o = np.random.normal(10, 100, n_outliers).reshape((n_outliers,-1))
    x6_o = np.random.normal(10, 100, n_outliers).reshape((n_outliers,-1))

    A_in = np.append(x1, x2, axis=1)
    A_in = np.append(A_in, x3, axis=1)
    A_in = np.append(A_in, x4, axis=1)
    A_in = np.append(A_in, x5, axis=1)
    A_in = np.append(A_in, y, axis=1)

    A_out = np.append(x1_o, x2_o, axis=1)
    A_out = np.append(A_out, x3_o, axis=1)
    A_out = np.append(A_out, x4_o, axis=1)
    A_out = np.append(A_out, x5_o, axis=1)
    A_out = np.append(A_out, x6_o, axis=1)

    A = np.append(A_in, A_out).reshape((n_points+n_outliers,-1))
    
    return A

def getSimpleSimulationData5D(n_points, n_outliers):
    '''produce n_number*2 matrix'''
    import numpy as np
    np.random.seed(42)
    
    x1 = np.linspace(-100, 100, n_points).reshape((n_points,-1))
    x2 = np.linspace(-100, 100, n_points).reshape((n_points,-1))
    x3 = np.random.normal(100, 200, n_points).reshape((n_points,-1))
    x4 = np.random.normal(200, 200, n_points).reshape((n_points,-1))
    y = -2*x1 + 5*x2 + 6*x3 + 7*x4

    x1_o = np.random.normal(0, 200, n_outliers).reshape((n_outliers,-1))
    x2_o = np.random.normal(40, 40, n_outliers).reshape((n_outliers,-1))
    x3_o = np.random.normal(100, 300, n_outliers).reshape((n_outliers,-1))
    x4_o = np.random.normal(-100, 100, n_outliers).reshape((n_outliers,-1))
    x5_o = np.random.normal(10, 100, n_outliers).reshape((n_outliers,-1))

    A_in = np.append(x1, x2, axis=1)
    A_in = np.append(A_in, x3, axis=1)
    A_in = np.append(A_in, x4, axis=1)
    A_in = np.append(A_in, y, axis=1)

    A_out = np.append(x1_o, x2_o, axis=1)
    A_out = np.append(A_out, x3_o, axis=1)
    A_out = np.append(A_out, x4_o, axis=1)
    A_out = np.append(A_out, x5_o, axis=1)

    A = np.append(A_in, A_out).reshape((n_points+n_outliers,-1))
    
    return A