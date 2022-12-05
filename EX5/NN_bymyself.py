import pandas as pd
import numpy as np
from scipy import  signal
from Networks import Network
from fullconnectedlayers import FClayers

def loss(y_true,y_predict):
    return (np.sum((y_predict-y_true)**2))/y_true.shape[0]

def loss_prime(y_true,y_predict):
    return y_predict-y_true

def sgolay( z, window_size, order, derivative=None):
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return signal.fftconvolve(Z, -r, mode='valid'), signal.fftconvolve(Z, -c, mode='valid')
    
    
    
    
data = pd.ExcelFile("exact.xlsx")
data_frame = data.parse('Sheet1')

R=data_frame['R']
J=data_frame['J']

R=np.array(R)
J=np.array(J)
R_test=np.hstack((np.array(-2),R))
J_test=np.hstack((np.array(3),J))


R_test=np.expand_dims(R_test,1)
J_test=np.expand_dims(J_test,1)


data_2D=np.concatenate((R_test,J_test),1)
#denoise
data_2D[1:]=sgolay(data_2D[1:],29,4)


input=data_2D[:-1,:]
output_truth_Rd=np.array(data_2D[1:,0],'float32')
output_truth_Jd=np.array(data_2D[1:,1],'float32')
output_truth_Jd=np.expand_dims(output_truth_Jd,1)
output_truth_Rd=np.expand_dims(output_truth_Rd,1)

arr_input=[]
arr_output_Rd=[]
arr_output_Jd=[]

arr_input.append(input)
arr_output_Rd.append(output_truth_Rd)
arr_output_Jd.append(output_truth_Jd)


net_Rd=Network()
net_Rd.add(FClayers((1000,2),(1000,1)))
net_Rd.setup_loss(loss,loss_prime)
net_Jd=Network()
net_Jd.add(FClayers((1000,2),(1000,1)))
net_Jd.setup_loss(loss,loss_prime)


net_Rd.fit(arr_input,arr_output_Rd,True,0.001,1000)
net_Jd.fit(arr_input,arr_output_Jd,False,0.001,1000)

print('a coe : ',net_Rd.layers[0].weights[0])
print('b coe : ',net_Rd.layers[0].weights[1])
print('c coe : ',net_Jd.layers[0].weights[0])
print('d coe : ',net_Jd.layers[0].weights[1])

