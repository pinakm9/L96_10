import os
import sys
#module_path = os.path.abspath(os.path.join('..'))
#if module_path not in sys.path:
#    sys.path.append(module_path)
import numpy as np
import tensorflow as tf
import wasserstein as tfw
import tensorflow_probability as tfp
tfd = tfp.distributions

distance=np.zeros((100,10))

# Code for shortening the object in time :
def shorten(arr, factor=10):
    arr_shape = list(arr.shape)
    arr_shape[0] = int(arr_shape[0]/factor)
    new_arr = np.zeros(arr_shape)
    for i in range(arr_shape[0]):
        new_arr[i] = arr[i*factor]
    return new_arr

# Change here for different observation gap and observation covariance parameter
mu=0.1
ob_gap=0.1

str_='/home/shashank/Lorenz_63/10_dim_L96'
os.chdir(str_)

#config1
ens_cov1=1.0  #Mcov=0.01,0.1,1.0
bias1=4.0

#config2
ens_cov2=0.5
bias2=2.0

ob_dim=5
N=200 
l_scale=0
alpha=1.0
# State and obs

# Access the two experiments and load ensembles:
#Go inside the data folder......................................
file_label1='bias={}_obs={}_ens={}_Mcov={},ocov={}_,gap={}_alpha=1.0_loc=None_r={}'.format(bias1,ob_dim,N,ens_cov1,mu,ob_gap,l_scale)
file_label2='bias={}_obs={}_ens={}_Mcov={},ocov={}_,gap={}_alpha=1.0_loc=None_r={}'.format(bias2,ob_dim,N,ens_cov2,mu,ob_gap,l_scale)

for k in range(10):
    os.chdir(str_+'/ob{}'.format(k+1)) 
    #Load the two ensembles....Convert the samples to tensors:
    os.chdir(str_+'/ob{}'.format(k+1)+'/'+file_label1)
    a_ens1=shorten(np.load(file_label1+'a_ensemble.npy'),factor=4).T
    os.chdir(str_+'/ob{}'.format(k+1)+'/'+file_label2)
    a_ens2=shorten(np.load(file_label2+'a_ensemble.npy'),factor=4).T
    for i in range(100):
        sample1=tf.convert_to_tensor(a_ens1[:,:,i],dtype=tf.float32)
        sample2=tf.convert_to_tensor(a_ens2[:,:,i],dtype=tf.float32)
        loss=tfw.sinkhorn_div_tf(sample1,sample2, epsilon=0.01, num_iters=200, p=2)
        distance[i,k]=tf.sqrt(loss)

os.chdir(str_+'/codes')
np.save('a_distance_between_bias,cov={}_{}and{}_{}_for_mu={},ob_gap={}_for_N={}_1to10_t=0to400'.format(bias1,ens_cov1,bias2,ens_cov2,mu,ob_gap,N),distance)
#Do this for the 10 observation realizations