
##########################
#
#  This is the status of the tensorflow implementation of the original numpy code used to generate figures in the milestone
#  There are still some parts which need to be transferred from the numpy version
#
###########################


# Remaining tasks:

# test performance against numpy version for spiral dataset
# change input layer sizes 2->784 and output layer sizes 2->10 to work with MNIST data 
# normalize the MNIST data
# setup data augmentation for MNIST
# run on GPU
# learning rate decay?


import tensorflow as tf
import numpy as np
import sys, os
import time
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import shutil
from icecream import ic


# whether to debug, train, or test
action  = sys.argv[1]


learning_rate = 0.001


x = tf.placeholder("float32", shape=[None, 2])
y = tf.placeholder("int32", shape=[None])

params = {}

def init_relu_layer_params(params, il, dim_i, dim_o):
    r = np.random.randn(dim_i, dim_o) * np.sqrt(2.0/dim_i)
    params['W'+str(il)] = tf.Variable(tf.constant(r, dtype=tf.float32), name='W'+str(il))
    
    r = np.zeros([1,dim_o])
    params['b'+str(il)] = tf.Variable(tf.constant(r, dtype=tf.float32), name='b'+str(il))
           
    
def init_orth_layer_params(params, il, dim_i, dim_o):
    r    = np.random.randn(dim_i, dim_o)
    Q,_  = np.linalg.qr(r)

    #Q = np.diag(np.ones(dim_i))

    params['O'+str(il)] = tf.Variable(tf.constant(Q, dtype=tf.float32), name='O'+str(il))

    
def nonlin_op(B):
    # todo:
    # port from numpy code


    

# every is the block size as defined in the milestone
every = 3

# layer_dims  = [784] + [800]*2 + [400]*2 + [100]*10 + [10] # performance 0.5
# layer_dims = [784] + [20]*30 + [10] # p ~ 0.6
# layer_dims = [784] + [10]*81 + [10]

layer_dims = [2] + [4]*15 + [2]

L = len(layer_dims)

# initialize the parameters

for il in range(L-1):
    l1 = layer_dims[il]
    l2 = layer_dims[il+1]
    if il%every==0:
        init_relu_layer_params(params, il, l1, l2)
    else:
        init_orth_layer_params(params, il, l1, l2)
        
As = [] # the activations
#A  = tf.reshape(x, [-1, 2])
A = x
As.append(A)

# network design
for il in range(L-1):
    l1 = layer_dims[il]
    l2 = layer_dims[il+1]
    if il%every==0:
        A = tf.matmul(A, params['W'+str(il)]) + params['b'+str(il)]
        #A = tf.nn.relu(A)
        A = tf.tanh(A)

        if il!=L-2: # do not use activation on last layer
            A = tf.maximum(A, 0.0)
            
        As.append(A)
    else:
        A = tf.matmul(A, params['O'+str(il)])
        #A = tf.matmul(params['O'+str(il)], tf.transpose(A))
        #A = tf.transpose(A)
        A = nonlin_op(A)
        As.append(A)

AL = A




myvars = [params[name] for name in params if ('W' in name)]
print('myvars with L2 loss len = ',len(myvars))
lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in myvars]) * 0.001


# question: will softmax cross entropy perform worse than binary cross entropy?

# binary cross entropy
#cost = tf.reduce_mean(y*tf.log(AL) + (1-y)*tf.log(AL)) + lossL2
# multiclass cross entropy
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=AL)) + lossL2


Adam        = tf.train.AdamOptimizer(learning_rate, beta1=0.8)
#Adam        = tf.train.GradientDescentOptimizer(learning_rate)
#train_op   = Adam.minimize(cost)

gradients   = tf.gradients(cost, As)



# set up accumulator operation to perform gradient descent after the entire batch has been processed
# this will be needed for a larger dataset like MNIST since orthogonal projection ops are slow

all_vars = []
grads_acc = []
for il in range(L-1):
    if il%every==0:
        all_vars.append(params['W'+str(il)])
        all_vars.append(params['b'+str(il)])
        grads_acc.append(tf.Variable(tf.zeros_like(params['W'+str(il)]), name='gW'+str(il)))
        grads_acc.append(tf.Variable(tf.zeros_like(params['b'+str(il)]), name='gb'+str(il)))
    else:
        all_vars.append(params['O'+str(il)])
        grads_acc.append(tf.Variable(tf.zeros_like(params['O'+str(il)]), name='gO'+str(il)))
        
grads_and_vars  = Adam.compute_gradients(cost, all_vars)


init_grads_acc_op = [tf.assign(g, tf.zeros_like(g)) for g in grads_acc] # call to reinitialize
acc_weight = tf.placeholder("float32", None)
add_grads_acc_op  = [grads_acc[i].assign(grads_acc[i] + acc_weight * gv[0]) for i,gv in enumerate(grads_and_vars)] # call to run update (add gradient)

grads_and_vars_avg = list(zip(grads_acc, all_vars))
update_params   = Adam.apply_gradients(grads_and_vars_avg)




# projection operations for orthogonal matrices
proj_ops = []
for il in range(L-1):
    if il%every!=0:
        # todo:
        # copy projection ops from numpy code
        # check tensorflow documentation for svd conventions
        

# saving operations
# seems much faster to do custom saving with numpy than using tf.train.saver
        
in_ps = {}
my_params = []
for il in range(L-1):
    if il%every==0:
        in_ps['W'+str(il)] = tf.placeholder("float32", shape=[layer_dims[il], layer_dims[il+1]])
        my_params.append(in_ps['W'+str(il)])
        in_ps['b'+str(il)] = tf.placeholder("float32", shape=[1,layer_dims[il+1]])
        my_params.append(in_ps['b'+str(il)])
    else:
        in_ps['O'+str(il)] = tf.placeholder("float32", shape=[layer_dims[il], layer_dims[il+1]])
        my_params.append(in_ps['O'+str(il)])
    
init_params_ops = []
ct = 0
for il in range(L-1):
    if il%every==0:
        init_params_ops.append(params['W'+str(il)].assign(in_ps['W'+str(il)]))
        init_params_ops.append(params['b'+str(il)].assign(in_ps['b'+str(il)]))                                           
    else:
        init_params_ops.append(params['O'+str(il)].assign(in_ps['O'+str(il)]))
        


        
sess = tf.Session()
sess.run(tf.global_variables_initializer())


if action=='debug':
    # later change this to load MNIST data
    x_train = np.load('data_spiral.npy')
    y_train = np.load('labels_spiral.npy')
    x_train = np.swapaxes(x_train, 0, 1)
    y_train = np.squeeze(y_train)
    
    print('data shape')
    print(x_train.shape)
    print('labels shape')
    print(y_train.shape)

    
    start_time = time.time()    

    c, ps_current  = sess.run([cost, params], feed_dict={x:x_train, y:y_train})
    print('cost = ',c)

    
    best_params = np.load('best_params.npy').item()    
    best_params_list = []
    for il in range(L-1):
        if il%every==0:
            best_params_list.append(best_params['W'+str(il)])
            best_params_list.append(best_params['b'+str(il)])
        else:
            best_params_list.append(best_params['O'+str(il)])
    
    sess.run(init_params_ops, feed_dict={i:d for i,d in zip(my_params, best_params_list)})
    
    c  = sess.run(cost, feed_dict={x:x_train, y:y_train})
    print('cost = ',c)

    

if action=='train' or action=='resume':

    if action=='train':
        savedir = sys.argv[2]
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        shutil.copyfile('nn.py', savedir+'nn.py')

        x_train = np.load('data_spiral.npy')
        y_train = np.load('labels_spiral.npy')


        start_epoch = 0
        
    if action=='resume':
        savedir = './'

        x_train = np.load('../data_spiral.npy')
        y_train = np.load('../labels_spiral.npy')

        model_num = int(sys.argv[2])
        saver.restore(sess, 'model%d'%model_num+'.ckpt')

        start_epoch = model_num
        
    x_train = np.swapaxes(x_train, 0, 1)
    y_train = np.squeeze(y_train)
    
    print('data shape')
    print(x_train.shape)
    print('labels shape')
    print(y_train.shape)

    
    # training
    costs = []
    all_errors = []
    best_cost = None
    best_grads = None
    best_error = None
    
    N = x_train.shape[0]
    batch_sz = N

    
    start_sim_time = time.time()
    
    for epoch in range(start_epoch,start_epoch+20001):

        start_time = time.time()
        
        batches = np.split(np.random.permutation(N), range(0,N,batch_sz))[1:]

        sess.run([init_grads_acc_op])
        
        errors = []
        
        # loop over batches
        for ib,batch in enumerate(batches):
            
            # now load the batch and augment the images
            #xs = x_train[batch]
            #ys = y_train[batch]

            xs = x_train
            ys = y_train
            
            aug_time = 0.0
            
            opt_time = time.time()
            aw = 1.0*len(batch)/N
            logits, c, ps_current, _  = sess.run([AL, cost, params, add_grads_acc_op], feed_dict={x:xs, y:ys, acc_weight:aw})
            costs.append(c)
            opt_time = time.time() - opt_time

            logits = np.squeeze(np.array(logits))
            preds = np.argmax(logits, axis=1)

            #if epoch%1000==0:
            #    ic(preds-ys)
            
            diff = ys - preds
            error_number = np.count_nonzero(diff)
            error = 1.0*error_number/len(batch)
            errors.append(error*aw)
            
            if best_cost==None or c<best_cost:
                best_cost = c
                #print('\nbest params set on epoch=',epoch,' for err ',best_error,' for cost ',c)
                
                
        avg_error = np.mean(errors)
        all_errors.append(avg_error)
        if best_error==None or avg_error<best_error:
            best_error = avg_error
            best_params = ps_current.copy()
        
        sess.run([update_params]) # finally do gradient descent
        
        svd_time = time.time()            
        sess.run(proj_ops)
        svd_time = time.time() - svd_time

        '''
        if epoch==20:
            ps = sess.run(params)
            for il in range(L-1):
                if il%every!=0:
                    print('check update il=%d'%il)
                    p = ps['O'+str(il)]
                    print(p-ps_old['O'+str(il)])
                    #print(np.dot(p, np.transpose(np.conj(p))))
            1./0
        '''

        if epoch%200==0:
            print('epoch ',epoch)
            print('cost = {:1.5f} error = {:1.5e} time = {:1.2f}'.format(best_cost, best_error, time.time()-start_time))
            print('profiler opt%1.2e'%opt_time+' svd%1.2e'%svd_time)

        if epoch>start_epoch and epoch%1000==0:
            np.save(savedir+'best_params', best_params)
            np.save(savedir+'costs', costs)
            np.save(savedir+'errors', all_errors)

            
            
    print('total simulation time',time.time()-start_sim_time)
            

    
if action=='validate':

    x_train = np.load('../data_spiral.npy')
    y_train = np.load('../labels_spiral.npy')
    x_train = np.swapaxes(x_train, 0, 1)
    y_train = np.squeeze(y_train)        

    print('data shape')
    print(x_train.shape)
    print('labels shape')
    print(y_train.shape)

    best_params = np.load('best_params.npy').item()    
    best_params_list = []
    for il in range(L-1):
        if il%every==0:
            best_params_list.append(best_params['W'+str(il)])
            best_params_list.append(best_params['b'+str(il)])
        else:
            best_params_list.append(best_params['O'+str(il)])
    
    sess.run(init_params_ops, feed_dict={i:d for i,d in zip(my_params, best_params_list)})
    
    logits, c  = sess.run([AL, cost], feed_dict={x:x_train, y:y_train})
    print('cost =',c)

    logits = np.squeeze(np.array(logits))
    preds = np.argmax(logits, axis=1)

    ic(preds-y_train)
                        
    diff = y_train - preds
    error_number = np.count_nonzero(diff)
    error = 1.0*error_number/200.0
    print('error =',error)
    

    print('\n\n\ngradients')
    gs = sess.run(gradients, feed_dict={x:x_train, y:y_train})
    for i,g in enumerate(gs):
        print(str(i), np.mean(g**2))
    

    # todo: 
    # make new predictions for a grid of coordinates to make figures
    #x_train = np.linspace(-
                          
        
