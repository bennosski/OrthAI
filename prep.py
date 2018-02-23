import numpy as np
import idx2numpy


y_train = idx2numpy.convert_from_file('data/train-labels-idx1-ubyte')
x_train = idx2numpy.convert_from_file('data/train-images-idx3-ubyte')

perm = np.random.permutation(60000)

y_val = y_train[perm[50000:]]
x_val = x_train[perm[50000:], :, :]

y_train = y_train[perm[:50000]]
x_train = x_train[perm[:50000], :, :]

y_test = idx2numpy.convert_from_file('data/t10k-labels-idx1-ubyte')
x_test = idx2numpy.convert_from_file('data/t10k-images-idx3-ubyte')

print x_train.shape
print y_train.shape
print x_val.shape
print y_val.shape
print x_test.shape
print y_test.shape

# norm the data

def norm(a):
    return a/127.5 - 1.0

x_train = norm(x_train)
y_train = y_train
x_val   = norm(x_val)
y_val   = y_val
x_test  = norm(x_test)
y_test  = y_test

np.save('x_train', x_train)
np.save('y_train', y_train)
np.save('x_val', x_val)
np.save('y_val', y_val)
np.save('x_test', x_test)
np.save('y_test', y_test)

