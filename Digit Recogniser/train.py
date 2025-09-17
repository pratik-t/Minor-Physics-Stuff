import numpy as np
import pandas as pd


def one_hot(y,m):
    one_hot_y = np.zeros([10,m])
    for i in range(len(y)):
        one_hot_y[y[i],i] = 1
    return one_hot_y


def relu(z):
    return(np.maximum(0,z))


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True)) # subtract max for stability
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)   # sum over columns


def forward_pass(x0, w1, b1, w2, b2):
    z1 = w1.T @ x0+b1
    x1 = relu(z1)
    z2 = w2.T @ x1+b2
    x2 = softmax(z2)
    return z1, x1, x2


def back_prop(m, x0, z1, x1, x2, y0, w2):
    eps2 = x2-y0
    dw2 = (1/m) * (x1 @ eps2.T)
    db2 = (1/m) * np.sum(eps2, axis=1, keepdims=True)
    eps1 = (w2.T @ eps2) * relu_derivative(z1)
    dw1 = (1/m) * (x0 @ eps1.T)
    db1 = (1/m) * np.sum(eps1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2


def update_params(alpha, w1, dw1, b1, db1, w2, dw2, b2, db2):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2

    return w1, b1, w2, b2


def accuracy(x2, y_lab):
    return int(100*np.sum(np.argmax(x2,axis=0)==y_lab)/y_lab.size)


df = pd.read_csv('data.csv')

labels = df['label'].values
features = df.drop('label', axis= 1)
features = (features.values / 255.0).T


model = np.load('model.npz')
w1 = model['w1']
b1 = model['b1']
w2 = model['w2']
b2 = model['b2']


epochs = 500
alpha = 0.01
m = 100

for i in range(epochs):
    set_number = 0
    while set_number + m <= 40000:
        training_features = features[:,set_number:set_number+m]
        training_labels = labels[set_number:set_number+m]
        training_one_hot_labels = one_hot(training_labels, m)
        
        z1, x1, x2 = forward_pass(training_features, w1, b1, w2, b2)
        dw1, db1, dw2, db2 = back_prop(m, training_features, z1, x1, x2, training_one_hot_labels, w2)
        w1, b1, w2, b2 = update_params(alpha, w1, dw1, b1, db1, w2, dw2, b2, db2)

        set_number+=m

    print(f'Iteration: {i+1} :: Accuracy: {accuracy(x2, training_labels)}')


np.savez('model.npz', w1=w1, b1=b1, w2=w2, b2=b2)
