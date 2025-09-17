import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def relu(z):
    return (np.maximum(0, z))


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True)) # subtract max for stability
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def forward_pass(x0, w1, b1, w2, b2):
    z1 = w1.T @ x0+b1
    x1 = relu(z1)
    z2 = w2.T @ x1+b2
    x2 = softmax(z2)
    return z1, x1, x2

df = pd.read_csv('data.csv')

labels = df['label'].values
features = df.drop('label', axis=1)
features = (features.values / 255.0).T

model = np.load('model.npz')
w1 = model['w1']
b1 = model['b1']
w2 = model['w2']
b2 = model['b2']


n = np.random.randint(40000, 42000)
digit = features[:, n].reshape(-1, 1)
label = labels[n]

z1, x1, x2 = forward_pass(digit, w1, b1, w2, b2)

prediction = np.argmax(x2)


tf_digit = features[:, n].reshape(1, -1)

tf_model = tf.keras.models.load_model('model_tf.keras')

tf_probs = tf_model.predict(tf_digit)
tf_prediction = np.argmax(tf_probs)

digit = features[:, n].reshape(28, 28)
plt.imshow(digit, cmap='gray')
plt.title(f"Scratch Predicted: {prediction}, TF Predicted: {tf_prediction}, Actual: {label}")
plt.axis('off')
plt.show()
