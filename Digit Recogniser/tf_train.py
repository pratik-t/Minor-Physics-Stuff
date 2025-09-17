import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


df = pd.read_csv('data.csv')
labels = df['label'].values
features = df.drop('label', axis=1)
features = features.values / 255.0

features = features[:40000]
labels = labels[:40000]

# One-hot encode labels for TensorFlow
labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=10)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')]
)

model.compile(loss = 'crossentropy', optimizer = 'Adam', metrics = ['accuracy'], jit_compile=True)

model.fit(features, labels_one_hot, epochs=500, batch_size=100)

model.save('model_tf.keras')