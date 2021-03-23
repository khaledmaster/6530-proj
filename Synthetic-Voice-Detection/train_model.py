



from preprocess import *
from swishnet import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from datetime import datetime
from keras.models import model_from_json, load_model
np.random.seed(7)
tf.get_logger().setLevel('INFO')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

train_dataset = pd.read_hdf('pre-processed-dataset/train_cnn.h5', key='traindf')
print(train_dataset.head())

validation_dataset = pd.read_hdf('pre-processed-dataset/validation_cnn', key='valdf')

test_dataset = pd.read_hdf('pre-processed-dataset/combined_cnn_test', key='testdf')

#0 is fake, 1 is real
X_train = np.array(train_dataset.feature.tolist())
y_train = np.array(train_dataset.class_label.tolist())
X_validation = np.array(validation_dataset.feature.tolist())
y_validation = np.array(validation_dataset.class_label.tolist())

X_test = np.array(test_dataset.feature.tolist())
y_test = np.array(test_dataset.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy_train = to_categorical(le.fit_transform(y_train))
yy_validation = to_categorical(le.fit_transform(y_validation))
yy_test = to_categorical(le.fit_transform(y_test))

#Data reshaping for CNN Architecture
num_rows = 40
num_columns = 87
X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns)
X_validation = X_validation.reshape(X_validation.shape[0], num_rows, num_columns)
num_labels = yy_train.shape[1]

X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns)


model = SwishNet(input_shape=(num_rows, num_columns), classes=2, width_multiply=2) #SwishNet Wide
# Compile the model
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
# Display model architecture summary
model.summary()

num_epochs = 15
num_batch_size = 32
start = datetime.now()
history = model.fit(X_train, yy_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_validation, yy_validation), verbose=1)
duration = datetime.now() - start
print("Training completed in time:", duration)

model.save("sn_40_87_width_2_accuracy_adam_poch_15_batch_32_binary_crossentropy.h5")
model = load_model("sn_40_87_width_2_accuracy_adam_poch_15_batch_32_binary_crossentropy.h5")
y_pred = model.predict(X_test)
print(y_pred)

# plt.style.use('dark_background')
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs_range = range(num_epochs)
# plt.figure(figsize=(30, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.grid()
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.grid()
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()