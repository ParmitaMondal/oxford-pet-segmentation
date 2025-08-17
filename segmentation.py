pip install -q git+https://github.com/tensorflow/examples.git

import os

from google.colab import drive

drive.mount("/content/gdrive", force_remount=True)
os.chdir("/content/gdrive/My Drive")

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
from IPython.display import clear_output
import matplotlib.pyplot as plt
import Utility

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 32
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(Utility.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(Utility.load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

for image, mask in train.take(1):
  sample_image, sample_mask = image, mask
Utility.display([sample_image, sample_mask])

train_dataset

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Conv2DTranspose

'''Building Model'''

model = Sequential()
model.add(Conv2D(16, kernel_size=(3), strides=(2), activation='relu',padding=('same'),input_shape=((128,128,3))))
model.add(Conv2D(32, kernel_size=(3), strides=(1,1),activation='relu',padding=('same')))
model.add(MaxPooling2D(pool_size=(2)))
model.add(Conv2D(64, kernel_size=(3), strides=(1,1), activation='relu',padding='same'))
model.add(Conv2DTranspose(64,kernel_size=(3), strides=(2), activation='relu',padding='same'))
model.add(Conv2DTranspose(3,kernel_size=(3),strides=(2), activation = 'relu',padding='same'))

'''Compiling Model'''

model.compile(optimizer=tf.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.summary()

tf.keras.utils.plot_model(model, show_shapes=True)

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

'''model fitting'''

model_history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS, validation_data=test_dataset,
                          callbacks=[Utility.DisplayCallback()])

for image, mask in test_dataset.take(3):
  pred_mask = model.predict(image)
  Utility.display([image[0],mask[0],Utility.create_mask(pred_mask)])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

loss[-1]

val_loss[-1]

model_history.history['accuracy'][-1]

model_history.history['val_accuracy'][-1]
