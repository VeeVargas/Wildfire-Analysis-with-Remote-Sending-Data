import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
import numpy as np
import glob
import os

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

USER_NAME = 'vee_vargas'
BUCKET = 'aus_fire_bucket'
FOLDER = 'training_data'
TRAINING_BASE = 'training_patches'
TESTING_BASE = 'testing_patches'
VAL_BASE = 'val_patches'

data_dir = '/content/collab' 
fire_dir = f'{data_dir}/fire_images'
non_fire_dir = f'{data_dir}/non_fire_images'


fire_images = []
for filename in glob.glob(f'{fire_dir}/*.png'):
    im=Image.open(filename)
    fire_images.append(im)

non_fire_images = []
for filename in glob.glob(f'{notfire_dir}/*.png'):
    im=Image.open(filename)
    non_fire_images.append(im)

# Generate images to balance classes
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

def gen_images(path, num, fire):
    '''
    Reads in image, coverts to array, reshapes, applies datagen 4 times.
    Repeats num times.
    '''
    if fire:
        for i in tqdm(range(1, num+1)):
            
            img = Image.open(f'{path}/fire.{i}.png')
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i=0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=f'{path}/', save_prefix='fire', save_format='png'):
                i += 1
                if i > 20:
                    break
    else:
        for i in tqdm(range(1, num+1)):
            img = Image.open(f'{path}/non_fire.{i}.png')
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i=0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=f'{path}/', save_prefix='non_fire', save_format='png'):
                i += 1
                if i > 4:
                    break
gen_images(non_fire_dir, 120, fire=False)
                    
# build model
batch_size = 16
img_height = 256
img_width = 256
epochs=50

X_train = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  shuffle=True,
  image_size=(img_height, img_width),
  batch_size=batch_size)

X_test = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


def model_CNN():
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=(img_height, img_width, 3))) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # Compile the model
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])    
    return model
   
model = model_CNN()
print('Model loaded.')

checkpoint = ModelCheckpoint(
            filepath='weights.hdf5',
            monitor = 'val_accuracy',
            verbose=2,
            save_best_only=True)
print('Checkpoint')

history = model.fit(
            X_train,
            steps_per_epoch=10,
            epochs=epochs,
            callbacks=[checkpoint],
            validation_data=X_test,
            validation_steps=10)
print('Finished Training')

model.summary()
model.save('mod3.h5')
#model.evaluate(X_test)

# PREDICT
def model_evaluate_val(ax, model, batch_size=1):
    X_test = image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      shuffle=True,
      image_size=(256, 256),
      batch_size=batch_size)

    return model.evaluate(X_test)

    images = []
    results = []
    labels = []

    for i, (image, label) in enumerate(X_test):
      prediction = model.predict(image)

      if (prediction < 0.5) != label:
        result = 'Correct'
      else:
        result = 'Incorrect'

      results.append(result)
      labels.append(label)

      images.append(image[0].numpy().astype("uint8"))
    #   if i > 10: break


    labels = ['Fire' if label==0 else 'Not fire' for label in labels]
    print('Labels updated.')    
    for i, (label, result, image) in enumerate(zip(labels, results, images)):
        ax[i//3, i%3].imshow(image)
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title('{}, Predicted {}'.format(label, result))
    return ax


model = load_model('m1.h5')
fig, ax = plt.subplots(2,3, figsize=(10,6))
fig.subplots_adjust(wspace=.05)
model_evaluate_val(ax, model)

fig.savefig('m1_predictions.jpeg')

