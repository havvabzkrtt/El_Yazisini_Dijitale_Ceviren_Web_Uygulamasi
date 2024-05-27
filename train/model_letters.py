import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D
from keras.models import Sequential,load_model
import keras


# Ne kadar süre çalışıyor ve ne kadar bellek harcıyor
import torch
from datetime import datetime
print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
start = datetime.now()



train_images = pd.read_csv("datasets/emnist-letters-train.csv",header=None)
test_images = pd.read_csv("datasets/emnist-letters-test.csv",header=None)
map_images = pd.read_csv("datasets/emnist-letters-mapping.txt",header=None) 


# Seperating labels from features in training and test data.
train_x = train_images.iloc[:,1:]  
train_y = train_images.iloc[:,0]  
train_x = train_x.values

test_x = test_images.iloc[:,1:]
test_y = test_images.iloc[:,0]
test_x = test_x.values


ascii_map = []
for i in map_images.values:
    ascii_map.append(i[0].split()[1])

# plt.imshow(np.rot90(np.fliplr(train_x[1].reshape(28,28))))

def rot_flip(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

train_x = np.apply_along_axis(rot_flip,1,train_x)
test_x = np.apply_along_axis(rot_flip,1,test_x)
# plt.imshow(train_x[2])
# train_x.shape

train_x = train_x.astype('float32')
train_x = train_x/255.0

test_x = test_x.astype('float32')
test_x = test_x/255.0

train_x = train_x.reshape(-1, 28,28, 1)   #Equivalent to (112800,28,28,1)
test_x = test_x.reshape(-1, 28,28, 1)   #Equivalent to (18800,28,28,1)

# CNN modeli oluşturma (BİZİM MODEL)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(28, activation='softmax'))

model.compile(optimizer = 'adam',loss= "sparse_categorical_crossentropy", metrics=['accuracy'])
# model.summary()


from keras.callbacks import EarlyStopping
# Early stopping geri çağrısını oluşturma
early_stopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)


history = model.fit(
    train_x,
    train_y,
    validation_data = (test_x,test_y),
    epochs = 10, callbacks=[early_stopper]
)

model.save("models/model_letters5.h5")

ascii_map = []
for i in map_images.values:
    ascii_map.append(i[0].split()[1])


# Adding character to associated ASCII Value
character = []
for i in ascii_map:
    character.append(chr(int(i)-1)) # int(i)-1  : -1 eklendi çünkü tahmin işleminde etiketlerde kayma vardı, gerçek etiket:A tahmin edilen:B oluyordu.

# plt.imshow(np.rot90(np.fliplr(train_x[1].reshape(28,28))))
character = pd.DataFrame(character)


ascii_map = pd.DataFrame(ascii_map)
ascii_map["Character"] = character
ascii_map.to_csv("mapping/emnist-letters-mapping5.csv",index=False,header=True)


print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
end = datetime.now()
delta = end - start
print('Difference is seconds:', delta.total_seconds())
exit()
