from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# Ne kadar süre çalışıyor ve ne kadar bellek harcıyor
import torch
from datetime import datetime
print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
start = datetime.now()


# MNIST veri setini yükleme ve eğitim/test setlerine ayırma
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Verileri CNN modeline uygun formata dönüştürme ve normalizasyon
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# CNN modeli oluşturma
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
model.add(Dropout(0.5))  # Önceki Dropout katmanı
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Yeni Dropout katmanı
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Bir diğer Dropout katmanı
model.add(Dense(10, activation='softmax'))

# Modeli derleme
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitme
history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# Modelin başarımını yazdırma
train_acc = history.history['accuracy']
test_acc = history.history['val_accuracy']
# print("Eğitim Doğruluğu:", train_acc)
# print("Test Doğruluğu:", test_acc)



model.save("./models/model_sayilar2.h5")


print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
end = datetime.now()
delta = end - start
print('Difference is seconds:', delta.total_seconds())
exit()



