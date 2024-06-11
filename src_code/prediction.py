import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
from keras.preprocessing import image # type: ignore

# Önceden eğitilmiş modeli yükle
model = tf.keras.models.load_model('./models/52-letters_model.h5')  

model_sayi = tf.keras.models.load_model('./models/model_sayilar2.h5')   # DAHA İYİ ÇALIŞIYOR(model_sayilar2.h5)

# Başka bir kod dosyasında CSV dosyasını okuma
ascii_map = pd.read_csv('./mapping/emnist-letters-mapping-son.csv')
# ascii_map_sayi = pd.read_csv('./mapping/emnis-mnist-mapping.csv')


# componets listesi ile
def list_predict_image(component):
    img = cv2.resize(component, (28, 28))
    x = np.expand_dims(img, axis=0)  # Batch boyutunu ekleyin
    x = x / 255.0  # Normalizasyon
    
    cl = model.predict(x)
    cl = list(cl[0])

    return ascii_map["Character"][cl.index(max(cl))]

def label_list(image_list):
    predicted_labels = []
    for img in image_list:
        #cv2.imshow("Letter", img)
        #cv2.waitKey(0)
        predicted_label = list_predict_image(img)
        predicted_label = predicted_label[0]
        predicted_labels.append(predicted_label)
    return predicted_labels



# image path ile 
def imagepath_predict_image(image_path):
    img = image.load_img(image_path, target_size=(28, 28))
    x = image.img_to_array(img)
    x = x / 255.0

    gray_image = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    gray_image = np.expand_dims(gray_image, axis=-1)
    gray_image = np.expand_dims(gray_image, axis=0)
    cl = model.predict(gray_image)
    cl = list(cl[0])

    return ascii_map["Character"][cl.index(max(cl))]

etiket_list = []
    

def label_list_imagepath(image_list):
    predicted_labels = []
    for img in image_list:
        predicted_label = list_predict_image(img)
        predicted_labels.append(predicted_label)
    return predicted_labels


from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image # type: ignore

# SADECE SAYILAR   / KÜTÜPHANELİ VERİSETİ
# componets listesi ile
def list_predict_image_number(component):
    
    img = cv2.resize(component, (28, 28))
    
    # Resmi modele uygun formata dönüştürme
    external_image_array = keras_image.img_to_array(img)
    external_image_array = external_image_array.reshape(1, 28, 28, 1) / 255.0  # Normalizasyon

    return external_image_array

def label_list_number(image_list):
    predicted_labels_numbers = []
    for img in image_list:
        # Modelde tahmin yapma
        #cv2.imshow("Harf", img)
        #cv2.waitKey(0)
        external_image_array = list_predict_image_number(img)
        predicted_label = np.argmax(model_sayi.predict(external_image_array), axis=-1)
        predicted_labels_numbers.append(predicted_label[0])
        
    return predicted_labels_numbers


