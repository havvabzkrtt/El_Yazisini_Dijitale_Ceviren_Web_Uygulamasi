import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
from keras.preprocessing import image

# Önceden eğitilmiş modeli yükle
model = tf.keras.models.load_model('./models/model1.h5')  # Örnek dosya adı

# Başka bir kod dosyasında CSV dosyasını okuma
ascii_map = pd.read_csv('./mapping/mapping.csv')



def list_predict_image(component):
    img = cv2.resize(component, (28, 28))
    x = np.expand_dims(img, axis=0)  # Batch boyutunu ekleyin
    x = x / 255.0  # Normalizasyon

    cl = model.predict(x)
    cl = list(cl[0])

    return ascii_map["Character"][cl.index(max(cl))]


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
    
def label_list(image_list):
    predicted_labels = []
    for img in image_list:
        predicted_label = list_predict_image(img)
        predicted_labels.append(predicted_label)
    return predicted_labels
