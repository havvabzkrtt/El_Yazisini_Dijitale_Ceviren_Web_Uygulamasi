# TAHMİN

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


import line_segmentation
import convert_and_letter_segmentation
import prediction


# read image, prepare it by resizing it to fixed height and converting it to grayscale
img = cv2.imread('./sablon_form/form_sablon1.png') 

segment_line_list = line_segmentation.line_segment(img)
def display_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
"""
# bu şekilde çalışıyor ama kötü tahmin ediyor, model ile alakalı olmalı  / # resim path'ten okumuyorsa
for line in segment_line_list:
    display_image(line)
    print(type(line))
    # Görüntüyü gri tonlamalı hale getirme
    # Veriyi PNG dosyasına kaydetmek için matplotlib'i kullanalım
    plt.imsave("./outputs/cikti1.png", line, cmap='gray') 
    resim_yolu = "./outputs/cikti1.PNG"            
    resim = cv2.imread(resim_yolu)
    donusturulmus_resim = convert_and_letter_segmentation.convert_img(resim)
    components = convert_and_letter_segmentation.letter_segment(donusturulmus_resim)
    break
    

# resim segment_line_list'ten değişikliğe uğramadan direkt geliyorsa 

# Dönüştürülmüş resmi göster

cv2.imshow("Dönüştürülmemiş Resim", resim)
cv2.imshow("Dönüştürülmüş----------------- Resim", donusturulmus_resim)
# cv2.imwrite("donusmus_resim.png", donusturulmus_resim)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
i = 0

# TAHMİNLER ÇOK KÖTÜ
for line in segment_line_list:
    donusturulmus_resim = convert_and_letter_segmentation.convert_img(line)
    components = convert_and_letter_segmentation.letter_segment(donusturulmus_resim)
    print(type(components[0]))
    print(components[0].shape)
    labels = prediction.label_list(components)
    print(i,". label: ", labels)
    i =+ 1
    break

# Dönüştürülmüş resmi göster
"""
cv2.imshow("Dönüştürülmemiş Resim", segment_line_list[0])
cv2.imshow("Dönüştürülmüş----------------- Resim", donusturulmus_resim)
# cv2.imwrite("donusmus_resim.png", donusturulmus_resim)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# dönüsüm icin calisan kısımlar 

# Resmi yükle
"""
resim_yolu = "cikti.PNG"            
resim = cv2.imread(resim_yolu)
print(resim)
print(type(resim))

donusturulmus_resim = convert_and_letter_segmentation.convert_img(resim)

components = convert_and_letter_segmentation.letter_segment(donusturulmus_resim)
"""





# etietleri listeden alma
"""
labels = prediction.label_list(components)
print(labels)
"""
# Tüm resimleri yan yana göstermek için subplot kullanalım
"""
plt.figure(figsize=(len(components) * 2, 2))
for i, component in enumerate(components, start=1):
    plt.subplot(1, len(components), i)
    plt.imshow(component, cmap='gray')
    plt.title(f"Prediction: {prediction.list_predict_image(component)}")
    plt.axis('off')

plt.show()
"""

