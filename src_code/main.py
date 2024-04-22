# TAHMİN

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


import line_segmentation
import convert_and_letter_segmentation
import prediction


# Ne kadar süre çalışıyor ve ne kadar bellek harcıyor
import torch
from datetime import datetime
print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
start = datetime.now()



# read image, prepare it by resizing it to fixed height and converting it to grayscale
#img = cv2.imread('./sablon_form/form_sablon3.png') 
img = cv2.imread('./sablon_form/form3.PNG') 
#img = cv2.imread('./sablon_form/form2.jpg') 
segment_line_list = line_segmentation.line_segment(img)


def display_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

# bu şekilde çalışıyor ama kötü tahmin ediyor, model ile alakalı olmalı  / # resim path'ten okumuyorsa
for line in segment_line_list:
    #display_image(line)
    print(type(line))
    # Görüntüyü gri tonlamalı hale getirme
    # Veriyi PNG dosyasına kaydetmek için matplotlib'i kullanalım
    """
    plt.imsave("./outputs/cikti1.png", line, cmap='gray') 
    resim_yolu = "./outputs/cikti1.PNG"            
    resim = cv2.imread(resim_yolu)
    """
    cv2.imshow("LineSegment Resim",line)
    donusturulmus_resim = convert_and_letter_segmentation.convert_img(line)
    components = convert_and_letter_segmentation.letter_segment(donusturulmus_resim)

    
    for idx, component in enumerate(components, start=1):
        cv2.imshow(f'Bileşen {idx}', component)
        cv2.waitKey(0)  # Bir tuşa basılana kadar beklet
        cv2.destroyAllWindows()  # Pencereyi kapat

    labelll = prediction.label_list(components)
    print("abel--------------------------------", labelll)
    """
    print(components[0].shape) 
    labels = prediction.label_list(components)
    print(labels)
    """
    

#cv2.imshow("Donusmus Resim",donusturulmus_resim)
cv2.waitKey(0)
cv2.imshow("LineSegment Resim1",segment_line_list[0])
cv2.waitKey(0)
"""
cv2.imshow("LineSegment Resim2",segment_line_list[1])
cv2.waitKey(0)

cv2.imshow("LineSegment Resim3",segment_line_list[2])
cv2.waitKey(0)

cv2.imshow("LineSegment Resim4",segment_line_list[3])
cv2.waitKey(0)
cv2.imshow("LineSegment Resim4",segment_line_list[4])
cv2.waitKey(0)
cv2.imshow("LineSegment Resim4",segment_line_list[5])
cv2.waitKey(0)
"""
cv2.destroyAllWindows()
"""
donusturulmus_resim = convert_and_letter_segmentation.convert_img(segment_line_list[1])
components = convert_and_letter_segmentation.letter_segment(donusturulmus_resim)

labelll2 = prediction.label_list(components)
print("Label2--------------------------------", labelll2)
"""


# TAHMİNLER ÇOK KÖTÜ
"""
for line in segment_line_list:
    donusturulmus_resim = convert_and_letter_segmentation.convert_img(line)
    components = convert_and_letter_segmentation.letter_segment(donusturulmus_resim)
    print(type(components[0]))
    print(components[0].shape)
    labels = prediction.label_list(components)
    print(i,". label: ", labels)
    i =+ 1
    break
"""
# Dönüştürülmüş resmi göster
"""
cv2.imshow("Dönüştürülmemiş Resim", segment_line_list[0])
cv2.imshow("Dönüştürülmüş----------------- Resim", donusturulmus_resim)
# cv2.imwrite("donusmus_resim.png", donusturulmus_resim)
cv2.waitKey(0)
cv2.destroyAllWindows()
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


print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
end = datetime.now()
delta = end - start
print('Difference is seconds:', delta.total_seconds())
exit()


