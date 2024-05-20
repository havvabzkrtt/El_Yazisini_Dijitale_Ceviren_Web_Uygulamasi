# TAHMİN

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# import src_code.line_segmentation2 as line_segmentation2
import line_segmentation 
import convert_and_letter_segmentation
import prediction


# Ne kadar süre çalışıyor ve ne kadar bellek harcıyor
import torch
from datetime import datetime
print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
start = datetime.now()


#img = cv2.imread('./sablon_form/form_sablon3.png') 
#img = cv2.imread('./sablon_form/form2.jpg') 
img = cv2.imread('./sablon_form/form1.PNG') 
num_rows = 7 # kaç satıra bölecek
index = 0  # her bir satır elde edildiğinde 1 artacak, bu sayıları içeren satır kontrolü için kullanılacak

segment_line_list = line_segmentation.line_segment(img,num_rows)


def display_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

for line in segment_line_list:
    #display_image(line)
    print(type(line))
    """
    plt.imsave("./outputs/cikti1.png", line, cmap='gray') 
    """
    """
    resim_yolu = "./sablon_form/telefon_no.PNG"            
    resim = cv2.imread(resim_yolu)
    cv2.imshow("Resim",resim)
    donusturulmus_resim = convert_and_letter_segmentation.convert_img(resim) # line
    """
    cv2.imshow("LineSegment Resim",line)
    donusturulmus_resim = convert_and_letter_segmentation.convert_img(line) # line

    
    components = convert_and_letter_segmentation.letter_segment(donusturulmus_resim)

    
    for idx, component in enumerate(components, start=1):
        cv2.imshow(f'Bileşen {idx}', component)
        cv2.waitKey(0)  # Bir tuşa basılana kadar beklet
        cv2.destroyAllWindows()  # Pencereyi kapat

    labelll = prediction.label_list(components)
    print("Label--------------------------------", labelll)
    """
    print(components[0].shape) 
    labels = prediction.label_list(components)
    print(labels)
    """
    

#cv2.imshow("Donusmus Resim",donusturulmus_resim)
cv2.waitKey(0)
cv2.imshow("LineSegment Resim1",segment_line_list[0])
cv2.waitKey(0)

cv2.destroyAllWindows()

print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
end = datetime.now()
delta = end - start
print('Difference is seconds:', delta.total_seconds())
exit()


