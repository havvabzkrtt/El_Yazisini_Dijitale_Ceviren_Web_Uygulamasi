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
row_index = 0  # her bir satır elde edildiğinde 1 artacak, bu sayıları içeren satır kontrolü için kullanılacak

segment_line_list = line_segmentation.line_segment(img,num_rows)

for line in segment_line_list:
    print(type(line))
    cv2.imshow("LineSegment Resim",line)
    cv2.waitKey(0)
    
    convert_image = convert_and_letter_segmentation.convert_img(line) # line
    cv2.imshow("Donusmus Resim",convert_image)
    cv2.waitKey(0)
    components = convert_and_letter_segmentation.letter_segment(convert_image)

    """
    for idx, component in enumerate(components, start=1):
        cv2.imshow(f'Bileşen {idx}', component)
        cv2.waitKey(0)  # Bir tuşa basılana kadar beklet
        cv2.destroyAllWindows()  # Pencereyi kapat
    """
    """
    label_number = prediction.label_list_number(components)
    print("Label Number: ", label_number)
    """
    if (row_index == 2):
        label_number = prediction.label_list_number(components)
        print("Label Number: ", label_number)
        print("Row index: ", row_index)
    else:
        label = prediction.label_list(components)
        print("Label: ", label)
        print("Row index: ", row_index)

    row_index += 1


cv2.destroyAllWindows()

print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
end = datetime.now()
delta = end - start
print('Difference is seconds:', delta.total_seconds())
exit()


