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


def process_line(line, row_index, indis_list):
    convert_image = convert_and_letter_segmentation.convert_img(line)
    cv2.imshow("Donusmus Resim", convert_image)
    cv2.waitKey(0)
    components = convert_and_letter_segmentation.letter_segment(convert_image)

    if row_index in indis_list:  # This list covers specific rows for numbers
        label_number = prediction.label_list_number(components)
        combined_label_number = "".join(map(str, label_number))
        return combined_label_number
    else:
        print("HATAYOK3-----------------------")
        label = prediction.label_list(components)
        combined_label = "".join(label).title()
        return combined_label

def extract_info(img, num_rows, dict_indis, indis_list):
    row_index = 0
    segment_line_list = line_segmentation.line_segment(img, num_rows)
    dictionary = {}

    for line in segment_line_list:
        print(type(line))
        cv2.imshow("LineSegment Resim", line)
        cv2.waitKey(0)
        
        processed_label = process_line(line, row_index, indis_list)
        dictionary[dict_indis[row_index]] = processed_label
        
        print(dictionary)
        print("Row index: ", row_index)
        row_index += 1

    print("Başarılı")
    return dictionary

def form_uni_info(img):
    print("HATAYOK0-----------------------")
    dict_indis = ["ID: ", "Name: ", "Surname: ", "University: ", "Faculty: ", "Department: ", "Student Number: "]
    indis_list = [0, 6]
    print("HATAYOK2-----------------------")
    return extract_info(img, 7, dict_indis, indis_list)

def personal_info(img):
    dict_indis = ["ID: ", "Identification Number: ", "Gender: ", "Birth Year: "]
    indis_list = [0, 1, 3]
    return extract_info(img, 4, dict_indis, indis_list)

def contact_info(img):
    dict_indis = ["ID: ", "Country: ", "City: ", "Zipcode: ", "Street Number: ", "Apartment Number: "]
    indis_list = [0, 3, 4, 5]
    return extract_info(img, 6, dict_indis, indis_list)


img = cv2.imread('./sablon_form/form7.PNG') 
result_dict = form_uni_info(img)

print("---------------------------------")
print(result_dict)
print("---------------------------------")
print(result_dict["ID: "])
print("---------------------------------")
cv2.destroyAllWindows()

print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
end = datetime.now()
delta = end - start
print('Difference is seconds:', delta.total_seconds())
exit()


