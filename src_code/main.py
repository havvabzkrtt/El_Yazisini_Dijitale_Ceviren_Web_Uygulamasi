# TAHMİN

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#app.py için bu şekilde kullanılcak
#from src_code.line_segmentation import line_segment 
#from src_code.convert_and_letter_segmentation import convert_img, letter_segment
#from src_code.prediction import label_list_number, label_list


from line_segmentation import line_segment 
from convert_and_letter_segmentation import convert_img, letter_segment
from prediction import label_list_number, label_list

print("yoksa burda mı??????")
# Ne kadar süre çalışıyor ve ne kadar bellek harcıyor



def process_line(line, row_index, indis_list):
    convert_image = convert_img(line)
    #cv2.imshow("Donusmus Resim", convert_image)
    #cv2.waitKey(0)
    components = letter_segment(convert_image)

    if row_index in indis_list:  # This list covers specific rows for numbers
        label_number = label_list_number(components)
        combined_label_number = "".join(map(str, label_number))
        return combined_label_number
    else:
        label = label_list(components)
        combined_label = "".join(label).title()
        return combined_label

def extract_info(img, num_rows, dict_indis, indis_list):
    row_index = 0
    segment_line_list = line_segment(img, num_rows)
    dictionary = {}

    for line in segment_line_list:
        print(type(line))
        #cv2.imshow("LineSegment Resim", line)
        #cv2.waitKey(0)
        
        processed_label = process_line(line, row_index, indis_list)
        dictionary[dict_indis[row_index]] = processed_label
        
        print(dictionary)
        print("Row index: ", row_index)
        row_index += 1

    print("Başarılı")
    return dictionary

def form_uni_info(img):
    dict_indis = ["ID: ", "Name: ", "Surname: ", "University: ", "Faculty: ", "Department: ", "Student Number: "]
    indis_list = [0, 6]
    print(extract_info(img, 7, dict_indis, indis_list))
    return extract_info(img, 7, dict_indis, indis_list)

def personal_info(img):
    dict_indis = ["ID: ", "Identification Number: ", "Gender: ", "Birth Year: "]
    indis_list = [0, 1, 3]
    return extract_info(img, 4, dict_indis, indis_list)

def contact_info(img):
    dict_indis = ["ID: ", "Country: ", "City: ", "Zipcode: ", "Street Number: ", "Apartment Number: "]
    indis_list = [0, 3, 4, 5]
    return extract_info(img, 6, dict_indis, indis_list)


img = cv2.imread('./sablon_form/uni_info1.PNG') 
result_dict = form_uni_info(img)

def process_image(form_type, image_path):
    img=cv2.imread(image_path)
    if form_type == "form1":  # Personal Info
        result = personal_info(img)
    elif form_type == "form2":  # University Info
        result = form_uni_info(img)
    elif form_type == "form3":  # Contact Info
        result = contact_info(img)
    else:
        result = {"processed_data": "Geçersiz form türü."}
    
    result.update({"form_type": form_type})
    return result


#print("---------------------------------")
#print(result_dict)
#print("---------------------------------")
#print(result_dict["ID: "])
#print("---------------------------------")

