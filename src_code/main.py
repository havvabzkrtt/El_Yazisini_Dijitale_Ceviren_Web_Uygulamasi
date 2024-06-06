# TAHMİN

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# app.py için bu şekilde kullanılcak

from src_code.line_segmentation import line_segment 
from src_code.convert_and_letter_segmentation import convert_img, letter_segment
from src_code.prediction import label_list_number, label_list

# main.py ı çalıştırırken

#from line_segmentation import line_segment 
#from convert_and_letter_segmentation import convert_img, letter_segment
#from prediction import label_list_number, label_list


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
        processed_label = process_line(line, row_index, indis_list)
        dictionary[dict_indis[row_index]] = processed_label
        
        print(dictionary)
        print("Row index: ", row_index)
        row_index += 1

    print("Başarılı")
    return dictionary

def uni_info(img):
    dict_indis = ["ID: ", "Name: ", "Surname: ", "University: ", "Faculty: ", "Department: ", "Student Number: "]
    indis_list = [0, 6] #rakam içeren indisler
    return extract_info(img, 7, dict_indis, indis_list)

def personal_info(img):
    dict_indis = ["ID: ", "Identification Number: ", "Gender: ", "Birth Year: ", "Birthplace: ", "Disease: ", "Phone Number: "]
    indis_list = [0, 1, 3, 6]
    return extract_info(img, 7, dict_indis, indis_list)

def address_info(img):
    dict_indis = ["ID: ", "Country: ", "City: ", "Zipcode: ", "Street Number: ", "Apartment Number: "]
    indis_list = [0, 3, 4, 5]
    return extract_info(img, 6, dict_indis, indis_list)


#img = cv2.imread('./sablon_form/2.png') 
#result_dict = uni_info(img)


from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 20)
        self.cell(0, 10, "HandyConvert Result", 0, 1, "C")
        self.ln(10)

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(5)

    def chapter_body(self, body):
        self.set_font("Arial", "", 12)
        for key, value in body.items():
            self.cell(0, 10, f"{key}: {value}", 0, 1)
        self.ln(10)

def save_dict_to_pdf(form_type, result, filename="result.pdf"):
    if form_type == 'personal_info':
        selected_option_text = 'Personal Informations'
    elif form_type == 'uni_info':
        selected_option_text = 'University Informations'
    elif form_type == 'address_info':
        selected_option_text = 'Address Informations'
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt=f"Form Type: {selected_option_text}", ln=True, align='L')
    pdf.ln(10)

    # Tablo içeriği
    pdf.set_font("Arial", "", 12)
    for key, value in result.items():
        pdf.cell(50, 10, txt=key, border=1)
        pdf.cell(0, 10, txt=str(value), border=1)
        pdf.ln()

    pdf.output(filename)
    print(f"PDF başarıyla '{filename}' olarak oluşturuldu.")



def process_image(form_type, img):
    result = {}  # Sonuç sözlüğü
    
    if form_type == "personal_info":  # Personal Info
        result_data = personal_info(img)
        print(result_data)
        result.update(result_data)
    elif form_type == "uni_info":  # University Info
        result_data = uni_info(img)
        print(result_data)
        result.update(result_data)
    elif form_type == "address_info":  # Address Info
        result_data = address_info(img)
        print(result_data)
        result.update(result_data)
    else:
        result = {"processed_data": "Geçersiz form türü."}
    
    #result.update({"form_type": form_type})
    
    # Sonucu PDF olarak kaydet
    save_dict_to_pdf(form_type, result)
    
    return result

# Örnek kullanım
# process_image("personal_info", img)

