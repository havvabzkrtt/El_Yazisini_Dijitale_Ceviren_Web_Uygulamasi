import cv2
import numpy as np
import matplotlib.pyplot as plt

def line_segment(image, num_rows):
    # Resmin boyutlarını alın
    height, width, _= image.shape

    # Her bir satırın yüksekliğini hesaplayın
    row_height = height // num_rows
    #print("Uzunluk:",row_height)
    # Parçalara bölünmüş resimleri saklamak için bir liste oluşturun
    row_images = []

    # Resmi parçalara böl
    for i in range(num_rows):
        # Başlangıç ve bitiş indekslerini hesaplayın
        start_row = i * row_height
        end_row = start_row + row_height

        # Parçayı alın
        row_image = image[start_row:end_row, :]

        # Parçayı listeye ekleyin
        row_images.append(row_image)

    return row_images


num_rows = 7

# read image, prepare it by resizing it to fixed height and converting it to grayscale
img = cv2.imread("./sablon_form/form1.PNG")


if img is None:
    print("Dosya okunamadı. Lütfen dosyanın mevcut olduğundan ve uygun bir görüntü formatında olduğundan emin olun.")
else:
    img = img.astype(np.uint8)
  

res_lines = line_segment(img,num_rows)

# Parçalara bölünmüş resimleri görselleştirin
plt.figure(figsize=(10, 10))
for i, row_image in enumerate(res_lines):
    plt.subplot(num_rows, 1, i + 1)
    plt.imshow(row_image, cmap='gray')
    plt.axis('off')
    plt.title(f"Row {i + 1}")
plt.tight_layout()
plt.show()
