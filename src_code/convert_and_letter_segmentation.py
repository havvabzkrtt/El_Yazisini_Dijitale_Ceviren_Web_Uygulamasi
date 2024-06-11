
import cv2
import numpy as np
import matplotlib.pyplot as plt

# resim segment_line_list'ten değişikliğe uğramadan direkt geliyorsa 
# arkaplan siyah, ön plan beyaz
def convert_img(input_img):
    # Girdi görüntüsünün zaten grayscale olup olmadığını kontrol et
    if len(input_img.shape) == 2:
        # Otsu'nun Binarizasyonu ile eşik değerini otomatik olarak belirle
        print("değişmediiii")
        siyah_beyaz_resim = input_img
    else:
        # Grayscale'e dönüştürme
        siyah_beyaz_resim = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # Otsu'nun Binarizasyonu ile eşik değerini otomatik olarak belirle
    _, siyah_beyaz_resim = cv2.threshold(siyah_beyaz_resim, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Her piksel değerine 10 ekle
    satir, sutun = siyah_beyaz_resim.shape
    for i in range(satir):
        for j in range(sutun):
            # Her piksel değerine 10 ekle (bu örnekte)
            if siyah_beyaz_resim[i, j] == 255:
                siyah_beyaz_resim[i, j] = 0
            else:
                siyah_beyaz_resim[i, j] = 255

    return siyah_beyaz_resim

# arkaplan beyaz, ön plan siyah
def convert_img2(input_img):
    # Girdi görüntüsünün zaten grayscale olup olmadığını kontrol et
    if len(input_img.shape) == 2:
        # Otsu'nun Binarizasyonu ile eşik değerini otomatik olarak belirle
        print("değişmediiii")
        siyah_beyaz_resim = input_img
    else:
        # Grayscale'e dönüştürme
        siyah_beyaz_resim = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # Otsu'nun Binarizasyonu ile eşik değerini otomatik olarak belirle
    _, siyah_beyaz_resim = cv2.threshold(siyah_beyaz_resim, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Her piksel değerine 10 ekle
    satir, sutun = siyah_beyaz_resim.shape
    for i in range(satir):
        for j in range(sutun):
            # Her piksel değerine 10 ekle (bu örnekte)
            if siyah_beyaz_resim[i, j] == 255:
                siyah_beyaz_resim[i, j] = 255
            else:
                siyah_beyaz_resim[i, j] = 0

    return siyah_beyaz_resim



def letter_segment(image):

    # Görüntüyü yükleme
    # image = cv2.imread('outputs/dr3.png', cv2.IMREAD_GRAYSCALE)
    # Bağlı bileşenlerin analizi ve istatistiklerin elde edilmesi
    # Bağlı bileşenlerin analizi ve istatistiklerin elde edilmesi
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

    # Ağırlık merkezlerine göre sıralama yapmak için bir liste hazırla
    sorting_list = []
    for i in range(1, num_labels):  # 0. etiket arka plan için olduğundan atlıyoruz
        sorting_list.append((centroids[i, 0], i))  # (X koordinatı, etiket numarası)

    # X koordinatına göre sırala
    sorted_indices = sorted(sorting_list, key=lambda x: x[0])


    components = []
    # Sıralı bileşenleri kaydet
    for idx, (_, label) in enumerate(sorted_indices):
        # Bileşenin sınırlayıcı kutusunu al
        x, y, w, h, area = stats[label]
        
        # Bileşeni ayır
        component = np.zeros_like(image)
        component[labels == label] = 255
        
        # Bileşeni önce 20x20 boyutuna getir
        resized_component = cv2.resize(component[y:y+h, x:x+w], (20, 20))
        
        # 20x20 boyutlu resmi 28x28'e çıkarmak için her yöne 4 piksel kenarlık ekle
        final_component = cv2.copyMakeBorder(resized_component, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        components.append(final_component)
    return components



"""

# Resmi yükle

resim_yolu = "./inputs/ahmet.png"            
resim = cv2.imread(resim_yolu)

donusturulmus_resim = convert_img(resim)

# Dönüştürülmüş resmi göster

cv2.imshow("Dönüştürülmemiş Resim", resim)
cv2.imshow("Dönüştürülmüş Resim", donusturulmus_resim)
# cv2.imwrite("donusmus_resim.png", donusturulmus_resim)
cv2.waitKey(0)
cv2.destroyAllWindows()


components = letter_segment(donusturulmus_resim)



cv2.imwrite("outputs/ahmet_donusmus.png", donusturulmus_resim)
for idx, component in enumerate(components, start=1):
    cv2.imshow(f'Bileşen {idx}', component)
    cv2.imwrite(f'outputs/ahmet_{idx}.png', component)
    cv2.waitKey(0)  # Bir tuşa basılana kadar beklet
    cv2.destroyAllWindows()  # Pencereyi kapat

"""


