
import cv2
import numpy as np

# Resmi yükle
"""
resim_yolu = "./src_code/merhaba.PNG"            
resim = cv2.imread(resim_yolu)
"""

# resim segment_line_list'ten değişikliğe uğramadan direkt geliyorsa 

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

"""

# resim path'ten okumuyorsa
def convert_img(input_img):
    # Resmi siyah beyaz yap
    siyah_beyaz_resim = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # Otsu'nun Binarizasyonu ile eşik değerini otomatik olarak belirle
    _, siyah_beyaz_resim = cv2.threshold(siyah_beyaz_resim, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Dönüştürülmüş resmi göster
    cv2.imshow("Dönüştürülmüş Resim", siyah_beyaz_resim)


    # Her piksel değerine 10 ekle
    satir, sutun = siyah_beyaz_resim.shape
    for i in range(satir):
        for j in range(sutun):
            # Her piksel değerine 10 ekle (bu örnekte)
            #print(siyah_beyaz_resim[i, j] )
            if(siyah_beyaz_resim[i, j] == 255):
                siyah_beyaz_resim[i, j] = 0
            else:
                siyah_beyaz_resim[i, j] = 255
    return siyah_beyaz_resim
"""

def letter_segment(input_siyah_beyaz_resim):

    # Bağlı bileşenlerin analizi ve istatistiklerin elde edilmesi
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(input_siyah_beyaz_resim)

    # Ağırlık merkezlerine göre sıralama yapmak için bir liste hazırla
    sorting_list = []
    for i in range(1, num_labels):  # 0. etiket arka plan için olduğundan atlıyoruz
        sorting_list.append((centroids[i, 0], i))  # (X koordinatı, etiket numarası)

    # X koordinatına göre sırala
    sorted_indices = sorted(sorting_list, key=lambda x: x[0])

    components = []  # Resimleri tutacak liste

    # Sıralı bileşenleri kaydet
    for idx, (_, label) in enumerate(sorted_indices):
        # Bileşenin sınırlayıcı kutusunu al
        x, y, w, h, area = stats[label]
        
        # Bileşeni ayır
        component = np.zeros_like(input_siyah_beyaz_resim)
        component[labels == label] = 255
        
        # Bileşenin sınırlayıcı kutusunu kontrol et
        if w > 28 or h > 28:
            # Bileşeni 28x28 boyutuna yeniden boyutlandır ve kenarlık bırak
            component = cv2.resize(component[y:y+h, x:x+w], (28, 28))
            border_size = 10 # Kenarlık boyutu
            component = cv2.copyMakeBorder(component, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
        
        # Her bileşeni ayrı bir dosyaya kaydet
        # cv2.imwrite(f'{idx}.png', component)
        
        # Bileşeni listeye ekle
        components.append(component)

    return components



# donusturulmus_resim = convert_img(resim)

# Dönüştürülmüş resmi göster
"""
cv2.imshow("Dönüştürülmemiş Resim", resim)
cv2.imshow("Dönüştürülmüş Resim", donusturulmus_resim)
# cv2.imwrite("donusmus_resim.png", donusturulmus_resim)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# components = letter_segment(donusturulmus_resim)

# listedeki her bir resmi tek tek görüntüle
"""
for idx, component in enumerate(components, start=1):
    cv2.imshow(f'Bileşen {idx}', component)
    cv2.waitKey(0)  # Bir tuşa basılana kadar beklet
    cv2.destroyAllWindows()  # Pencereyi kapat
"""

