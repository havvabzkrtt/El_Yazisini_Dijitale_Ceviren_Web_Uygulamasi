




import cv2
import numpy as np
from matplotlib import pyplot as plt

def word_segment(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Gauss bulanıklığı uygula
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Kenar tespiti yap
    edged = cv2.Canny(blurred, 50, 150)
    
    # Dilasyon ve erozyon işlemleriyle kenarları genişlet ve daralt
    kernel = np.ones((5, 15), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=1)
    
    # Kontur tespiti
    cnt, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Kontur tespiti başarılı mı?
    if len(cnt) == 0 or hierarchy is None:
        print("No contours found")
        return []
    else:
        print(f"Contours found: {len(cnt)}")

    # Tespit edilen konturları görüntü üzerinde göster
    image_contours = image.copy()
    segments = []
    sorted_segments = []  # Sıralanmış segment listesi
    for i, c in enumerate(cnt):
        # Konturların boyutlarını filtrele
        x, y, w, h = cv2.boundingRect(c)
        if 50 < w < 600 and 20 < h < 200:  # Genişlik ve yükseklik kısıtlamalarını düzenledim
            cv2.rectangle(image_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Segmenti ekle
            segment = image[y:y+h, x:x+w]
            segments.append((x, segment))  # (x, segment) tuple'ı eklendi

    # x koordinatına göre sırala
    sorted_segments = sorted(segments, key=lambda x: x[0])

    # Görüntüyü göster
    
    #plt.figure(figsize=(10, 10))
    #plt.imshow(image_contours)
    #plt.title("Filtered Detected Contours")
    #plt.show()
    
    return [segment for _, segment in sorted_segments]  # Sıralanmış segmentler



"""

# Fonksiyonu deneme
# Gerekli kütüphaneleri içe aktar
import cv2
import matplotlib.pyplot as plt

# Fonksiyonu deneme
image_path = "./sablon_form/uni_tekli.PNG"
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
segments = word_segment(image)

# Parçalara bölünmüş resimleri görselleştir
plt.figure(figsize=(6, 4))
num_segments = len(segments) - 1  # İlk segmenti çıkartıyoruz
for i, segment in enumerate(segments[1:], 1):  # İlk segmenti atlamak için enumerate(segments[1:], 1) kullanıyoruz
    plt.subplot(num_segments, 1, i)
    plt.imshow(segment, cmap='gray')
    plt.axis('off')
    plt.title(f"Word {i}")
plt.tight_layout()
plt.show()

"""