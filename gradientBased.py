import cv2
import os
import matplotlib.pyplot as plt

os.makedirs('cache', exist_ok=True)

img = cv2.imread('images/foto.jpg', 0)

if img is None:
    print("File tidak ditemukan")
    exit()

# Sobel Gradient
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Magnitude gradient
grad = cv2.magnitude(sobelx, sobely)
grad = cv2.convertScaleAbs(grad)

# Canny Edge
canny = cv2.Canny(img, 100, 200)

# Simpan
cv2.imwrite('cache/grad_sobel.png', grad)
cv2.imwrite('cache/grad_canny.png', canny)

# Tampil
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title('Sumber')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title('Magnitudo gradien')
plt.imshow(grad, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title('Sharpening')
plt.imshow(canny, cmap='gray')
plt.axis('off')

plt.show()