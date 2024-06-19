import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image = cv2.imread('cropped_image2.png', cv2.IMREAD_GRAYSCALE)

# Применение порогового значения для создания маски белых областей
_, mask = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)

# Увеличение маски, чтобы захватить окрестности белых областей
kernel = np.ones((3,3), np.uint8)
mask_dilated = cv2.dilate(mask, kernel, iterations=2)

# Инверсия маски
mask_inv = cv2.bitwise_not(mask_dilated)

# Применение инверсированной маски к изображению
result = cv2.bitwise_and(image, image, mask=mask_inv)

# Визуализация результатов
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Исходное изображение")
axes[1].imshow(result, cmap='gray')
axes[1].set_title("Изображение после удаления сетки")
for ax in axes:
    ax.axis('off')
plt.show()




