import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображений
image1 = cv2.imread('cropped_image1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('cropped_image2.png', cv2.IMREAD_GRAYSCALE)  # Путь ко второму изображению
kernel = np.ones((3,3), np.uint8)

# Получение размеров изображений
height1, width1 = image1.shape
height2, width2 = image2.shape

# Определение минимальных размеров
min_height = min(height1, height2)
min_width = min(width1, width2)

# Обрезка изображений до минимальных размеров
image1_cropped = image1[:min_height, :min_width]
image2_cropped = image2[:min_height, :min_width]

# Инвертирование изображений
image1_inverted = cv2.bitwise_not(image1_cropped)
image2_inverted = cv2.bitwise_not(image2_cropped)
image1_blurred = cv2.GaussianBlur(image1_inverted, (5, 5), 0)
image2_blurred = cv2.GaussianBlur(image2_inverted, (9, 9), 0)
image2_thresh = cv2.morphologyEx(image2_blurred, cv2.MORPH_OPEN, kernel)

# Нормализация изображений
image1_normalized = cv2.normalize(image1_blurred, None, 0, 255, cv2.NORM_MINMAX)
image2_normalized = cv2.normalize(image2_thresh, None, 0, 255, cv2.NORM_MINMAX)

# Применение адаптивного порога для определения "белого"
image1_thresh = cv2.adaptiveThreshold(image1_normalized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)
image2_thresh = cv2.adaptiveThreshold(image2_normalized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)

# Применение морфологической операции для устранения пересвеченного фона
image1_thresh = cv2.morphologyEx(image1_thresh, cv2.MORPH_OPEN, kernel)
# image1_thresh = cv2.morphologyEx(image1_thresh, cv2.MORPH_CLOSE, kernel)

image1_thresh = cv2.morphologyEx(image1_thresh, cv2.MORPH_OPEN, kernel)
# image1_thresh = cv2.morphologyEx(image1_thresh, cv2.MORPH_CLOSE, kernel)

image2_thresh = cv2.morphologyEx(image2_thresh, cv2.MORPH_OPEN, kernel)
# image2_thresh = cv2.morphologyEx(image2_thresh, cv2.MORPH_CLOSE, kernel)

image2_thresh = cv2.morphologyEx(image2_thresh, cv2.MORPH_OPEN, kernel)
# image2_thresh = cv2.morphologyEx(image2_thresh, cv2.MORPH_CLOSE, kernel)

# Вычисление разности изображений
difference = cv2.absdiff(image1_thresh, image2_thresh)

# Визуализация результатов
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image1_thresh, cmap='gray')
axes[0].set_title("Физ.модель ")
axes[1].imshow(image2_thresh, cmap='gray')
axes[1].set_title("Теор.модель")
axes[2].imshow(difference, cmap='gray')
axes[2].set_title("Разность изображений")
for ax in axes:
    ax.axis('off')
plt.show()

# Сохранение результатов (опционально)
cv2.imwrite('/mnt/data/image1_inverted_thresh.png', image1_thresh)
cv2.imwrite('/mnt/data/image2_inverted_thresh.png', image2_thresh)
cv2.imwrite('/mnt/data/difference_inverted.png', difference)
