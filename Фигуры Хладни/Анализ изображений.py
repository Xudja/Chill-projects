
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображений
image1 = cv2.imread('cropped_image1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('cropped_image2.png', cv2.IMREAD_GRAYSCALE)  # Путь ко второму изображению

# Получение размеров изображений
height1, width1 = image1.shape
height2, width2 = image2.shape

# Определение минимальных размеров
min_height = min(height1, height2)
min_width = min(width1, width2)

# Обрезка изображений до минимальных размеров
image1_cropped = image1[:min_height, :min_width]
image2_cropped = image2[:min_height, :min_width]

# Обнаружение границ с использованием алгоритма Кэнни
edges1 = cv2.Canny(image1_cropped, 100, 200)
edges2 = cv2.Canny(image2_cropped, 100, 200)

# Сравнение контуров
difference = cv2.absdiff(edges1, edges2)

# Визуализация результатов
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(edges1, cmap='gray')
axes[0].set_title("Контуры изображения 1")
axes[1].imshow(edges2, cmap='gray')
axes[1].set_title("Контуры изображения 2")
axes[2].imshow(difference, cmap='gray')
axes[2].set_title("Различия в контурах")
for ax in axes:
    ax.axis('off')
plt.show()

