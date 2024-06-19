import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def image_to_vector(image):
    # Преобразуем изображение в вектор
    return image.flatten()


def compare_images_cosine(imageA, imageB):
    # Преобразуем изображения в векторы
    vectorA = image_to_vector(imageA)
    vectorB = image_to_vector(imageB)

    # Вычисляем косинусное сходство
    similarity = cosine_similarity([vectorA], [vectorB])
    return similarity[0][0]


# Загрузка изображений
imageA = cv2.imread('cropped_image2.png', cv2.IMREAD_GRAYSCALE)
imageB = cv2.imread('cropped_image1.png', cv2.IMREAD_GRAYSCALE)

# Изменяем размер изображений до одинакового размера (если необходимо)
imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))

# Сравнение изображений
similarity = compare_images_cosine(imageA, imageB)

print(f"Cosine Similarity: {similarity}")

# Отображаем изображения
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(imageA, cmap='gray')
axes[0].set_title("Image A")
axes[1].imshow(imageB, cmap='gray')
axes[1].set_title("Image B")
for ax in axes:
    ax.axis('off')
plt.show()
