import cv2
import matplotlib.pyplot as plt

# Загрузка изображений
image1 = cv2.imread('квадрат 2.JPG')
image2 = cv2.imread('вадрат.JPG')  # Путь ко второму изображению

# Получение размеров изображений
height1, width1 = image1.shape[:2]
height2, width2 = image2.shape[:2]

# Определение минимальных размеров
min_height = min(height1, height2)
min_width = min(width1, width2)

# Функция для обрезки изображения до заданного размера
def crop_image(image, width, height):
    return image[:height, :width]

# Обрезка изображений
cropped_image1 = crop_image(image1, min_width, min_height)
cropped_image2 = crop_image(image2, min_width, min_height)

# Сохранение обрезанных изображений (опционально)
cv2.imwrite('cropped_image1.png', cropped_image1)
cv2.imwrite('cropped_image2.png', cropped_image2)

# Визуализация обрезанных изображений
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(cropped_image1, cv2.COLOR_BGR2RGB))
axes[0].set_title("Обрезанное изображение 1")
axes[1].imshow(cv2.cvtColor(cropped_image2, cv2.COLOR_BGR2RGB))
axes[1].set_title("Обрезанное изображение 2")
for ax in axes:
    ax.axis('off')
plt.show()
