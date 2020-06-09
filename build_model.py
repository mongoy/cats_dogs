"""
Создание и тренировка CNN
"""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import matplotlib.pyplot as plt
# from google.colab import files # для отлавке в GC


# Размер картинок
SIZE = 224

# Загрузка датасета с кошками и собаками
# метаданные tfds.load('cats_vs_dogs', split=['train[:100%]'], with_info=True, as_supervised=True)
train, _ = tfds.load('cats_vs_dogs', split=['train[:100%]'], with_info=True, as_supervised=True)

# Просмотр картинок 10 шт для примера
# for img, lable in train[0].take(10):
#     plt.figure()
#     plt.imshow(img)
#     print(lable)


# Преобразование картинок
def resize_image(img, label):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (SIZE, SIZE))
    img = img/255.0
    return img, label


#
train_resized = train[0].map(resize_image)
# Разделим датасет на части
train_batches = train_resized.shuffle(1000).batch(16)

# Строит сверточную нейросеть
# Загрузим базовый слой от CNN MobileNetV2
base_layers = tf.keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3), include_top=False)
base_layers.trainable = False
# Создадим модель сети
model = tf.keras.Sequential([
    base_layers,
    GlobalAveragePooling2D(),
    Dropout(0.2),
    Dense(1)
])
# Актевируем сеть
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

# Тренировка сети
model.fit(train_batches, epochs=1)

# Тренировка сети 2
model.fit(train_batches, epochs=2)

# Сохраняем веса - результат тренировки
model.save_weights("my_checkpoint")

# для GC загрузка картинок - проверка готово CNN
# files.upload()

# Проверка результатов тренировки
# for i in range(14):
#   img = load_img(f'{i+1}.jpg')
#   img_array = img_to_array(img)
#   img_resized, _ = resize_image(img_array, _)
#   img_expended = np.expand_dims(img_resized, axis=0)
#   prediction = model.predict(img_expended)[0][0]
#   pred_label = 'КОТ' if prediction < 0.5 else 'СОБАКА'
#   plt.figure()
#   plt.imshow(img)
#   plt.title(f'{pred_label} {prediction}')
