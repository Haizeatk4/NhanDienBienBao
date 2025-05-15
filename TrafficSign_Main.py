# 2.2. Chuẩn bị dữ liệu và tiền xử lý

# 2.2.1. Thu thập dữ liệu
import pandas as pd

df_train = pd.read_csv("Train.csv")
print("Sample data:")
print(df_train.head())

# 2.2.2. Tiền xử lý dữ liệu
import os
import cv2
import numpy as np
from tensorflow import keras
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

images = []
labels = []

for index, row in df_train.iterrows():
    class_id = row["ClassId"]
    img_path = os.path.join("Train", str(class_id), row["Path"].split("/")[-1])
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (32, 32))
        img = img / 255.0
        images.append(img)
        labels.append(class_id)

X = np.array(images)
y = to_categorical(labels, num_classes=43)

# Tăng cường dữ liệu
train_datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

train_generator = train_datagen.flow(
    X, y,
    batch_size=32,
    subset='training'
)
val_generator = train_datagen.flow(
    X, y,
    batch_size=32,
    subset='validation'
)

# 2.3. Phát triển mô hình phân loại CNN

# 2.3.1. Kiến trúc mô hình
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')  # 43 lớp biển báo
])

# 2.3.2. Thông số huấn luyện
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(train_generator,
          validation_data=val_generator,
          epochs=20)

# 2.3.3. Lưu mô hình
model.save("model.h5")
