import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

def load_data(dataset_path, img_size=(224, 224)):
    images = []
    labels = []

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)

        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)

                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)
                img = img / 255.0

                images.append(img)
                labels.append(0 if 'messy' in folder.lower() else 1)  # 0 = Messy, 1 = Organized

    return np.array(images), np.array(labels)


dataset_path = 'dataset'
X, y = load_data(dataset_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)


base_model.trainable = True


for layer in base_model.layers[:100]:
    layer.trainable = False


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

history_finetune = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=32)

model.save('room_cleanliness_model.h5')



def predict_image(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)


    prediction = model.predict(img)


    return f"Organized Room : {prediction}" if prediction > 0.5 else f"Messy Room : {prediction}"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = predict_image(frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, result, (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Room Cleanliness Detection - Press Q to Quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()