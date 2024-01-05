import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 20
EPOCHS = 3

# Load and preprocess the dataset
def load_dataset(dataset_path):
    images = []
    labels = []

    try:
        for filename in os.listdir(dataset_path):
            if filename.endswith(".png"):
                filepath = os.path.join(dataset_path, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, IMAGE_SIZE)
                
                label = int(filename.split('_')[-1][0])

                images.append(img)
                labels.append(label)
        
        return np.array(images), np.array(labels)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def preprocess_images(images):
    images = images / 255.0
    return images


def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(6, activation='softmax'))
    return model


dataset_path = "training_images (copy)"
images, labels = load_dataset(dataset_path)

if images is None or labels is None:
    print("Dataset loading failed. Please check your dataset path and format.")
    exit()

processed_images = preprocess_images(images)


label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)


X_train, X_val, y_train, y_val = train_test_split(
    processed_images, encoded_labels, test_size=0.2, random_state=42
)

X_train = X_train.reshape(X_train.shape + (1,))
X_val = X_val.reshape(X_val.shape + (1,))

model = build_model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) / BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val)
)

test_loss, test_acc = model.evaluate(X_val, y_val)
print(f"Test accuracy: {test_acc}")

model.save("finger_count_model")
