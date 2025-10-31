# ===========================================
# 🧠 Experiment 3: AI for Medical Diagnosis based on MRI/X-ray Data
# ===========================================

# 🎯 Objective:
# Build a Convolutional Neural Network (CNN) to classify MRI/X-ray images
# into healthy vs. diseased categories using deep learning.

# ===========================================
# 📘 Step 1: Import Required Libraries
# ===========================================
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ===========================================
# 📂 Step 2: Dataset Directory Setup
# ===========================================
# Directory structure should be like:
# dataset/
# ├── train/
# │   ├── Normal/
# │   └── Diseased/
# └── test/
#     ├── Normal/
#     └── Diseased/

train_dir = "dataset/train"
test_dir = "dataset/test"

# ===========================================
# 🧹 Step 3: Data Preprocessing & Augmentation
# ===========================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

# ===========================================
# 🏗️ Step 4: CNN Model Architecture
# ===========================================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

# ===========================================
# ⚙️ Step 5: Compile the Model
# ===========================================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ===========================================
# 🧠 Step 6: Model Training
# ===========================================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_data,
    epochs=20,
    validation_data=test_data,
    callbacks=[early_stop],
    verbose=1
)

# ===========================================
# 📈 Step 7: Performance Visualization
# ===========================================
plt.figure(figsize=(10,4))

# Accuracy Plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='blue')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss', color='red')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ===========================================
# 🧪 Step 8: Model Evaluation
# ===========================================
loss, acc = model.evaluate(test_data, verbose=0)
print(f"✅ Final Test Accuracy: {acc*100:.2f}%")
print(f"✅ Final Test Loss: {loss:.4f}")

# ===========================================
# 💾 Step 9: Save Model
# ===========================================
model.save("AI_Medical_Diagnosis_Model.h5")
print("💾 Model saved as 'AI_Medical_Diagnosis_Model.h5'")

# ===========================================
# 📊 Step 10: Summary
# ===========================================
model.summary()

# ===========================================
# 🧩 Step 11: Visualize Few Predictions
# ===========================================
import numpy as np
from tensorflow.keras.preprocessing import image

# Fetch some test images for prediction
for i in range(3):
    img_path, label = test_data.filepaths[i], test_data.labels[i]
    img = image.load_img(img_path, target_size=(150,150))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]
    result = "Diseased" if pred > 0.5 else "Normal"

    plt.imshow(image.load_img(img_path))
    plt.title(f"Actual: {'Diseased' if label==1 else 'Normal'} | Predicted: {result}")
    plt.axis('off')
    plt.show()
