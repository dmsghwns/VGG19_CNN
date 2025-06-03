import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import subprocess
import numpy as np

# Disable pydevd file validation warning
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# Ensure TensorFlow version compatibility
print(f"Current TensorFlow Version: {tf.__version__}")
if not tf.__version__.startswith('2.'):
    print("Installing TensorFlow 2.17.0 for GPU compatibility...")
    !pip install tensorflow==2.17.0 -q
    import tensorflow as tf

# Check GPU availability and memory usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU Available: {gpus}")
    # Enable memory growth to prevent memory allocation issues
    for gpu in gpus:
         tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.get_strategy()
    # Check GPU memory usage with nvidia-smi
    try:
        memory_info = subprocess.check_output(['nvidia-smi']).decode('utf-8')
        print("GPU Memory Usage:\n", memory_info)
    except:
        print("Unable to check GPU memory usage. Ensure nvidia-smi is available.")
else:
    print("No GPU found. Falling back to CPU. Please select L4 GPU in 'Runtime > Change runtime type'.")
    strategy = tf.distribute.get_strategy()

# Mount Google Drive for checkpoint saving
from google.colab import drive
drive.mount('/content/drive')

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0, 1]

# Enhanced data augmentation
# Note: We will resize the images *after* they are yielded by the generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2
)

# Define the resize function (to be applied to batches)
def resize_images_batch(images):
    # Ensure images is a TensorFlow tensor before resizing
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    return tf.image.resize(images, [224, 224])

# Create data generators
train_generator = datagen.flow(x_train, y_train, batch_size=64)
# Validation generator (no augmentation)
val_generator = ImageDataGenerator().flow(x_test, y_test, batch_size=64)

# Wrap generators to apply resize_images_batch to each yielded batch
def resized_generator(generator):
    for images, labels in generator:
        # Resize the images batch
        resized_images = resize_images_batch(images)
        # labels might need reshaping if using sparse_categorical_crossentropy and coming from flow
        # ensure labels are correct shape, e.g., (batch_size,) or (batch_size, 1)
        yield resized_images, labels

train_generator_resized = resized_generator(train_generator)
val_generator_resized = resized_generator(val_generator)

# Define and compile VGG19 model
with strategy.scope():
    # input_shape는 resize 후의 크기인 (224, 224, 3)로 설정
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Fine-tuning: Unfreeze the last 5 layers
    base_model.trainable = True
    for layer in base_model.layers[:-5]:
        layer.trainable = False

    # Add custom top layers with batch normalization
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])

    # 학습률
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Load weights if checkpoint exists
checkpoint_path = "/content/drive/MyDrive/vgg19_cifar10_checkpoint.weights.h5"
if os.path.exists(checkpoint_path):
    try:
        # 모델 구조가 일치해야 로드 가능
        model.load_weights(checkpoint_path)
        print("Loaded weights from checkpoint.")
    except Exception as e:
        print(f"Error loading checkpoint weights: {e}")
        print("Starting training without loading weights.")

# Define checkpoint callback to save model weights
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    save_freq='epoch', # Save at the end of each epoch
    verbose=1
)

# Train the model using the resized data generators
print("Starting model training...")
# 에포크
history = model.fit(
    train_generator_resized,
    epochs=10,
    validation_data=val_generator_resized,
    steps_per_epoch=len(x_train) // 64, # Number of batches per epoch
    validation_steps=len(x_test) // 64, # Number of validation batches
    callbacks=[checkpoint_callback]
)
print("Training completed.")

# Plot training and validation loss/accuracy
plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Predict and visualize a sample image from the test set
# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Select the first image from the test set
sample_image = x_test[0:1]  # Shape: (1, 32, 32, 3)
sample_image_resized = resize_images_batch(sample_image)  # Shape: (1, 224, 224, 3)

# Make a prediction
prediction = model.predict(sample_image_resized)
predicted_class = np.argmax(prediction, axis=1)[0]
actual_class = y_test[0][0]

# Print the prediction result
print(f"Predicted Class: {class_names[predicted_class]}")
print(f"Actual Class: {class_names[actual_class]}")

# Visualize the image with prediction
plt.figure(figsize=(3, 3))
plt.imshow(x_test[0])
plt.title(f"Predicted: {class_names[predicted_class]}\nActual: {class_names[actual_class]}")
plt.axis('off')
plt.show()

# Print final GPU memory usage
try:
    memory_info = subprocess.check_output(['nvidia-smi']).decode('utf-8')
    print("Final GPU Memory Usage:\n", memory_info)
except:
    print("Unable to check final GPU memory usage.")
