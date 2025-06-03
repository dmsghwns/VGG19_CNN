import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Ensure TensorFlow version compatibility (2.17.0 for TPU)
print(f"Current TensorFlow Version: {tf.__version__}")
if not tf.__version__.startswith('2.'):
    print("Installing TensorFlow 2.17.0 for TPU compatibility...")
    !pip install tensorflow==2.17.0 -q
    import tensorflow as tf

# Check TPU availability and initialize
try:
  tpu_address = os.environ.get('COLAB_TPU_ADDR')
  if tpu_address:
      print(f"TPU Address Found: grpc://{tpu_address}")
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + tpu_address)
      tf.config.experimental_connect_to_cluster(resolver)
      tf.tpu.experimental.initialize_tpu_system(resolver)
      strategy = tf.distribute.TPUStrategy(resolver)
      print("TPU initialized successfully.")
      tpu_devices = tf.config.list_logical_devices('TPU')
      print(f"Available TPU devices: {tpu_devices}")
  else:
      print("TPU address not found. Please check runtime settings.")
      raise ValueError("Ensure 'TPU' is selected in 'Runtime > Change runtime type' and restart the runtime.")
except Exception as e:
    print(f"Error initializing TPU: {e}")
    print("Falling back to GPU/CPU. To use TPU, verify TPU runtime and try again.")
    strategy = tf.distribute.get_strategy()

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0, 1]

# Resize images to 224x224 for VGG19
def resize_images(images):
    return tf.image.resize(images, [224, 224])

x_train = resize_images(x_train)
x_test = resize_images(x_test)

# Data augmentation for better generalization
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Define and compile VGG19 model within TPU strategy scope
with strategy.scope():
    # Load VGG19 with ImageNet weights, excluding top layers
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Freeze base model layers
    base_model.trainable = False

    # Add custom top layers for CIFAR-10
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train the model with data augmentation
print("Starting model training...")
history = model.fit(datagen.flow(x_train, y_train, batch_size=256),
                    epochs=5,
                    validation_data=(x_test, y_test),
                    steps_per_epoch=len(x_train) // 256)
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
