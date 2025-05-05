import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalMaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt

# Set random seed for reproducibility
tf.random.set_seed(42)

# Define paths
train_dir = './archive/Dataset'
img_height, img_width = 150, 150  # Increased image size
batch_size = 32  # Reduced batch size

# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training', shuffle=True,
)

# Validation data generator
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation', shuffle=False,
)

# Build an enhanced CNN model
model = Sequential([

    Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.2),


    GlobalMaxPooling2D(),

    # Dense layers
    Dense(256, activation='relu'),
    Dropout(0.25),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model with adjusted learning rate and metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
     metrics=['accuracy', Precision(), Recall()]
)

# Create callbacks
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=12,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-5,
    verbose=1
)

# Train the model
history = model.fit(
    train_generator,
    # steps_per_epoch=train_generator.samples // batch_size,
    epochs=150,  # Increased epochs
    validation_data=validation_generator,
   

    # validation_steps=validation_generator.samples // batch_size,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
        ],      
)

# Load the best model
best_model = tf.keras.models.load_model('best_model.keras')

# Evaluate the best model
train_results = best_model.evaluate(train_generator)
val_results = best_model.evaluate(validation_generator)

print(f"\nBest Model Training Metrics:")
print(f"Accuracy: {train_results[1]:.4f}")
print(f"Precision: {train_results[2]:.4f}")
print(f"Recall: {train_results[3]:.4f}")

print(f"\nBest Model Validation Metrics:")
print(f"Accuracy: {val_results[1]:.4f}")
print(f"Precision: {val_results[2]:.4f}")
print(f"Recall: {val_results[3]:.4f}")
# Plot training history
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Plot the training history
plot_training_history(history)