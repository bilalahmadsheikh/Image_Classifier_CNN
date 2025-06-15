import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import json
import cv2
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(42)
np.random.seed(42)

# SIMPLIFIED SETTINGS
IMG_SIZE = (64, 64)  # Smaller size - Roman numerals don't need high resolution
BATCH_SIZE = 32      # Larger batch size for stability

# Dataset paths
base_dir = "Dataset"  # Make this configurable
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# MINIMAL preprocessing - just basic normalization
def simple_preprocess_image(image_path, target_size=IMG_SIZE, debug=False):
    """Minimal preprocessing - keep it simple!"""
    try:
        # Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = image_path
            
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # ONLY do essential preprocessing:
        
        # 1. Resize to target size
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        # 2. Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # 3. Optional: Ensure consistent contrast (simple approach)
        # If image is too dark, brighten it slightly
        if np.mean(img_normalized) < 0.3:
            img_normalized = np.clip(img_normalized * 1.2, 0, 1)
        
        if debug:
            plt.figure(figsize=(10, 3))
            plt.subplot(1, 3, 1)
            plt.imshow(img, cmap='gray')
            plt.title('Original')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(img_resized, cmap='gray')
            plt.title('Resized')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(img_normalized, cmap='gray')
            plt.title('Normalized')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return img_normalized
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

# SIMPLIFIED data augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,      # Minimal rotation
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,         # Minimal zoom
    horizontal_flip=False,   # Never flip Roman numerals
    vertical_flip=False,
    fill_mode='constant',
    cval=0.0
)

val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

# Load data
train_data = train_gen.flow_from_directory(
    train_dir, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='categorical',
    shuffle=True,
    color_mode='grayscale'
)

val_data = val_gen.flow_from_directory(
    val_dir, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='categorical',
    shuffle=False,
    color_mode='grayscale'
)

test_data = test_gen.flow_from_directory(
    test_dir, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='categorical',
    shuffle=False,
    color_mode='grayscale'
)

# SIMPLIFIED model - much smaller and faster
def create_simple_model(input_shape, num_classes):
    model = Sequential([
        # First conv block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second conv block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third conv block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Classification layers
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Create and compile model
num_classes = len(train_data.class_indices)
model = create_simple_model((IMG_SIZE[0], IMG_SIZE[1], 1), num_classes)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.arange(num_classes),
    y=train_data.classes
)
class_weight_dict = dict(enumerate(class_weights))

# Callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "simple_roman_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.0001,
    verbose=1
)

# Training
steps_per_epoch = max(1, train_data.samples // BATCH_SIZE)
validation_steps = max(1, val_data.samples // BATCH_SIZE)

print(f"Training with {steps_per_epoch} steps per epoch")

history = model.fit(
    train_data,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_data,
    validation_steps=validation_steps,
    epochs=30,  # Reduced epochs
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weight_dict,
    verbose=1
)

# Simple prediction function
def predict_simple(image_path, debug=False):
    """Simple prediction with minimal preprocessing"""
    try:
        img_processed = simple_preprocess_image(image_path, debug=debug)
        if img_processed is None:
            return None
        
        img_batch = np.expand_dims(np.expand_dims(img_processed, axis=-1), axis=0)
        predictions = model.predict(img_batch, verbose=0)
        
        class_labels = list(train_data.class_indices.keys())
        pred_idx = np.argmax(predictions[0])
        confidence = predictions[0][pred_idx]
        
        predicted_class = class_labels[pred_idx]
        
        if debug:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(img_processed, cmap='gray')
            plt.title('Processed Image')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.bar(class_labels, predictions[0])
            plt.title('Predictions')
            plt.xticks(rotation=45)
            plt.ylabel('Confidence')
            plt.tight_layout()
            plt.show()
        
        return predicted_class, confidence, predictions[0]
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

# Test the simplified approach
def test_simple_classification():
    """Test with minimal preprocessing"""
    print("\n=== TESTING SIMPLIFIED APPROACH ===")
    
    # Test images from your original code
    test_images = [
        {
            "path": r"D:\projects\ai_project_oel\Dataset\train\iv\4_cap_50.png",
            "expected": "IV (4)",
            "description": "Training image IV"
        },
        {
            "path": r"C:\Users\bilaa\OneDrive\Pictures\5_test.jpg", 
            "expected": "V (5)",
            "description": "Handwritten V"
        }
    ]
    
    for test_case in test_images:
        img_path = test_case["path"]
        expected = test_case["expected"]
        description = test_case["description"]
        
        print(f"\n--- Testing {description} ---")
        print(f"Path: {img_path}")
        print(f"Expected: {expected}")
        
        if os.path.exists(img_path):
            result = predict_simple(img_path, debug=True)
            if result:
                predicted_class, confidence, all_preds = result
                print(f"Prediction: {predicted_class.upper()} ({confidence*100:.1f}%)")
                
                # Show top 3 predictions
                class_labels = list(train_data.class_indices.keys())
                top_indices = np.argsort(all_preds)[::-1][:3]
                print("Top 3 predictions:")
                for i, idx in enumerate(top_indices):
                    print(f"  {i+1}. {class_labels[idx].upper()} - {all_preds[idx]*100:.1f}%")
            else:
                print("❌ Prediction failed")
        else:
            print(f"❌ Image not found: {img_path}")
            print("💡 Please update the path or place your test images in the correct location")

# Alternative function for testing any image
def test_custom_image(image_path, expected_label=None):
    """Test a single custom image"""
    print(f"\n=== TESTING CUSTOM IMAGE ===")
    print(f"Path: {image_path}")
    if expected_label:
        print(f"Expected: {expected_label}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    result = predict_simple(image_path, debug=True)
    if result:
        predicted_class, confidence, all_preds = result
        print(f"Prediction: {predicted_class.upper()} ({confidence*100:.1f}%)")
        
        # Show all predictions
        class_labels = list(train_data.class_indices.keys())
        print("\nAll predictions:")
        for label, prob in zip(class_labels, all_preds):
            print(f"  {label.upper()}: {prob*100:.1f}%")
        
        return predicted_class, confidence
    else:
        print("❌ Prediction failed")
        return None

# Function to test multiple images from a directory
def test_directory(directory_path, max_images=5):
    """Test multiple images from a directory"""
    print(f"\n=== TESTING DIRECTORY: {directory_path} ===")
    
    if not os.path.exists(directory_path):
        print(f"❌ Directory not found: {directory_path}")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(directory_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        print("❌ No image files found in directory")
        return
    
    # Test up to max_images
    test_files = image_files[:max_images]
    print(f"Testing {len(test_files)} images...")
    
    results = []
    for filename in test_files:
        img_path = os.path.join(directory_path, filename)
        print(f"\n--- Testing: {filename} ---")
        
        result = predict_simple(img_path, debug=False)  # Set debug=True to see each image
        if result:
            predicted_class, confidence, _ = result
            print(f"Prediction: {predicted_class.upper()} ({confidence*100:.1f}%)")
            results.append((filename, predicted_class, confidence))
        else:
            print("❌ Failed to process")
            results.append((filename, "ERROR", 0.0))
    
    # Summary
    print(f"\n=== SUMMARY FOR {directory_path} ===")
    for filename, prediction, confidence in results:
        print(f"{filename}: {prediction} ({confidence*100:.1f}%)")
    
    return results

# Run evaluation
if test_data.samples > 0:
    test_loss, test_acc = model.evaluate(test_data, verbose=1)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# Plot training history
if history:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run the actual tests
print("\n" + "="*50)
print("RUNNING TESTS WITH YOUR SPECIFIC IMAGES")
print("="*50)

# Test your specific images
test_simple_classification()

# Example usage for custom testing:
print("\n" + "="*50)  
print("ADDITIONAL TESTING FUNCTIONS AVAILABLE:")
print("="*50)
print("1. test_custom_image('path/to/image.jpg', 'Expected Label')")
print("2. test_directory('path/to/image/folder', max_images=5)")
print("3. test_simple_classification() - tests your predefined images")

# Uncomment these lines to test additional images:
# test_custom_image(r"C:\path\to\your\image.jpg", "Expected Roman Numeral")
# test_directory(r"C:\path\to\your\test\folder", max_images=10)

print("\n=== SIMPLIFIED APPROACH BENEFITS ===")
print("✓ Faster training (30 epochs vs 100)")
print("✓ Smaller model (less overfitting)")
print("✓ Minimal preprocessing (preserves image quality)")
print("✓ Better generalization")
print("✓ Easier to debug and understand")
print("✓ Proper image path handling")
print("✓ Multiple testing functions available")