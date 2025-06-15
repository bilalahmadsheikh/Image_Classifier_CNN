import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy import ndimage
from skimage import morphology, filters, measure
import json
import argparse
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class RomanNumeralTester:
    def __init__(self, model_path=r"simple_roman_model.keras", img_size=(64, 64)):
        """Initialize the Roman Numeral Tester
        
        Args:
            model_path (str): Path to the trained model file
            img_size (tuple): Target image size for preprocessing
        """
        self.img_size = img_size
        self.model = None
        self.class_labels = None
        
        # Define default class labels BEFORE calling load_model
        self.default_class_labels = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
        
        # Load the model
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                print("Please make sure the model file exists.")
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            print(f"Loading model from: {model_path}")
            self.model = load_model(model_path)
            print("✓ Model loaded successfully!")
            
            # Try to load class labels if they exist
            label_path = model_path.replace('.keras', '_labels.json').replace('.h5', '_labels.json')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    self.class_labels = json.load(f)
                print(f"✓ Class labels loaded: {self.class_labels}")
            else:
                self.class_labels = self.default_class_labels.copy()
                print(f"⚠️ Using default class labels: {self.class_labels}")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Make sure the model file exists and is compatible.")
            raise
    
    def save_class_labels(self, labels, model_path="simple_roman_model.keras"):
        """Save class labels to JSON file"""
        try:
            label_path = model_path.replace('.keras', '_labels.json').replace('.h5', '_labels.json')
            with open(label_path, 'w') as f:
                json.dump(labels, f)
            print(f"✓ Class labels saved to: {label_path}")
        except Exception as e:
            print(f"❌ Error saving class labels: {e}")
    
    def assess_image_quality(self, img):
        """Assess if image needs more intensive preprocessing"""
        try:
            # Convert to float for calculations
            img_float = img.astype(np.float32) / 255.0
            
            # Calculate various quality metrics
            mean_intensity = np.mean(img_float)
            std_intensity = np.std(img_float)
            
            # Edge detection to check clarity
            edges = cv2.Canny((img_float * 255).astype(np.uint8), 50, 150)
            edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
            
            # Noise estimation using Laplacian variance
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            
            # Contrast measurement
            contrast = std_intensity / (mean_intensity + 1e-8)
            
            # Decision criteria for enhanced preprocessing
            needs_enhancement = (
                mean_intensity < 0.4 or  # Too dark
                mean_intensity > 0.8 or  # Too bright
                std_intensity < 0.1 or   # Low contrast
                edge_density < 0.05 or   # Too few edges (blurry)
                laplacian_var < 100 or   # Low sharpness
                contrast < 0.3           # Poor contrast
            )
            
            return needs_enhancement, {
                'mean_intensity': mean_intensity,
                'std_intensity': std_intensity,
                'edge_density': edge_density,
                'laplacian_var': laplacian_var,
                'contrast': contrast
            }
        except Exception as e:
            print(f"Error in image quality assessment: {e}")
            return False, {}
    
    def enhanced_preprocess_image(self, img, target_size=None):
        """Enhanced preprocessing for challenging images"""
        if target_size is None:
            target_size = self.img_size
            
        try:
            # 1. Resize first
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # 2. Noise reduction
            img_denoised = cv2.bilateralFilter(img_resized, 9, 75, 75)
            
            # 3. Adaptive histogram equalization for better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            img_enhanced = clahe.apply(img_denoised)
            
            # 4. Morphological operations to clean up the image
            kernel = np.ones((2, 2), np.uint8)
            
            # Close small gaps
            img_closed = cv2.morphologyEx(img_enhanced, cv2.MORPH_CLOSE, kernel)
            
            # 5. Adaptive thresholding to better separate foreground/background
            img_thresh = cv2.adaptiveThreshold(
                img_closed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 6. Find and clean the main connected component
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                255 - img_thresh, 8, cv2.CV_32S
            )
            
            if num_labels > 1:
                # Find the largest component (excluding background)
                largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                
                # Create mask for largest component
                mask = (labels == largest_component).astype(np.uint8) * 255
                
                # Apply mask to get clean image
                img_clean = cv2.bitwise_and(255 - img_thresh, mask)
                img_final = 255 - img_clean
            else:
                img_final = img_thresh
            
            # 7. Final smoothing
            img_final = cv2.GaussianBlur(img_final, (3, 3), 0)
            
            # 8. Normalize to [0, 1]
            img_normalized = img_final.astype(np.float32) / 255.0
            
            return img_normalized
            
        except Exception as e:
            print(f"Error in enhanced preprocessing: {e}")
            # Fallback to simple preprocessing
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            return img_resized.astype(np.float32) / 255.0
    
    def simple_preprocess_image(self, img, target_size=None):
        """Simple preprocessing for clear images"""
        if target_size is None:
            target_size = self.img_size
            
        try:
            # Simple preprocessing for clear images
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Optional: Ensure consistent contrast
            if np.mean(img_normalized) < 0.3:
                img_normalized = np.clip(img_normalized * 1.2, 0, 1)
            
            return img_normalized
        except Exception as e:
            print(f"Error in simple preprocessing: {e}")
            return None
    
    def preprocess_image(self, image_path, force_enhanced=False, show_steps=False):
        """Adaptive preprocessing with visualization"""
        try:
            # Load image
            if isinstance(image_path, str):
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"Could not load image: {image_path}")
            else:
                img = image_path
            
            # Assess image quality
            needs_enhancement, quality_metrics = self.assess_image_quality(img)
            
            # Choose preprocessing method
            if force_enhanced or needs_enhancement:
                preprocessing_type = "Enhanced"
                img_processed = self.enhanced_preprocess_image(img)
            else:
                preprocessing_type = "Simple"
                img_processed = self.simple_preprocess_image(img)
            
            # Show preprocessing steps if requested
            if show_steps and img_processed is not None:
                self.visualize_preprocessing_steps(img, img_processed, preprocessing_type, quality_metrics)
            
            return img_processed, preprocessing_type, quality_metrics
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None, None, None
    
    def visualize_preprocessing_steps(self, original_img, processed_img, preprocessing_type, quality_metrics):
        """Visualize the preprocessing steps"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Preprocessing Pipeline - {preprocessing_type}', fontsize=16)
            
            # Original image
            axes[0, 0].imshow(original_img, cmap='gray')
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Resized image
            img_resized = cv2.resize(original_img, self.img_size, interpolation=cv2.INTER_AREA)
            axes[0, 1].imshow(img_resized, cmap='gray')
            axes[0, 1].set_title('Resized')
            axes[0, 1].axis('off')
            
            # Final processed image
            axes[0, 2].imshow(processed_img, cmap='gray')
            axes[0, 2].set_title(f'{preprocessing_type} Processed')
            axes[0, 2].axis('off')
            
            # Quality metrics
            metrics_text = f"Preprocessing: {preprocessing_type}\n\n"
            for key, value in quality_metrics.items():
                metrics_text += f"{key.replace('_', ' ').title()}: {value:.3f}\n"
            
            axes[1, 0].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
            axes[1, 0].set_title('Quality Metrics')
            axes[1, 0].axis('off')
            
            # Histogram comparison
            axes[1, 1].hist(original_img.flatten(), bins=50, alpha=0.7, label='Original', color='blue')
            axes[1, 1].hist((processed_img * 255).flatten(), bins=50, alpha=0.7, label='Processed', color='red')
            axes[1, 1].set_title('Intensity Histograms')
            axes[1, 1].legend()
            axes[1, 1].set_xlabel('Pixel Intensity')
            axes[1, 1].set_ylabel('Frequency')
            
            # Edge detection comparison
            edges_original = cv2.Canny(original_img, 50, 150)
            edges_processed = cv2.Canny((processed_img * 255).astype(np.uint8), 50, 150)
            
            # Combine edge images for comparison
            edge_comparison = np.zeros((edges_original.shape[0], edges_original.shape[1], 3))
            edge_comparison[:, :, 0] = edges_original / 255.0  # Red for original
            edge_comparison[:, :, 1] = edges_processed / 255.0  # Green for processed
            
            axes[1, 2].imshow(edge_comparison)
            axes[1, 2].set_title('Edge Comparison\n(Red: Original, Green: Processed)')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in visualization: {e}")
    
    def predict_image(self, image_path, show_preprocessing=True, show_predictions=True, force_enhanced=False):
        """Predict Roman numeral from image with full visualization"""
        print(f"\n{'='*60}")
        print(f"TESTING IMAGE: {os.path.basename(image_path) if isinstance(image_path, str) else 'Array Input'}")
        print(f"{'='*60}")
        
        if self.model is None:
            print("❌ Model not loaded properly")
            return None
        
        # Preprocess image
        img_processed, preprocessing_type, quality_metrics = self.preprocess_image(
            image_path, force_enhanced=force_enhanced, show_steps=show_preprocessing
        )
        
        if img_processed is None:
            print("❌ Failed to preprocess image")
            return None
        
        print(f"✓ Preprocessing: {preprocessing_type}")
        
        # Make prediction
        try:
            img_batch = np.expand_dims(np.expand_dims(img_processed, axis=-1), axis=0)
            predictions = self.model.predict(img_batch, verbose=0)
            
            # Get prediction results
            pred_idx = np.argmax(predictions[0])
            confidence = predictions[0][pred_idx]
            predicted_class = self.class_labels[pred_idx]
            
            # Print results
            print(f"\n🎯 PREDICTION RESULTS:")
            print(f"Predicted: {predicted_class.upper()} (Confidence: {confidence*100:.1f}%)")
            
            # Show top 3 predictions
            top_indices = np.argsort(predictions[0])[::-1][:3]
            print(f"\nTop 3 Predictions:")
            for i, idx in enumerate(top_indices):
                print(f"  {i+1}. {self.class_labels[idx].upper()}: {predictions[0][idx]*100:.1f}%")
            
            # Visualize predictions
            if show_predictions:
                self.visualize_predictions(img_processed, predictions[0], predicted_class, confidence)
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': predictions[0],
                'preprocessing_type': preprocessing_type,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None
    
    def visualize_predictions(self, processed_img, predictions, predicted_class, confidence):
        """Visualize prediction results"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Show processed image
            axes[0].imshow(processed_img, cmap='gray')
            axes[0].set_title(f'Processed Image\nPrediction: {predicted_class.upper()} ({confidence*100:.1f}%)')
            axes[0].axis('off')
            
            # Show prediction probabilities
            bars = axes[1].bar(self.class_labels, predictions)
            axes[1].set_title('Prediction Probabilities')
            axes[1].set_xlabel('Roman Numerals')
            axes[1].set_ylabel('Confidence')
            axes[1].tick_params(axis='x', rotation=45)
            
            # Highlight the predicted class
            max_idx = np.argmax(predictions)
            bars[max_idx].set_color('red')
            bars[max_idx].set_alpha(0.8)
            
            # Add confidence values on bars
            for i, (bar, prob) in enumerate(zip(bars, predictions)):
                if prob > 0.01:  # Only show values > 1%
                    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in prediction visualization: {e}")
    
    def compare_preprocessing_methods(self, image_path):
        """Compare simple vs enhanced preprocessing"""
        print(f"\n{'='*60}")
        print(f"COMPARING PREPROCESSING METHODS")
        print(f"Image: {os.path.basename(image_path) if isinstance(image_path, str) else 'Array Input'}")
        print(f"{'='*60}")
        
        if self.model is None:
            print("❌ Model not loaded properly")
            return None
        
        # Load original image
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"❌ Could not load image: {image_path}")
                return None
        else:
            img = image_path
        
        # Process with both methods
        img_simple = self.simple_preprocess_image(img)
        img_enhanced = self.enhanced_preprocess_image(img)
        
        if img_simple is None or img_enhanced is None:
            print("❌ Failed to preprocess images")
            return None
        
        # Make predictions with both
        try:
            img_batch_simple = np.expand_dims(np.expand_dims(img_simple, axis=-1), axis=0)
            img_batch_enhanced = np.expand_dims(np.expand_dims(img_enhanced, axis=-1), axis=0)
            
            pred_simple = self.model.predict(img_batch_simple, verbose=0)
            pred_enhanced = self.model.predict(img_batch_enhanced, verbose=0)
            
            # Get results
            simple_idx = np.argmax(pred_simple[0])
            enhanced_idx = np.argmax(pred_enhanced[0])
            
            simple_class = self.class_labels[simple_idx]
            enhanced_class = self.class_labels[enhanced_idx]
            
            simple_conf = pred_simple[0][simple_idx]
            enhanced_conf = pred_enhanced[0][enhanced_idx]
            
            # Print comparison
            print(f"\n📊 COMPARISON RESULTS:")
            print(f"Simple Preprocessing:   {simple_class.upper()} ({simple_conf*100:.1f}%)")
            print(f"Enhanced Preprocessing: {enhanced_class.upper()} ({enhanced_conf*100:.1f}%)")
            
            # Visualize comparison
            self._visualize_comparison(img, img_simple, img_enhanced, pred_simple[0], pred_enhanced[0], 
                                     simple_class, enhanced_class, simple_conf, enhanced_conf)
            
            return {
                'simple': {'class': simple_class, 'confidence': simple_conf, 'predictions': pred_simple[0]},
                'enhanced': {'class': enhanced_class, 'confidence': enhanced_conf, 'predictions': pred_enhanced[0]}
            }
            
        except Exception as e:
            print(f"❌ Error in comparison: {e}")
            return None
    
    def _visualize_comparison(self, img, img_simple, img_enhanced, pred_simple, pred_enhanced, 
                            simple_class, enhanced_class, simple_conf, enhanced_conf):
        """Helper method to visualize preprocessing comparison"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Preprocessing Method Comparison', fontsize=16)
            
            # Original image
            axes[0, 0].imshow(img, cmap='gray')
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Simple preprocessing
            axes[0, 1].imshow(img_simple, cmap='gray')
            axes[0, 1].set_title(f'Simple Processing\n{simple_class.upper()} ({simple_conf*100:.1f}%)')
            axes[0, 1].axis('off')
            
            # Enhanced preprocessing
            axes[0, 2].imshow(img_enhanced, cmap='gray')
            axes[0, 2].set_title(f'Enhanced Processing\n{enhanced_class.upper()} ({enhanced_conf*100:.1f}%)')
            axes[0, 2].axis('off')
            
            # Prediction comparisons
            axes[1, 0].bar(self.class_labels, pred_simple)
            axes[1, 0].set_title('Simple - Predictions')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].set_ylabel('Confidence')
            
            axes[1, 1].bar(self.class_labels, pred_enhanced)
            axes[1, 1].set_title('Enhanced - Predictions')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].set_ylabel('Confidence')
            
            # Difference plot
            diff = pred_enhanced - pred_simple
            colors = ['red' if d < 0 else 'green' for d in diff]
            axes[1, 2].bar(self.class_labels, diff, color=colors, alpha=0.7)
            axes[1, 2].set_title('Difference (Enhanced - Simple)')
            axes[1, 2].tick_params(axis='x', rotation=45)
            axes[1, 2].set_ylabel('Confidence Difference')
            axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in comparison visualization: {e}")
    
    def test_multiple_images(self, image_paths, show_individual=False):
        """Test multiple images and show summary"""
        results = []
        
        print(f"\n{'='*60}")
        print(f"TESTING {len(image_paths)} IMAGES")
        print(f"{'='*60}")
        
        for i, img_path in enumerate(image_paths):
            print(f"\n--- Image {i+1}/{len(image_paths)}: {os.path.basename(img_path)} ---")
            
            if not os.path.exists(img_path):
                print(f"❌ File not found: {img_path}")
                results.append(None)
                continue
            
            result = self.predict_image(
                img_path, 
                show_preprocessing=show_individual, 
                show_predictions=show_individual
            )
            results.append(result)
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY RESULTS")
        print(f"{'='*60}")
        
        for i, (img_path, result) in enumerate(zip(image_paths, results)):
            filename = os.path.basename(img_path)
            if result:
                pred = result['predicted_class'].upper()
                conf = result['confidence'] * 100
                preprocessing = result['preprocessing_type']
                status = "✓" if conf > 70 else "⚠️"
                print(f"{status} {filename}: {pred} ({conf:.1f}%) [{preprocessing}]")
            else:
                print(f"❌ {filename}: FAILED")
        
        return results
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            print("❌ No model loaded")
            return None
        
        print("\n📊 MODEL INFORMATION:")
        print(f"Input shape: {self.model.input_shape}")
        print(f"Output shape: {self.model.output_shape}")
        print(f"Number of classes: {len(self.class_labels)}")
        print(f"Class labels: {self.class_labels}")
        print(f"Total parameters: {self.model.count_params()}")
        
        return {
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'num_classes': len(self.class_labels),
            'class_labels': self.class_labels,
            'total_params': self.model.count_params()
        }

# Example usage and testing functions
def main():
    """Main function with example usage"""
    try:
        # Check if model file exists
        model_path = "simple_roman_model.keras"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("Please make sure you have trained the model first and the file exists.")
            return
        
        # Initialize tester
        tester = RomanNumeralTester(model_path)
        
        # Show model info
        tester.get_model_info()
        
        # Example test images (update these paths)
        test_images = [
            r"D:\projects\ai_project_oel\Dataset\test\ii\2_cap_453.png",
            r"C:\Users\bilaa\OneDrive\Pictures\5_testt.jpg",
            # Add more image paths here
        ]
        
        print("\n🚀 Roman Numeral Model Tester")
        print("=" * 50)
        
        # Test individual images
        for img_path in test_images:
            if os.path.exists(img_path):
                print(f"\n🔍 Testing: {img_path}")
                
                # Basic prediction
                result = tester.predict_image(img_path)
                
                # Compare preprocessing methods
                if result:
                    tester.compare_preprocessing_methods(img_path)
            else:
                print(f"⚠️ Image not found: {img_path}")
        
        # Test multiple images at once (if any exist)
        existing_images = [img for img in test_images if os.path.exists(img)]
        if existing_images:
            tester.test_multiple_images(existing_images)
        else:
            print("\n⚠️ No test images found. Please update the image paths in the test_images list.")
            
    except Exception as e:
        print(f"❌ Error in main function: {e}")

def test_single_image(image_path, model_path="simple_roman_model.keras"):
    """Quick function to test a single image"""
    try:
        tester = RomanNumeralTester(model_path)
        return tester.predict_image(image_path, show_preprocessing=True, show_predictions=True)
    except Exception as e:
        print(f"❌ Error testing image: {e}")
        return None

def compare_image_preprocessing(image_path, model_path="simple_roman_model.keras"):
    """Quick function to compare preprocessing methods"""
    try:
        tester = RomanNumeralTester(model_path)
        return tester.compare_preprocessing_methods(image_path)
    except Exception as e:
        print(f"❌ Error comparing preprocessing: {e}")
        return None

def quick_test(image_path, model_path="simple_roman_model.keras"):
    """Quickly test an image with minimal output"""
    try:
        tester = RomanNumeralTester(model_path)
        result = tester.predict_image(image_path, show_preprocessing=False, show_predictions=True)
        return result
    except Exception as e:
        print(f"❌ Error in quick test: {e}")
        return None

def detailed_test(image_path, model_path="simple_roman_model.keras"):
    """Test an image with full visualization"""
    try:
        tester = RomanNumeralTester(model_path)
        result = tester.predict_image(image_path, show_preprocessing=True, show_predictions=True)
        if result:
            tester.compare_preprocessing_methods(image_path)
        return result
    except Exception as e:
        print(f"❌ Error in detailed test: {e}")
        return None

if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Test Roman Numeral Classification Model')
    parser.add_argument('--image', type=str, help='Path to image file to test')
    parser.add_argument('--model', type=str, default='simple_roman_model.keras', help='Path to model file')
    parser.add_argument('--compare', action='store_true', help='Compare preprocessing methods')
    parser.add_argument('--enhanced', action='store_true', help='Force enhanced preprocessing')
    
    args = parser.parse_args()
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image file not found: {args.image}")
        else:
            try:
                tester = RomanNumeralTester(args.model)
                
                if args.compare:
                    tester.compare_preprocessing_methods(args.image)
                else:
                    tester.predict_image(args.image, force_enhanced=args.enhanced)
            except Exception as e:
                print(f"❌ Error: {e}")
    else:
        main()

print("\n" + "="*60)
print("ROMAN NUMERAL MODEL TESTER - READY!")
print("="*60)
print("Available functions:")
print("1. test_single_image('path/to/image.jpg')")
print("2. compare_image_preprocessing('path/to/image.jpg')")
print("3. quick_test('path/to/image.jpg')")
print("4. detailed_test('path/to/image.jpg')")
print("\nExample usage:")
print("detailed_test(r'C:\\Users\\bilaa\\OneDrive\\Pictures\\5_test.jpg')")
print("="*60)
