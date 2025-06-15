from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import base64
import io
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import sqlite3
from pathlib import Path
import re
from werkzeug.utils import secure_filename

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class RomanNumeralClassifier:
    def __init__(self, model_path="simple_roman_model.keras"):
        self.img_size = (64, 64)
        self.model = None
        self.class_labels = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']
        self.roman_to_number = {
            'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'ix': 9, 'v': 5,
            'vi': 6, 'vii': 7, 'viii': 8, 'x': 10
        }
        self.number_to_roman = {v: k for k, v in self.roman_to_number.items()}
        self.load_model(model_path)
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for tracking predictions"""
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                actual_label TEXT,
                predicted_label TEXT,
                confidence REAL,
                is_accurate BOOLEAN,
                preprocessing_method TEXT,
                timestamp DATETIME,
                processing_time REAL,
                quality_metrics TEXT
            )
        ''')
        conn.commit()
        conn.close()
        
    def load_model(self, model_path):
        """Load the trained model with validation"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = load_model(model_path)
            
            # Validate model structure
            input_shape = self.model.input_shape
            output_shape = self.model.output_shape
            
            print(f"✓ Model loaded successfully from {model_path}")
            print(f"✓ Model input shape: {input_shape}")
            print(f"✓ Model output shape: {output_shape}")
            print(f"✓ Expected classes: {len(self.class_labels)}")
            
            # Test prediction with dummy data to catch issues early
            try:
                dummy_input = np.random.random((1, 64, 64, 1)).astype(np.float32)
                test_pred = self.model.predict(dummy_input, verbose=0)
                print(f"✓ Model test prediction successful: {test_pred.shape}")
            except Exception as test_error:
                print(f"⚠️  Model test prediction failed: {test_error}")
                raise
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
            
    def extract_actual_label_from_filename(self, filename):
        """Extract actual roman numeral from filename (first digit)"""
        try:
            # Extract first digit from filename
            match = re.search(r'(\d+)', filename)
            if match:
                digit = int(match.group(1))
                if 1 <= digit <= 10:
                    return self.number_to_roman[digit]
            return None
        except:
            return None
    
    def no_preprocessing(self, img, target_size=None, debug=False):
        """
        MINIMAL preprocessing - only essential steps for model compatibility
        - Resize to expected model input size
        - Convert to float32
        - NO normalization or other processing
        """
        if target_size is None:
            target_size = self.img_size
            
        if debug:
            print("MINIMAL PREPROCESSING MODE - Only resize + dtype conversion")
            print(f"Original image shape: {img.shape}")
            print(f"Target size: {target_size}")
        
        try:
            # MUST resize to model's expected input size
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # Only convert to float32 for model compatibility
            img_float = img_resized.astype(np.float32)
            
            if debug:
                print(f"After resize: {img_resized.shape}")
                print(f"After dtype conversion: {img_float.dtype}")
                print(f"Range: {np.min(img_float)} - {np.max(img_float)}")
            
            return img_float
            
        except Exception as e:
            print(f"Error in minimal preprocessing: {e}")
            return None
    
    def simple_preprocess_image(self, img, target_size=None, debug=False):
        """
        EXACT SAME preprocessing as used in training - this is critical for consistency!
        Minimal preprocessing - keep it simple and match training exactly!
        """
        if target_size is None:
            target_size = self.img_size
            
        try:
            if debug:
                print("PREPROCESSING MODE - Applying training-consistent preprocessing")
                print(f"Original image shape: {img.shape}")
                print(f"Target size: {target_size}")
            
            # 1. Resize to target size (same interpolation as training)
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # 2. Normalize to [0, 1] (same as training)
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # 3. Optional: Ensure consistent contrast (EXACT same logic as training)
            # If image is too dark, brighten it slightly
            if np.mean(img_normalized) < 0.3:
                img_normalized = np.clip(img_normalized * 1.2, 0, 1)
            
            if debug:
                print(f"Image shape after resize: {img_resized.shape}")
                print(f"Image mean after normalization: {np.mean(img_normalized):.3f}")
                print(f"Image std after normalization: {np.std(img_normalized):.3f}")
                print(f"Image min/max: {np.min(img_normalized):.3f}/{np.max(img_normalized):.3f}")
            
            return img_normalized
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None
    def convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    def assess_image_quality(self, img):
        """Basic image quality assessment for reporting purposes"""
        try:
            img_float = img.astype(np.float32) / 255.0
            mean_intensity = np.mean(img_float)
            std_intensity = np.std(img_float)
            
            # Basic edge detection for quality metrics
            edges = cv2.Canny(img, 50, 150)
            edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
            
            # Blur detection using Laplacian variance
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            contrast = std_intensity / (mean_intensity + 1e-8)
            
            # Convert all numpy types to Python native types
            quality_metrics = {
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'edge_density': float(edge_density),
                'laplacian_var': float(laplacian_var),
                'contrast': float(contrast),
                'brightness_adjusted': bool(mean_intensity < 0.3)  # Explicitly convert to Python bool
            }
            
            # Ensure all values are JSON serializable
            return self.convert_to_json_serializable(quality_metrics)
            
        except Exception as e:
            print(f"Error in quality assessment: {e}")
            return {}

    def predict_image(self, img, filename="", debug=False, use_preprocessing=False):
        """Make prediction on image with optional preprocessing"""
        start_time = datetime.now()
        
        try:
            # Validate input image
            if img is None:
                return {'error': 'Invalid image input'}
                
            if debug:
                print(f"Input image shape: {img.shape}")
                print(f"Input image dtype: {img.dtype}")
            
            # Get quality metrics for reporting
            quality_metrics = self.assess_image_quality(img)
            
            # Choose preprocessing method based on parameter
            if use_preprocessing:
                img_processed = self.simple_preprocess_image(img, debug=debug)
                preprocessing_method = "Training-Consistent Preprocessing"
            else:
                img_processed = self.no_preprocessing(img, debug=debug)
                preprocessing_method = "Minimal Preprocessing (Resize + Float32)"
                
            if img_processed is None:
                return {'error': 'Failed to process image'}
            
            # CONSISTENT batch preparation regardless of preprocessing method
            # Ensure we have the right shape: (batch_size, height, width, channels)
            if len(img_processed.shape) == 2:  # Grayscale (H, W)
                img_batch = np.expand_dims(np.expand_dims(img_processed, axis=-1), axis=0)  # (1, H, W, 1)
            elif len(img_processed.shape) == 3:  # Already has channel (H, W, C)
                img_batch = np.expand_dims(img_processed, axis=0)  # (1, H, W, C)
            else:
                return {'error': f'Unexpected image shape: {img_processed.shape}'}
                
            if debug:
                print(f"Processed image shape: {img_processed.shape}")
                print(f"Batch shape for model: {img_batch.shape}")
                print(f"Batch dtype: {img_batch.dtype}")
                print(f"Batch range: {np.min(img_batch)} - {np.max(img_batch)}")
            
            # Validate batch shape matches model expectations
            expected_shape = self.model.input_shape
            if img_batch.shape[1:] != expected_shape[1:]:  # Skip batch dimension
                return {'error': f'Shape mismatch: got {img_batch.shape[1:]}, expected {expected_shape[1:]}'}
            
            # Make prediction with error handling
            try:
                predictions = self.model.predict(img_batch, verbose=0)
            except Exception as pred_error:
                return {'error': f'Model prediction failed: {str(pred_error)}'}
            
            # Validate predictions
            if predictions is None or len(predictions) == 0:
                return {'error': 'Model returned no predictions'}
                
            pred_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][pred_idx])
            predicted_class = self.class_labels[pred_idx]
            
            # Extract actual label from filename
            actual_label = self.extract_actual_label_from_filename(filename)
            is_accurate = actual_label == predicted_class if actual_label else None
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Store in database
            self.store_prediction(
                filename, actual_label, predicted_class, confidence,
                is_accurate, preprocessing_method, processing_time, quality_metrics
            )
            
            # Generate visualizations with error handling
            try:
                original_b64 = self.image_to_base64(img)
                
                # For processed image visualization
                if use_preprocessing:
                    processed_display = (img_processed * 255).astype(np.uint8)
                else:
                    # For minimal preprocessing, img_processed is already in 0-255 range
                    processed_display = np.clip(img_processed, 0, 255).astype(np.uint8)
                    
                processed_b64 = self.image_to_base64(processed_display)
                prediction_chart_b64 = self.generate_prediction_chart(predictions[0])
            except Exception as viz_error:
                print(f"Visualization error: {viz_error}")
                original_b64 = processed_b64 = prediction_chart_b64 = ""
            
            return {
                'predicted_class': predicted_class.upper(),
                'confidence': float(confidence),  # Ensure Python float
                'actual_label': actual_label.upper() if actual_label else None,
                'is_accurate': bool(is_accurate) if is_accurate is not None else None,  # Ensure Python bool
                'preprocessing_method': preprocessing_method,  
                'processing_time': float(processing_time),  # Ensure Python float
                'quality_metrics': self.convert_to_json_serializable(quality_metrics),
                'all_predictions': [float(x) for x in predictions[0].tolist()],  # Ensure Python floats
                'debug_info': {
                    'input_shape': list(img.shape),  # Convert tuple to list
                    'processed_shape': list(img_processed.shape),
                    'batch_shape': list(img_batch.shape),
                    'model_input_shape': str(self.model.input_shape)
                } if debug else {},
                'visualizations': {
                    'original_image': original_b64,
                    'processed_image': processed_b64,
                    'prediction_chart': prediction_chart_b64
                },
                'detailed_predictions': self.format_detailed_predictions(predictions[0])
            }
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': f'Prediction failed: {str(e)}'}
            
    def store_prediction(self, filename, actual, predicted, confidence, is_accurate, method, proc_time, quality):
        """Store prediction in database with proper JSON serialization"""
        try:
            conn = sqlite3.connect('predictions.db')
            cursor = conn.cursor()
            
            # Ensure all data is JSON serializable
            json_safe_quality = self.convert_to_json_serializable(quality)
            json_safe_confidence = float(confidence) if confidence is not None else None
            json_safe_is_accurate = bool(is_accurate) if is_accurate is not None else None
            json_safe_proc_time = float(proc_time) if proc_time is not None else None
            
            cursor.execute('''
                INSERT INTO predictions 
                (filename, actual_label, predicted_label, confidence, is_accurate, 
                preprocessing_method, timestamp, processing_time, quality_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                filename, 
                actual, 
                predicted, 
                json_safe_confidence, 
                json_safe_is_accurate, 
                method, 
                datetime.now(), 
                json_safe_proc_time, 
                json.dumps(json_safe_quality)
            ))
            conn.commit()
            conn.close()
        
        except Exception as e:
            print(f"Database error: {e}")
            import traceback
            traceback.print_exc()
           
    def get_statistics(self):
        """Get prediction statistics"""
        try:
            conn = sqlite3.connect('predictions.db')
            cursor = conn.cursor()
            
            # Total predictions
            cursor.execute('SELECT COUNT(*) FROM predictions')
            total_predictions = cursor.fetchone()[0]
            
            # Successful predictions (where actual label was available and correct)
            cursor.execute('SELECT COUNT(*) FROM predictions WHERE is_accurate = 1')
            successful_predictions = cursor.fetchone()[0]
            
            # Accuracy rate
            cursor.execute('SELECT COUNT(*) FROM predictions WHERE actual_label IS NOT NULL')
            total_with_labels = cursor.fetchone()[0]
            
            accuracy = (successful_predictions / total_with_labels * 100) if total_with_labels > 0 else 0
            
            # High confidence predictions (>90%)
            cursor.execute('SELECT COUNT(*) FROM predictions WHERE confidence > 0.9')
            high_confidence = cursor.fetchone()[0]
            
            # Average processing time
            cursor.execute('SELECT AVG(processing_time) FROM predictions')
            avg_processing_time = cursor.fetchone()[0] or 0
            
            # Recent predictions
            cursor.execute('''
                SELECT filename, predicted_label, confidence, is_accurate, timestamp, preprocessing_method
                FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''')
            recent_predictions = cursor.fetchall()
            
            # Preprocessing method distribution
            cursor.execute('''
                SELECT preprocessing_method, COUNT(*) 
                FROM predictions 
                GROUP BY preprocessing_method
            ''')
            preprocessing_stats = cursor.fetchall()
            
            conn.close()
            
            return {
                'total_predictions': total_predictions,
                'successful_predictions': successful_predictions,
                'accuracy_rate': round(accuracy, 1),
                'high_confidence_count': high_confidence,
                'avg_processing_time': round(avg_processing_time, 3),
                'recent_predictions': recent_predictions,
                'preprocessing_stats': preprocessing_stats
            }
        except Exception as e:
            return {
                'total_predictions': 0,
                'successful_predictions': 0,  
                'accuracy_rate': 0,
                'high_confidence_count': 0,
                'avg_processing_time': 0,
                'recent_predictions': [],
                'preprocessing_stats': []
            }
            
    def image_to_base64(self, img):
        """Convert image to base64 string"""
        try:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        except:
            return ""
            
    def generate_prediction_chart(self, predictions):
        """Generate prediction probability chart"""
        try:
            plt.figure(figsize=(12, 6))
            bars = plt.bar(self.class_labels, predictions)
            
            # Highlight the highest prediction
            max_idx = np.argmax(predictions)
            bars[max_idx].set_color('#6366f1')
            bars[max_idx].set_alpha(0.8)
            
            plt.title('Prediction Probabilities', fontsize=14, fontweight='bold')
            plt.xlabel('Roman Numerals')
            plt.ylabel('Confidence')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for i, (bar, prob) in enumerate(zip(bars, predictions)):
                if prob > 0.01:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            chart_str = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{chart_str}"
        except Exception as e:
            plt.close()
            return ""
            
    def format_detailed_predictions(self, predictions):
        """Format detailed predictions for UI with JSON-safe types"""
        detailed = []
        sorted_indices = np.argsort(predictions)[::-1]  # Sort by confidence descending
        
        for i in sorted_indices:
            label = self.class_labels[i]
            prob = predictions[i]
            detailed.append({
                'label': label.upper(),
                'probability': float(prob),  # Ensure it's Python float, not numpy float
                'percentage': round(float(prob) * 100, 1),
                'roman_number': int(self.roman_to_number[label])  # Ensure it's Python int
            })
        return detailed

# Initialize classifier
classifier = RomanNumeralClassifier()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Single image prediction endpoint with improved error handling"""
    filepath = None
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Validate file type
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Unsupported file type: {file_ext}. Allowed: {", ".join(allowed_extensions)}'}), 400
        
        # Get parameters
        debug = request.form.get('debug', 'false').lower() == 'true'
        use_preprocessing = request.form.get('preprocessing', 'false').lower() == 'true'
        
        if debug:
            print(f"🔍 Debug mode enabled")
            print(f"📁 File: {file.filename}")
            print(f"🔧 Preprocessing: {use_preprocessing}")
        
        # Create secure filename with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        secure_name = secure_filename(file.filename)
        filename = f"{timestamp}_{secure_name}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save file with error handling
        try:
            file.save(filepath)
            if debug:
                print(f"✅ File saved to: {filepath}")
        except Exception as save_error:
            return jsonify({'error': f'Failed to save file: {str(save_error)}'}), 500
        
        # Verify file was saved and is readable
        if not os.path.exists(filepath):
            return jsonify({'error': 'File was not saved properly'}), 500
            
        # Load and validate image
        try:
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return jsonify({'error': 'Could not read image file. Please ensure it\'s a valid image.'}), 400
                
            if debug:
                print(f"📷 Image loaded successfully: {img.shape}")
                
        except Exception as img_error:
            return jsonify({'error': f'Failed to load image: {str(img_error)}'}), 400
            
        # Make prediction with error handling
        try:
            result = classifier.predict_image(
                img, 
                filename=secure_name,  # Use original secure filename for analysis
                debug=debug, 
                use_preprocessing=use_preprocessing
            )
            
            if debug:
                print(f"🎯 Prediction completed")
                print(f"📊 Result keys: {list(result.keys())}")
                
        except Exception as pred_error:
            return jsonify({'error': f'Prediction failed: {str(pred_error)}'}), 500
        
        # Validate result
        if 'error' in result:
            return jsonify(result), 500
            
        return jsonify(result)
        
    except Exception as e:
        error_msg = f'Unexpected error: {str(e)}'
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500
        
    finally:
        # Always clean up the uploaded file
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                if debug:
                    print(f"🗑️ Cleaned up: {filepath}")
            except Exception as cleanup_error:
                print(f"⚠️ Could not clean up file {filepath}: {cleanup_error}")

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get prediction statistics"""
    try:
        stats = classifier.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Process multiple images"""
    try:
        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'No image files provided'}), 400
            
        # Get preprocessing option for batch
        use_preprocessing = request.form.get('preprocessing', 'false').lower() == 'true'
        
        results = []
        successful = 0
        failed = 0
        
        for file in files:
            if file.filename == '':
                continue
                
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                result = classifier.predict_image(img, filename, use_preprocessing=use_preprocessing)
                if 'error' not in result:
                    successful += 1
                else:
                    failed += 1
                    
                results.append({
                    'filename': filename,
                    'result': result
                })
            else:
                failed += 1
                results.append({
                    'filename': filename,
                    'result': {'error': 'Could not load image'}
                })
            
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
        
        return jsonify({
            'batch_results': results,
            'summary': {
                'total_processed': len(results),
                'successful': successful,
                'failed': failed,
                'success_rate': round((successful / len(results) * 100), 1) if results else 0,
                'preprocessing_used': use_preprocessing
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if classifier.model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        return jsonify({
            'model_info': {
                'input_shape': str(classifier.model.input_shape),
                'output_shape': str(classifier.model.output_shape),
                'num_classes': len(classifier.class_labels),
                'class_labels': classifier.class_labels,
                'total_params': int(classifier.model.count_params())
            },
            'processing_info': {
                'image_size': classifier.img_size,
                'default_mode': 'No Preprocessing (Raw)',
                'optional_preprocessing': 'Training-Consistent Preprocessing',
                'supported_formats': ['PNG', 'JPG', 'JPEG', 'BMP', 'TIFF']
            },
            'preprocessing_options': {
                'no_preprocessing': {
                    'description': 'Raw image input - only dtype conversion to float32',
                    'steps': ['Convert to float32 for model compatibility']
                },
                'with_preprocessing': {
                    'description': 'Training-consistent preprocessing',
                    'steps': [
                        '1. Resize to 64x64 using INTER_AREA interpolation',
                        '2. Normalize pixel values to [0, 1] range',
                        '3. Apply brightness adjustment if mean < 0.3 (multiply by 1.2)'
                    ]
                }
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test_preprocessing', methods=['POST'])
def test_preprocessing():
    """Test both preprocessing modes with improved error handling"""
    filepath = None
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Create secure filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        secure_name = secure_filename(file.filename)
        filename = f"{timestamp}_{secure_name}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file.save(filepath)
        
        # Load image
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Get original image stats
        original_stats = {
            'shape': list(img.shape),  # Convert to list for JSON serialization
            'dtype': str(img.dtype),
            'mean': float(np.mean(img)),
            'std': float(np.std(img)),
            'min': int(np.min(img)),
            'max': int(np.max(img))
        }
        
        # Test no preprocessing
        try:
            no_preprocess_img = classifier.no_preprocessing(img, debug=True)
            no_preprocess_stats = {
                'shape': list(no_preprocess_img.shape),
                'dtype': str(no_preprocess_img.dtype),
                'mean': float(np.mean(no_preprocess_img)),
                'std': float(np.std(no_preprocess_img)),
                'min': float(np.min(no_preprocess_img)),
                'max': float(np.max(no_preprocess_img))
            } if no_preprocess_img is not None else None
        except Exception as e:
            no_preprocess_stats = {'error': str(e)}
            no_preprocess_img = None
        
        # Test with preprocessing
        try:
            processed_img = classifier.simple_preprocess_image(img, debug=True)
            processed_stats = {
                'shape': list(processed_img.shape),
                'dtype': str(processed_img.dtype),
                'mean': float(np.mean(processed_img)),
                'std': float(np.std(processed_img)),
                'min': float(np.min(processed_img)),
                'max': float(np.max(processed_img))
            } if processed_img is not None else None
        except Exception as e:
            processed_stats = {'error': str(e)}
            processed_img = None
        
        # Get quality assessment
        try:
            quality_metrics = classifier.assess_image_quality(img)
        except Exception as e:
            quality_metrics = {'error': str(e)}
        
        # Generate visualizations with error handling
        try:
            original_b64 = classifier.image_to_base64(img)
        except:
            original_b64 = ""
            
        try:
            no_preprocess_b64 = classifier.image_to_base64(
                np.clip(no_preprocess_img, 0, 255).astype(np.uint8)
            ) if no_preprocess_img is not None else ""
        except:
            no_preprocess_b64 = ""
            
        try:
            processed_b64 = classifier.image_to_base64(
                (processed_img * 255).astype(np.uint8)
            ) if processed_img is not None else ""
        except:
            processed_b64 = ""
        
        result = {
            'original_stats': original_stats,
            'no_preprocessing_stats': no_preprocess_stats,
            'with_preprocessing_stats': processed_stats,
            'quality_metrics': quality_metrics,
            'visualizations': {
                'original_image': original_b64,
                'no_preprocessing_image': no_preprocess_b64,
                'with_preprocessing_image': processed_b64
            },
            'comparison': {
                'no_preprocessing_changes': 'Only dtype conversion: uint8 -> float32',
                'with_preprocessing_changes': {
                    'resized': f"{original_stats['shape']} -> {processed_stats['shape']}" if processed_stats and 'error' not in processed_stats else "Failed",
                    'normalized': f"[{original_stats['min']}-{original_stats['max']}] -> [{processed_stats['min']:.3f}-{processed_stats['max']:.3f}]" if processed_stats and 'error' not in processed_stats else "Failed",
                    'brightness_adjusted': quality_metrics.get('brightness_adjusted', False) if 'error' not in quality_metrics else False
                }
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Clean up
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
@app.route('/debug/status', methods=['GET'])
def debug_status():
    """Debug endpoint to check server and model status"""
    try:
        return jsonify({
            'server_status': 'running',
            'model_loaded': classifier.model is not None,
            'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
            'upload_folder_writable': os.access(app.config['UPLOAD_FOLDER'], os.W_OK),
            'model_classes': classifier.class_labels,
            'preprocessing_methods': [
                'No Preprocessing (Default)',
                'Training-Consistent Preprocessing'
            ],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'server_status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier.model is not None,
        'default_mode': 'No Preprocessing (Raw)',
        'optional_mode': 'Training-Consistent Preprocessing',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🏛️ Roman Numeral Classifier Backend Starting...")
    print("="*60)
    print("🔧 NEW: Default mode with NO preprocessing!")
    print("📊 Processing Modes Available:")
    print("  ✓ DEFAULT: No preprocessing (raw image, only dtype conversion)")
    print("  ✓ OPTIONAL: Training-consistent preprocessing")
    print("📊 Features Available:")
    print("  ✓ Single image prediction with optional preprocessing")
    print("  ✓ Batch processing with preprocessing choice")
    print("  ✓ Accuracy tracking based on filename analysis") 
    print("  ✓ Real-time statistics and analytics")
    print("  ✓ Image quality assessment and reporting")
    print("  ✓ Preprocessing comparison testing endpoint")
    print("  ✓ Visualization generation")
    print("  ✓ SQLite database for prediction tracking")
    print("="*60)
    print("🔍 Key Changes Made:")
    print("  • DEFAULT: No preprocessing - raw image input")
    print("  • OPTIONAL: Use 'preprocessing=true' parameter for training-consistent preprocessing")
    print("  • Added comparison mode in /test_preprocessing")
    print("  • Updated statistics to track preprocessing method used")
    print("="*60)
    print("🚀 Server running on http://localhost:5000")
    print("📖 Usage:")
    print("  • POST /predict with 'preprocessing=false' (default) for raw input")
    print("  • POST /predict with 'preprocessing=true' for trained preprocessing")
    print("  • POST /test_preprocessing to compare both modes side-by-side")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)