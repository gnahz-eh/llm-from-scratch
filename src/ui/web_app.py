"""
Web UI for LLM Training Progress Visualization
"""
import json
import os
import threading
import time
from flask import Flask, render_template, jsonify
from queue import Queue
import webbrowser

class ProgressTracker:
    """Tracks training progress and section completion status"""
    
    def __init__(self):
        self.progress_data = {
            'sections': {
                '1': {'name': 'Device Setup & Initialization', 'status': 'pending', 'progress': 0},
                '2': {'name': 'Tokenizer Initialization', 'status': 'pending', 'progress': 0},
                '3': {'name': 'Initial Model Testing (Untrained)', 'status': 'pending', 'progress': 0},
                '4': {'name': 'Model Inference Testing', 'status': 'pending', 'progress': 0},
                '5': {'name': 'Data Preparation', 'status': 'pending', 'progress': 0},
                '6': {'name': 'Data Loaders Creation', 'status': 'pending', 'progress': 0},
                '7': {'name': 'Loss Calculation (Untrained)', 'status': 'pending', 'progress': 0},
                '8': {'name': 'Model Training', 'status': 'pending', 'progress': 0},
                '9': {'name': 'Advanced Generation Testing', 'status': 'pending', 'progress': 0},
                '10': {'name': 'Dependency Version Check', 'status': 'pending', 'progress': 0},
                '11': {'name': 'Loading Pre-trained GPT-2 Weights', 'status': 'pending', 'progress': 0},
                '12': {'name': 'Create Pre-trained Model', 'status': 'pending', 'progress': 0},
                '13': {'name': 'Pre-trained Model Testing', 'status': 'pending', 'progress': 0}
            },
            'training': {
                'epoch': 0,
                'total_epochs': 10,
                'train_loss': [],
                'val_loss': [],
                'tokens_seen': []
            },
            'generation': {
                'input_text': '',
                'output_text': '',
                'temperature': 0,
                'top_k': 0
            },
            'logs': [],
            'model_results': {
                'untrained_output': '',
                'trained_output': '',
                'advanced_output': '',
                'pretrained_output': ''
            },
            'inference_test': {
                'target_text': '',
                'predicted_text': '',
                'accuracy': 0
            },
            'data_stats': {
                'total_characters': 0,
                'total_tokens': 0,
                'train_tokens': 0,
                'val_tokens': 0
            },
            'loss_stats': {
                'initial_train_loss': 0,
                'initial_val_loss': 0,
                'final_train_loss': 0,
                'final_val_loss': 0
            }
        }
        self.lock = threading.Lock()
    
    def update_section_status(self, section_num, status, progress=100):
        """Update section completion status"""
        with self.lock:
            if str(section_num) in self.progress_data['sections']:
                self.progress_data['sections'][str(section_num)]['status'] = status
                self.progress_data['sections'][str(section_num)]['progress'] = progress
    
    def update_training_progress(self, epoch, train_loss, val_loss, tokens_seen):
        """Update training progress"""
        with self.lock:
            self.progress_data['training']['epoch'] = epoch
            self.progress_data['training']['train_loss'].append(train_loss)
            self.progress_data['training']['val_loss'].append(val_loss)
            self.progress_data['training']['tokens_seen'].append(tokens_seen)
            
            # Store detailed epoch information
            if 'epoch_details' not in self.progress_data['training']:
                self.progress_data['training']['epoch_details'] = []
            
            epoch_info = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'tokens_seen': tokens_seen,
                'timestamp': time.strftime('%H:%M:%S')
            }
            self.progress_data['training']['epoch_details'].append(epoch_info)
    
    def update_generation_result(self, input_text, output_text, temperature, top_k):
        """Update generation results"""
        with self.lock:
            # Store the latest generation result
            new_result = {
                'input_text': input_text,
                'output_text': output_text,
                'temperature': temperature,
                'top_k': top_k,
                'timestamp': time.strftime('%H:%M:%S')
            }
            
            # Keep a history of generation results
            if 'generation_history' not in self.progress_data:
                self.progress_data['generation_history'] = []
            
            self.progress_data['generation_history'].append(new_result)
            
            # Keep only last 5 generation results
            if len(self.progress_data['generation_history']) > 5:
                self.progress_data['generation_history'] = self.progress_data['generation_history'][-5:]
            
            # Update current generation data
            self.progress_data['generation'] = new_result
    
    def add_log(self, message, level='info'):
        """Add log message"""
        with self.lock:
            timestamp = time.strftime('%H:%M:%S')
            self.progress_data['logs'].append({
                'timestamp': timestamp,
                'message': message,
                'level': level
            })
            # Keep only last 50 logs
            if len(self.progress_data['logs']) > 50:
                self.progress_data['logs'] = self.progress_data['logs'][-50:]
    
    def update_model_result(self, result_type, output):
        """Update model generation results for comparison"""
        with self.lock:
            if result_type in ['untrained', 'trained', 'advanced', 'pretrained']:
                self.progress_data['model_results'][f'{result_type}_output'] = output
    
    def update_inference_test(self, target_text, predicted_text, accuracy):
        """Update inference test results"""
        with self.lock:
            self.progress_data['inference_test'] = {
                'target_text': target_text,
                'predicted_text': predicted_text,
                'accuracy': accuracy
            }
    
    def update_data_stats(self, total_characters, total_tokens, train_tokens, val_tokens):
        """Update data statistics"""
        with self.lock:
            self.progress_data['data_stats'] = {
                'total_characters': total_characters,
                'total_tokens': total_tokens,
                'train_tokens': train_tokens,
                'val_tokens': val_tokens
            }
    
    def update_loss_stats(self, loss_type, train_loss, val_loss):
        """Update loss statistics"""
        with self.lock:
            if loss_type == 'initial':
                self.progress_data['loss_stats']['initial_train_loss'] = train_loss
                self.progress_data['loss_stats']['initial_val_loss'] = val_loss
            elif loss_type == 'final':
                self.progress_data['loss_stats']['final_train_loss'] = train_loss
                self.progress_data['loss_stats']['final_val_loss'] = val_loss
    
    def get_progress_data(self):
        """Get current progress data"""
        with self.lock:
            return json.loads(json.dumps(self.progress_data))

# Global progress tracker instance
progress_tracker = ProgressTracker()

def create_app():
    """Create Flask application"""
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                static_folder=os.path.join(os.path.dirname(__file__), 'static'))
    
    @app.route('/')
    def index():
        """Main dashboard page"""
        return render_template('dashboard.html')
    
    @app.route('/api/progress')
    def get_progress():
        """API endpoint to get current progress"""
        return jsonify(progress_tracker.get_progress_data())
    
    return app

def start_web_server(port=5000):
    """Start the web server in a separate thread"""
    app = create_app()
    
    def run_server():
        app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait a moment for server to start, then open browser
    time.sleep(1)
    webbrowser.open(f'http://127.0.0.1:{port}')
    
    return server_thread

def log_section_start(section_num, section_name):
    """Log section start"""
    progress_tracker.update_section_status(section_num, 'running', 0)
    progress_tracker.add_log(f"Started section {section_num}: {section_name}")

def log_section_complete(section_num, section_name):
    """Log section completion"""
    progress_tracker.update_section_status(section_num, 'completed', 100)
    progress_tracker.add_log(f"Completed section {section_num}: {section_name}", 'success')

def log_training_epoch(epoch, train_loss, val_loss, tokens_seen):
    """Log training epoch progress"""
    progress_tracker.update_training_progress(epoch, train_loss, val_loss, tokens_seen)
    progress_tracker.add_log(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

def log_generation_result(input_text, output_text, temperature, top_k):
    """Log text generation result"""
    progress_tracker.update_generation_result(input_text, output_text, temperature, top_k)
    progress_tracker.add_log(f"Generated text with temp={temperature}, top_k={top_k}", 'success')

def log_message(message, level='info'):
    """Log general message"""
    progress_tracker.add_log(message, level)

def log_model_result(result_type, output):
    """Log model generation result for comparison"""
    progress_tracker.update_model_result(result_type, output)
    if result_type == 'untrained':
        progress_tracker.add_log(f"🤖 UNTRAINED model output captured", 'warning')
    elif result_type == 'trained':
        progress_tracker.add_log(f"📈 TRAINED model output captured", 'info')
    elif result_type == 'advanced':
        progress_tracker.add_log(f"🎯 ADVANCED generation output captured", 'info')
    elif result_type == 'pretrained':
        progress_tracker.add_log(f"🏆 PRETRAINED model output captured", 'success')

def log_inference_test(target_text, predicted_text, accuracy):
    """Log inference test results"""
    progress_tracker.update_inference_test(target_text, predicted_text, accuracy)
    progress_tracker.add_log(f"🎯 Inference test: {accuracy:.1f}% accuracy", 'info')

def log_data_stats(total_characters, total_tokens, train_tokens, val_tokens):
    """Log data preparation statistics"""
    progress_tracker.update_data_stats(total_characters, total_tokens, train_tokens, val_tokens)
    progress_tracker.add_log(f"📊 Data prepared: {total_characters:,} chars, {total_tokens:,} tokens", 'info')

def log_loss_stats(loss_type, train_loss, val_loss):
    """Log loss statistics"""
    progress_tracker.update_loss_stats(loss_type, train_loss, val_loss)
    if loss_type == 'initial':
        progress_tracker.add_log(f"📉 Initial losses - Train: {train_loss:.4f}, Val: {val_loss:.4f}", 'warning')
    elif loss_type == 'final':
        progress_tracker.add_log(f"📈 Final losses - Train: {train_loss:.4f}, Val: {val_loss:.4f}", 'success')

def log_pretrained_loading(progress_msg):
    """Log pretrained model loading progress"""
    progress_tracker.add_log(f"⬇️ {progress_msg}", 'info')