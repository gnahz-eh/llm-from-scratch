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
                '1': {'name': 'Library Dependencies Check', 'status': 'pending', 'progress': 0},
                '2': {'name': 'Model Configuration', 'status': 'pending', 'progress': 0},
                '3': {'name': 'Data Preparation', 'status': 'pending', 'progress': 0},
                '4': {'name': 'Model Creation', 'status': 'pending', 'progress': 0},
                '5': {'name': 'Model Inference Testing', 'status': 'pending', 'progress': 0},
                '6': {'name': 'Loss Calculation', 'status': 'pending', 'progress': 0},
                '7': {'name': 'Version Information', 'status': 'pending', 'progress': 0},
                '8': {'name': 'GPT-2 Model Loading', 'status': 'pending', 'progress': 0},
                '9': {'name': 'Training Code', 'status': 'pending', 'progress': 0},
                '10': {'name': 'Generation with Temperature Scaling', 'status': 'pending', 'progress': 0},
                '11': {'name': 'Dependency Versions', 'status': 'pending', 'progress': 0},
                '12': {'name': 'Loading Pre-trained GPT-2 Weights', 'status': 'pending', 'progress': 0},
                '13': {'name': 'Create and Load Pre-trained Model', 'status': 'pending', 'progress': 0},
                '14': {'name': 'Text Generation with Pre-trained Model', 'status': 'pending', 'progress': 0}
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
            'logs': []
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