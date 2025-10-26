# LLM Training Dashboard

This project now includes a web-based dashboard to visualize your LLM training progress in real-time.

## Features

- **Real-time Progress Tracking**: Monitor each section of the training pipeline as it executes
- **Training Visualization**: Live charts showing training and validation loss over epochs
- **Generation Results**: Display text generation results with temperature and top-k settings
- **Live Logs**: Real-time log stream with different message types (info, success, warning, error)
- **Statistics Dashboard**: Current epoch, total tokens processed, and loss metrics

## Installation

First, install the required dependencies:

```bash
pip install flask transformers
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Usage

Simply run your main script as usual:

```bash
python -m src.main
```

The web dashboard will automatically:
1. Start a Flask web server on port 5000
2. Open your default browser to http://127.0.0.1:5000
3. Display real-time progress as the script executes

## Dashboard Sections

### 1. Section Progress
- Shows completion status of all 14 sections in main.py
- Color-coded indicators: 
  - **Gray**: Pending
  - **Yellow**: Currently running
  - **Green**: Completed

### 2. Training Statistics
- Current epoch progress
- Latest training loss
- Total tokens processed
- Progress bar for overall training completion

### 3. Training Loss Chart
- Real-time line chart showing training and validation loss
- Updates automatically during training epochs

### 4. Text Generation Results
- Shows input text and generated output
- Displays generation parameters (temperature, top-k)
- Updates when generation sections complete

### 5. Live Logs
- Real-time log stream with timestamps
- Color-coded by message type
- Auto-scrolls to show latest messages

## Technical Details

- **Backend**: Flask web server running on port 5000
- **Frontend**: Vanilla HTML/CSS/JavaScript with Chart.js
- **Real-time Updates**: Polling every 1 second via REST API
- **Thread Safety**: Progress tracking uses thread locks for concurrent access
- **Graceful Degradation**: Script runs normally even if web UI dependencies are missing

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, modify the port in `src/ui/web_app.py`:

```python
server_thread = start_web_server(port=8080)  # Change to available port
```

### Dependencies Missing
If Flask is not installed, the script will run normally without the web UI. Install Flask to enable the dashboard:

```bash
pip install flask
```

### Browser Doesn't Open
If the browser doesn't open automatically, manually navigate to:
http://127.0.0.1:5000

## File Structure

```
src/
├── ui/
│   ├── __init__.py
│   ├── web_app.py              # Flask application and progress tracking
│   └── templates/
│       └── dashboard.html      # Web dashboard interface
├── main.py                     # Modified with progress tracking
└── utils/
    └── train.py               # Modified with UI integration
```

## Customization

You can customize the dashboard by:
- Modifying `dashboard.html` for UI changes
- Adjusting update frequency in the JavaScript (currently 1 second)
- Adding new progress tracking points in your code using the log functions
- Changing chart colors and styles in the CSS

The dashboard provides a comprehensive view of your LLM training process, making it easier to monitor progress and identify any issues during training.