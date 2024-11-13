from flask import Flask, render_template, send_from_directory, jsonify, request
import os
import logging
import sys
import time
import threading
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io
import base64
import json

app = Flask(__name__)

# Global variables
last_modified_time = 0
training_complete = False
training_thread = None
model = None
test_dataset = None

def get_random_predictions():
    global model, test_dataset
    if model is None or test_dataset is None:
        return []
    
    results = []
    indices = np.random.choice(len(test_dataset), 10, replace=False)
    
    model.eval()
    with torch.no_grad():
        for idx in indices:
            image, true_label = test_dataset[idx]
            
            # Get prediction
            output = model(image.unsqueeze(0).cuda())
            pred_label = output.argmax(dim=1).item()
            
            # Convert image to base64 for display
            img_array = (image.squeeze().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            results.append({
                'image': img_str,
                'prediction': pred_label,
                'true_label': true_label
            })
    
    return results

def run_training(config):
    global training_complete, model, test_dataset
    print("Starting model training with config:", config)
    try:
        import train
        # Create a metrics file with initial values
        initial_metrics = {
            'epoch': 0,
            'model1_loss': 0.0,
            'model1_accuracy': 0.0,
            'model2_loss': 0.0,
            'model2_accuracy': 0.0
        }
        with open(os.path.join('static', 'current_metrics.json'), 'w') as f:
            json.dump(initial_metrics, f)

        # Run the training with config
        train.train_model(
            [16, 32], [32, 64],
            optimizer=config['optimizer'],
            batch_size=int(config['batchSize']),
            num_epochs=int(config['epochs'])
        )

        # After training, load the model and test dataset
        from train import CNN
        model = CNN().cuda()
        model.load_state_dict(torch.load('mnist_cnn.pth'))
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
                
        print("Training completed successfully!")
        training_complete = True
    except Exception as e:
        print(f"Error during training: {str(e)}")
        logging.error(f"Training error: {str(e)}", exc_info=True)
        training_complete = True

@app.route('/get_predictions')
def get_predictions():
    predictions_path = os.path.join('static', 'test_predictions.json')
    if os.path.exists(predictions_path):
        try:
            with open(predictions_path, 'r') as f:
                predictions = json.load(f)
                results = []
                for pred in predictions:
                    image = np.array(pred['image']).squeeze()
                    img = Image.fromarray((image * 255).astype(np.uint8))
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    results.append({
                        'image': img_str,
                        'true_label': pred['true_label'],
                        'model1_prediction': pred['model1_prediction'],
                        'model2_prediction': pred['model2_prediction']
                    })
                return jsonify({'predictions': results})
        except Exception as e:
            print(f"Error loading predictions: {e}")
    return jsonify({'predictions': []})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def send_static(filename):
    return send_from_directory('static', filename)

@app.route('/check_update')
def check_update():
    plot_path = os.path.join('static', 'training_plot.png')
    metrics_path = os.path.join('static', 'current_metrics.json')
    final_metrics_path = os.path.join('static', 'final_metrics.json')
    
    # Initialize metrics with numeric values
    current_metrics = {
        'epoch': 0,
        'model1_loss': 0.0,
        'model1_accuracy': 0.0,
        'model2_loss': 0.0,
        'model2_accuracy': 0.0
    }
    
    try:
        # If training is complete, try to load final metrics
        if training_complete and os.path.exists(final_metrics_path):
            try:
                with open(final_metrics_path, 'r') as f:
                    content = f.read()
                    if content.strip():  # Check if file is not empty
                        current_metrics = json.loads(content)
                        print(f"Loaded final metrics: {current_metrics}")
            except Exception as e:
                print(f"Error reading final metrics: {str(e)}")
                # If final metrics fail, try current metrics as fallback
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        content = f.read()
                        if content.strip():
                            current_metrics = json.loads(content)
        # If training is still ongoing or final metrics don't exist, load current metrics
        elif os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                content = f.read()
                if content.strip():
                    loaded_metrics = json.loads(content)
                    # Handle epoch separately from other metrics
                    current_metrics['epoch'] = int(loaded_metrics.get('epoch', 0))
                    # Handle other numeric metrics
                    for key in ['model1_loss', 'model1_accuracy', 'model2_loss', 'model2_accuracy']:
                        try:
                            value = loaded_metrics.get(key, 0.0)
                            current_metrics[key] = float(value)
                        except (TypeError, ValueError):
                            current_metrics[key] = 0.0
    except Exception as e:
        print(f"Error handling metrics: {str(e)}")
        logging.error(f"Metrics handling error: {str(e)}", exc_info=True)

    plot_updated = False
    if os.path.exists(plot_path):
        global last_modified_time
        current_modified_time = os.path.getmtime(plot_path)
        plot_updated = current_modified_time > last_modified_time
        if plot_updated:
            last_modified_time = current_modified_time

    response_data = {
        'updated': plot_updated,
        'timestamp': time.time(),
        'training_complete': training_complete,
        'metrics': current_metrics
    }
    print(f"Sending response: {response_data}")
    return jsonify(response_data)

@app.route('/start_training', methods=['POST'])
def start_training():
    global training_thread, training_complete, last_modified_time
    
    # Handle purge request
    if request.json.get('purge'):
        # Reset all metrics files
        initial_metrics = {
            'epoch': 0,
            'model1_loss': 0.0,
            'model1_accuracy': 0.0,
            'model2_loss': 0.0,
            'model2_accuracy': 0.0
        }
        for filename in ['current_metrics.json', 'final_metrics.json']:
            file_path = os.path.join('static', filename)
            with open(file_path, 'w') as f:
                json.dump(initial_metrics, f)
        return jsonify({'status': 'purged'})
    
    # Regular training start code...
    if training_thread and training_thread.is_alive():
        return jsonify({'status': 'error', 'message': 'Training already in progress'})
    
    config = request.json
    training_complete = False
    last_modified_time = 0
    
    training_thread = threading.Thread(
        target=run_training,
        args=(config,)
    )
    training_thread.start()
    
    return jsonify({'status': 'started'})

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Reset metrics files on startup
    initial_metrics = {
        'epoch': 0,
        'model1_loss': 0.0,
        'model1_accuracy': 0.0,
        'model2_loss': 0.0,
        'model2_accuracy': 0.0
    }
    for filename in ['current_metrics.json', 'final_metrics.json']:
        file_path = os.path.join('static', filename)
        with open(file_path, 'w') as f:
            json.dump(initial_metrics, f)
    
    # Start the Flask app
    app.run(debug=True, use_reloader=False)