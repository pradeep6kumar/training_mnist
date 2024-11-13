import threading
import queue
from flask import jsonify

class TrainingManager:
    def __init__(self):
        self.training_thread = None
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        
    def start_training(self):
        if self.training_thread and self.training_thread.is_alive():
            return
        
        self.stop_event.clear()
        self.training_thread = threading.Thread(target=self.training_worker)
        self.training_thread.daemon = True
        self.training_thread.start()
        
    def stop_training(self):
        if self.training_thread and self.training_thread.is_alive():
            self.stop_event.set()
            self.training_thread.join()
        
        # Clear any remaining items in queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
    
    def training_worker(self):
        # Import train_model function here to avoid circular imports
        from train import train_model
        
        # Call train_model with stop_event
        train_model(epochs=10, device='cuda', train_loader=train_loader, 
                    model=model, criterion=criterion, optimizer=optimizer, 
                    stop_event=self.stop_event)
    
    def purge_data(self):
        # Stop any ongoing training first
        self.stop_training()
        # Then proceed with purge operations
        # ... your existing purge code ...

@app.route('/purge', methods=['POST'])
def purge():
    try:
        training_manager.purge_data()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/train', methods=['POST'])
def train():
    try:
        training_manager.start_training()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Create a global training manager instance
training_manager = TrainingManager() 