import torch
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def train_model(epochs, device, train_loader, model, criterion, optimizer, stop_event):
    model.train()
    for epoch in range(epochs):
        if stop_event.is_set():
            print("Training stopped.")
            break
        
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            if stop_event.is_set():
                print("Training stopped.")
                break
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:    # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        # Generate and save plot after each epoch
        plot_training_progress(epoch + 1, loss.item())
    
    print('Finished Training')

def plot_training_progress(epoch, loss):
    plt.figure(figsize=(10, 5))
    plt.title(f"Training Progress - Epoch {epoch}")
    plt.plot(range(epoch), [loss] * epoch, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert plot to base64 string
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Save base64 string to a file or send it to the frontend
    with open('static/training_plot.txt', 'w') as f:
        f.write(plot_base64)
    
    plt.close() 