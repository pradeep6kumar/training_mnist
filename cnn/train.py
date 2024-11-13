import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import logging
import random
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import json

class CNN(nn.Module):
    def __init__(self, kernels=[32, 64]):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, kernels[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(kernels[0], kernels[1], kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(kernels[1] * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.kernel_config = kernels

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def update_plot(losses1, accuracies1, losses2, accuracies2, current_dir, current_metrics=None):
    plt.figure(figsize=(12, 5))
    
    # Create subplot for losses
    plt.subplot(1, 2, 1)
    iterations = range(len(losses1))
    plt.plot(iterations, losses1, label='Model 1 Loss', color='blue')
    plt.plot(iterations, losses2, label='Model 2 Loss', color='red', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()

    # Create subplot for accuracies
    plt.subplot(1, 2, 2)
    plt.plot(iterations, accuracies1, label='Model 1 Accuracy', color='blue')
    plt.plot(iterations, accuracies2, label='Model 2 Accuracy', color='red', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Comparison')
    plt.legend()

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(current_dir, 'static', 'training_plot.png')
    plt.savefig(plot_path)
    plt.close()

    # Save current metrics
    if current_metrics:
        metrics_path = os.path.join(current_dir, 'static', 'current_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(current_metrics, f)

def save_metrics(epoch, model1_loss, model1_acc, model2_loss, model2_acc):
    metrics = {
        'epoch': int(epoch + 1),
        'model1_loss': float(model1_loss),
        'model1_accuracy': float(model1_acc / 100),
        'model2_loss': float(model2_loss),
        'model2_accuracy': float(model2_acc / 100)
    }
    with open(os.path.join('static', 'current_metrics.json'), 'w') as f:
        json.dump(metrics, f)

def train_model(kernel_config1, kernel_config2, optimizer='adam', batch_size=128, num_epochs=1):
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define current_dir at the start
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create static directory if it doesn't exist
    static_dir = os.path.join(current_dir, 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # Print configurations
    print("Starting training process...")
    print(f"Model 1 kernels: {kernel_config1}")
    print(f"Model 2 kernels: {kernel_config2}")
    print(f"Training on device: {device}")

    # MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    model1 = CNN(kernel_config1).to(device)
    model2 = CNN(kernel_config2).to(device)
    
    # Update hyperparameters with config values
    learning_rate = 0.001
    
    # Initialize optimizers based on config
    if optimizer.lower() == 'adam':
        optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
        optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)
    else:  # SGD
        optimizer1 = optim.SGD(model1.parameters(), lr=learning_rate, momentum=0.9)
        optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate, momentum=0.9)

    # Update the plot update frequency based on epochs
    update_frequency = 100 if num_epochs > 1 else 10
    
    criterion = nn.CrossEntropyLoss()

    # Initialize metrics file with zeros
    initial_metrics = {
        'epoch': 0,
        'model1_loss': 0.0,
        'model1_accuracy': 0.0,
        'model2_loss': 0.0,
        'model2_accuracy': 0.0
    }
    
    metrics_path = os.path.join(current_dir, 'static', 'current_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(initial_metrics, f)

    # Training loop
    losses1, accuracies1 = [], []
    losses2, accuracies2 = [], []
    iteration_count = 0
    
    # Keep track of last known good metrics
    last_metrics = initial_metrics.copy()

    for epoch in range(num_epochs):
        # Update and save epoch number immediately
        current_metrics = {
            'epoch': epoch + 1,
            'model1_loss': last_metrics['model1_loss'],
            'model1_accuracy': last_metrics['model1_accuracy'],
            'model2_loss': last_metrics['model2_loss'],
            'model2_accuracy': last_metrics['model2_accuracy']
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(current_metrics, f)
            f.flush()
            os.fsync(f.fileno())

        # Force immediate plot update at epoch start
        update_plot(losses1, accuracies1, losses2, accuracies2, current_dir, current_metrics)

        model1.train()
        model2.train()
        running_loss1 = running_loss2 = 0.0
        correct1 = correct2 = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            total += labels.size(0)

            # Train Model 1
            optimizer1.zero_grad()
            outputs1 = model1(images)
            loss1 = criterion(outputs1, labels)
            loss1.backward()
            optimizer1.step()
            running_loss1 += loss1.item()
            _, predicted1 = torch.max(outputs1.data, 1)
            correct1 += (predicted1 == labels).sum().item()

            # Train Model 2
            optimizer2.zero_grad()
            outputs2 = model2(images)
            loss2 = criterion(outputs2, labels)
            loss2.backward()
            optimizer2.step()
            running_loss2 += loss2.item()
            _, predicted2 = torch.max(outputs2.data, 1)
            correct2 += (predicted2 == labels).sum().item()

            # Calculate current metrics
            current_loss1 = running_loss1 / (i + 1)
            current_loss2 = running_loss2 / (i + 1)
            current_acc1 = 100 * correct1 / total
            current_acc2 = 100 * correct2 / total

            # Update last known good metrics only if values are valid
            if current_loss1 > 0 and current_loss2 > 0:
                last_metrics.update({
                    'epoch': epoch + 1,  # Keep the current epoch number
                    'model1_loss': current_loss1,
                    'model1_accuracy': current_acc1 / 100,
                    'model2_loss': current_loss2,
                    'model2_accuracy': current_acc2 / 100
                })

            progress_bar.set_postfix({
                'Loss1': f'{current_loss1:.4f}',
                'Acc1': f'{current_acc1:.2f}%',
                'Loss2': f'{current_loss2:.4f}',
                'Acc2': f'{current_acc2:.2f}%'
            })

            iteration_count += 1
            
            # Update metrics more frequently at epoch boundaries
            if i == 0 or i == len(train_loader) - 1 or iteration_count % update_frequency == 0:
                losses1.append(current_loss1)
                losses2.append(current_loss2)
                accuracies1.append(current_acc1)
                accuracies2.append(current_acc2)
                
                # Use last known good metrics for updates
                update_plot(losses1, accuracies1, losses2, accuracies2, current_dir, last_metrics)
                
                # Save current metrics
                if current_loss1 > 0 and current_loss2 > 0:
                    save_metrics(epoch, current_loss1, current_acc1, current_loss2, current_acc2)

    # Save final metrics using last known good values
    final_metrics = {
        'epoch': num_epochs,
        'model1_loss': last_metrics['model1_loss'],
        'model1_accuracy': last_metrics['model1_accuracy'],
        'model2_loss': last_metrics['model2_loss'],
        'model2_accuracy': last_metrics['model2_accuracy']
    }
    
    print("Saving final metrics:", final_metrics)
    
    # Save final metrics to both current and final metrics files
    metrics_path = os.path.join(current_dir, 'static', 'current_metrics.json')
    final_metrics_path = os.path.join(current_dir, 'static', 'final_metrics.json')
    
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f)
    with open(final_metrics_path, 'w') as f:
        json.dump(final_metrics, f)

    # Save models
    torch.save({
        'kernel_config': kernel_config1,
        'model_state_dict': model1.state_dict(),
    }, os.path.join(current_dir, 'static', 'model1.pth'))

    torch.save({
        'kernel_config': kernel_config2,
        'model_state_dict': model2.state_dict(),
    }, os.path.join(current_dir, 'static', 'model2.pth'))

    # Generate and save test predictions for both models
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        test_predictions = []
        indices = np.random.choice(len(test_dataset), 10, replace=False)
        
        for idx in indices:
            image, label = test_dataset[idx]
            output1 = model1(image.unsqueeze(0).to(device))
            output2 = model2(image.unsqueeze(0).to(device))
            pred1 = output1.argmax(dim=1).item()
            pred2 = output2.argmax(dim=1).item()
            
            test_predictions.append({
                'image': image.numpy().tolist(),
                'true_label': label,
                'model1_prediction': pred1,
                'model2_prediction': pred2
            })
        
        predictions_path = os.path.join(current_dir, 'static', 'test_predictions.json')
        with open(predictions_path, 'w') as f:
            json.dump(test_predictions, f)