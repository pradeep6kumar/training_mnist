```
mnist_cnn/
│
├── app.py
├── train.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── HowTo.md

```


# How to Run the MNIST CNN Training with Flask

## Prerequisites
- Python 3.x
- PyTorch with CUDA support
- Flask
- Matplotlib
- tqdm

## Installation
1. Clone the repository or create the directory structure as shown above.
2. Install the required packages:
   ```bash
   pip install torch torchvision flask matplotlib tqdm
   ```

## Running the Application
1. Start the training process:
   ```bash
   python train.py
   ```
   This will train the CNN on the MNIST dataset and log the training progress.

2. Once training is complete, start the Flask server:
   ```bash
   python app.py
   ```

3. Open your web browser and go to `http://127.0.0.1:5000/` to view the training results.

## Notes
- The training will log the loss and accuracy to the console.
- The training plot will be saved in the `static` directory and displayed 
on the web page.


## Summary of the Implementation


The train.py script trains a simple CNN on the MNIST dataset, logs the training process, and saves a plot of the loss and accuracy.
The app.py script sets up a Flask server to serve an HTML page displaying the training results.
The index.html file displays the training plot.
The HowTo.md file provides instructions on how to set up and run the project.
Make sure to have the necessary libraries installed and a compatible GPU for CUDA support. You can adjust the batch size and other hyperparameters as needed.