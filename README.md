# 2D Graph Neural Network for Shape Classification

## Introduction

This project demonstrates the use of **Graph Neural Networks (GNNs)** to classify synthetic 2D shapes (triangles, rectangles, circles, hexagons, ellipses). It covers the full machine learning pipeline from data generation, preprocessing, model training, evaluation, and visualization, allowing experimentation with various configurations and hyperparameters.

## Features:
- Generates synthetic 2D shapes and applies data augmentation (scaling, rotation).
- Preprocesses data into graph representations for GNNs.
- Implements a configurable GNN model using `config.json`.
- Includes early stopping to prevent overfitting.
- Provides visualization of predictions and evaluation metrics.

## Prerequisites and Required Libraries

To run this project, you need to have the following software installed:

- Python 3.8 or higher
- pip (for installing Python packages)
- Virtual environment (optional but recommended): For a clean environment, it's recommended to create a virtual environment before installing the dependencies:
- Required libraries: torch, torch-geometric, numpy, matplotlib

### Additional Information
- This project uses PyTorch and PyTorch Geometric for implementing the Graph Neural Network.
- The dataset is synthetic, and the shape generation process is random. You can adjust the number of samples via `config.json file`.

## Project Structure

The project is organized into the following files:
```
/2DAGGNet_Exercise
	├── config.json # Configuration file for hyperparameters and settings
	├── utils.py # Utility functions for configuration loading and early stopping
	├── SyntheticDataGenerationScript.py # Data generation script to create synthetic shapes
	├── PreprocessingData.py # Script to preprocess the shapes into graph format
	├── GNNModel.py # Model definition and training code for the GNN
	├── EvaluationAndVisualization.py # Evaluation and visualization of model performance
	├── start.py # Script to run the entire pipeline with one command
	└── README.md # This README file
```

### File Descriptions:

- **`config.json`**: Contains the hyperparameters and configurable parameters for the entire pipeline.
- **`SyntheticDataGenerationScript.py`**: Generates synthetic 2D shapes and applies augmentation (scaling and rotation).
- **`PreprocessingData.py`**: Converts the generated shapes into graph structures for GNN input.
- **`GNNModel.py`**: Defines the Graph Neural Network model, trains it, and saves the trained weights.
- **`EvaluationAndVisualization.py`**: Evaluates the trained model and visualizes its performance on the test set.
- **`start.py`**: Runs the entire pipeline automatically (data generation, training, evaluation, and visualization).
- **`utils.py`**: Contains utility functions like configuration loading and early stopping implementation.

## Setup Instructions

### Step 1: Clone the Repository

Clone the repository to your local machine:

```bash
git clone <repository_url>
cd 2DAGNNet
```

### Step 2: Install Dependencies

Create a virtual environment (optional but recommended) and install the required libraries:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
pip install torch torch-geometric numpy matplotlib
```

### Step 3: Modify Configuration (Optional)

You can modify the training parameters, data generation settings, and model configurations in the `config.json` file. Here are some of the key parameters you can modify:

- **`data.num_samples`**: Number of shapes to generate.
- **`data.train_split`, `data.validation_split`, `data.test_split`**: Split ratios for the dataset.
- **`model.num_layers`**: Number of layers in the GNN model.
- **`training.learning_rate`, `training.weight_decay`**: Training parameters for optimization.
- **`augmentation.rotation`, `augmentation.scaling`**: Enable or disable data augmentation techniques.

### Step 4: Running the Project

1. **Run the Full Pipeline**:

To run the entire pipeline (data generation, preprocessing, model training, and evaluation), simply execute the `start.py` script:

```bash
python start.py
```
This will automatically run the following steps:

1. SyntheticDataGenerationScript.py: Generates synthetic 2D shapes and applies augmentation.
2. PreprocessingData.py: Preprocesses the shapes into graph data structures.
3. GNNModel.py: Trains the GNN model using the preprocessed data.
4. EvaluationAndVisualization.py: Evaluates the model and visualizes the results.

2. **Monitor Training**: During the training process, you’ll see the loss printed every epoch. If the validation loss stops improving for several epochs (controlled by `early_stopping_patience`), the training will halt early to prevent overfitting.
   
3. **Model Saving**: After the training completes, the model will be saved as `trained_model.pth`. You can reload this model for further analysis or inference.

4. **Evaluation and Visualization**: After training, the script will evaluate the model on the test set and display visualizations of the predictions. These visualizations show the predicted labels for each graph, allowing you to visually inspect the performance of the model.

## Model Architecture and Training

### Model Architecture

#### Graph Neural Network (GNN) Model

The model uses **Graph Convolutional Networks (GCN)** for classifying 2D shapes. The architecture consists of multiple graph convolution layers followed by global pooling, which aggregates node-level features into graph-level features.

##### Key Components:
- **GCNConv Layers**: These layers perform graph convolution, aggregating information from neighboring nodes.
- **Global Mean Pooling**: After the node features are updated by the GCN layers, global pooling is used to aggregate node-level features into a fixed-size vector representing the entire graph.
- **Fully Connected Layer**: A final fully connected layer takes the pooled features and outputs the class prediction for each graph.

##### Model Flow:
1. **Input**: A graph, represented by node features (`x`), edge indices (`edge_index`), and a batch tensor (`batch`) to map nodes to their respective graphs.
2. **GCN Layers**: Multiple GCN layers are applied sequentially to learn node representations.
3. **Global Pooling**: The node representations are aggregated using global mean pooling.
4. **Output**: The fully connected layer produces the final classification output.

### Training Process

The model is trained using the **Adam optimizer** and the **cross-entropy loss function**. Here's the training process:
1. **Forward Pass**: The graph is passed through the GNN, which aggregates node features through GCN layers.
2. **Loss Calculation**: Cross-entropy loss is used to measure the difference between the predicted and true labels.
3. **Backpropagation**: Gradients are calculated and the model weights are updated.
4. **Validation**: During training, the model's performance is evaluated on a validation set. If performance stops improving, training will halt early using the **EarlyStopping** mechanism.

### Early Stopping

The model uses **early stopping** to prevent overfitting. If the validation loss does not improve for a specified number of epochs (`early_stopping_patience`), training will stop early.

### Hyperparameter Tuning

The model's performance can be improved by tuning the following hyperparameters in the `config.json` file:
- **Learning rate** (`training.learning_rate`)
- **Number of layers** (`model.num_layers`)
- **Hidden layer size** (`model.hidden_dim`)
- **Dropout rate** (`model.dropout`)

Experiment with different values to improve the accuracy of the model.

## Model Evaluation and Visualization

### Evaluation

Once the model is trained, we evaluate its performance on the test set. The evaluation includes:
1. **Accuracy**: The percentage of correctly classified graphs in the test set.
2. **Visualization**: We visualize the predictions, displaying the predicted labels for the test graphs.

### Visualization

The `EvaluationAndVisualization.py` script generates visualizations that include:
- **Node positions**: Each node is plotted based on its x, y coordinates.
- **Graph edges**: The edges are drawn connecting the nodes.
- **Predicted labels**: Each graph's predicted label is shown in the title of the plot.

**Example output**:
- A triangle with its predicted label, showing the actual shape structure and the predicted label in the title.
```plaintext
Graph Classification: Triangle
```

This indicates the model classified the graph as a "Triangle."

## Customization and Experimentation

This project is designed to be flexible, allowing you to experiment with different configurations:
**Custom Data Generation**: Modify the `SyntheticDataGenerationScript.py` to generate additional or more complex shapes. You can change the shapes in the dataset or even add new ones (e.g., polygons, stars).
**Model Configuration**: You can experiment with different GNN layers, such as replacing `GCNConv` with `GATConv` (Graph Attention Network) or `SAGEConv` (GraphSAGE). Experiment with different architectures by modifying the number of layers, hidden dimensions, or adding residual connections.
**Hyperparameter Tuning**: Adjust the learning rate, batch size, weight decay, and other training parameters via the `config.json` file to improve the model’s performance.
**Additional Augmentation Techniques**: Add more augmentation strategies, such as flipping, translation, or adding noise, to increase the robustness of the model.

### Scaling the Dataset
- Increase the number of samples in `config.json` for a larger dataset, which will improve the model’s ability to generalize
