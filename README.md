# XAI Assignment #2: DeepXplore

## Project Structure

- `test.py`: Main evaluation script using DeepXplore to find model disagreements and measure coverage.
- `deepxplore.py`: Core implementation of the DeepXplore algorithm (Gradient Ascent for coverage and disagreement).
- `utils.py`: Utility functions for image conversion (`to_image`) and neuron coverage calculation.
- `data/`: Directory for CIFAR-10 dataset storage.
- `results/`: Directory for saving disagreement-inducing adversarial samples and visualization.

### 1. Model Configuration

This assignment compares two different pre-trained architectures to find cross-model inconsistencies:
- **Model 1**: ResNet-56 (CIFAR-10)
- **Model 2**: ResNet-44 (CIFAR-10)

### 2. DeepXplore Evaluation

The `test.py` script performs joint optimization to generate inputs that trigger different predictions between the two models while maximizing neuron coverage.

**Key Parameters (configured in `test.py`):**
- `lambda_1`: Weight for neuron coverage maximization (Default: `2`).
- `lambda_2`: Weight for model disagreement maximization (Default: `1`).
- `threshold`: Neuron activation threshold (Default: `0.6`).
- `max_samples`: Maximum number of original test images to sample (Default: `1000`).

### Execution

Run the evaluation script to generate adversarial samples and measure coverage:

```bash
# Run DeepXplore on CIFAR-10
python test.py