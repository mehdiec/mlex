# Exercice Machine Learning TheraPanacea

The result on the val set is saved in the `ml_exercise_therapanacea/resnet_val.txt` file.

## Project Structure

- `pl_module.py`: Contains the PyTorch Lightning module implementation (`ClassificationPLModule`)
- `model.py`: Defines the model architectures and building blocks
- `data_module.py`: Implements the data module for handling data loading and preprocessing
- `data_loader.py`: Contains the dataset class implementation
- `train.py`: Main training script with configuration handling
- `eda.ipynb`: Notebook for exploratory data analysis on the training data
- `resnet_config.yaml`: Example configuration file for resnet model
- `config_infer.yaml`: Example configuration file for inference
- `ml_exercise_therapanacea/resnet_train.proba`: Probability scores for the training set used for model evaluation
- `ml_exercise_therapanacea/efficientnet_train.proba`: Probability scores for the training set used for model evaluation

## Technical Implementation Details

### Model Training Strategy

This implementation handles a classification task with particular emphasis on addressing class imbalance and ensuring model generalization.

#### Architecture & Framework
- Implemented using PyTorch Lightning for structured training and clean code organization
- Customizable classification head with configurable depth and dropout rates
- Binary classification setup utilizing BCEWithLogitsLoss

#### Data Handling

1. **Class Imbalance Handling**
   - Implemented WeightedRandomSampler for balanced batch sampling
   - Configurable positive class weighting in loss function
   - Stratified dataset splitting to maintain class distributions

2. **Augmentation Strategy**
   - Differential augmentation pipeline:
     - Minority class: Aggressive augmentation suite including:
       ```python
       - Geometric: RandomRotate90, ElasticTransform, GridDistortion
       - Intensity: GaussNoise, BrightnessContrast adjustments
       - Quality: Blur variations, CLAHE, Sharpening
       ```
     - Majority class: Conservative augmentation with basic transforms
     - Validation: Standard normalization and resizing only (ImageNet-like (e.g. ResNet))

#### Training Optimization
- **Optimizer**: AdamW with configurable learning rate and weight decay
- **Learning Rate Management**:
  - Multiple scheduler options:
    - Cosine Annealing
    - One Cycle Policy 
    - ReduceLROnPlateau
  - Dynamic learning rate adjustment based on validation metrics

The One Cycle Policy was chosen for its rapid convergence and better generalization through cyclic learning rates, while requiring minimal hyperparameter tuning. While it needs careful setup of max learning rate and total steps, its efficiency makes it ideal for short training cycles.

#### Monitoring & Quality Control
- Metric tracking using CometML
- Early stopping monitoring validation HTER
- Best model checkpointing based on validation metrics
- train/validation/test split with stratification

#### Misc
- Configuration using YAML 

## Results at a Glance
Full metrics, confusion matrices, and timing benchmarks live in **`eda.ipynb` → “Results” section**.  

## Model Performance Summary

| Set   | Model        | HTER  | FAR   | FRR   | ROC AUC |
|--------|--------------|--------|--------|--------|---------|
| Test  | ResNet       | 0.069 | 0.058 | 0.081 | 0.979 |
| Test  | EfficientNet | 0.081 | 0.083 | 0.080 | 0.977 |

 
## Requirements


```bash
conda create -n mlex python=3.10 -y
conda activate mlex
pip install -r requirements.txt
```

## Usage

### Training

To train the model:

```bash
python train.py --config path/to/config.yaml
```

### Configuration

The training can be configured through command line arguments or a YAML configuration file. Key parameters include:

- Model parameters:
  - `model_type`: Name of the backbone model (e.g. resnet, cnn_encoder)
  - `num_classes`: Number of output classes (1 for binary)
  - `freeze`: Whether to freeze backbone weights
  - `pretrained`: Use pretrained weights
  - `dim_base`: Base dimension for CNN encoder (32)
  - `classification_head_depth`: Depth of classification head
  - `classification_head_dropout`: Dropout rate
  - `channels`: Number of input channels (3 for RGB)

- Training parameters:
  - `num_epochs`: Number of training epochs
  - `batch_size`: Batch size for training
  - `lr`: Learning rate
  - `weight_decay`: Weight decay for optimization
  - `scheduler`: Learning rate scheduler type (e.g. one_cycle)
  - `pos_weight`: Positive class weight for imbalanced datasets

- General settings:
  - `device_index`: GPU device index
  - `root_dir`: Root directory path
  - `checkpoint_dir`: Directory for saving checkpoints
  - `checkpoint_path`: Path to load existing checkpoint

### Inference

To train the model:

```bash
python infer.py --config path/to/config.yaml
```

### Configuration

The inference can be configured through command line arguments or a YAML configuration file. Key parameters include:

- Model parameters:
  - `model_type`: Name of the backbone model (e.g. resnet, efficientnet)
  - `num_classes`: Number of output classes (1 for binary)
  - `classification_head_depth`: Depth of classification head

- Inference parameters:
  - `batch_size`: Batch size for inference
  - `get_proba`: Whether to save probability scores
  - `val_threshold`: Classification threshold for binary predictions

- General settings:
  - `device_index`: GPU device index
  - `checkpoint_path`: Path to load model checkpoint
  - `input_dir`: Directory containing input images
  - `output_file`: Path to save predictions