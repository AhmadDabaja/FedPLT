import os
import sys

from hyperparameters import DATASET_NAME, CLIENT_COUNT, ALPHA, SEED
from src.DataModule.utils import plot_sample_distribution, plot_stacked_class_distribution

# Dataset Specification
seed = SEED
data_name = DATASET_NAME
client_count = CLIENT_COUNT
alpha = ALPHA

# Paths
dataset_path = f'./Datasets/{data_name} clients_{client_count} dirichlet_{str(alpha).replace(".", ",")}'
data_file_path = os.path.join(dataset_path, f'seed_{seed}')

# Check if directory exists
if not os.path.isdir(dataset_path):
    print(f"[ERROR] Dataset directory does not exist:\n→ {dataset_path}")
    sys.exit(1)

# Check if the file exists
if not os.path.isfile(data_file_path):
    print(f"[ERROR] Required data file not found:\n→ {data_file_path}")
    sys.exit(1)

# If everything exists, proceed
plot_sample_distribution(data_file_path)
plot_stacked_class_distribution(data_file_path)
