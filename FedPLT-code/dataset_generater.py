import os
from src.DataModule.DataLoader import dirichlet_drawer
from src.DataModule.utils import save_dataloaders, plot_sample_distribution, plot_stacked_class_distribution
import src.SFL.Tools as tools
from hyperparameters import *



# Parameters
seeds = list(range(2, 102, 10))
uniform_data_count=False

# Dataset Directory
dataset_dir = f"./Datasets/{DATASET_NAME} clients_{CLIENT_COUNT} dirichlet_{str(ALPHA).replace('.', ',')}"
os.makedirs(dataset_dir, exist_ok=True)

for seed in seeds:
    tools.SEED = seed
    tools.set_seed()

    training_loaders, validation_loaders, testing_loaders = dirichlet_drawer(dataset=DATASET_NAME,
                                                                             clients_count=CLIENT_COUNT,
                                                                             alphas=[ALPHA],
                                                                             batch_size=BATCH_SIZE,
                                                                             uniform_data_count=uniform_data_count)


    save_dataloaders(training_loaders, validation_loaders, testing_loaders, f'{dataset_dir}/seed_{seed}')