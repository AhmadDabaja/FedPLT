import hashlib
import json
import os
import sys
import time

import torch
from torch.nn import CrossEntropyLoss

from src.DataModule.utils import load_dataloaders
from src.SFL.Simulation import Simulation
import src.SFL.Tools as tools

from src.Scheme.FedPLTServerProxy import PLTServerProxy
from src.Scheme.FedPLTClientProxy import PLTClientProxy
from src.Scheme.FedPLTHistory import CustomHistory
from src.utils import create_deactivation_function, get_model_class_by_name

from hyperparameters import *


############################### Configuring the Simulation ###############################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tools.SEED = SEED
tools.set_seed()

model = get_model_class_by_name(MODEL_NAME)
model.one_random_batch = BATCH_SAMPLING

deactivation_function = create_deactivation_function(CLIENT_COUNT,
                                                     model.layer_sizes,
                                                     PAUSE_VECTORS,
                                                     DEACTIVATION_NAME,)

simulation_parameters = {'model': model.__name__,
                         'clients': str(CLIENT_COUNT),
                         'participation count': str(COMMUNICATION_COST),
                         'rounds': str(ROUND_COUNT),
                         'epochs': str(EPOCHS_COUNT),
                         'dataset': DATASET_NAME,
                         'heterogeneity level': ALPHA,
                         'batches': str(BATCH_SIZE),
                         'deactivation': deactivation_function.__name__,
                         'aggregation method': AGGREGATION_METHOD,
                         'original_ocs': 'yes' if ORIGINAL_OCS else 'no',
                         'learning rate': LEARNING_RATE,
                         'seed': str(SEED)
                         }

################################ Preparing the Simulation ################################

# Paths
dataset_path = f'./Datasets/{DATASET_NAME} clients_{CLIENT_COUNT} dirichlet_{str(ALPHA).replace(".", ",")}'
data_file_path = os.path.join(dataset_path, f'seed_{SEED}')

# Check if directory exists
if not os.path.isdir(dataset_path):
    print(f"[ERROR] Dataset directory does not exist:\n→ {dataset_path}")
    sys.exit(1)

# Check if the file exists
if not os.path.isfile(data_file_path):
    print(f"[ERROR] Required data file not found:\n→ {data_file_path}")
    sys.exit(1)

training_loaders, validation_loaders, testing_loaders = load_dataloaders(data_file_path,
                                                                         BATCH_SIZE,
                                                                         True)

server_config = {'deactivation_function': deactivation_function,
                 'aggregation_method': AGGREGATION_METHOD,
                 'client_selection_count': COMMUNICATION_COST,
                 'original_ocs': ORIGINAL_OCS}

client_configs = [{'epochs': EPOCHS_COUNT,
                   'criterion': CrossEntropyLoss(),
                   'learning rate': LEARNING_RATE,
                   'device': device} for _ in range(CLIENT_COUNT)]

server_proxy = PLTServerProxy(model, server_config, client_configs)
client_proxies = [PLTClientProxy(training_loader, validation_loader, testing_loader) for
                  training_loader, validation_loader, testing_loader in
                  zip(training_loaders, validation_loaders, testing_loaders)]

history = CustomHistory(CLIENT_COUNT, simulation_parameters)

custom_simulation = Simulation(rounds_count=ROUND_COUNT,
                               server_proxy=server_proxy,
                               client_proxies=client_proxies,
                               history=history)

################################# Starting the Simulation ################################

custom_simulation.start_simulation()

################################### Saving the Results ###################################

history = custom_simulation.get_history()

# Save the simulation
serialized_string = json.dumps(simulation_parameters, sort_keys=True)
hash_value = hashlib.md5(serialized_string.encode())
simulation_name = f'Simulation_{hash_value.hexdigest()}_{time.strftime("%m-%d-%Y_%H:%M:%S")}'
history.save(f'./results', simulation_name)

