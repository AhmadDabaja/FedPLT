# Global Parameters
SEED = 42

# FL parameters
CLIENT_COUNT = 100
ROUND_COUNT = 500

# Clients Parameters
DATASET_NAME = 'FashionMnist'                   # or 'Cifar10'
ALPHA = 0.1

MODEL_NAME = "FedPLTDenseFashionModel"          # or "FedPLTDenseCifar10Model" or "FedPLTConvFashionModel"

EPOCHS_COUNT = 3
BATCH_SIZE = 64                                 # -1 for full batch
LEARNING_RATE = 0.01
BATCH_SAMPLING = False


# FedPLT parameters
PAUSE_VECTORS = [([100, 100, 100, 100], 0.1),
                ([20, 20, 20, 20], 0.9)]
DEACTIVATION_NAME = "PLT(10% Full, 90%x20%)"
COMMUNICATION_COST = 10
ORIGINAL_OCS = False
AGGREGATION_METHOD = 'PartDSAvg'                # 'DSAvg' or 'PartDSAvg' or 'PartAvg'

