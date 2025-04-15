How to Run a FedPLT Simulation
==============================

To run a FedPLT simulation, follow these steps:

I. Set Simulation Hyperparameters
---------------------------------
Configure the simulation hyperparameters in the "hyperparameter.py" file.

II. Generate the Dataset (Skip if already generated)
----------------------------------------------------
1. Open the "dataset_generator.py" file and set the following:
   - The "seeds" list.
   - The "uniform_data_count" value.
2. Run the "dataset_generator.py" script to generate the dataset.
3. (Optional) To visualize the dataset distribution:
   - Set the configuration parameters in "plot_distribution.py".
   - Run the script to generate the visualization.

III. Run the Simulation
-----------------------
Execute the "run_simulation.py" script.
The FedPLT simulation will start, and results will be saved automatically in the "results/" directory.

Note
----
If you have any questions, don’t hesitate to contact us.
