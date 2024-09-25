# TREAT
[NeurIPS 2024] Physics-Informed Regularization for Domain-Agnostic Dynamical System Modeling 

## Data Generation
To generate the simulated datasets, go to the data directory first
```
cd data
```
where we have a separate script for generating each dataset. Hyperparameters used for datasets evaluated in the paper are saved as default values for each argument.
To generate the *simple spring* dataset, run
```
python generate_dataset.py
```
To generate the *forced spring* dataset, run
```
python generate_dataset_external.py
```
To generate the *damped spring* dataset, run
```
python generate_dataset_damped.py
```
To generate the *pendulum* dataset, run
```
python generate_dataset_pendulum.py
```

## Training 
For all training procedures, use "--n-balls 5" for the spring datasets and use "--n-balls 3" for the pendulum dataset. We use the --data option to distinguish the dataset to be trained on. Supported dataset types include simple_spring, damped_spring, forced_spring, pendulum.

### Training TANGO and LGODE

To start the training of TANGO and LGODE, use run_model.py. To run LGODE, use the command 
```
python run_model.py --data simple_spring --n-balls 5 --train_cut 20000 --test_cut 5000 --sample-percent-train 0.6 --sample-percent-train 0.6 
```
with appropriate arguments.
To run TANGO, provide a non-negative value for the --reverse_f_lambda option 
```
python run_model.py --data simple_spring --n-balls 5 --train_cut 20000 --test_cut 5000 --sample-percent-train 0.6 --sample-percent-train 0.6 --reverse_f_lambda 0.5
```
To run the ablation using the time-reversal loss following the original definition of time-reversal, use the --use_trsode option
```
python run_model.py --data simple_spring --n-balls 5 --train_cut 20000 --test_cut 5000 --sample-percent-train 0.6 --sample-percent-train 0.6 --reverse_f_lambda 0.5 --use_trsode
```
To run the ablation using the difference between groud truth and backward trajectories, use the --reverse_gt_lambda option instead
```
python run_model.py --data simple_spring --n-balls 5 --train_cut 20000 --test_cut 5000 --sample-percent-train 0.6 --sample-percent-train 0.6 --reverse_gt_lambda 0.5
```
### Training HODEN, TRS-ODEN, and TRS-ODEN_GNN
To train these models, use the run_models_trsode.py script, and use the --function_type option to specify the model.

Running HODEN
```
python run_models_trsode.py --function_type hamiltonian --data simple_spring --n-balls 5 --train_cut 20000 --test_cut 5000 --sample-percent-train 0.6 --sample-percent-train 0.6 --reverse_f_lambda 0.5
```
Running TRS-ODEN
```
python run_models_trsode.py --function_type ode --data simple_spring --n-balls 5 --train_cut 20000 --test_cut 5000 --sample-percent-train 0.6 --sample-percent-train 0.6 --reverse_f_lambda 0.5
```
Running TRS-ODEN_GNN
```
CUDA_VISIBLE_DEVICES=0 python run_models_trsode.py --function_type gnn --data simple_spring --n-balls 5 --train_cut 20000 --test_cut 5000 --sample-percent-train 0.6 --sample-percent-train 0.6 --reverse_f_lambda 0.5
```
### Training LatentODE
To train LatentODE, go to latent_ode folder and then run 
```
python run_model.py --data simple_spring --n-balls 5 --latent-ode --train_cut 20000 --test_cut 5000 --sample-percent-train 0.6 --sample-percent-train 0.6 -l 15 -u 1000 -g 50 --rec-layers 4 --gen-layers 2 --rec-dim 100 --lr 1e-3
```

