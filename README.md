# TREAT
[NeurIPS 2024] Physics-Informed Regularization for Domain-Agnostic Dynamical System Modeling 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/wanjiaZhao1203/TREAT/blob/main/LICENSE)


[**[Project Page]**](https://treat-ode.github.io/) [**[Paper]**](https://arxiv.org/pdf/2310.06427) 


## setup dataset
### Simulate Data
To generate the simulated datasets, go to the data directory first
```
cd data
```
where we have a separate script fo generating each dataset. Hyperparameters used for datasets evaluated in the paper are saved as default values for each argument. 
##### Spring Datasets (suffix=simple,damped,forced)
```python
python generate_dataset_suffix.py
```
An example data of simple spring with 5000/2000 for train/test is in `data/example_data` folder. 
##### Pendulum Datasets 
```python
python generate_dataset_pendulum.py
```
##### Attractor Datasets 
```python
python generate_dataset_attractor.py
```

### Motion Capture dataset
The raw data were obtained via [CMU Motion Capture Database](http://mocap.cs.cmu.edu/) 
The walk and dance dataset are provided in `data/motion_walk` and `data/motion_dance` folder. 

You can also download .asf and .amc. from the website and use the following command to get other trajectory:
```python
python motion_preprocess.py
python motion_get_data.py
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

