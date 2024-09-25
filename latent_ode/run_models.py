###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import time
import datetime
import argparse
import numpy as np
import pandas as pd
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim

import lib.utils as utils
# from lib.plotting import *

from lib.rnn_baselines import *
from lib.ode_rnn import *
from lib.create_latent_ode_model import create_LatentODE_model
from lib.latent_ode_loader import ParseData
from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from lib.diffeq_solver import DiffeqSolver
# from mujoco_physics import HopperPhysics

from lib.utils import compute_loss_all_batches
from pathlib import Path

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('-n',  type=int, default=100, help="Size of the dataset")
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=256)
parser.add_argument('--viz', action='store_true', help="Show plots while training")

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")

parser.add_argument('--dataset', type=str, default='periodic', help="Dataset to load. Available: physionet, activity, hopper, periodic")
parser.add_argument('-s', '--sample-tp', type=float, default=None, help="Number of time points to sub-sample."
	"If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")

parser.add_argument('-c', '--cut-tp', type=int, default=None, help="Cut out the section of the timeline of the specified length (in number of points)."
	"Used for periodic function demo.")

parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset."
	"Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")

parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")
parser.add_argument('--z0-encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")

parser.add_argument('--classic-rnn', action='store_true', help="Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only.")
parser.add_argument('--rnn-cell', default="gru", help="RNN Cell type. Available: gru (default), expdecay")
parser.add_argument('--input-decay', action='store_true', help="For RNN: use the input that is the weighted average of impirical mean and previous value (like in GRU-D)")

parser.add_argument('--ode-rnn', action='store_true', help="Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only.")

parser.add_argument('--rnn-vae', action='store_true', help="Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss.")

parser.add_argument('-l', '--latents', type=int, default=6, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")

parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")

parser.add_argument('--poisson', action='store_true', help="Model poisson-process likelihood for the density of events in addition to reconstruction.")
parser.add_argument('--classif', action='store_true', help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")

parser.add_argument('--linear-classif', action='store_true', help="If using a classifier, use a linear classifier instead of 1-layer NN")
parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")

parser.add_argument('--device', type=int, default=0)
parser.add_argument('--data', type=str, default='simple_spring')
parser.add_argument('--sample-percent-train', type=float, default=0.6, help='Percentage of training observtaion data')
parser.add_argument('--sample-percent-test', type=float, default=0.6, help='Percentage of testing observtaion data')
parser.add_argument('--total_ode_step', type=int, default=60, help='total number of ode steps')
parser.add_argument('--extrap_num', type=int, default=40, help='extrap num ')
parser.add_argument('--train_cut', type=int, default=20, help='maximum number of train samples')
parser.add_argument('--test_cut', type=int, default=5, help='maximum number of test samples')
parser.add_argument('--name', type=str, default="LatentODE",help="LGODE,DCGODE")

args = parser.parse_args()

device = torch.device("cuda:%d" % args.device if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3] 
utils.makedirs(args.save)


if args.data == "simple_spring":
	args.suffix = '_springs_simpled_long5'
	args.dataset='data/simple_spring'

if args.data == "damped_spring":
	args.suffix = '_springs_damped5'
	args.dataset='data/damped_spring'

if args.data == "forced_spring":
	args.suffix = '_forced_spring5'
	args.dataset='data/forced_spring'

if args.data == "charged":
	args.suffix = '_charged5'
	args.dataset='data/charged'

if args.data == "pendulum":
	args.suffix = '_pendulum3'
	args.dataset='data/pendulum'

#####################################################################################################

if __name__ == '__main__':
	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)

	experimentID = args.load
	if experimentID is None:
		# Make a new experiment ID
		experimentID = int(SystemRandom().random()*100000)
	

	start = time.time()
	
	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)

	utils.makedirs("results/")

	##################################################################
	print("Loading dataset: " + args.dataset)
	dataloader = ParseData(args.dataset, suffix=args.suffix, mode='extrap', args=args, device=device)
	test_loader, test_batch = dataloader.load_data(sample_percent=args.sample_percent_test,
									batch_size=args.batch_size, data_type="test", cut_num=args.test_cut)
	train_loader, train_batch = dataloader.load_data(sample_percent=args.sample_percent_train, 
									batch_size=args.batch_size, data_type="train", cut_num=args.train_cut)

	input_dim = dataloader.feature

	data_obj = {"dataset_obj": None, 
		"train_dataloader": utils.inf_generator(train_loader), 
		"test_dataloader": utils.inf_generator(test_loader),
		"input_dim": input_dim,
		"n_train_batches": train_batch,
		"n_test_batches": test_batch
	}

	classif_per_tp = False
	if ("classif_per_tp" in data_obj):
		# do classification per time point rather than on a time series as a whole
		classif_per_tp = data_obj["classif_per_tp"]

	if args.classif and (args.dataset == "hopper" or args.dataset == "periodic"):
		raise Exception("Classification task is not available for MuJoCo and 1d datasets")

	n_labels = 1
	if args.classif:
		if ("n_labels" in data_obj):
			n_labels = data_obj["n_labels"]
		else:
			raise Exception("Please provide number of labels for classification task")

	##################################################################
	# Create the model
	obsrv_std = 0.01
	if args.dataset == "hopper":
		obsrv_std = 1e-3 

	obsrv_std = torch.Tensor([obsrv_std]).to(device)

	z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

	if args.rnn_vae:
		if args.poisson:
			print("Poisson process likelihood not implemented for RNN-VAE: ignoring --poisson")

		# Create RNN-VAE model
		model = RNN_VAE(input_dim, args.latents, 
			device = device, 
			rec_dims = args.rec_dims, 
			concat_mask = True, 
			obsrv_std = obsrv_std,
			z0_prior = z0_prior,
			use_binary_classif = args.classif,
			classif_per_tp = classif_per_tp,
			linear_classifier = args.linear_classif,
			n_units = args.units,
			input_space_decay = args.input_decay,
			cell = args.rnn_cell,
			n_labels = n_labels,
			train_classif_w_reconstr = (args.dataset == "physionet")
			).to(device)


	elif args.classic_rnn:
		if args.poisson:
			print("Poisson process likelihood not implemented for RNN: ignoring --poisson")

		if args.extrap:
			raise Exception("Extrapolation for standard RNN not implemented")
		# Create RNN model
		model = Classic_RNN(input_dim, args.latents, device, 
			concat_mask = True, obsrv_std = obsrv_std,
			n_units = args.units,
			use_binary_classif = args.classif,
			classif_per_tp = classif_per_tp,
			linear_classifier = args.linear_classif,
			input_space_decay = args.input_decay,
			cell = args.rnn_cell,
			n_labels = n_labels,
			train_classif_w_reconstr = (args.dataset == "physionet")
			).to(device)
	elif args.ode_rnn:
		# Create ODE-GRU model
		n_ode_gru_dims = args.latents
				
		if args.poisson:
			print("Poisson process likelihood not implemented for ODE-RNN: ignoring --poisson")

		if args.extrap:
			raise Exception("Extrapolation for ODE-RNN not implemented")

		ode_func_net = utils.create_net(n_ode_gru_dims, n_ode_gru_dims, 
			n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)

		rec_ode_func = ODEFunc(
			input_dim = input_dim, 
			latent_dim = n_ode_gru_dims,
			ode_func_net = ode_func_net,
			device = device).to(device)

		z0_diffeq_solver = DiffeqSolver(input_dim, rec_ode_func, "euler", args.latents, 
			odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
	
		model = ODE_RNN(input_dim, n_ode_gru_dims, device = device, 
			z0_diffeq_solver = z0_diffeq_solver, n_gru_units = args.gru_units,
			concat_mask = True, obsrv_std = obsrv_std,
			use_binary_classif = args.classif,
			classif_per_tp = classif_per_tp,
			n_labels = n_labels,
			train_classif_w_reconstr = (args.dataset == "physionet")
			).to(device)
	elif args.latent_ode:
		model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, 
			classif_per_tp = classif_per_tp,
			n_labels = n_labels)
	else:
		raise Exception("Model not specified")

	##################################################################

	# if args.viz:
	# 	viz = Visualizations(device)

	##################################################################
	
	#Load checkpoint and evaluate the model
	if args.load is not None:
		utils.get_ckpt_model(args.load, model, device)
		# exit()

	# for pred_length_cut in [20, 30, 40, 50, 60]:
	# test_res = compute_loss_all_batches(model,
	# 				data_obj["test_dataloader"], args,
	# 				n_batches = data_obj["n_test_batches"],
	# 				experimentID = experimentID,
	# 				device = device,
	# 				n_traj_samples = 3, kl_coef = 0)
	#
	# message = 'Epoch 0 [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | mse {:.6f} | rmse {:.6f}|'.format(
	# 		test_res["loss"].detach(), test_res["likelihood"].detach(),
	# 		test_res["mse"], test_res["rmse"])
	# print(message)

	##################################################################
	# Training

	log_dir = os.path.join("/home/zijiehuang", "latent_logs", '%s_%d_%s' % (args.data, args.total_ode_step, args.name))

	# args.alias + "_" + args.z0_encode./home/zijiehuang/LGODE_logsr + "_" + args.data + "_" + str(
	#     args.sample_percent_train) + "_" + args.mode + "_" + str(experimentID) + ".log"
	Path(log_dir).mkdir(parents=True, exist_ok=True)

	logname = 'niters%d_lr%f_traincut%d_testcut%d_observ-ratio_train%.2f_test%.2f.log' % (
		args.niters, args.lr,  args.train_cut, args.test_cut, args.sample_percent_train, args.sample_percent_test
	)





	logger = utils.get_logger(logpath=os.path.join(log_dir,logname), filepath=os.path.abspath(__file__))
	logger.info(input_command)
	logger.info(str(args))

	optimizer = optim.Adamax(model.parameters(), lr=args.lr)

	num_batches = data_obj["n_train_batches"]
	best_test_mse = np.inf
	# import pdb
	for itr in range(1, num_batches * (args.niters + 1)):
		optimizer.zero_grad()
		utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)

		wait_until_kl_inc = 10
		if itr // num_batches < wait_until_kl_inc:
			kl_coef = 0.
		else:
			kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))
		

		batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
		# pdb.set_trace()
		train_res = model.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
		train_res["loss"].backward()
		optimizer.step()

		n_iters_to_viz = 1
		if itr % (n_iters_to_viz * num_batches) == 0:
			with torch.no_grad():

				test_res = compute_loss_all_batches(model, 
					data_obj["test_dataloader"], args,
					n_batches = data_obj["n_test_batches"],
					experimentID = experimentID,
					device = device,
					n_traj_samples = 3, kl_coef = kl_coef)

				message_test = 'Epoch {:04d} [Test seq(cond on sampled tp)] | Loss{: .6f} | Likelihood{: .6f} | ms{: .6f} | rmse{: .6f} | '.format(
					itr//num_batches, test_res["loss"].detach(), test_res["likelihood"].detach(),test_res["mse"], test_res["rmse"])




				logger.info("Experiment " + str(experimentID))
				logger.info(message_test)
				logger.info("KL coef: {}".format(kl_coef))
				logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
				logger.info("Train CE loss (one batch): {}".format(train_res["ce_loss"].detach()))
				
				if "auc" in test_res:
					logger.info("Classification AUC (TEST): {:.4f}".format(test_res["auc"]))

				if "mse" in test_res:
					logger.info("Test MSE: {:.6f}".format(test_res["mse"]))

				if "mape" in test_res:
					logger.info("Test MAPE: {:.6f}".format(test_res["mape"]))

				if "accuracy" in train_res:
					logger.info("Classification accuracy (TRAIN): {:.4f}".format(train_res["accuracy"]))

				if "accuracy" in test_res:
					logger.info("Classification accuracy (TEST): {:.4f}".format(test_res["accuracy"]))

				if "pois_likelihood" in test_res:
					logger.info("Poisson likelihood: {}".format(test_res["pois_likelihood"]))

				if "ce_loss" in test_res:
					logger.info("CE loss: {}".format(test_res["ce_loss"]))

				if test_res["mse"] < best_test_mse:
					best_test_mse = test_res["mse"].data.item()
					best_test_rmse = test_res["rmse"].data.item()
					message_test_best = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Best mse {:.6f}'.format(
						itr//num_batches, test_res["mse"].detach())

					ckpt_path = os.path.join(args.save, "experiment_" + str(
						experimentID) + "_"  + args.data + "_" + str(
						args.total_ode_step) + "_" + args.name +"_obratio_"+str(args.sample_percent_train)+ "_epoch_" + str(itr // num_batches) + "_mse_" + str(
						best_test_mse) + "_rmse_" + str(
						best_test_rmse) + '.ckpt')

					torch.save({
						'args': args,
						'state_dict': model.state_dict(),
					}, ckpt_path)


			# Plotting
			# if args.viz:
			# 	with torch.no_grad():
			# 		test_dict = utils.get_next_batch(data_obj["test_dataloader"])
			#
			# 		print("plotting....")
			# 		if isinstance(model, LatentODE) and (args.dataset == "periodic"): #and not args.classic_rnn and not args.ode_rnn:
			# 			plot_id = itr // num_batches // n_iters_to_viz
			# 			viz.draw_all_plots_one_dim(test_dict, model,
			# 				plot_name = file_name + "_" + str(experimentID) + "_{:03d}".format(plot_id) + ".png",
			# 			 	experimentID = experimentID, save=True)
			# 			plt.pause(0.01)

