from lib.likelihood_eval import *
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import torch.nn as nn
import torch
import numpy as np


class VAE_Baseline(nn.Module):
    def __init__(self, input_dim, latent_dim,
                 z0_prior, device,
                 obsrv_std=0.01,
                 ):
        super(VAE_Baseline, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device

        self.obsrv_std = torch.Tensor([obsrv_std]).to(device)

        self.z0_prior = z0_prior

    def get_gaussian_likelihood(self, truth, pred_y, temporal_weights, mask):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]

        # Compute likelihood of the data under the predictions
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
        mask = mask.repeat(pred_y.size(0), 1, 1, 1)
        log_density_data = masked_gaussian_log_density(pred_y, truth_repeated,
                                                       obsrv_std=self.obsrv_std, mask=mask,
                                                       temporal_weights=temporal_weights)  # 【num_traj,num_sample_traj] [250,3]
        log_density_data = log_density_data.permute(1, 0)
        log_density = torch.mean(log_density_data, 1)

        # shape: [n_traj_samples]
        return log_density




    def get_energy_gaussian_likelihood(self, truth, pred_y, n_ball, temporal_weights, k,mask):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]

        # Compute likelihood of the data under the predictions
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
        mask = mask.repeat(pred_y.size(0), 1, 1, 1)
        gt_energy=compute_average_energy(truth_repeated,n_ball,k,mask)
        pred_y_energy=compute_average_energy(pred_y,n_ball,k,mask)
        res=compute_energy_likelihood(gt_energy, pred_y_energy,mse,temporal_weights=temporal_weights)
        return res


    def get_f_r_gaussian_likelihood(self, pred_y, pred_y_reverse, temporal_weights, mask):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]

        # Compute likelihood of the data under the predictions

        mask = mask.repeat(pred_y.size(0), 1, 1, 1)
        log_density_data = masked_gaussian_log_density(pred_y_reverse,pred_y,
                                                       obsrv_std=self.obsrv_std, mask=mask,
                                                       temporal_weights=temporal_weights)  # 【num_traj,num_sample_traj] [250,3]
        log_density_data = log_density_data.permute(1, 0)
        log_density = torch.mean(log_density_data, 1)

        # shape: [n_traj_samples]
        return log_density

    def get_mse(self, truth, pred_y, mask=None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        n_traj, n_tp, n_dim = truth.size()

        # Compute likelihood of the data under the predictions
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
        mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute likelihood of the data under the predictions
        log_density_data = compute_mse(pred_y, truth_repeated, mask=mask)
        # shape: [1]
        return torch.mean(log_density_data)
    def get_mape(self, truth, pred_y, mask=None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        n_traj, n_tp, n_dim = truth.size()

        # Compute likelihood of the data under the predictions
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
        mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute likelihood of the data under the predictions
        log_density_data = compute_mape(truth_repeated,pred_y, mask=mask)
        # shape: [1]
        return torch.mean(log_density_data)

    def get_energy_mse(self,truth, pred_y,n_ball,temporal_weights,k,mask=None):
        n_traj, n_tp, n_dim = truth.size()
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
        mask = mask.repeat(pred_y.size(0), 1, 1, 1)
        gt_energy = compute_average_energy(truth_repeated, n_ball, k, mask)
        pred_y_energy = compute_average_energy(pred_y, n_ball, k, mask)
        diff = gt_energy- pred_y_energy
        mse = torch.mean(diff ** 2)
        return mse

    def get_f_r_mse(self, pred_y,pred_y_reverse, mask=None):
        # pred_y_reverse shape [n_traj_samples, n_traj, n_tp, n_dim]
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]

        mask = mask.repeat(pred_y.size(0), 1, 1, 1)
        # Compute likelihood of the data under the predictions
        log_density_data = compute_mse(pred_y_reverse, pred_y, mask=mask)
        # shape: [1]
        return torch.mean(log_density_data)






    def compute_all_losses(self, batch_dict_encoder, batch_dict_decoder, batch_dict_graph, n_traj_samples=1,
                           reverse_f_lambda=1.,reverse_gt_lambda=1.,energy_lambda=1):
        # Condition on subsampled points
        # Make predictions for all the points

        pred_y, pred_y_reverse,n_ball, info, temporal_weights = self.get_reconstruction(batch_dict_encoder, batch_dict_decoder,
                                                                                 batch_dict_graph,
                                                                                 n_traj_samples=n_traj_samples)
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]

        # print("get_reconstruction done -- computing likelihood")
        fp_mu = info["first_point"]
        # fp_std = fp_std.abs()
        # fp_distr = Normal(fp_mu, fp_std)

        # assert(torch.sum(fp_std < 0) == 0.)
        # kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

        # if torch.isnan(kldiv_z0).any():
        # 	print(fp_mu)
        # 	# print(fp_std)
        # 	raise Exception("kldiv_z0 is Nan!")

        # Mean over number of latent dimensions
        # kldiv_z0 shape: [n_traj_samples, n_traj, n_latent_dims] if prior is a mixture of gaussians (KL is estimated)
        # kldiv_z0 shape: [1, n_traj, n_latent_dims] if prior is a standard gaussian (KL is computed exactly)
        # shape after: [n_traj_samples]
        # kldiv_z0 = torch.mean(kldiv_z0,(1,2))

        # Compute likelihood of all the points


        Forward_gt_rec_likelihood = self.get_gaussian_likelihood(
            batch_dict_decoder["data"], pred_y, temporal_weights,
            mask=batch_dict_decoder["mask"])  # negative value

        Forward_gt_mse = self.get_mse(
            batch_dict_decoder["data"], pred_y,
            mask=batch_dict_decoder["mask"])  # [1]
        Forward_gt_mape = self.get_mape(
            batch_dict_decoder["data"], pred_y,
            mask=batch_dict_decoder["mask"])  # [1]
        ## loss for forward and backward

        Reverse_gt_rec_likelihood = self.get_gaussian_likelihood(
            batch_dict_decoder["data"], pred_y_reverse, temporal_weights,
            mask=batch_dict_decoder["mask"])  # negative value

        Reverse_gt_mse = self.get_mse(
            batch_dict_decoder["data"], pred_y_reverse,
            mask=batch_dict_decoder["mask"])  # [1]


        Reverse_f_rec_likelihood = self.get_f_r_gaussian_likelihood(
            pred_y, pred_y_reverse, temporal_weights,
            mask=batch_dict_decoder["mask"])  # negative value

        Reverse_f_mse = self.get_f_r_mse(
            pred_y, pred_y_reverse,
            mask=batch_dict_decoder["mask"])  # [1]



        # Forward_gt_energy_rec_likelihood=self.get_energy_gaussian_likelihood(batch_dict_decoder["data"], pred_y,n_ball,temporal_weights,k=.1,mask=batch_dict_decoder["mask"])
        # energy_mse=self.get_energy_mse(batch_dict_decoder["data"], pred_y,n_ball,temporal_weights,k=.1,mask=batch_dict_decoder["mask"])


        # loss

        # loss = - torch.logsumexp(Forward_gt_rec_likelihood, 0) -reverse_f_lambda* torch.logsumexp(Reverse_f_rec_likelihood, 0)-reverse_gt_lambda* torch.logsumexp(Reverse_gt_rec_likelihood, 0)-energy_lambda*torch.logsumexp(Forward_gt_energy_rec_likelihood,0)
        # if torch.isnan(loss):
        #     loss = - torch.mean(Forward_gt_rec_likelihood, 0) - reverse_f_lambda* torch.mean(Reverse_f_rec_likelihood, 0)-reverse_gt_lambda* torch.mean(Reverse_gt_rec_likelihood, 0)-energy_lambda*torch.mean(Forward_gt_energy_rec_likelihood,0)

        loss = - torch.logsumexp(Forward_gt_rec_likelihood, 0) -reverse_f_lambda* torch.logsumexp(Reverse_f_rec_likelihood, 0)-reverse_gt_lambda* torch.logsumexp(Reverse_gt_rec_likelihood, 0)
        if torch.isnan(loss):
            loss = - torch.mean(Forward_gt_rec_likelihood, 0) - reverse_f_lambda* torch.mean(Reverse_f_rec_likelihood, 0)-reverse_gt_lambda* torch.mean(Reverse_gt_rec_likelihood, 0)

        results = {}
        results["loss"] = torch.mean(loss)
        results["likelihood"] = torch.mean(Forward_gt_rec_likelihood).data.item() + reverse_f_lambda*torch.mean(Reverse_f_rec_likelihood).data.item()+ reverse_gt_lambda*torch.mean(Reverse_gt_rec_likelihood).data.item()
        results["mse"] = torch.mean(Forward_gt_mse).data.item() + reverse_f_lambda*torch.mean(Reverse_f_mse).data.item()+ reverse_gt_lambda*torch.mean(Reverse_gt_mse).data.item()
        results["forward_gt_mse"] = torch.mean(Forward_gt_mse).data.item()
        results["reverse_f_mse"] = torch.mean(Reverse_f_mse).data.item()
        results["reverse_gt_mse"] = torch.mean(Reverse_gt_mse).data.item()
        results["forward_gt_mape"] = torch.mean(Forward_gt_mape).data.item()


        # results['energy_mse']= torch.mean(energy_mse).data.item()

        # results["kl_first_p"] =  torch.mean(kldiv_z0).detach().data.item()
        # results["std_first_p"] = torch.mean(fp_std).detach().data.item()

        return results,batch_dict_decoder["data"], pred_y, pred_y_reverse,








