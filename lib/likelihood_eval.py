from lib.likelihood_eval import *
import torch





def gaussian_log_likelihood(mu, data, obsrv_std):
	log_p = ((mu - data) ** 2) / (2 * obsrv_std * obsrv_std)
	neg_log_p = -1*log_p
	return neg_log_p

def generate_time_weight(n_timepoints,n_dims):
	value_min = 1
	value_max = 2
	interval = (value_max - value_min)/(n_timepoints-1)

	value_list = [value_min + i*interval for i in range(n_timepoints)]
	value_list= torch.FloatTensor(value_list).view(-1,1)

	value_matrix= torch.cat([value_list for _ in range(n_dims)],dim = 1)

	return value_matrix


def compute_masked_likelihood(mu, data, mask, likelihood_func,temporal_weights=None):
	# Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

	log_prob = likelihood_func(mu, data)  # [n_traj, n_traj_samples, n_timepoints, n_dims]
	if temporal_weights!= None:
		weight_for_times = torch.cat([temporal_weights for _ in range(n_dims)],dim = 1)
		weight_for_times = weight_for_times.to(mu.device)
		weight_for_times = weight_for_times.repeat(n_traj_samples, n_traj, 1, 1)
		log_prob_masked = torch.sum(log_prob * mask * weight_for_times, dim=2)  # [n_traj, n_traj_samples, n_dims]
	else:
		unnormalized_map = log_prob / torch.maximum(1e-9*torch.ones_like(mu), torch.abs(mu))
		unnormalized_map = torch.sum(unnormalized_map * mask, dim=2)

		log_prob_masked = torch.sum(log_prob * mask, dim=2)  # [n_traj, n_traj_samples, n_dims]
		# norm_masked = torch.sum(abs(mu) * mask, dim=2)

	# unnormalized_map = log_prob_masked / norm_masked

	timelength_per_nodes = torch.sum(mask.permute(0,1,3,2),dim=3)
	assert (not torch.isnan(timelength_per_nodes).any())
	mse_log_prob_masked_normalized = torch.div(log_prob_masked , timelength_per_nodes) #【n_traj_sample, n_traj, feature], average each feature by dividing time length
	# Take mean over the number of dimensions
	res_mse = torch.mean(mse_log_prob_masked_normalized, -1) # 【n_traj_sample, n_traj], average among features.
	res_mse = res_mse.transpose(0,1)

	mape_log_prob_masked_normalized = torch.div(unnormalized_map ,
										   timelength_per_nodes)  # 【n_traj_sample, n_traj, feature], average each feature by dividing time length
	# Take mean over the number of dimensions
	res_mape = torch.mean(mape_log_prob_masked_normalized, -1)  # 【n_traj_sample, n_traj], average among features.
	res_mape = res_mape.transpose(0, 1)


	return res_mse,res_mape


def compute_masked_likelihood_old(mu, data, mask, likelihood_func):
	# Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()


	log_prob = likelihood_func(mu, data)  # [n_traj, n_traj_samples, n_timepoints, n_dims]
	log_prob_masked = torch.sum(log_prob * mask, dim=2)  # [n_traj, n_traj_samples, n_dims]

	timelength_per_nodes = torch.sum(mask.permute(0, 1, 3, 2), dim=3)
	assert (not torch.isnan(timelength_per_nodes).any())
	log_prob_masked_normalized = torch.div(log_prob_masked,
										   timelength_per_nodes)  # 【n_traj_sample, n_traj, feature], average each feature by dividing time length
	# Take mean over the number of dimensions
	res = torch.mean(log_prob_masked_normalized, -1)  # 【n_traj_sample, n_traj], average among features.
	res = res.transpose(0, 1)
	return res


def masked_gaussian_log_density(mu, data, obsrv_std, mask,temporal_weights=None):

	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

	assert(data.size()[-1] == n_dims)

	# Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
	func = lambda mu, data: gaussian_log_likelihood(mu, data, obsrv_std = obsrv_std)
	res,_ = compute_masked_likelihood(mu, data,mask, func,temporal_weights)
	return res


def mse(mu,data):
	return  (mu - data) ** 2
def mape(mu,data):
	return abs(mu - data)


def compute_mse(mu, data, mask):

	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
	assert(data.size()[-1] == n_dims)

	res ,_= compute_masked_likelihood(mu, data, mask, mse)
	return res
def compute_mape(mu, data, mask):
	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
	assert (data.size()[-1] == n_dims)

	_,res = compute_masked_likelihood(mu, data, mask, mape)
	return res

def compute_average_energy(mu,n_ball,k,mask):
	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

	mu_nball= mu.view(-1, n_ball, n_timepoints, n_dims)
	n_traj_pre=mu_nball.size()[0]
	# 提取速度数据
	vel= mu[:, :, :, 2:]

	# 提取位置数据
	positions = mu[:, :, :, :2]

	# 计算动能
	kinetic_energies = 0.5 * torch.sum(vel ** 2, dim=-1)

	# 计算弹性势能
	potential_energies = torch.zeros(n_traj_pre, n_ball, n_timepoints).to(mu.device)
	for i in range(n_ball):
		for j in range(i + 1, n_ball):
			displacement = positions[:, i, :, :] - positions[:, j, :, :]
			displacement_magnitude = torch.norm(displacement, dim=-1)
			# potential_energies = potential_energies.to(device)
			# displacement_magnitude = displacement_magnitude.to(device)

			potential_energies[:, i, :] += 0.5 * k * displacement_magnitude ** 2

	total_potential_energy = torch.sum(potential_energies, dim=1)
	total_energy_per_moment = kinetic_energies + total_potential_energy
	mask_sliced = mask[:, 0, :, 0]  # 取第二和第四维度的第一个元素
	mask_expanded = mask_sliced.unsqueeze(1).expand_as(total_energy_per_moment)
	average_energy_per_trajectory_masked = torch.mean(total_energy_per_moment * mask_expanded, dim=1)

	return average_energy_per_trajectory_masked  #[n_traj_pre]

def compute_energy_likelihood(mu_energy, data_energy, likelihood_func, temporal_weights=None):
	# Compute the likelihood per patient and per attribute so that we don't prioritize patients with more measurements

	# Compute the likelihood
	log_prob = likelihood_func(mu_energy, data_energy)  # [n_traj_samples]

	# If temporal weights are provided, apply them
	if temporal_weights is not None:
		weight_for_times = temporal_weights.to(mu_energy.device)
		log_prob_weighted = log_prob * weight_for_times
	else:
		log_prob_weighted = log_prob

	# Take mean over the number of dimensions
	res = torch.mean(log_prob_weighted)  # Scalar value

	return res






	

