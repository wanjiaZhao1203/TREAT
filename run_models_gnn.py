import os
import sys
from lib.non_graph_loader import ParseData
from tqdm import tqdm
import argparse
import numpy as np
from random import SystemRandom
import torch
import torch.optim as optim
import lib.utils as utils
from torch.distributions.normal import Normal
from lib.trsoden import ODENetwork
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import pdb


def compute_loss_all_batches(model, encoder, graph, decoder, n_batches, device, function_type):

    total = {}
    total["loss"] = 0
    total["mse"] = 0
    total["mape"] = 0
    total["forward_gt_mse"] = 0
    total["reverse_f_mse"] = 0

    n_test_batches = 0

    model.eval()
    print("Computing loss... ")
    if function_type == 'hamiltonian':
        for i in tqdm(range(n_batches)):
            batch_dict_encoder = utils.get_next_batch(encoder, device)
            batch_dict_decoder = utils.get_next_batch(decoder, device)
            batch_dict_graph = utils.get_next_batch_new(graph, device)
                
            results,_,_ = model.compute_loss(batch_dict_encoder, batch_dict_decoder, batch_dict_graph)

            for key in total.keys():
                if key in results:
                    var = results[key]
                    if isinstance(var, torch.Tensor):
                        var = var.detach().item()
                    total[key] += var

            n_test_batches += 1

            del batch_dict_encoder,batch_dict_decoder,results
    else:
        with torch.no_grad():
            for i in tqdm(range(n_batches)):
                batch_dict_encoder = utils.get_next_batch(encoder, device)
                batch_dict_decoder = utils.get_next_batch(decoder, device)
                batch_dict_graph = utils.get_next_batch_new(graph, device)
                    
                results,_,_ = model.compute_loss(batch_dict_encoder, batch_dict_decoder, batch_dict_graph)

                for key in total.keys():
                    if key in results:
                        var = results[key]
                        if isinstance(var, torch.Tensor):
                            var = var.detach().item()
                        total[key] += var

                n_test_batches += 1

                del batch_dict_encoder,batch_dict_decoder,results

    if n_test_batches > 0:
        for key, value in total.items():
            total[key] = total[key] / n_test_batches


    return total

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('--n-balls', type=int, default=5,
                    help='Number of objects in the dataset.')
parser.add_argument('--niters', type=int, default=500)
parser.add_argument('--lr', type=float, default=2e-4, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=128)
parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--save-graph', type=str, default='plot/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="name of ckpt. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")
parser.add_argument('--data', type=str, default='spring', help="spring,charged,motion,spring_external")
parser.add_argument('--z0-encoder', type=str, default='GTrans', help="GTrans")
parser.add_argument('-l', '--latents', type=int, default=16, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default=64, help="Dimensionality of the recognition model .")
parser.add_argument('--ode-dims', type=int, default=128, help="Dimensionality of the ODE func")
parser.add_argument('--rec-layers', type=int, default=2, help="Number of layers in recognition model ")
parser.add_argument('--n-heads', type=int, default=1, help="Number of heads in GTrans")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers  ODE func ")
parser.add_argument('--extrap', type=str, default="True",
                    help="Set extrapolation mode. If this flag is not set, run interpolation mode.")
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--sample-percent-train', type=float, default=0.6, help='Percentage of training observtaion data')
parser.add_argument('--sample-percent-test', type=float, default=0.6, help='Percentage of testing observtaion data')
parser.add_argument('--augment_dim', type=int, default=0, help='augmented dimension')
parser.add_argument('--edge_types', type=int, default=2, help='edge number in NRI')
parser.add_argument('--odenet', type=str, default="NRI", help='NRI')
parser.add_argument('--solver', type=str, default="rk4", help='dopri5,rk4,euler')
parser.add_argument('--l2', type=float, default=1e-3, help='l2 regulazer')
parser.add_argument('--optimizer', type=str, default="AdamW", help='Adam, AdamW')
parser.add_argument('--clip', type=float, default=10, help='Gradient Norm Clipping')
parser.add_argument('--cutting_edge', type=bool, default=True, help='True/False')
parser.add_argument('--extrap_num', type=int, default=40, help='extrap num ')
parser.add_argument('--rec_attention', type=str, default="attention")
parser.add_argument('--alias', type=str, default="run")
parser.add_argument('--train_cut', type=int, default=2000, help='maximum number of train samples')
parser.add_argument('--test_cut', type=int, default=500, help='maximum number of test samples')
parser.add_argument('--total_ode_step', type=int, default=60, help='total number of ode steps')
parser.add_argument('--dataset', type=str, default='data', help='dataset directory')
parser.add_argument('--tensorboard_dir', type=str, default='tensorboards', help='tensorboard root directory')
parser.add_argument('--warmup_epoch', type=int, default=20, help='number of warmup epoch to train with forward mse only')
parser.add_argument('--reverse_f_lambda', type=float, default=0, help='weight of reverse_f mse after warmup')
parser.add_argument('--reverse_gt_lambda', type=float, default=0, help='weight of reverse_gt mse after warmup')
parser.add_argument('--energy_lambda', type=float, default=0, help='weight of energy mse after warmup')
parser.add_argument('--device', type=int, default=0, help='running device')
parser.add_argument('--Tmax', type=float, default=2000, help='optimazor')
parser.add_argument('--eta_min', type=float, default=0, help='optimazor')
parser.add_argument('--visdata_dir', type=str, default='visdata', help='vis root directory')
parser.add_argument('--function_type', type=str, default='gnn', help='function type for ode network')
parser.add_argument('--time_augment', action='store_true', default=False, help='time augment for ode network')
parser.add_argument('--nb_units', type=int, default=1000, help='nb_units for ode network')
parser.add_argument('--nb_layers', type=int, default=1, help='nb_layers for ode network')
parser.add_argument('--activation', type=str, default='tanh', help='activation for ode network')
parser.add_argument('--name', type=str, default="GNN",help="GNN" )
parser.add_argument('--feature_dim', type=int, default=3, help="3d")

args = parser.parse_args()
assert (int(args.rec_dims % args.n_heads) == 0)

if args.data == "simple_spring":
    args.suffix = '_springs_simpled_long5'
    args.dataset='data/simple_spring'

if args.data == "simple_spring_3d":
    args.suffix = '_3d_springs_simpled5'
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

if args.data == "motion":
	args.dataset = 'data/motion'
	args.suffix = '_motion'
	args.total_ode_step = 49
	args.n_balls = 31
if args.data == "motion_dance":
	args.dataset = 'data/motion_dance'
	args.suffix = '_motion_dance'
	args.total_ode_step = 49
	args.n_balls = 31

task = 'extrapolation' if args.extrap == 'True' else 'intrapolation'

############ CPU AND GPU related, Mode related, Dataset Related
if torch.cuda.is_available():
    print("Using GPU" + "-" * 80)
    device = torch.device("cuda:%d"%args.device)
else:
    print("Using CPU" + "-" * 80)
    device = torch.device("cpu")

if args.extrap == "True":
    print("Running extrap mode" + "-" * 80)
    args.mode = "extrap"
elif args.extrap == "False":
    print("Running interp mode" + "-" * 80)
    args.mode = "interp"

#####################################################################################################

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    ############ Saving Path and Preload.
    file_name = os.path.basename(__file__)[:-3]  # run_models
    utils.makedirs(args.save)
    utils.makedirs(args.save_graph)

    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random() * 100000)

    ############ Loading Data
    print("Loading dataset: " + args.dataset)
    dataloader = ParseData(args.dataset, suffix=args.suffix, mode=args.mode, args=args)
    test_encoder, test_decoder, test_graph, test_batch = dataloader.load_data(sample_percent=args.sample_percent_test,
                                                                              batch_size=args.batch_size,
                                                                              data_type="test",
                                                                              cut_num=args.test_cut)
    train_encoder, train_decoder, train_graph, train_batch = dataloader.load_data(
        sample_percent=args.sample_percent_train, batch_size=args.batch_size, data_type="train",
        cut_num=args.train_cut)

    input_dim = dataloader.feature

    ############ Command Related
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)

    ############ Model Select
    # Create the model
    obsrv_std = 0.01
    # pdb.set_trace()
    obsrv_std = torch.Tensor([obsrv_std]).to(device)

    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

    model = ODENetwork(nb_object=args.n_balls, lambda_trs=args.reverse_f_lambda,
                       learning_rate=args.lr, function_type=args.function_type, 
                       time_augment=args.time_augment, nb_units=args.nb_units, 
                       nb_layers=args.nb_layers, nb_coords=args.feature_dim, activation=args.activation, args=args,
                       device=device
                       ).to(device)

    ##################################################################
    # Load checkpoint and evaluate the model
    if args.load is not None:
        ckpt_path = os.path.join(args.save, args.load)
        utils.get_ckpt_model(ckpt_path, model, device)
        # exit()

    ##################################################################
    # Training
    # pdb.set_trace()
    # log_dir = os.path.join("./home/zijiehuang/LGODE_logs/", '%s_%s'%(args.data, task))
    log_dir = os.path.join("/home/zijiehuang", "PIGODE_logs", '%s_%d_%s' % (args.data, args.total_ode_step, args.name))

    # args.alias + "_" + args.z0_encode./home/zijiehuang/LGODE_logsr + "_" + args.data + "_" + str(
    #     args.sample_percent_train) + "_" + args.mode + "_" + str(experimentID) + ".log"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logname = 'niters%d_lr%f-%d-%f_warmup-epoch%d_reverse_f_lambda%.2f_reverse_gt_lambda%.2f_traincut%d_testcut%d_observ-ratio_train%.2f_test%.2f.log' % (
        args.niters, args.lr, args.Tmax, args.eta_min,
        args.warmup_epoch, args.reverse_f_lambda,
        args.reverse_gt_lambda, args.train_cut, args.test_cut, args.sample_percent_train, args.sample_percent_test
    )
    logger = utils.get_logger(logpath=os.path.join(log_dir,logname), filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info(str(args))
    logger.info(args.alias)

    # Optimizer

    if args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.Tmax, args.eta_min)

    # pdb.set_trace()
    # wait_until_kl_inc = 10
    best_test_mse = np.inf
    best_train_mse = np.inf

    n_iters_to_viz = 1

    # #weight of reverse:
    # reverse_f_lambda=None
    # reverse_gt_lambda=None

    writer = SummaryWriter(log_dir=os.path.join(
        args.tensorboard_dir,
        '%s_%d_%s' % (args.data, args.total_ode_step, args.name),
        'observe_ratio_train%.2f_test%.2f' % (args.sample_percent_train, args.sample_percent_test),
        'niters%d_lr%f-%d-%f_warmup_epoch%d_reverse_f_lambda%.2f_reverse_gt_lambda%.2f_traincut%d_testcut%d' % (
            args.niters, args.lr, args.Tmax, args.eta_min,
            args.warmup_epoch, args.reverse_f_lambda,
            args.reverse_gt_lambda, args.sample_percent_train, args.sample_percent_test)
    ))

    def train_single_batch(model, batch_dict_encoder, batch_dict_decoder, batch_dict_graph=None):

        optimizer.zero_grad()
        train_res,_,_ = model.compute_loss(batch_dict_encoder, batch_dict_decoder, batch_dict_graph)

        loss = train_res["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        loss_value = loss.data.item()

        del loss
        torch.cuda.empty_cache()
        # train_res, loss
        # return loss_value, train_res["mse"], train_res["likelihood"],train_res["energy_mse"],train_res["forward_gt_mse"],train_res["reverse_f_mse"],train_res["reverse_gt_mse"]
        return loss_value, train_res["mse"], train_res["forward_gt_mse"],train_res["reverse_f_mse"]


    def train_epoch(epo):
        model.train()
        loss_list = []
        mse_list = []
        forward_gt_mse_list =[]
        reverse_f_mse_list = []
        reverse_gt_mse_list = []
        likelihood_list = []
        # energy_mse_list=[]
        kl_first_p_list = []
        std_first_p_list = []

        torch.cuda.empty_cache()

        for itr in tqdm(range(train_batch)):
            batch_dict_encoder = utils.get_next_batch(train_encoder, device)
            batch_dict_decoder = utils.get_next_batch(train_decoder, device)

            batch_dict_graph = None
            if args.function_type:
                 batch_dict_graph = utils.get_next_batch_new(train_graph, device)

            loss, mse, forward_gt_mse,reverse_f_mse = train_single_batch(model, batch_dict_encoder, batch_dict_decoder, batch_dict_graph)

            loss_list.append(loss), mse_list.append(mse), forward_gt_mse_list.append(
                forward_gt_mse), reverse_f_mse_list.append(reverse_f_mse)
            
            del batch_dict_encoder,  batch_dict_decoder
            # train_res, loss
            torch.cuda.empty_cache()

        scheduler.step()

        message_train = 'Epoch {:04d} | [Train seq (cond on sampled tp)] | Loss {:.6f} |  Forward gt MSE {:.6f} | Reverse f MSE {:.6f}'.format(
            epo,
            np.mean(loss_list),
            np.mean(forward_gt_mse_list), np.mean(reverse_f_mse_list))

        # return message_train ,np.mean(energy_mse_list), np.mean(forward_gt_mse_list), np.mean(reverse_f_mse_list),np.mean(reverse_gt_mse_list)
        return message_train , np.mean(forward_gt_mse_list), np.mean(reverse_f_mse_list)




    for epo in range(1, args.niters + 1):
        reverse_f_lambda = args.reverse_f_lambda
        reverse_gt_lambda = args.reverse_gt_lambda
        # message_train,train_energy_mse,train_forward_gt_mse,train_reverse_f_mse,train_reverse_gt_mse = train_epoch(epo)
        message_train,train_forward_gt_mse,train_reverse_f_mse = train_epoch(epo)

        if epo % n_iters_to_viz == 0:
            model.eval()

            test_res= compute_loss_all_batches(model, test_encoder, test_graph, test_decoder,
                                                n_batches=test_batch, device=device, 
                                                function_type=args.function_type)

            message_test = 'Epoch {:04d} [Test seq (cond on sampled tp)] | r_f_lambda {:.4f} | Loss {:.6f} | Forward gt MSE {:.6f} | Forward gt MAPE {:.6f} |Reverse f MSE {:.6f}'.format(
                epo, reverse_f_lambda,
                test_res["loss"],
                test_res["forward_gt_mse"], 
                test_res["mape"], 
                test_res["reverse_f_mse"])

            logger.info("Experiment " + str(experimentID))
            logger.info(message_train)
            logger.info(message_test)
            # logger.info(
            # "KL coef: {}".format(kl_coef))
            print("data: %s, encoder: %s, lr: %s, epoch: %s, train_sample: %s,test_sample: %s, mode: %s, reverse_f_lambda: %s" % (
                args.data, args.z0_encoder, str(args.lr), str(args.niters), str(args.sample_percent_train), str(args.sample_percent_test), args.mode,reverse_f_lambda))

            if test_res["forward_gt_mse"] < best_test_mse:
                best_test_mse = test_res["forward_gt_mse"]
                best_test_mape = test_res["mape"]
                message_test_best = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Best forward gt  mse {:.6f}| Best epoch mape {:.6f}'.format(epo,
                                                                                                        best_test_mse, test_res["mape"])

                ckpt_path = os.path.join(args.save, "experiment_" + str(
                    experimentID) + "_" + args.data + "_" + str(
                    args.total_ode_step) + "_" + args.name + "_obratio_" + str(
                    args.sample_percent_train) + "_rflambda_" + str(args.reverse_f_lambda) + "_epoch_" + str(
                    epo) + "_mse_" + str(
                    best_test_mse) + "_mape_" + str(
                    best_test_mape) + '.ckpt')
                torch.save({
                    'args': args,
                    'state_dict': model.state_dict(),
                }, ckpt_path)




                # groundtruth_dir = os.path.join(
                #     args.visdata_dir,
                #     '%s_%s' % (args.data, task),
                #     'observe_ratio_train%.2f_test%.2f' % (args.sample_percent_train, args.sample_percent_test),
                #     'train_cut%d_test_cut%d' % (args.train_cut, args.test_cut),
                #     'reverse_f_lambda%.2f_reverse_gt_lambda%.2f' % (args.reverse_f_lambda, args.reverse_gt_lambda)
                # )
                #
                # forward_dir = os.path.join(
                #     args.visdata_dir,
                #     '%s_%s' % (args.data, task),
                #     'observe_ratio_train%.2f_test%.2f' % (args.sample_percent_train, args.sample_percent_test),
                #     'train_cut%d_test_cut%d' % (args.train_cut, args.test_cut),
                #     'reverse_f_lambda%.2f_reverse_gt_lambda%.2f_energy_lambda%.2f' % (args.reverse_f_lambda, args.reverse_gt_lambda,args.energy_lambda)
                # )
                #
                # reverse_dir = os.path.join(
                #     args.visdata_dir,
                #     '%s_%s' % (args.data, task),
                #     'observe_ratio_train%.2f_test%.2f' % (args.sample_percent_train, args.sample_percent_test),
                #     'train_cut%d_test_cut%d' % (args.train_cut, args.test_cut),
                #     'reverse_f_lambda%.2f_reverse_gt_lambda%.2f_energy_lambda%.2f' % (args.reverse_f_lambda, args.reverse_gt_lambda,args.energy_lambda)
                # )

                # # Create directories
                # Path(groundtruth_dir).mkdir(parents=True, exist_ok=True)
                # Path(forward_dir).mkdir(parents=True, exist_ok=True)
                # Path(reverse_dir).mkdir(parents=True, exist_ok=True)

                # # Save files
                # np.save(os.path.join(groundtruth_dir, 'groundtruth_trajectory.npy'), gt.cpu().detach().numpy())
                # np.save(os.path.join(forward_dir, 'forward_trajectory.npy'), f.cpu().detach().numpy())
                # np.save(os.path.join(reverse_dir, 'reverse_trajectory.npy'), r.cpu().detach().numpy())


                logger.info(message_test_best)

            if train_forward_gt_mse < best_train_mse:
                best_train_mse = train_forward_gt_mse
                message_train_best = 'Epoch {:04d} [Train seq (cond on sampled tp)] | Best forward gt  mse {:.6f}'.format(
                    epo,best_train_mse)
                logger.info(message_train_best)




            writer.add_scalar('train_MSE/train_forward_gt_mse', train_forward_gt_mse, epo)
            writer.add_scalar('train_MSE/train_reverse_f_mse', train_reverse_f_mse,epo)
            # writer.add_scalar('train_MSE/train_reverse_gt_mse', train_reverse_gt_mse, epo)
            # writer.add_scalar('train_MSE/train_energy_mse', train_energy_mse, epo)

            writer.add_scalar('test_MSE/test_forward_gt_mse', test_res["forward_gt_mse"], epo)
            writer.add_scalar('test_MSE/test_reverse_f_mse', test_res["reverse_f_mse"], epo)
            # writer.add_scalar('test_MSE/test_reverse_gt_mse', test_res["reverse_gt_mse"], epo)
            # writer.add_scalar('test_MSE/test_energy_mse', test_res["energy_mse"], epo)



            # writer.add_scalar('Weight/FGT_RF', test_res["forward_gt_mse"] / test_res["reverse_f_mse"], epo)
            # writer.add_scalar('Weight/FGT_RGT', test_res["forward_gt_mse"] / test_res["reverse_gt_mse"], epo)

            torch.cuda.empty_cache()

    writer.close()

