import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import lib.utils as utils
import pdb


class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, reverse_ode_func, method, args,
                 odeint_rtol=1e-3, odeint_atol=1e-4, device=torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.device = device
        self.ode_func = ode_func
        self.reverse_ode_func = reverse_ode_func
        self.args = args
        self.num_atoms = args.n_balls
        self.use_trsode = args.use_trsode

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

        # graph related
        self.rel_rec, self.rel_send = self.compute_rec_send()

    def compute_rec_send(self):
        off_diag = np.ones([self.num_atoms, self.num_atoms]) - np.eye(self.num_atoms)
        rel_rec = np.array(self.encode_onehot(np.where(off_diag)[0]),
                           dtype=np.float32)  # every node as one-hot[10000], (20,5)
        rel_send = np.array(self.encode_onehot(np.where(off_diag)[1]), dtype=np.float32)  # every node as one-hot,(20,5)
        rel_rec = torch.FloatTensor(rel_rec).to(self.device)
        rel_send = torch.FloatTensor(rel_send).to(self.device)

        return rel_rec, rel_send

    def forward(self, first_point, time_steps_to_predict, graph, backwards=False):
        '''

        :param first_point: 【n_sample,b*n_ball,d]
        :param time_steps_to_predict: [t]
        :param graph: [2, num_edge]
        :param backwards:
        :return:
        '''
        # whether to padding 0 to the time series
        # print('time_steps_to_predict shape',time_steps_to_predict.size())
        ispadding = False
        if time_steps_to_predict[0] != 0:
            ispadding = True
            time_steps_to_predict = torch.cat(
                (torch.zeros(1, device=time_steps_to_predict.device), time_steps_to_predict))
        # print('time_steps_to_predict shape after padding', time_steps_to_predict.size())
        # pdb.set_trace()
        time_steps_to_predict_reverse = torch.flip(time_steps_to_predict.max() - time_steps_to_predict, dims=[0])
        # time_steps_to_predict_reverse = time_steps_to_predict.max() - time_steps_to_predict
        # print('time_steps_to_predict_reverse shape', time_steps_to_predict_reverse.shape)

        n_traj_samples, n_traj, feature = first_point.size()[0], first_point.size()[1], first_point.size()[2]

        first_point_augumented = first_point.view(-1, self.num_atoms, feature)  # [n_sample*b, n_ball,d]
        if self.args.augment_dim > 0:
            aug = torch.zeros(first_point_augumented.shape[0], first_point_augumented.shape[1],
                              self.args.augment_dim).to(self.device)
            first_point_augumented = torch.cat([first_point_augumented, aug], 2)
            feature += self.args.augment_dim

        # duplicate graph w.r.t num_sample_traj
        graph_augmented = torch.cat([graph for _ in range(n_traj_samples)], dim=0)

        rel_type_onehot = torch.FloatTensor(first_point_augumented.size(0), self.rel_rec.size(0),
                                            self.args.edge_types).to(self.device)  # [b,20,2]
        rel_type_onehot.zero_()
        rel_type_onehot.scatter_(2, graph_augmented.view(first_point_augumented.size(0), -1, 1), 1)  # [b,20,2]
        # rel_type_onehot[b,20,1]: edge value, [b,20,0] :1-edge value.
        # pdb.set_trace()
        self.ode_func.set_graph(rel_type_onehot, self.rel_rec, self.rel_send, self.args.edge_types)

        pred_y = odeint(self.ode_func, first_point_augumented, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol,
                        method=self.ode_method)  # [time_length, n_sample*b,n_ball, d]

        if not self.use_trsode:
            pred_y_reverse_flipped = odeint(self.reverse_ode_func, pred_y[-1], time_steps_to_predict_reverse,
                                            rtol=self.odeint_rtol, atol=self.odeint_atol,
                                            method=self.ode_method)  # [time_length, n_sample*b,n_ball, d]

            pred_y_reverse = torch.flip(pred_y_reverse_flipped, dims=[0])
            # pdb.set_trace()
            assert torch.all(pred_y_reverse[-1] == pred_y[-1]), "Tensors are not identical"
        else:
            xr = first_point_augumented
            ls_xr = [xr]
            dts = time_steps_to_predict[1:] - time_steps_to_predict[:-1]
            for i in range(time_steps_to_predict.shape[0] - 1):
                dt = dts[i]

                dxr1 = - self.ode_func(None, xr) * dt
                dxr2 = - self.ode_func(None, xr + 0.5 * dxr1) * dt
                dxr3 = - self.ode_func(None, xr + 0.5 * dxr2) * dt
                dxr4 = - self.ode_func(None, xr + dxr3) * dt
                dxr = (1 / 6) * (dxr1 + 2 * dxr2 + 2 * dxr3 + dxr4)
                xr = xr + dxr
                ls_xr.append(xr)

            pred_y_reverse = torch.stack(ls_xr, dim=0)
            assert torch.all(pred_y_reverse[0] == pred_y[0]), "Tensors are not identical"
        '''
        pred_y = self.ode_func(time_steps_to_predict, first_point_augumented)
        pred_y = pred_y.repeat(time_steps_to_predict.shape[0], 1, 1,1)
        '''
        # print('pred_y_reverse size', pred_y_reverse.size())
        # print('first_point size(): n_traj_samples, n_traj, feature',first_point.size())
        # print("pred_y_reverse 0 size:", pred_y_reverse[0].size())

        # print('pred_y_reverse_flipped',pred_y_reverse_flipped)
        # print("pred_y_reverse_flipped 0:", pred_y_reverse_flipped[0])
        # print('pred_y Final',pred_y[-1])
        # print('pred_y_reverse Final',pred_y_reverse[-1])
        # print('pred_y_reverse 0', pred_y_reverse[0])

        if ispadding:
            pred_y = pred_y[1:, :, :, :]
            time_steps_to_predict = time_steps_to_predict[1:]
            # pdb.set_trace()
        pred_y = pred_y.view(time_steps_to_predict.size(0), -1, pred_y.size(3))  # [t,n_sample*b*n_ball, d]

        pred_y = pred_y.permute(1, 0, 2)  # [n_sample*b*n_ball, time_length, d]
        pred_y = pred_y.view(n_traj_samples, n_traj, -1, feature)  # [n_sample, b*n_ball, time_length, d]

        # assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
        assert (pred_y.size()[0] == n_traj_samples)
        assert (pred_y.size()[1] == n_traj)

        if self.args.augment_dim > 0:
            pred_y = pred_y[:, :, :, :-self.args.augment_dim]

        if ispadding:
            pred_y_reverse = pred_y_reverse[1:, :, :, :]
            time_steps_to_predict_reverse = time_steps_to_predict_reverse[1:]

        pred_y_reverse = pred_y_reverse.view(time_steps_to_predict_reverse.size(0), -1,
                                             pred_y_reverse.size(3))  # [t,n_sample*b*n_ball, d]

        pred_y_reverse = pred_y_reverse.permute(1, 0, 2)  # [n_sample*b*n_ball, time_length, d]
        pred_y_reverse = pred_y_reverse.view(n_traj_samples, n_traj, -1,
                                             feature)  # [n_sample, b*n_ball, time_length, d]

        # assert(torch.mean(pred_y_reverse[:, :, 0, :]  - first_point) < 0.001)
        assert (pred_y_reverse.size()[0] == n_traj_samples)
        assert (pred_y_reverse.size()[1] == n_traj)

        if self.args.augment_dim > 0:
            pred_y_reverse = pred_y_reverse[:, :, :, :-self.args.augment_dim]

        # assert torch.all(pred_y_reverse[-1] == pred_y[-1]), "Tensors are not identical"

        return pred_y, pred_y_reverse

        print("After return pred_y_reverse shape:", pred_y_reverse.shape)
        print("pred_y shape:", pred_y.shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot


#
# class DiffeqSolver(nn.Module):
#     def __init__(self, ode_func, method,args,
#             odeint_rtol = 1e-3, odeint_atol = 1e-4, device = torch.device("cpu")):
#         super(DiffeqSolver, self).__init__()
#
#         self.ode_method = method
#         self.device = device
#         self.ode_func = ode_func
#         self.args = args
#         self.num_atoms = args.n_balls
#
#
#         self.odeint_rtol = odeint_rtol
#         self.odeint_atol = odeint_atol
#
#         # graph related
#         self.rel_rec, self.rel_send = self.compute_rec_send()
#
#
#     def compute_rec_send(self):
#         off_diag = np.ones([self.num_atoms, self.num_atoms]) - np.eye(self.num_atoms)
#         rel_rec = np.array(self.encode_onehot(np.where(off_diag)[0]),
#                            dtype=np.float32)  # every node as one-hot[10000], (20,5)
#         rel_send = np.array(self.encode_onehot(np.where(off_diag)[1]), dtype=np.float32)  # every node as one-hot,(20,5)
#         rel_rec = torch.FloatTensor(rel_rec).to(self.device)
#         rel_send = torch.FloatTensor(rel_send).to(self.device)
#
#         return rel_rec, rel_send
#
#
#     def forward(self, first_point, time_steps_to_predict, graph,backwards = False):
#         '''
#
#         :param first_point: 【n_sample,b*n_ball,d]
#         :param time_steps_to_predict: [t]
#         :param graph: [2, num_edge]
#         :param backwards:
#         :return:
#         '''
#         #whether to padding 0 to the time series
#         ispadding = False
#         if time_steps_to_predict[0] != 0:
#             ispadding = True
#             time_steps_to_predict = torch.cat((torch.zeros(1,device=time_steps_to_predict.device),time_steps_to_predict))
#
#
#
#         n_traj_samples, n_traj,feature = first_point.size()[0], first_point.size()[1],first_point.size()[2]
#         first_point_augumented = first_point.view(-1,self.num_atoms,feature) #[n_sample*b, n_ball,d]
#         if self.args.augment_dim > 0:
#             aug = torch.zeros(first_point_augumented.shape[0],first_point_augumented.shape[1], self.args.augment_dim).to(self.device)
#             first_point_augumented = torch.cat([first_point_augumented, aug], 2)
#             feature += self.args.augment_dim
#
#         # duplicate graph w.r.t num_sample_traj
#         graph_augmented = torch.cat([graph for _ in range(n_traj_samples)], dim=0)
#
#         rel_type_onehot = torch.FloatTensor(first_point_augumented.size(0), self.rel_rec.size(0),
#                                                 self.args.edge_types).to(self.device)  # [b,20,2]
#         rel_type_onehot.zero_()
#         rel_type_onehot.scatter_(2, graph_augmented.view(first_point_augumented.size(0), -1, 1), 1)  # [b,20,2]
#         # rel_type_onehot[b,20,1]: edge value, [b,20,0] :1-edge value.
#
#         self.ode_func.set_graph(rel_type_onehot,self.rel_rec,self.rel_send,self.args.edge_types)
#
#
#         pred_y = odeint(self.ode_func, first_point_augumented, time_steps_to_predict,
#             rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method) #[time_length, n_sample*b,n_ball, d]
#
#         '''
#         pred_y = self.ode_func(time_steps_to_predict, first_point_augumented)
#         pred_y = pred_y.repeat(time_steps_to_predict.shape[0], 1, 1,1)
#         '''
#
#         if ispadding:
#             pred_y = pred_y[1:,:,:,:]
#             time_steps_to_predict = time_steps_to_predict[1:]
#
#         pred_y = pred_y.view(time_steps_to_predict.size(0), -1, pred_y.size(3)) #[t,n_sample*b*n_ball, d]
#
#         pred_y = pred_y.permute(1,0,2) #[n_sample*b*n_ball, time_length, d]
#         pred_y = pred_y.view(n_traj_samples,n_traj,-1,feature) #[n_sample, b*n_ball, time_length, d]
#
#         #assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
#         assert(pred_y.size()[0] == n_traj_samples)
#         assert(pred_y.size()[1] == n_traj)
#
#         if self.args.augment_dim > 0:
#             pred_y = pred_y[:, :, :, :-self.args.augment_dim]
#
#         return pred_y
#
#     def encode_onehot(self,labels):
#         classes = set(labels)
#         classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
#                         enumerate(classes)}
#         labels_onehot = np.array(list(map(classes_dict.get, labels)),
#                                  dtype=np.int32)
#         return labels_onehot
#
#
#


class GraphODEFuncT(nn.Module):
    def __init__(self, ode_func_net):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(GraphODEFuncT, self).__init__()

        self.ode_func_net = ode_func_net  # input: x, edge_index
        self.nfe = 0

    def forward(self, z, t_local, backwards=False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        self.nfe += 1
        # print(self.nfe)
        grad = self.ode_func_net(z)

        if backwards:
            grad = -grad
        return grad

    def set_graph(self, rec_type, rel_rec, rel_send, edge_types):
        # print(self.nfe)
        for layer in self.ode_func_net.gcs:
            layer.base_conv.rel_type = rec_type
            layer.base_conv.rel_rec = rel_rec
            layer.base_conv.rel_send = rel_send
            layer.base_conv.edge_types = edge_types
        self.nfe = 0


class GraphODEFunc(nn.Module):
    def __init__(self, ode_func_net, device=torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(GraphODEFunc, self).__init__()

        self.device = device
        self.ode_func_net = ode_func_net  # input: x, edge_index
        self.nfe = 0

    def forward(self, t_local, z, backwards=False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        self.nfe += 1
        # print(self.nfe)
        grad = self.ode_func_net(z)

        if backwards:
            grad = -grad
        return grad

    def set_graph(self, rec_type, rel_rec, rel_send, edge_types):
        # print(self.nfe)
        for layer in self.ode_func_net.gcs:
            layer.base_conv.rel_type = rec_type
            layer.base_conv.rel_rec = rel_rec
            layer.base_conv.rel_send = rel_send
            layer.base_conv.edge_types = edge_types
        self.nfe = 0


class ODEFunc(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_func_net, device=torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc, self).__init__()

        self.input_dim = input_dim
        self.device = device

        utils.init_network_weights(ode_func_net)
        self.gradient_net = ode_func_net

    def forward(self, t_local, y, backwards=False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)


class ReverseGraphODEFunc(nn.Module):
    def __init__(self, ode_func_net, device=torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ReverseGraphODEFunc, self).__init__()

        self.device = device
        self.ode_func_net = ode_func_net  # input: x, edge_index
        self.nfe = 0

    def forward(self, t_local, z, backwards=False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        self.nfe += 1
        # print(self.nfe)
        grad = self.ode_func_net(z)

        if backwards:
            grad = -grad
        return -grad

    def set_graph(self, rec_type, rel_rec, rel_send, edge_types):
        # print(self.nfe)
        for layer in self.ode_func_net.gcs:
            layer.base_conv.rel_type = rec_type
            layer.base_conv.rel_rec = rel_rec
            layer.base_conv.rel_send = rel_send
            layer.base_conv.edge_types = edge_types
        self.nfe = 0


class ReverseODEFunc(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_func_net, device=torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ReverseODEFunc, self).__init__()

        self.input_dim = input_dim
        self.device = device

        utils.init_network_weights(ode_func_net)
        self.gradient_net = ode_func_net

    def forward(self, t_local, y, backwards=False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """

        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return -grad

    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)








