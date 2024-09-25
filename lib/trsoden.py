import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lib.gnn_models import GNN
from lib.diffeq_solver import GraphODEFuncT
import numpy as np

import pdb
class Gradient(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return torch.autograd.grad(y, x)[0]

def append_activation(activation, layers: list):
    if activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    else:
        raise NotImplementedError

class ODEFunc(nn.Module):
    def __init__(self, dim=1, time_augment=False,
                 nb_units=1000, nb_layers=1, activation='tanh'):
        super(ODEFunc, self).__init__()
        self.time_augment = time_augment
        layers = []
        if time_augment:
            layers.append(nn.Linear(2 * dim + 1, nb_units))
        else:
            layers.append(nn.Linear(2 * dim, nb_units))

        append_activation(activation, layers)
        for _ in range(nb_layers - 1):
            layers.append(nn.Linear(nb_units, nb_units))
            append_activation(activation, layers)
        layers.append(nn.Linear(nb_units, 2 * dim, bias=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, t):
        '''
        inputs: x or [x, t]
        '''
        if self.time_augment:
            inputs = torch.cat([x, t], dim=1)
        else:
            inputs = x
        return self.layers(inputs)

class HamiltionEquation(nn.Module):
    def __init__(self, dim=1, time_augment=False,
                 nb_units=1000, nb_layers=1, activation='tanh'):
        super(HamiltionEquation, self).__init__()
        self.time_augment = time_augment
        self.dim = dim
        self.units = nb_units

        q_layers = []
        p_layers = []
        if self.time_augment:
            q_layers.append(nn.Linear(self.dim + 1, self.units))
            p_layers.append(nn.Linear(self.dim + 1, self.units))
        else:
            q_layers.append(nn.Linear(self.dim, self.units))
            p_layers.append(nn.Linear(self.dim, self.units))
        append_activation(activation, q_layers)
        append_activation(activation, p_layers)
        for _ in range(nb_layers - 1):
            q_layers.append(nn.Linear(self.units, self.units))
            p_layers.append(nn.Linear(self.units, self.units))
            append_activation(activation, q_layers)
            append_activation(activation, p_layers)
        
        q_layers.append(nn.Linear(self.units, 1, bias=False))
        p_layers.append(nn.Linear(self.units, 1, bias=False))
        
        self.q_layers, self.p_layers = nn.Sequential(*q_layers), nn.Sequential(*p_layers)

    def forward(self, x, t):
        '''
        inputs: x or [x, t]
        '''
    
        if self.time_augment:
            raise NotImplementedError
            q_inputs = torch.cat([x[:, :self.dim], t], dim=1).requires_grad_()
            p_inputs = torch.cat([x[:, self.dim:], t], dim=1).requires_grad_()
        else:
            q_inputs = x[:, :self.dim].requires_grad_()
            p_inputs = x[:, self.dim:].requires_grad_()

        v = self.q_layers(q_inputs).sum()
        # pdb.set_trace()
        k = self.p_layers(p_inputs).sum()
        dq = torch.autograd.grad(k, p_inputs, create_graph=True)[0]
        dp = - torch.autograd.grad(v, q_inputs, create_graph=True)[0]
        
        return torch.cat([dq, dp], dim=1)

class ODENetwork(nn.Module):
    def __init__(self, nb_object=1, nb_coords=2, function_type='ode', time_augment=False,
                 nb_units=1000, nb_layers=1, activation='tanh', 
                 lambda_trs=0.0, learning_rate=2e-4, args=None, device=None):
        super(ODENetwork, self).__init__()
        self.dim = int(nb_object * nb_coords)
        if function_type == 'gnn':
            self.dim = nb_coords
        latent_dim = nb_coords * 2
        self.augment = time_augment
        self.nb_object = nb_object
        self.nb_coords = nb_coords
        self.args = args
        self.device = device
        
        self.lambda_trs = lambda_trs
        self.lr = learning_rate
        self.function_type = function_type

        if function_type == 'ode':
            self.func = ODEFunc(self.dim , time_augment, nb_units, nb_layers, activation)
        elif function_type == 'hamiltonian':
            self.func = HamiltionEquation(self.dim , time_augment, nb_units, nb_layers, activation)
        elif function_type == 'gnn':
            assert time_augment == False
            if args.augment_dim > 0:
                raise NotImplementedError
            else:
                ode_input_dim = latent_dim

            ode_func_net = GNN(in_dim=ode_input_dim, n_hid=args.ode_dims, out_dim=ode_input_dim,
                n_heads=args.n_heads, n_layers=args.gen_layers, dropout=args.dropout,
                conv_name = args.odenet, aggregate="add")
            
            self.func = GraphODEFuncT(ode_func_net=ode_func_net)
            self.rel_rec, self.rel_send = self.compute_rec_send()
            
        else:
            raise NotImplementedError

    def encode_onehot(self,labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot
    
    def compute_rec_send(self):
        off_diag = np.ones([self.nb_object, self.nb_object]) - np.eye(self.nb_object)
        rel_rec = np.array(self.encode_onehot(np.where(off_diag)[0]),
                           dtype=np.float32)  # every node as one-hot[10000], (20,5)
        rel_send = np.array(self.encode_onehot(np.where(off_diag)[1]), dtype=np.float32)  # every node as one-hot,(20,5)
        rel_rec = torch.FloatTensor(rel_rec).to(self.device)
        rel_send = torch.FloatTensor(rel_send).to(self.device)

        return rel_rec, rel_send
    
    def solve(self, ts, x0, graph=None):
        '''
        ts: batch_de["time_steps"], shape [n_timepoints] 
        x0: expected [batchsize, n x spatial dim x 2] is not gnn else [batchsize, n, d]
        '''

        if self.function_type == 'gnn':
            assert graph is not None

            rel_type_onehot = torch.FloatTensor(x0.size(0), self.rel_rec.size(0),
                                                    self.args.edge_types).to(self.device)  # [b,20,2]
            rel_type_onehot.zero_()
            # pdb.set_trace()
            rel_type_onehot.scatter_(2, graph.view(x0.size(0), -1, 1), 1) 

            self.func.set_graph(rel_type_onehot,self.rel_rec,self.rel_send,self.args.edge_types)
        
        if ts[0] != 0:
            ts = torch.cat([torch.zeros(1), ts])
        dts = ts[1:] - ts[:-1]
        x = x0
        if self.function_type == 'gnn':
            xr = torch.cat([x0[:, :, :self.dim], -x0[:, :, self.dim:]], dim=-1)
        else:
            xr = torch.cat([x0[:, :self.dim], -x0[:, self.dim:]], dim=1)

        ls_x, ls_xr = [], []

        for i in range(ts.shape[0]-1):
            t = ts[i].repeat(x.shape[0], 1)
            tr = -t
            dt = dts[i]

            if self.augment:
                dx1 = self.func(x, t) * dt
                dx2 = self.func(x + 0.5 * dx1, t + 0.5 * dt) * dt
                dx3 = self.func(x + 0.5 * dx2, t + 0.5 * dt) * dt
                dx4 = self.func(x + dx3, t + dt) * dt
                dx = (1 / 6) * (dx1 + 2 * dx2 + 2 * dx3 + dx4)
                x = x + dx
                ls_x.append(x)

                dxr1 = - self.func(xr, tr) * dt
                dxr2 = - self.func(xr + 0.5 * dxr1, tr - 0.5 * dt) * dt
                dxr3 = - self.func(xr + 0.5 * dxr2, tr - 0.5 * dt) * dt
                dxr4 = - self.func(xr + dxr3, tr - dt) * dt
                dxr = (1 / 6) * (dxr1 + 2 * dxr2 + 2 * dxr3 + dxr4)
                xr = xr + dxr
                ls_xr.append(xr)
            else:  # Leapfrog solver for autonomous systems
                # Forward time evolution
                if self.function_type == 'gnn':
                    q, p = x[:, :, :self.dim], x[:, :, self.dim:]
                    p = p + 0.5 * dt * self.func(x, t)[:, :, self.dim:] 
                    q = q + 1.0 * dt * self.func(torch.cat([q, p], dim=-1), t)[:, :, :self.dim] 
                    p = p + 0.5 * dt * self.func(torch.cat([q, p], dim=-1), t)[:, :, self.dim:] 

                    # Backward time evolution
                    qr, pr = xr[:, :, :self.dim], xr[:, :, self.dim:]
                    pr = pr - 0.5 * dt * self.func(xr, t)[:, :, self.dim:]
                    qr = qr - 1.0 * dt * self.func(torch.cat([qr, pr], dim=-1), t)[:, :, :self.dim] 
                    pr = pr - 0.5 * dt * self.func(torch.cat([qr, pr], dim=-1), t)[:, :, self.dim:] 
                else:
                    q, p = x[:, :self.dim], x[:, self.dim:]
                    p = p + 0.5 * dt * self.func(x, t)[:, self.dim:] 
                    q = q + 1.0 * dt * self.func(torch.cat([q, p], dim=-1), t)[:, :self.dim] 
                    p = p + 0.5 * dt * self.func(torch.cat([q, p], dim=-1), t)[:, self.dim:] 

                    # Backward time evolution
                    qr, pr = xr[:, :self.dim], xr[:, self.dim:]
                    pr = pr - 0.5 * dt * self.func(xr, t)[:, self.dim:]
                    qr = qr - 1.0 * dt * self.func(torch.cat([qr, pr], dim=-1), t)[:, :self.dim] 
                    pr = pr - 0.5 * dt * self.func(torch.cat([qr, pr], dim=-1), t)[:, self.dim:] 
                
                x = torch.cat([q, p], dim=-1)
                ls_x.append(x)
                xr = torch.cat([qr, pr], dim=-1)
                ls_xr.append(xr)

        stack_dim = 1 if self.function_type != 'gnn' else 2
        return torch.stack(ls_x, dim=stack_dim), torch.stack(ls_xr, dim=stack_dim)

    def first_point_imputation(self, batch_enc, batch_dec):
        '''
        batch_enc["data"] : [b x n_objects, D]
        batch_enc["time_steps"] : [b x n_objects]

        batch_dec["data"] : [b x n_objects, T, D]
        batch_dec["time_steps"] : [b x n_objects, T]
        '''
        dec_indices_orig_init_value = (batch_dec["time_first"] == 0).nonzero(as_tuple=False).squeeze()
        # pdb.set_trace()
        time_intervals = (batch_dec["time_first"] - batch_enc["time_steps"]).unsqueeze(1)
        computed_init_states = batch_enc["data"]*batch_dec["time_first"].unsqueeze(1)/time_intervals - batch_dec["data"][:,0,:]*batch_enc["time_steps"].unsqueeze(1)/time_intervals 
        computed_init_states[dec_indices_orig_init_value] = batch_dec["data"][dec_indices_orig_init_value, 0, :] 

        return computed_init_states

    def compute_loss(self, batch_enc, batch_dec, graph=None):
        init_states = self.first_point_imputation(batch_enc, batch_dec) #b x n_objects, D
        b = init_states.shape[0] // self.nb_object
        
        ts = batch_dec["time_steps"].to(self.device)
        padding = False
        if ts[0] != 0:
            ts = torch.cat([torch.zeros(1).to(self.device), ts]).to(self.device)
            padding = True
        # pdb.set_trace()
        if self.function_type != 'gnn':
            q = init_states[:, :self.nb_coords].reshape(b, self.nb_object, -1).reshape(b, -1) # [b, N x D//2]
            p = init_states[:, self.nb_coords:].reshape(b, self.nb_object, -1).reshape(b, -1)
            
            x0 = torch.cat([q, p], dim=-1)

            
            X, Xr = self.solve(ts, x0) # [B, T, N x d x 2]

            T, D = X.shape[1], X.shape[2] // 2
            Xq, Xp = X[:,:,:D], X[:,:,D:]
            Xrq, Xrp = Xr[:,:,:D], Xr[:,:,D:]
            
            Xq = Xq.reshape(b, T, self.nb_object, self.nb_coords).permute(0, 2, 1, 3).reshape(-1, T, self.nb_coords)
            Xp = Xp.reshape(b, T, self.nb_object, self.nb_coords).permute(0, 2, 1, 3).reshape(-1, T, self.nb_coords)
            Xrq = Xrq.reshape(b, T, self.nb_object, self.nb_coords).permute(0, 2, 1, 3).reshape(-1, T, self.nb_coords)
            Xrp = Xrp.reshape(b, T, self.nb_object, self.nb_coords).permute(0, 2, 1, 3).reshape(-1, T, self.nb_coords)
            
        else:
            x0 = init_states.reshape(b, self.nb_object, -1)  

            X, Xr = self.solve(ts, x0, graph) # [B, N, T, d x 2]
            T, D = X.shape[2], X.shape[3]
            X, Xr = X.reshape(init_states.shape[0], T, D), Xr.reshape(init_states.shape[0], T, D) # [B x N, T, d x 2]
            
            Xq, Xp = X[:,:,:D], X[:,:,D:]
            Xrq, Xrp = Xr[:,:,:D], Xr[:,:,D:]


        if padding:
            mask = batch_dec["mask"]
        else:
            mask = batch_dec["mask"][:, 1:, :]
            batch_dec["data"] = batch_dec["data"][:, 1:, :]
            
        # X = torch.cat([Xq, Xp], dim=2)
        timelength_per_nodes = torch.sum(mask.permute(0,2,1),dim=2)

        # mask = mask.reshape(b, self.nb_object, T, -1).permute(0,2,1,3).reshape(b, T, -1)
        # pdb.set_trace()
        forward_diff = torch.square(torch.cat([Xq, Xp], dim=2) - batch_dec["data"]) * mask
        forward_diff = forward_diff.sum(dim=1) / timelength_per_nodes
        l_ode = torch.mean(forward_diff)

        fr_diff = torch.square(torch.cat([Xq, -Xp], dim=2) - torch.cat([Xrq, Xrp], dim=2)) * mask
        fr_diff = fr_diff.sum(dim=1) / timelength_per_nodes
        l_trs = torch.mean(fr_diff)

        l = l_ode + self.lambda_trs * l_trs
        
        #sum over time dim
        forward_mape = torch.abs(torch.cat([Xq, Xp], dim=2) - batch_dec["data"]) / torch.maximum(1e-9*torch.ones_like(batch_dec["data"]), torch.abs(batch_dec["data"])) 
        forward_mape = torch.sum(forward_mape * mask, dim=1)
        # pdb.set_trace()
        # forward_mape = forward_mape / torch.sum(torch.abs(batch_dec["data"]) * mask, dim=1)
        forward_mape = forward_mape / timelength_per_nodes
        forward_mape = torch.mean(forward_mape)

        results = {}
        results["loss"] = l
        results["mse"] = l_ode.data.item() + l_trs.data.item()
        results["forward_gt_mse"] = l_ode.data.item()
        results["reverse_f_mse"] = l_trs.data.item()
        results["mape"] = forward_mape.data.item()

        return results, batch_dec["data"], torch.cat([Xq, Xp], dim=2)

        





        
