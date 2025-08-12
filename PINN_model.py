import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import utils.helper_functions_PINN as hf_PINN
import os
import utils.fun as fun

class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
class DNN(nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f"layer{i}", nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.net.add_module(f"tanh{i}", nn.Tanh())
            else:
                self.net.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

class PINN:

    def __init__(self, N_colloc, filename, data_loss_batch, hidden_layers):

        self.N_colloc = N_colloc
        self.filename = filename
        self.data_loss_batch = data_loss_batch
        self.hidden_layers = hidden_layers
        self.model = []
        self.optimizer = []
        self.dataset = []
        self.dataloader = []
        self.loss_data = []
        self.X_norm = []
        self.Y_norm = []
        self.X_min = []
        self.X_max = []
        self.Y_min = []
        self.Y_max =  []
        self.param_combinations = []
        self.x_min = 0
        self.x_max = 0


    def create_normalized_data(self, filename):
        df = pd.read_csv(os.path.join("Input", filename))
        input_list = []
        output_list = []
        N_x = 201
        N_D = 20
        for idx, row in df.iterrows():
            # Inputs
            dV_ges = float(row["dV_ges"]) / 3.6 * 1e-6
            eps_0 = float(row["eps_0"])
            phi_0 = float(row["phi_0"])
            x_arr = np.array([float(v) for v in row["x"].split(",")])                # shape (N_x,)
            # Outputs
            V_dis_arr = np.array([float(v) for v in row["V_dis"].split(",")])        # (N_x,)
            V_c_arr   = np.array([float(v) for v in row["V_c"].split(",")])          # (N_x,)
            phi32_arr = np.array([float(v) for v in row["phi_32"].split(",")])       # (N_x,)
            N_arrs    = [np.array([float(v) for v in row[f"N_{j}"].split(",")]) for j in range(N_D)]  # list of (N_x,)

            for i in range(N_x):
                inp = [dV_ges, eps_0, phi_0, x_arr[i]]
                out = [V_dis_arr[i], V_c_arr[i], phi32_arr[i]] + [N_arrs[j][i] for j in range(N_D)]
                input_list.append(inp)
                output_list.append(out)

        X = np.array(input_list, dtype=np.float32)
        Y = np.array(output_list, dtype=np.float32)
        Y = np.maximum(Y, 0)  # Set negative values to 0

        # Remove all-zero output columns automatically
        # col_nonzero = np.any(Y != 0, axis=0)
        # Y = Y[:, col_nonzero]
        # kept_output_cols = np.where(col_nonzero)[0]  # Save mapping for later

        # Min-max normalization for inputs
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)

        # Min-max normalization for outputs
        Y_min, Y_max = Y.min(axis=0), Y.max(axis=0)
        Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-8)

        # Build DataFrame for parameter extraction
        params_df = pd.DataFrame(input_list, columns=["dV_ges", "eps_0", "phi_0", "x"])
        param_combinations = params_df.drop("x", axis=1).drop_duplicates().values
        x_min = params_df["x"].min()
        x_max = params_df["x"].max()

        return X_norm, Y_norm, X_min, X_max, Y_min, Y_max, param_combinations, x_min, x_max

    def denormalize(self, Y_norm, Y_min, Y_max):
        return Y_norm * (Y_max - Y_min + 1e-8) + Y_min

    def normalize_inputs(self, inputs, X_min, X_max):
        return (inputs - X_min) / (X_max - X_min + 1e-8)
    
    def get_collocation_points(self, param_combinations, x_min, x_max, N_colloc=402):
        collocation_points = []
        for params in param_combinations:
            dV_ges, eps_0, phi_0 = params
            x_vals = np.linspace(x_min, x_max, N_colloc)
            for x in x_vals:
                collocation_points.append([dV_ges, eps_0, phi_0, x])
        return np.array(collocation_points)

    def get_boundary_points(self, param_combinations, x_min):
        boundary_points = []
        for params in param_combinations:
            dV_ges, eps_0, phi_0 = params
            boundary_points.append([dV_ges, eps_0, phi_0, x_min])
        return np.array(boundary_points)

    def _pos(self, x, eps=1e-12):
        # strictly ≥ 0, smooth near 0; add tiny eps to avoid divide-by-zero downstream
        return F.softplus(x) + eps
    
    def pde_loss_GPT(self, model, X_norm, x_phys, Y_min, Y_max, X_min, X_max, x_min, x_max, idx_colloc, Set, Sub, debug=False):
        X_norm.requires_grad_(True)
        Y_pred_norm = model(X_norm)
        Y_pred = self.denormalize(Y_pred_norm, torch.tensor(Y_min, device=Y_pred_norm.device, dtype=Y_pred_norm.dtype),
                                        torch.tensor(Y_max, device=Y_pred_norm.device, dtype=Y_pred_norm.dtype))
        X_in = self.denormalize(X_norm, torch.tensor(X_min, device=X_norm.device, dtype=X_norm.dtype),
                                torch.tensor(X_max, device=X_norm.device, dtype=X_norm.dtype))

        V_dis, V_c, phi_32 = Y_pred[:,0], Y_pred[:,1], Y_pred[:,2]
        N_j = Y_pred[:,3:]
        device, dtype = V_dis.device, V_dis.dtype
        B, N_d = N_j.shape


        # Per-sample grads (see point 10)
        dxnorm_dxphys = 1.0 / (x_max - x_min + 1e-8)
        def grad_wrt_x(y):
            g=[]
            for i in range(y.shape[0]):
                gi = torch.autograd.grad(y[i], X_norm, retain_graph=True, create_graph=True)[0][i,3]
                g.append(gi)
            return torch.stack(g)*dxnorm_dxphys
        
        # enforce non-negativity for physics & compute grads on these transformed vars
        V_dis = self._pos(V_dis); V_c = self._pos(V_c); phi_32 = self._pos(phi_32); N_j = self._pos(N_j)
        
        dV_dis_dx  = grad_wrt_x(V_dis)
        dV_c_dx    = grad_wrt_x(V_c)
        dphi32_dx  = grad_wrt_x(phi_32)
        dN_dx      = torch.stack([grad_wrt_x(N_j[:,j]) for j in range(N_j.shape[1])], dim=1)

        # Prepare NumPy inputs per sample & call physics
        u_dis_list=[]; u_c_list=[]; du_dis_list=[]; du_c_list=[]
        dV_coal_list=[]; dV_sed_list=[]; tau_dd_list=[]; S32_list=[]; h_c_list=[]; v_sj_list=[]
        eps_p = Sub.eps_p

        X_in_np  = X_in.detach().cpu().numpy()
        V_dis_np = V_dis.detach().cpu().numpy()
        V_c_np   = V_c.detach().cpu().numpy()
        phi_np   = phi_32.detach().cpu().numpy()
        N_np     = N_j.detach().cpu().numpy()
        x_np     = x_phys.detach().cpu().numpy()
        

        for i in range(len(idx_colloc)):
            sim = hf_PINN.make_sim_instance(Set, Sub)  # see helper snippet below
            sim.Sub.dV_ges = float(X_in_np[i,0]); sim.Sub.eps_0 = float(X_in_np[i,1]); sim.Sub.phi_0 = float(X_in_np[i,2])
            _, _, sim.d_j, _ = fun.initialize_boundary_conditions(sim.Sub.eps_0, sim.Sub.phi_0, 2.5*sim.Sub.phi_0, 'Output', sim.Set.N_D, plot=False)
            (u_dis_i, u_c_i, du_dis_i, du_c_i,
            dV_coal_i, dV_sed_i, tau_dd_i, S32_i, h_c_i, v_sj_i) = sim.get_terms(
                np.array([V_dis_np[i]]), np.array([V_c_np[i]]),
                np.array([phi_np[i]]),  np.array([N_np[i,:]]),
                np.array([x_np[i]]))

            # Scalars / length-1 arrays -> float
            u_dis_list.append( float(np.atleast_1d(u_dis_i)[0]) )
            u_c_list.append(   float(np.atleast_1d(u_c_i)[0]) )
            du_dis_list.append(float(np.atleast_1d(du_dis_i)[0]) )
            du_c_list.append( float(np.atleast_1d(du_c_i)[0]) )
            dV_coal_list.append(float(np.atleast_1d(dV_coal_i)[0]))
            dV_sed_list.append( float(np.atleast_1d(dV_sed_i)[0]) )
            tau_dd_list.append( float(np.atleast_1d(tau_dd_i)[0]) )
            S32_list.append(    float(np.atleast_1d(S32_i)[0]) )
            h_c_list.append(    float(np.atleast_1d(h_c_i)[0]) )

            # Sedimentation velocity per class -> 1D (N_d,)
            v_sj_1d = np.asarray(v_sj_i).reshape(-1)  # from (N_d,1) or (N_d,) to (N_d,)
            v_sj_list.append(v_sj_1d)

        # Convert back to tensors
        u_dis     = torch.tensor(u_dis_list, device=device, dtype=dtype)      # (B,)
        u_c       = torch.tensor(u_c_list,   device=device, dtype=dtype)      # (B,)
        du_dis_dx = torch.tensor(du_dis_list,device=device, dtype=dtype)      # (B,)
        du_c_dx   = torch.tensor(du_c_list,  device=device, dtype=dtype)      # (B,)
        dV_coal   = torch.tensor(dV_coal_list, device=device, dtype=dtype)    # (B,)
        dV_sed    = torch.tensor(dV_sed_list,  device=device, dtype=dtype)    # (B,)
        tau_dd    = torch.tensor(tau_dd_list,  device=device, dtype=dtype)    # (B,)
        S_32      = torch.tensor(S32_list,     device=device, dtype=dtype)    # (B,)
        h_c       = torch.tensor(h_c_list,     device=device, dtype=dtype)    # (B,)
        v_sj      = torch.tensor(np.stack(v_sj_list, axis=0), device=device, dtype=dtype)  # (B, N_d)


        if debug:
            print("---- pde_loss_GPT debug shapes ----")
            print("B =", B, ", N_d =", N_d)
            print("u_dis     :", tuple(u_dis.shape))
            print("u_c       :", tuple(u_c.shape))
            print("du_dis_dx :", tuple(du_dis_dx.shape))
            print("du_c_dx   :", tuple(du_c_dx.shape))
            print("dV_coal   :", tuple(dV_coal.shape))
            print("dV_sed    :", tuple(dV_sed.shape))
            print("tau_dd    :", tuple(tau_dd.shape))
            print("S_32      :", tuple(S_32.shape))
            print("h_c       :", tuple(h_c.shape))
            print("N_j       :", tuple(N_j.shape))
            print("dN_dx     :", tuple(dN_dx.shape))
            print("v_sj      :", tuple(v_sj.shape))
            # assertions: all scalars 1D (B,), class arrays (B,N_d)
            assert u_dis.ndim==1 and u_dis.shape[0]==B
            assert u_c.ndim==1   and u_c.shape[0]==B
            assert du_dis_dx.ndim==1 and du_dis_dx.shape[0]==B
            assert du_c_dx.ndim==1   and du_c_dx.shape[0]==B
            assert dV_coal.ndim==1 and dV_coal.shape[0]==B
            assert dV_sed.ndim==1  and dV_sed.shape[0]==B
            assert tau_dd.ndim==1  and tau_dd.shape[0]==B
            assert S_32.ndim==1    and S_32.shape[0]==B
            assert h_c.ndim==1     and h_c.shape[0]==B
            assert N_j.shape == (B, N_d)
            assert dN_dx.shape == (B, N_d)
            assert v_sj.shape == (B, N_d), f"Expected (B, N_d) got {v_sj.shape}"
            print("All shapes look good ✅")
            print("-----------------------------------")

        # Residuals
        eq1 = -(u_dis * dV_dis_dx + V_dis * du_dis_dx) - dV_coal + (1/eps_p)*dV_sed
        eq2 = -(u_c   * dV_c_dx   + V_c   * du_c_dx)   + (1-eps_p)*dV_coal - (1/eps_p)*dV_sed
        eq3 = -(u_dis * dphi32_dx + phi_32 * du_dis_dx) + phi_32/(6*tau_dd) + S_32

        # eq4: watch orientation (we made v_sj as [N, N_d])
        eq4 = -(u_c.unsqueeze(1) * dN_dx + N_j * du_c_dx.unsqueeze(1)) - N_j * (v_sj / h_c.unsqueeze(1))

        res_loss = eq1.pow(2).mean() + eq2.pow(2).mean() + eq3.pow(2).mean() + eq4.pow(2).mean()
        return res_loss




    def boundary_loss_GPT(self, model, X_bc_norm, X_min, X_max, Y_min, Y_max, Set):
        Y_pred_norm = model(X_bc_norm)
        Y_pred = self.denormalize(Y_pred_norm,
                        torch.tensor(Y_min, device=Y_pred_norm.device, dtype=Y_pred_norm.dtype),
                        torch.tensor(Y_max, device=Y_pred_norm.device, dtype=Y_pred_norm.dtype))
        Y_pred = self._pos(Y_pred)  # make all components ≥ 0 before comparing to BC targets
        V_dis_0, V_c_0 = hf_PINN.get_parameters_boundary(Set)
        X_bc = self.denormalize(X_bc_norm, torch.tensor(X_min), torch.tensor(X_max))
        eps_0 = X_bc[:,1].numpy()
        phi_0 = X_bc[:,2]
        N_j_0 = np.zeros((len(eps_0), Set.N_D), dtype=np.float32)
        for i in range(len(eps_0)):
            _, n_in, d_in, _ = fun.initialize_boundary_conditions(eps_0[i], float(phi_0[i].item()), 2.5*float(phi_0[i].item()), 'Output', Set.N_D, plot=False)
            factor = (eps_0[i] * V_c_0) / ((np.pi/6)*np.sum(n_in * d_in**3) + 1e-12)
            Nj = (factor * np.round(n_in, decimals=0)).astype(np.float32)
            N_j_0[i,:] = Nj[:Set.N_D]  # truncate if needed
        N_j_0 = torch.from_numpy(N_j_0).to(Y_pred.device, Y_pred.dtype)
        
        bc_target = torch.zeros_like(Y_pred)
        bc_target[:,0] = V_dis_0  # V_dis(x=0)
        bc_target[:,1] = V_c_0    # V_c(x=0)
        bc_target[:,2] = phi_0    # phi_32(x=0)
        bc_target[:,3:] = N_j_0   # N_j(x=0)

        bc_loss = ((Y_pred[:,:3] - bc_target[:,:3])**2).mean() + ((Y_pred[:,3:] - bc_target[:,3:])**2).mean()
        return bc_loss
    
    def create_model(self):
        self.X_norm, self.Y_norm, self.X_min, self.X_max, self.Y_min, self.Y_max, self.param_combinations, self.x_min, self.x_max = self.create_normalized_data(self.filename)
        layers = [self.X_norm.shape[1], *self.hidden_layers, self.Y_norm.shape[1]]
        self.model = DNN(layers)
        print('\nDNN model created succesfully. Structure: ', layers, '\n')
    
    def pre_training(self, epochs, lr=1e-3):
        self.dataset = MyDataset(self.X_norm, self.Y_norm)
        self.dataloader = DataLoader(self.dataset, batch_size=self.data_loss_batch, shuffle=True)
        self.loss_data = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        print('Start of DNN model training.')
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in self.dataloader:
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.loss_data(preds, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * xb.size(0)
            print(f"[DNN] Epoch {epoch+1:05d}/{epochs:05d}, Loss: {total_loss/len(self.dataset):.6e}")

    def training(self, epochs, lr=5e-4):
        # Build all collocation and boundary points
        collocation_points = self.get_collocation_points(self.param_combinations, self.x_min, self.x_max, self.N_colloc)
        boundary_points = self.get_boundary_points(self.param_combinations, self.x_min)
        # Build normalized model inputs for pde & bc loss
        X_in_norm = torch.from_numpy(self.normalize_inputs(collocation_points, self.X_min, self.X_max)).float()
        # X_in_norm.clamp_(min=0)
        x_in = torch.from_numpy(collocation_points[:, 3]).float()
        X_bc_norm = torch.from_numpy(self.normalize_inputs(boundary_points, self.X_min, self.X_max)).float()

        ode_loss_batch = self.data_loss_batch # Same batch for conistency of training in epoch loop
        bc_loss_batch = min(len(X_bc_norm), self.data_loss_batch) # If parameter combinations fewer than data batch, use all boundary points per batch. If you have more, randomly sample 32.

        Set, Sub = hf_PINN.sim_init()
        self.optimizer.param_groups[0]['lr'] = lr

        self.model.train()
        print('Start of PINN model training.')
        for epoch in range(epochs):
            total, tot_d, tot_b, tot_o, step = 0.0, 0.0, 0.0, 0.0, 0.0
            for xb, yb in self.dataloader:
                self.optimizer.zero_grad()
                preds = self.model(xb)
                data_loss = self.loss_data(preds, yb)
                # Randomly sample collocation and BC points for this batch
                idx_bc = torch.randint(0, X_bc_norm.shape[0], (bc_loss_batch,))
                bc_loss  = self.boundary_loss_GPT(self.model, X_bc_norm[idx_bc], self.X_min, self.X_max, self.Y_min, self.Y_max, Set)
                idx_colloc = torch.randint(0, X_in_norm.shape[0], (ode_loss_batch,))               
                ode_loss = self.pde_loss_GPT(self.model, X_in_norm[idx_colloc], x_in[idx_colloc], self.Y_min, self.Y_max, self.X_min, self.X_max, self.x_min, self.x_max, idx_colloc, Set, Sub)
                loss = data_loss + ode_loss + bc_loss
                loss.backward()
                self.optimizer.step()
                step += 1
                tot_d  += data_loss.item()
                tot_b  += bc_loss.item()
                tot_o  += ode_loss.item()
                total  += loss.item()
            print(f"[PINN] Epoch {epoch+1:05d}/{epochs:05d}, total={total/step:.3e} | data={tot_d/step:.3e} | bc={tot_b/step:.3e} | ode={tot_o/step:.3e}")
