import numpy as np
from scipy.optimize import newton
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import utils.utils_pinn.sim_parameters as sp
import utils.utils_pinn.sim_model_PINN as mm # mm for mechanistic model
import utils.utils_pinn.fun as fun

# Funktion berechnet Höhe eines Kreissegments auf Basis des Kreisradius r und der Fläche A
def getHeight(A, r):
    eq = lambda h: A - r**2 * np.arccos(1 - h / r) + (r - h) * np.sqrt(2 * r * h - h**2)
    h0 = r / 2
    if A < 0:
        #print('Querschnitt kleiner Null: ' + str(A))
        return 0
    elif A > np.pi * r**2:
        #print('Querschnitt größer als zulässig: ' + str(A))
        return 2*r
    return newton(eq, h0)

def getHeightArray(A, r):
    h = np.zeros_like(A)
    for i in range(len(h)):
        h[i] = getHeight(A[i], r)
    return h

# Funktion berechnet die Fläche eines Kreissegments auf Basis des Kreisradiuses r und der Höhe h des Segments
def getArea(h, r):
    return r**2 * np.arccos(1 - h / r) - (r - h) * np.sqrt(2 * r * h - h**2)


def predict_outputs_np(
    model,
    dV_ges, eps_0, phi_0, x,          # scalars except x may be scalar or array
    X_min, X_max, Y_min, Y_max,       # arrays from create_normalized_data
    clamp_inputs=True,
    ensure_nonneg=False
):
    import numpy as np
    import torch

    # 0) detect scalar vs vector x
    is_scalar = (np.ndim(x) == 0)
    x_vec = np.array([x], dtype=np.float32) if is_scalar else np.asarray(x, dtype=np.float32).reshape(-1)
    N = x_vec.shape[0]

    # 1) inputs in physical units
    dV_si = dV_ges / 3.6 * 1e-6  # m^3/s

    # **No ragged array — build (N,4) properly**
    X_dim = np.column_stack((
        np.full(N, dV_si, dtype=np.float32),
        np.full(N, eps_0, dtype=np.float32),
        np.full(N, phi_0, dtype=np.float32),
        x_vec
    ))  # shape: (N, 4)

    # 2) normalize
    X_min = np.asarray(X_min, dtype=np.float32)
    X_max = np.asarray(X_max, dtype=np.float32)
    Y_min = np.asarray(Y_min, dtype=np.float32)
    Y_max = np.asarray(Y_max, dtype=np.float32)

    X_norm = (X_dim - X_min) / (X_max - X_min + 1e-8)
    if clamp_inputs:
        X_norm = np.clip(X_norm, 0.0, 1.0)

    # 3) model forward (batch)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X_norm).to(device).float()   # (N, 4)
        y_norm = model(x_t).cpu().numpy()                   # (N, n_outputs)

    # 4) denormalize
    Y = y_norm * (Y_max - Y_min + 1e-8) + Y_min            # (N, n_outputs)
    if ensure_nonneg:
        Y = np.clip(Y, 0.0, None)

    # 5) return format
    return Y[0] if is_scalar else [Y[i].copy() for i in range(N)]



def sim_init():
    Set = sp.Settings(N_x=201, L=1.3, D=0.2, h_c_0=0.1, h_dis_0=0.03, N_D=20)
    SubSys = sp.Substance_System()
    SubSys.update('in_silico_substance.xlsx')
    return Set, SubSys

def make_sim_instance(Set, Sub):
    return mm.input_simulation(Set, Sub)


def get_parameters_boundary(Set):
    D = Set.D
    dl = Set.dl
    A = Set.A
    h_c_0 = 0.1 + 0.03
    h_dis_0 = 0.001
    h_d_0 = D - h_c_0 - h_dis_0
    A_d_0 = getArea(h_d_0, D/2)
    A_c_0 = getArea(h_c_0, D/2)
    A_dis_0 = A - A_d_0 - A_c_0
    V_dis_0 = A_dis_0 * dl
    V_c_0 = A_c_0 * dl
    return V_dis_0, V_c_0


# ================================= Sanity check for physics terms at ONE sample (INIT) ==================================


@torch.no_grad()
def sanity_physics_terms(model, Set, Sub, X_min, X_max, Y_min, Y_max, dV_ges_raw, eps_0, phi_0, x_val):
    """
    dV_ges_raw : same units as your CSV (L/h). We convert to SI here.
    eps_0, phi_0, x_val: scalars
    """

    # --- 1) Build the normalized input exactly like training ---
    dV_si = dV_ges_raw / 3.6 * 1e-6  # m^3/s
    x_dim = np.array([[dV_si, eps_0, phi_0, x_val]], dtype=np.float32)
    x_norm = (x_dim - X_min) / (X_max - X_min + 1e-8)
    x_norm = np.clip(x_norm, 0.0, None)

    model.eval()
    x_t = torch.from_numpy(x_norm).float()
    y_pred_norm = model(x_t)
    # denormalize to physical units
    Ymin_t = torch.tensor(Y_min, dtype=y_pred_norm.dtype)
    Ymax_t = torch.tensor(Y_max, dtype=y_pred_norm.dtype)
    y_dim = y_pred_norm * (Ymax_t - Ymin_t) + Ymin_t

    # --- 2) enforce non-negativity like in pde_loss/bc_loss ---
    def _pos(t, eps=1e-12):
    # same transform you used in losses to keep outputs >= 0
        return F.softplus(t) + eps
    
    y_dim = _pos(y_dim)

    # unpack outputs (B=1)
    V_dis  = float(y_dim[0, 0].item())
    V_c    = float(y_dim[0, 1].item())
    phi_32 = float(y_dim[0, 2].item())
    N_j    = y_dim[0, 3:].cpu().numpy().reshape(-1)  # (N_d,)

    # --- 3) Build mechanistic sim for THIS sample ---
    sim = make_sim_instance(Set, Sub)
    sim.Sub.dV_ges = float(dV_si)
    sim.Sub.eps_0  = float(eps_0)
    sim.Sub.phi_0  = float(phi_0)

    # set droplet class diameters for this sample
    _, _, sim.d_j, _ = fun.initialize_boundary_conditions(
        sim.Sub.eps_0, sim.Sub.phi_0, 2.5*sim.Sub.phi_0, 'Output', sim.Set.N_D, plot=False
    )

    # --- 4) Compute physics terms at this point ---
    (u_dis, u_c, du_dis_dx, du_c_dx,
     dV_coal, dV_sed, tau_dd, S32, h_c, v_sj) = sim.get_terms(
        np.array([V_dis]), np.array([V_c]),
        np.array([phi_32]), np.array([N_j]),
        np.array([x_val])
    )

    # Coerce to nice shapes for printing
    u_dis     = float(np.atleast_1d(u_dis)[0])
    u_c       = float(np.atleast_1d(u_c)[0])
    du_dis_dx = float(np.atleast_1d(du_dis_dx)[0])
    du_c_dx   = float(np.atleast_1d(du_c_dx)[0])
    dV_coal   = float(np.atleast_1d(dV_coal)[0])
    dV_sed    = float(np.atleast_1d(dV_sed)[0])
    tau_dd    = float(np.atleast_1d(tau_dd)[0])
    S32       = float(np.atleast_1d(S32)[0])
    h_c       = float(np.atleast_1d(h_c)[0])
    v_sj      = np.asarray(v_sj).reshape(-1)  # (N_d,)

    print("\n=== Sanity check @ one point ===")
    print(f"Inputs: dV_ges={dV_ges_raw:.6g} L/h  (SI {dV_si:.3e} m^3/s), eps_0={eps_0:.6g}, phi_0={phi_0:.6g} m, x={x_val:.6g} m")
    print("Model outputs (after Softplus):")
    print(f"  V_dis  = {V_dis:.6e} m^3")
    print(f"  V_c    = {V_c:.6e} m^3")
    print(f"  phi_32 = {phi_32:.6e} m")
    print(f"  N_j[0:5] = {N_j[:5]}")
    print(f"  dl = {sim.Set.dl} m")
    print("Physics terms:")
    print(f"  u_dis     = {u_dis:.6e} m/s")
    print(f"  u_c       = {u_c:.6e} m/s")
    print(f"  du_dis/dx = {du_dis_dx:.6e} 1/s")
    print(f"  du_c/dx   = {du_c_dx:.6e} 1/s")
    print(f"  dV_coal   = {dV_coal:.6e} m^3/s")
    print(f"  dV_sed    = {dV_sed:.6e} m^3/s")
    print(f"  tau_dd    = {tau_dd:.6e} s")
    print(f"  S32       = {S32:.6e} m/s")
    print(f"  h_c       = {h_c:.6e} m")
    print(f"  v_sj shape= {v_sj.shape}  sample: {v_sj[:5]}")
    print("================================\n")

    # return dictionary if you want to inspect programmatically
    return {
        "V_dis": V_dis, "V_c": V_c, "phi_32": phi_32, "N_j": N_j,
        "u_dis": u_dis, "u_c": u_c, "du_dis_dx": du_dis_dx, "du_c_dx": du_c_dx,
        "dV_coal": dV_coal, "dV_sed": dV_sed, "tau_dd": tau_dd, "S32": S32,
        "h_c": h_c, "v_sj": v_sj
    }

# after you built: model, Set, Sub, X_min, X_max, Y_min, Y_max, param_combinations, x_min, x_max
# sanity_physics_terms(model, Set, Sub, X_min, X_max, Y_min, Y_max,
#                      dV_ges_raw=1000,  # L/h
#                      eps_0=0.25,
#                      phi_0=500e-6,    # m
#                      x_val=0.65)      # m

# ================================== Sanity check for physics terms at ONE sample (END) ==================================

