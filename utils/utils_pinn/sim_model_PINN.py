import numpy as np
import utils.utils_pinn.helper_functions as hf
from utils.utils_pinn.helper_functions import getHeightArray


class input_simulation:

    def __init__(self, Settings, Substance_System):

        self.Set = Settings
        self.Sub = Substance_System

        self.V_dis = []
        self.V_d = []
        self.V_c = []
        self.phi_32 = []
        self.N_j = []

        self.u_dis = []
        self.u_d = []
        self.u_c = []
        self.u_0 = 0
        self.d_j = []


    def tau(self, h, d_32, ID, sigma, r_s_star):
        La_mod = (self.Sub.g * self.Sub.delta_rho / sigma) ** 0.6 * d_32 * h**0.2
        ra = 0.5 * d_32 * (1 - (1 - 4.7 / (4.7 + La_mod)) ** 0.5)
        if ID == "d":
            rf = d_32 * 0.3025 * (1 - (4.7 / (4.7 + La_mod))) ** 0.5
        else:
            rf = d_32 * 0.5239 * (1 - (4.7 / (4.7 + La_mod))) ** 0.5
        tau = (7.68* self.Sub.eta_c* (ra ** (7 / 3)/ (self.Sub.H_cd ** (1 / 6) * sigma ** (5 / 6) * rf * r_s_star)))

        return tau

    def henschke_input(self, V_dis, V_c, phi_32, sigma, r_s_star):

        D = self.Set.D
        dl = self.Set.dl
        dV = 0
        tau_di = 9e9  # Koaleszenzzeit hoch gewählt, damit quasi keine stattfindet, wenn V_dis < 0
        tau_dd = tau_di
        h_dis = 0
        V_A = (np.pi * (D**2) / 4)*dl
        V_d = V_A - V_dis - V_c
        
        if phi_32 <= 0:
            phi_32 = self.Sub.phi_0 / 10
        if V_dis > 0:
            h_c = hf.getHeight(V_c / dl, D / 2)
            h_d = hf.getHeight(V_d / dl, D / 2)
            Ay = 2 * dl * (2 * (D / 2) * h_d - h_d**2) ** 0.5
            h_dis = max(D - h_c - h_d , 0.0001)
            tau_di = self.tau(h_dis, phi_32, "I", sigma, r_s_star)
            tau_dd = self.tau(h_dis, phi_32, "d", sigma, r_s_star)
            if (tau_di > 0):
                dV = 2 * Ay * phi_32 / (3 * tau_di * self.Sub.eps_p)
            else:
                dV = 0
            if (tau_dd==0):
                tau_dd = 9e9

        return dV, tau_dd


    def velocities(self, V_dis, V_c, N_j, x):
        dl = self.Set.dl
        eps_0 = self.Sub.eps_0
        eps_p = self.Sub.eps_p
        u_0 = (self.Sub.dV_ges / (np.pi * self.Set.D**2 / 4))
        self.u_0 = u_0
        A_A = self.Set.A
        u_dis = u_0 * (1 - x**2.9) 
        d_j = self.d_j
        A_dis = V_dis / dl
        A_c = V_c / dl
        A_d = A_A - A_c - A_dis
        eps_c = np.sum(N_j * (d_j**3) * (np.pi/6)) / max(V_c, 1e-12)   #check
        du_dis_dx = -u_0 * 2.9 * x **1.9
        du_c_dx = u_0 * np.ones(len(x))        
        u_c = (u_0*A_A*(eps_0-1)+u_dis*A_dis*(1-eps_p))/(A_c*(eps_c-1) + 1e-12)
        u_d = (u_0*A_A - u_dis*A_dis - u_c*A_c) / (A_d + 1e-12)
        du_c_dx = (A_dis*(1-eps_p)/(A_c*(eps_c-1) + 1e-12)) * du_dis_dx
        return u_dis, u_c, u_d, du_dis_dx, du_c_dx
    
    def swarm_sedimenation_velocity(self, V_c, N_j):
        d_j = self.d_j
        v_sed = np.zeros((len(d_j),), dtype=float)   # 1D
        if (V_c > 0):
            eps = np.sum(N_j * (d_j**3) * (np.pi/6)) / max(V_c, 1e-12)
        else:
            eps = self.Sub.eps_0
        for j in range(len(d_j)):
            v_sed[j] = ((self.Sub.g * self.Sub.delta_rho / (18 * self.Sub.eta_c))* (d_j[j] ** 2)* (1 - eps))
        return v_sed # shape (N_d,)

    def sedimentation_rate(self, V_c, N_j):
        d_j = self.d_j
        D = self.Set.D
        dl = self.Set.dl
        V_s = 0
        v_sed = self.swarm_sedimenation_velocity(V_c ,N_j)
        h_c = 1
        if (V_c>0):
            h_c = hf.getHeight(V_c / dl, D / 2)
            V_s = np.sum((N_j * v_sed / h_c) * (np.pi/6) * d_j**3)
        else:
            print('V_c negative!', V_c)
        return V_s
    
    def source_term_32(self, V_dis, V_c, phi_32, N_j):
        D = self.Set.D
        dl = self.Set.dl
        d_j = self.d_j
        S32 = 0
        v_sed = self.swarm_sedimenation_velocity(V_c ,N_j)
        h_c = 1
        if (V_dis>0 and V_c>0):
            h_c = hf.getHeight(V_c / dl, D / 2)
            S32 = ((np.pi/6) * phi_32 / (V_dis * self.Sub.eps_p)) * np.sum(N_j * v_sed * d_j**2 * (d_j - phi_32) / h_c)
        else:
            S32 = 0
        return S32
    
    def h_c_array(self, V_c):
        D = self.Set.D
        dl = self.Set.dl
        h_c_arr = 1
        if (V_c > 0):
            h_c_arr = hf.getHeight(V_c / dl, D / 2)
        else:
            print('V_c negative!')
        return h_c_arr
    
    
    def get_terms(self, V_dis, V_c, phi_32, N_j, x):
        u_dis, u_c, _, du_dis_dx, du_c_dx = self.velocities(V_dis, V_c, N_j, x)
        dV_coal, tau_dd = self.henschke_input(V_dis, V_c, phi_32, self.Sub.sigma, self.Sub.r_s_star)
        dV_sed = self.sedimentation_rate(V_c, N_j)
        S_32 = self.source_term_32(V_dis, V_c, phi_32, N_j)
        h_c = self.h_c_array(V_c)
        v_sj = self.swarm_sedimenation_velocity(V_c, N_j)

        return u_dis, u_c, du_dis_dx, du_c_dx, dV_coal, dV_sed, tau_dd, S_32, h_c, v_sj