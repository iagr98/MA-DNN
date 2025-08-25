import numpy as np
import pandas as pd
import os
from utils.utils_pinn import helper_functions as hf

# Settings-Objekt beinhaltet Simulationsparameter und Abscheidergeometrie


class Settings:

    def __init__(self, N_x=101, L=0.56, D=0.15, h_c_0=0.055, h_dis_0=0.04, N_D=20):
        # simulation time
        self.T = 300
        self.N_t = 2*self.T + 1
        self.t = np.linspace(0, self.T, self.N_t)
        self.dt = self.t[1] - self.t[0]

        # settler geometry
        self.N_x = N_x
        self.L = L
        self.D = D
        self.x = np.linspace(0, self.L, self.N_x)
        self.dl = self.x[1] - self.x[0]
        self.A = np.pi / 4 * self.D**2
        self.delta_V = self.A * self.dl

        # droplet classes

        self.N_D = N_D

        # Annahme der Anfangsbedingungen
        self.h_c_0 = h_c_0 + h_dis_0
        self.h_dis_0 = h_dis_0

    def reduce_Nx(
        self, dNx=10, T=100
    ):  # T anpassen, wenn T in Klasse Settings geändert wird!!!!
        self.T = T
        self.t = np.linspace(0, self.T, self.N_t)
        self.N_x -= dNx
        self.x = np.linspace(0, self.L, self.N_x)
        self.dl = self.x[1] - self.x[0]
        print(
            "Number of grid points reduced from "
            + str(self.N_x + dNx)
            + " to "
            + str(self.N_x)
        )

    def set_Nt(self, N_t):
        N_t_old = self.N_t
        self.N_t = N_t
        self.t = np.linspace(0, self.T, self.N_t)
        self.dt = self.t[1] - self.t[0]
        print(
            "Number of time steps changed from " + str(N_t_old) + " to " + str(self.N_t)
        )

    def set_Nx(self, N_x):
        self.N_x = N_x
        self.x = np.linspace(0, self.L, self.N_x)
        self.dl = self.x[1] - self.x[0]
        self.delta_V = self.A * self.dl

    def set_T(self, T):
        self.T = T
        self.t = np.linspace(0, self.T, self.N_t)
        self.dt = self.t[1] - self.t[0]

    def set_L(self, L):
        self.L = L
        self.x = np.linspace(0, self.L, self.N_x)
        self.dl = self.x[1] - self.x[0]


# Folgende Klasse beinhaltet weitere Parameter und liest Stoffwerte und Messdaten aus Excel-Dateien aus


class Substance_System:

    def __init__(self):
        # constants
        self.g = 9.81
        self.eps_p = 0.9
        self.eps_di = 1.0
        self.H_cd = 1e-20

        # Substance parameters
        self.rho_c = 0  # Density of conti phase [kg/m³]
        self.rho_d = 0  # Density of dispers phase [kg/m³]
        self.sigma = 0  # Surface tension [N/m]
        self.eta_c = 0  # Viscosity conti phase [Pas]
        self.eta_d = 0  # Viscosity disp phase [Pas]
        self.o_to_w = 0  # Ratio of organic and water phase [-]
        self.dV_ges = 0  # Total Volume Flow Rate [m^3/s]
        self.phi_0 = 0  # Sauter-Diameter at the beginning [m]
        self.r_s_star = 0  # r_s_star parameter [-]
        self.h_p_star = 0  # h_p_star parameter [-]

        # Calculated parameters
        self.delta_rho = 0  # Density difference [kg/m^3]
        self.eps_0 = 0  # Hold-Up Feed [-]
        self.s = 0  # slip parameter [-]
        self.eta_dis = 0  # Viscosity dpz [-]
        self.light_in_heavy = (
            0  # boolean whether light phase is dispersed in heavy phase
        )

        # Data
        self.x_exp = []
        self.h_p_exp = []
        self.x_sim = []
        self.h_p_sim = []

    def set_o_to_w(self, o_w):
        self.o_to_w = o_w
        if self.light_in_heavy:
            self.eps_0 = self.o_to_w / (1 + self.o_to_w)
        else:
            self.eps_0 = 1 - self.o_to_w / (1 + self.o_to_w)

    def update(self, excel_file):

        # Get the absolute path to the repo root (one folder above utils)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Build the full path to the Excel file in Input/
        excel_path = os.path.join(repo_root, "Input", excel_file)

        import_data = pd.read_excel(excel_path, sheet_name="Parameters", index_col=0)

        # Update experimental parameter
        self.rho_c = import_data["ρ_c"]["Value"]
        self.rho_d = import_data["ρ_d"]["Value"]
        self.sigma = import_data["σ"]["Value"]
        self.eta_c = import_data["η_c"]["Value"]
        self.eta_d = import_data["η_d"]["Value"]
        self.o_to_w = import_data["o_w"]["Value"]
        self.dV_ges = (
            import_data["dV_ges"]["Value"] / 3.6 * 1e-6
        )  # convert into SI unit
        self.phi_0 = import_data["phi_0"]["Value"]
        self.r_s_star = import_data["r_s_star"]["Value"]
        self.h_p_star = import_data["h_p_star"]["Value"]

        # calculate more substance parameters
        if self.rho_d < self.rho_c:
            self.light_in_heavy = True
            self.eps_0 = self.o_to_w / (1 + self.o_to_w)
        else:
            self.light_in_heavy = False
            self.eps_0 = 1 - self.o_to_w / (1 + self.o_to_w)

        self.delta_rho = abs(self.rho_c - self.rho_d)
        self.s = 1 - np.exp(-12300 * self.rho_c / self.delta_rho * self.sigma**3)
        self.eta_dis = hf.yaron(self.eta_c, self.eta_d, self.eps_0)

        print("\nUpdated Parameters with Excel")
