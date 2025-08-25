'''
 Model of Backi et al. 2018
 Modifications:
 2 phase: l-l
 no mixer - assumed droplet distribution
 
 functions to be imported into model_backi.py
'''

# functions
def get_stokes_vel(D, Delta_rho, eta_c, g):
    # stokes velocity calculation
    v_s = g*D**2*Delta_rho/(18*eta_c)
    return v_s

def get_swarm_vel(D, Delta_rho, eta_c, g, epsilon):
    '''
    swarm sedimentation Richardson-Zaki with n=2 from Kampwerth 2020
    input: D: droplet diameter in m
    input: Delta_rho: density difference of dispersed and continuous phase in kg/m3
    input: eta_c: continuous phase viscosity in Pa*s
    input: g: gravitational acceleration in m/s2
    input: epsilon: hold up of dispersed phase in 1
    output: v_s: swarm velocity in m/s
    '''
    n = 2
    v_0 = get_stokes_vel(D, Delta_rho, eta_c, g)
    v_s = v_0 * (1 - epsilon)**(n-1)
    return v_s 
   
def get_tau_x(L,A,q):
    # horizontal residence time calculation of a segment of length L, crosssectional area A and volume flow q
    tau_h = L*A/q
    return tau_h

def get_A_x(r,h_w):
    import numpy as np
    import utils.utils_pinn.constants as constants
    # crosssectional area (normal vector in x direction) of separator dependent on height of water h_w and radius of separator r
    if h_w >= 2*r - constants.EPS:
        return np.pi*r**2
    if h_w <= constants.EPS: # return area of segment equal to 1e-3 m height (to avoid division by zero)
        return constants.EPS
    return r**2/2*(2*np.arccos((r-h_w)/r) - np.sin(2*np.arccos((r-h_w)/r)))

def get_A_y(r,h_i,dL): 
    # crosssectional area (normal vector in y direction) of interface dependent on height of interface h_i and radius of separator r for axial segment length dL
    # catch error if h_i is smaller eq. zero (no DGTS in separator)
    import numpy as np
    import utils.utils_pinn.constants as constants
    root = h_i*(2*r - h_i)
    if root < constants.EPS:
        return 2*dL*constants.EPS
    else:
        A_i = 2*dL*np.sqrt(root)
    return A_i

def get_factor_dAx_dh(l,r,h):
    # derivative factor for crosssectional area (normal vector in x direction) of separator dependent on height h and separator length l
    import numpy as np
    if h >= 2*r or h <= 0:
        return 1e12 # return infinity if h is equal larger than 2*r or smaller than zero
    else:
        return 1/(2*l*(h*(2*r-h))**0.5)

def get_tau_y(h, v_s):
    '''
    vertical residence time of a droplet with swarm velocity v_S for height h
    input: h: height in m
    input: v_S: swarm velocity in m/s
    output: tau_v: vertical residence time in s
    '''
    # vertical residence time of a droplet with diameter D with stokes velocity for height h
    tau_v = h/v_s
    return tau_v

def get_V_d(D):
    # Volume of droplet with diameter m
    import numpy as np
    V_d = np.pi/6*D**3
    return V_d

def get_droplet_classes(d_32, N_D, path, s=0.32, d_max=3e-3, plot=False):
    '''
    calculate log-normal DSD function from sauter mean diameter d32[m]
    input: d_32: sauter mean diameter of droplets in m
    input: d_max: maximum droplet diameter in m
    input: s: standard deviation of volume-based log-normal distribution (Kraume2004)
    input: plot: if True plot the distribution
    input: path: path to store results
    return n_count_rel: relative number-based probability of droplets for each class based on the derived volume-based log normal distribution
    return d_bins_center: bin center of droplet classes

    see Kraume, Influence of Physical Properties on Drop Size Distributions of Stirred Liquid-Liquid Dispersions, 2004 
    and Ye, Effect of temperature on mixing and separation of stirred liquid/liquid dispersions over a wide range of dispersed phase fractions, 2023
    '''
    import numpy as np
    from scipy.stats import lognorm
    from scipy import stats
    import matplotlib.pyplot as plt

    # statistical representative number of droplets for volume distribution
    N_vol = int(1e6)
    # lognorm volume distribution of d/d_32
    dist = lognorm(s) 
    
    # define bin edges (diameter class) equidistantly from numberClasses
    x = np.linspace(0,d_max/d_32,N_D+1)
    
    if plot==True:
        # plot lognorm distribution
        fig, ax = plt.subplots(1, 1)
        ax.plot(x*d_32*1e6,dist.pdf(x))
        ax.set_ylim([0,1.2*max(dist.pdf(x))])
        ax.set_xlim([0,d_max*1e6])
        ax.set_xlabel('$d / \mathrm{\mu m}$')
        ax.set_ylabel('$q_3 / \mathrm{\mu m}^-1$')
        ax.set_title('Volume-based probability density distribution \n $d_{32}$=' + str(d_32*1e6) +'$\mu m$'
                    + ', $d_{max}$=' + str(d_max*1e6) + '$\mu m$'
                    + ', \n number of classes=' + str(N_D))
        # save plot
        fig.savefig(path + '\lognorm_dist.png', dpi=1000)
        fig.savefig(path + '\lognorm_dist.eps', dpi=1000)
        fig.savefig(path + '\lognorm_dist.svg', dpi=1000)
        
    # divide sample points into bins hist[0] is the count and hist[1] the edges of bins
    hist = np.histogram(dist.rvs(N_vol, random_state= 1),bins=x, density=False)
    # return middle value of bins boundary values
    d_bins = hist[1]*d_32
    d_bins_center = np.zeros(len(d_bins)-1)
    for i in range(len(d_bins)-1):
        d_bins_center[i] = (d_bins[i]+d_bins[i+1])/2
        
    # transform volume based absolute distribution to number based relative distribution
    v_count_abs = hist[0]
    n_count_abs = np.zeros(len(v_count_abs))
    v_count_rel = np.zeros(len(v_count_abs))
    for i in range(len(v_count_abs)):
        n_count_abs[i] = v_count_abs[i]*6/(np.pi*d_bins_center[i]**3)
        v_count_rel[i] = v_count_abs[i]/sum(v_count_abs)
    # normalize number distribution
    n_count_rel = np.zeros(len(v_count_abs))
    for i in range(len(v_count_abs)):
        n_count_rel[i] = n_count_abs[i]/sum(n_count_abs)
        
    # optional plotting of transformed distribution
    if plot==True:
        fig, ax = plt.subplots(1, 1)
        ax.plot(d_bins_center*1e6,v_count_rel, label='Volume-based')
        ax.plot(d_bins_center*1e6,n_count_rel, label='Number-based')
        # ax.set_ylim([0,1])
        ax.set_xlim([0,d_max*1e6])
        ax.set_xlabel('$d / \mathrm{\mu m}$')
        ax.set_ylabel('$h $')
        ax.set_xlim([0,d_max*1e6])
        ax.set_ylim([0,1.2*max(np.append(v_count_rel, n_count_rel))])
        ax.set_title('Relative distribution \n $d_{32}$=' + str(d_32*1e6) +'$\mu m$'
                    + ', $d_{max}$=' + str(d_max*1e6) + '$\mu m$'
                    + ', \n number of classes=' + str(N_D))
        ax.legend()
        # save plot
        fig.savefig(path + '\lognorm_dist_rel.png', dpi=1000)
        fig.savefig(path + '\lognorm_dist_rel.eps', dpi=1000)
        fig.savefig(path + '\lognorm_dist_rel.svg', dpi=1000)
        
        # plot histogram of number distribution
        fig, ax = plt.subplots(1, 1)
        ax.bar(d_bins_center*1e6,n_count_rel, width=0.5*d_bins_center[1]*1e6)
        ax.set_xlabel('$d / \mathrm{\mu m}$')
        ax.set_ylabel('$h $')
        ax.set_title('Relative number-based distribution \n $d_{32}$=' + str(d_32*1e6) +'$\mu m$'
                    + ', $d_{max}$=' + str(d_max*1e6) + '$\mu m$'
                    + ', \n number of classes=' + str(N_D))
        fig.savefig(path + '\lognorm_dist_rel_n.png', dpi=1000)
        fig.savefig(path + '\lognorm_dist_rel_n.eps', dpi=1000)
        fig.savefig(path + '\lognorm_dist_rel_n.svg', dpi=1000)
        # plot histogram of volume distribution
                # plot histogram of number distribution
        fig, ax = plt.subplots(1, 1)
        ax.bar(d_bins_center*1e6,v_count_rel, width=0.5*d_bins_center[1]*1e6)
        ax.set_xlabel('$d / \mathrm{\mu m}$')
        ax.set_ylabel('$h $')
        ax.set_title('Relative volume-based distribution \n $d_{32}$=' + str(d_32*1e6) +'$\mu m$'
                    + ', $d_{max}$=' + str(d_max*1e6) + '$\mu m$'
                    + ', \n number of classes=' + str(N_D))
        fig.savefig(path + '\lognorm_dist_rel_v.png', dpi=1000)
        fig.savefig(path + '\lognorm_dist_rel_v.eps', dpi=1000)
        fig.savefig(path + '\lognorm_dist_rel_v.svg', dpi=1000)

    return n_count_rel, d_bins_center

def get_totalNumber_water_inlet(hold_up, d_32, d_max, V_mix, path, N_D):
    '''
    calculates the total number of droplets entering the separator for a given hold up and volume of mixing that follows the volume-based lognormal distribution (Kraume2004)
    input: hold_up: hold up of org. in aq. phase entering the separator in 1
    input: d_32: sauter mean diameter of droplets in m
    input: d_max: maximum droplet diameter in m
    input: V_mix: Volume of mixer (volume of first aq. phase segment) in m3
    output: N_in_total: total number of droplets entering the separator in 1
    '''
    # use minimize to calculate number of droplets
    from scipy import optimize

    N_in_total = 1e4 # initial guess
    
    # relative number distribution at inlet 
    n_count_rel, d_bins = get_droplet_classes(d_32, N_D, d_max=d_max, path=path)
    
    def f(N_in_total):

        # volume of dispered phase in m3
        V_disp = 0
        for i in range(N_D):
            V_disp = V_disp + N_in_total*n_count_rel[i]*get_V_d(d_bins[i])
        # hold up of water in separator
        hold_up_calc = V_disp/V_mix
        return hold_up_calc-hold_up
    
    N_in_total = optimize.newton(f,N_in_total,rtol=1e-4)
        
    # converet number of droplets to integer
    N_in_total = int(N_in_total)
    # calculate hold up for found number of droplets
    hold_up_calc = f(N_in_total) + hold_up
    # print results
    # print hold_up_calc with 4 digits
    hold_up_calc = round(hold_up_calc,4)
    # print('hold up: ' + str(hold_up_calc))
    # print('number of droplets: ' + str(N_in_total))
    return N_in_total, n_count_rel, d_bins

def get_sauter_mean_diameter_stirrer(We):
    '''
    calculates the sauter mean diameter of droplets in m based on the weber number
    input: We=n**2*D**3*rho_disp/sigma: weber number in 1
    output: d_32: sauter mean diameter of droplets in m
    see Kraume 2004 et al.
    '''
    import numpy as np
    import utils.utils_pinn.constants as constants
    c_2 = 1 # constant depending on stirrer geometry
    n = 0.6 # for breakage dominant mixing processes
    d_32 = constants.D_STIRRER*c_2*(We)**(n)
    return d_32

def get_sauter_mean_diameter(n_count_abs, d_bins):
    '''
    calculates the sauter mean diameter of droplets 
    input: n_count_abs: absolute number-based probability of droplets for each class in 1
    input: d_bins_center: bin center of droplet classes in m
    output: d_32: sauter mean diameter of droplets in m
    '''
    import numpy as np
    if np.sum(n_count_abs) != 0:
        v = 0
        o = 0
        for i in range(len(n_count_abs)):
            v = v + n_count_abs[i]*d_bins[i]**3
            o = o + n_count_abs[i]*d_bins[i]**2
        return v/o
    else:
        return 0

def get_coalescence_time(d_32, h_p, r_v, Delta_rho, sigma, eta_c, g, Ha, at_interface=True):
    '''
    calculates the coalescence time of droplets in s
    input: d_32: sauter mean diameter of droplets in m
    input: h_p: height of the dense-packed zone in m
    input: r_v: asymetric film drainage parameter
    input: Delta_rho: density difference of dispersed and continuous phase in kg/m3
    input: sigma: interfacial tension between dispersed and continuous phase in N/m
    input: eta_c: continuous phase viscosity in Pa*s
    input: g: gravitational acceleration in m/s2
    input: Ha: Hamaker constant in J
    input: at_interface: if True coalescence at interface, if False coalescence at droplet-droplet contact
    output: tau: coalescence time of droplets in s
    '''
    import numpy as np
    # check if droplets are at interface or droplet-droplet contact
    if d_32 <= 0:
        return np.inf
    La = (Delta_rho*g/sigma)**0.6 *h_p**0.2 *d_32
    root = np.sqrt(1 - 4.7 / (La+4.7))
    r_fc = 0.3025 * d_32 * root # contact area radius droplet to droplet
    r_fi = np.sqrt(3) * r_fc # contact area radius droplet to interface
    r_a  = 0.5 * d_32 * (1 - root) # radius of the channel contour
    if at_interface:
        r_f = r_fi
    else:
        r_f = r_fc
    tau = ((6*np.pi)**(7/6) * eta_c * (r_a)**(7/3)) / \
            (4 * sigma**(5/6) * Ha**(1/6) * r_f * r_v) 
    return tau

def initialize_boundary_conditions(epsilon_in, d_32, d_max, path, N_D, plot=False):
    '''
    calculates the boundary conditions for the separator model of Backi et al. 2018
    input: epsilon_in: hold up of water in inlet in 1
    input: d_32: sauter mean diameter of droplets in m
    input: d_max: maximum droplet diameter in m
    input: path: path to store results
    output: hold_up_calc: hold up of water in separator in 1
    output: n_in: number of droplets in each class at inlet
    output: d_bins: bin center of droplet classes in m
    output: N_in_total: total number of droplets entering the separator in 1
    '''
    import utils.utils_pinn.constants as constants
    V_mix = get_A_x(constants.R, 2*constants.R)*constants.L/constants.N_S
    N_in_total, n_in_rel, d_bins = get_totalNumber_water_inlet(epsilon_in,d_32,d_max=d_max, V_mix=V_mix, path=path, N_D=N_D) # total number of droplets at inlet
    # relative dropsize distribution at inlet
    get_droplet_classes(d_32, N_D, d_max=d_max, path=path, plot=plot)
    # convert to absolute number of droplets at inlet
    n_in = n_in_rel*N_in_total
    # inlet flow rates
    # volume of dispered phase in m3
    V_disp = 0
    for i in range(N_D):
        V_disp = V_disp + n_in[i]*get_V_d(d_bins[i])
    # hold up of water in separator
    hold_up_calc = V_disp/V_mix
    return (hold_up_calc, n_in, d_bins, N_in_total)

def calculate_separator(y,u,p):
    '''
    calculates the separator model of Backi et al. 2018
    input: y: state variables
    input: u: input variables
    input: p: parameter variables
    output: dy_dt_RHS: right hand side of ODE
    output: q_w: water flow rate in each segment
    output: epsilon_w: hold up of water in each segment
    output: q_o: organic flow rate in each segment
    output: q_dp: dispersed phase flow rate in each segment
    output: n: number of droplets in each segment and class
    output: pos: position of droplets in each segment and class
    output: d32_dp: sauter mean diameter of droplets in each segment
    output: tau_di: coalescence time of droplets in each segment
    output: dV_c: volume flow rate of coalescence in each segment
    output: dV_w_dp: volume flow rate of water leaving segment due to coalescence and build up of dense-packed zone
    output: dV_s: volume flow rate of sedimentation in each segment
    output: dV_si: volume flow rate of sedimentation in each segment
    output: dn_dp: number of droplets leaving segment from dense-packed zone
    '''
    import utils.utils_pinn.constants as constants
    import numpy as np
    
    # assign diff variables
    h_w = y[0]
    h_l = y[1]
    h_dp = y[2]
    
    # assign input variables
    q_w_in = u[0]
    q_w_out = u[1]
    q_o_out = u[2]
    n_in = u[3]
    epsilon_in = u[4]
    D = p[0]
    r_v = p[1]

    dy_dt_RHS = np.zeros(3)
    # declaration of variables for each segment and droplet class
    tau_x = np.zeros(constants.N_S)
    tau_y = np.zeros((constants.N_S, constants.N_D))
    dV_s = np.zeros((constants.N_S, constants.N_D))
    dV_w = np.zeros((constants.N_S, constants.N_D))
    n_to_dp = np.zeros((constants.N_S, constants.N_D))
    v_s = np.zeros((constants.N_S, constants.N_D)) # calculated from previous hold-up
    n_dpz = np.zeros((constants.N_S, constants.N_D)) 
    # declaration of variables for each segment
    d32_dp = np.zeros(constants.N_S)
    tau_di = np.zeros(constants.N_S)
    dV_c = np.zeros(constants.N_S)
    dV_w_dp = np.zeros(constants.N_S) 
    dV_si = np.zeros(constants.N_S)
    d32_aq = np.zeros(constants.N_S)
    # declaration of variables entering a segment / N_S + 1 is outlet
    q_w = np.zeros(constants.N_S + 1)
    epsilon_w = np.zeros(constants.N_S + 1)
    q_w_o = np.zeros(constants.N_S + 1)
    q_o = np.zeros(constants.N_S + 1)
    q_dp = np.zeros(constants.N_S + 1)
    # declaration of variables entering a segment and for each class
    n = np.zeros((constants.N_S + 1, constants.N_D))
    pos = np.zeros((constants.N_S + 1, constants.N_D))
    # declaration of variables for each droplet class
    V_d = np.zeros(constants.N_D)
  
    # droplet specific properties
    V_d = get_V_d(D)
    
    # boundary conditions (inlet)
    q_w[0] = q_w_in
    epsilon_w[0] = epsilon_in
    q_o[0] = 0
    q_dp[0] = 0
    n[0,:] = n_in
    pos[0,:] = constants.POS_IN
    
    A_w = get_A_x(constants.R,h_w)
    A_y = get_A_y(constants.R,h_dp,constants.L/constants.N_S)
    
    # calculation of rate terms for each segment
    for i in range(constants.N_S):
        ## sedimentation for every segment
        # residence times
        tau_x[i] = get_tau_x(constants.L/constants.N_S,A_w,q_w[i])
        v_s[i,:] = get_swarm_vel(D,constants.DELTA_RHO,constants.ETA_W,constants.G,epsilon_w[i])
        tau_y[i,:] = get_tau_y(h_w-pos[i,:],v_s[i,:])
        
        #calculate available stream of dispersed phase in aq.
        q_w_o[i] = q_w[i]*epsilon_w[i] 
            
        # for every droplet class
        for k in range(constants.N_D):           
            # partial sedimentation of class k
            if tau_x[i] < tau_y[i,k]:
                # droplet arrive at dense-packed zone
                n_to_dp[i,k] = n[i,k]*(tau_x[i]*v_s[i,k]/(h_w - pos[i,k]))
                # update next segment
                n[i+1,k] = n[i,k]*(1 - tau_x[i]*v_s[i,k]/(h_w - pos[i,k]))
                # position update
                if pos[i,k] + v_s[i,k]*tau_x[i] > h_w:
                    pos[i+1,k] = h_w
                else:   
                    pos[i+1,k] = pos[i,k] + v_s[i,k]*tau_x[i]
            # full sedimentation of class k
            else:
                n_to_dp[i,k] = n[i,k]
                n[i+1,k] = 0
                pos[i+1,k] = y[0] - constants.EPS
            # calculate rate of sedimentation for ODE 
            if np.sum(n[i,:]) != 0:
                dV_s[i,k] = q_w_o[i]* ((n[i,k] - n[i+1,k])/np.sum(n[i,:]))  
            else:
                dV_s[i,k] = 0
        # calculate rate of sedimentation for this segment & rate of DSD to dense-packed zone
        dV_si[i] = np.sum(dV_s[i,:])
        # sauter mean diameter of droplets sedimenting to dense-packed zone
        d32_aq[i] = get_sauter_mean_diameter(n_to_dp[i,:], D[:])
        
        ## dense-packed zone
        # distribution of droplets in this segment by combining droplets from previous segment and sedimenting droplets
        if i == 0:
            # calculation by absolute numbers
            n_dpz[i,:] = n_to_dp[i,:]
        else:
            # assume mixing of previous segment distribution in dense-packed zone and sedimentating droplets
            if q_dp[i] != 0:
                n_dpz[i,:] = n_dpz[i-1,:] + n_to_dp[i,:]
            else:
                n_dpz[i,:] = n_to_dp[i,:]
        d32_dp[i] = get_sauter_mean_diameter(n_dpz[i,:], D)   
        # coalescence rate calculation
        if h_dp > h_w + d32_dp[i]/2:
            tau_di[i] = get_coalescence_time(d32_dp[i], h_dp-h_w, r_v, constants.DELTA_RHO, constants.SIGMA, constants.ETA_W, constants.G, constants.HA)
        else:
            tau_di[i] = np.inf
        dV_c[i] = 2*d32_dp[i]*A_y*constants.EPSILON_DI / (3*tau_di[i])
        
        # calculation of remaining flows
        # calculation of pure water stream from heavy phase due to coalescence and build up of dense-packed zone
        dV_w_dp[i] = (dV_si[i] - dV_c[i]) * (1 / constants.EPSILON_DP - 1)

        ## update of convective volume flows
        q_o[i+1] = q_o[i] + dV_c[i]
        q_w[i+1] = np.max([q_w[i] - dV_si[i] - dV_w_dp[i], 0])
        q_dp[i+1] = np.max([q_dp[i] + dV_si[i] + dV_w_dp[i] - dV_c[i], 0])

        # update hold-up in aq. phase by volume balance of dispersed droplets in aq. phase
        epsilon_w[i+1] = (epsilon_w[i]*q_w[i] - dV_si[i]) / q_w[i+1]

    # check if dense-packed zone is empty
    if h_dp < h_w + constants.EPS:
        dV_c_tot = 0
        dV_w_dp_tot = np.sum(dV_s) * (1 / constants.EPSILON_DP - 1)
    else:
        dV_c_tot = np.sum(dV_c)
        dV_w_dp_tot = np.sum(dV_w_dp)
    dV_s_tot = np.sum(dV_s) 
    
    # assigning dy_dt for each segment
    dy_dt_RHS[0] = (q_w_in - q_w_out - dV_s_tot - dV_w_dp_tot) * get_factor_dAx_dh(constants.L,constants.R,h_w)
    dy_dt_RHS[1] = (q_w_in - q_o_out - q_w_out)* get_factor_dAx_dh(constants.L,constants.R,h_l)
    dy_dt_RHS[2] = (q_w_in - q_w_out - dV_c_tot)* get_factor_dAx_dh(constants.L,constants.R,h_dp)
    
    # pack results, flow rates and number distributions into tuple
    results = (dy_dt_RHS, q_w, epsilon_w, q_o, q_dp, n, pos, d32_dp, tau_di, dV_c, dV_w_dp, dV_s, dV_si, n_dpz)
    return results

def dy_dt_RHS(t, y, u, p):
    dy_dt_RHS = calculate_separator(y,u,p)[0]
    return dy_dt_RHS
