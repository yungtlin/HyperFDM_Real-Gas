###########
# Modules #
###########
import numpy as np 
import matplotlib.pyplot as plt 

from matplotlib.pyplot import cm
from FDM import Zhong, FDM_interp

########
# Mesh #
########
# FDM meshes for 2D flow past cylinder
class Mesh:
    ########
    # Init #
    ########
    # n: # of mesh [xi, eta] 
    # R: cylinder radius
    # Computational domain is restricted to (xi, eta) \in [0, 1]\times[0, 1]
    def __init__(self, N, R, stencil=5, beta=180):
        print("Initializing mesh (%i, %i)..."%(N[0], N[1]))
        self.N = N #
        self.r_c = R # Radius of the cylinder
        self.beta = beta/180*np.pi # swept angle of the computational domain

        self.eta = np.linspace(0, 1, N[0])
        self.xi = np.linspace(-0.5, 0.5, N[1])
        
        self.Xi, self.Eta = np.meshgrid(self.xi, self.eta)
        self.deta = 1/(N[0] - 1) 
        self.dxi = 1/(N[1] - 1)

        # Bools
        self.is_H_init = False

        # Set stencils
        self.set_FDM_stencil(stencil)


    def set_FDM_stencil(self, stencil):
        self.stencil = stencil

        # For symmetric interpolation
        n_m = int((self.stencil - 1)/2)
        loc = np.arange(-n_m, n_m + 1)*self.dxi
        loc = np.delete(loc, n_m)
        self.interp_center = FDM_interp(0, loc)

        # For boundary point interpolation
        # left     
        loc = np.arange(1, self.stencil)*self.dxi
        self.interp_left = FDM_interp(0, loc)

        # right 
        loc = np.arange(-self.stencil + 1, 0)*self.dxi
        self.interp_rght = FDM_interp(0, loc)

        # FDM derivative
        self.coeffs_xi = np.zeros((stencil, stencil))
        self.coeffs_xixi = np.zeros((stencil, stencil))

        loc = np.arange(stencil)
        for idx in range(stencil):
            N = stencil
            N0 = stencil - (idx + 1)  # # of right points
            M = 1
            M0 = 0
            form = [N, N0, M, M0]

            loc_shft = loc - idx
            loc_xi = loc_shft*self.dxi

            # first derivatives
            c_a, c_b = Zhong(1, form, loc_xi, alpha=0)
            self.coeffs_xi[idx] = c_a

            # second derivatives
            c_a, c_b = Zhong(2, form, loc_xi, alpha=0)
            self.coeffs_xixi[idx] = c_a

        if self.is_H_init:
            self.update_all()

    def init_H_polynomial(self, a_0=0.5, a_1=1, p=2):
        h_0 = a_0*self.r_c
        h_1 = a_1*self.r_c
        ha = (h_1 - h_0)

        xi_max = np.max(np.abs(self.xi))
        xi = self.xi/xi_max
        H0 = h_0 + ha*xi**p

        Ht0 = H0*0
        self.update_H(H0, Ht0)


    def update_H(self, H, Ht):
        if len(H) != self.N[1] or len(Ht) != self.N[1]:
            raise ValueError("Incorrect Input Dimension for Mesh update_H!")

        self.H = H 
        self.Ht = Ht

        self.is_H_init = True
        self.update_all()

    def force_symmetric(self, H, Ht):
        # Centerline symmetric
        N_xi = H.shape[0]
        is_H_odd = (N_xi % 2 == 1)

        if is_H_odd:
            idx_c = int((N_xi - 1)/2)
            n_m = int((self.stencil - 1)/2)

            # find neighbors
            nghbr_list = np.arange(idx_c - n_m, idx_c + n_m + 1,dtype=np.int32)
            nghbr_list = np.delete(nghbr_list, n_m) # remove center
            
            H_nghbr = H[nghbr_list]
            Ht_nghbr = Ht[nghbr_list]

            H[idx_c] = np.sum(self.interp_center*H_nghbr)
            Ht[idx_c] = np.sum(self.interp_center*Ht_nghbr)

        # Boundary line symmetric
        # left
        n_s = self.interp_left.shape[0]
        
        H_nghbr = H[1:n_s + 1]
        Ht_nghbr = Ht[1:n_s + 1]
        H[0] = np.sum(self.interp_left*H_nghbr)
        Ht[0] = np.sum(self.interp_left*Ht_nghbr)

        # right
        n_s = self.interp_rght.shape[0]
        
        H_nghbr = H[-n_s - 1:-1]
        Ht_nghbr = Ht[-n_s - 1:-1]
        H[-1] = np.sum(self.interp_rght*H_nghbr)
        Ht[-1] = np.sum(self.interp_rght*Ht_nghbr)

    # NEED TO VERIFY #
    def update_all(self):
        # FDM approach
        self.update_FDM_H_xi()
        H_xi = self.H_xi
        H_xixi = self.H_xixi
        beta = self.beta

        # General Generation
        Theta = beta*self.Xi
        R_h = (self.H + self.r_c).reshape((1, -1))
        R = (R_h - self.r_c)*self.Eta + self.r_c

        sinT = np.sin(Theta)
        cosT = np.cos(Theta)

        self.X = R*(-cosT)
        self.Y = R*sinT

        H = self.H.reshape((1, -1))
        Ht = self.Ht.reshape((1, -1))

        Eta = self.Eta 
        Xi = self.Xi

        R = H*Eta + self.r_c

        ## First Spatial Derivatives ##
        self.x_xi = -H_xi*Eta*cosT + R*sinT*beta
        self.y_xi =  H_xi*Eta*sinT + R*cosT*beta

        self.x_eta = -H*cosT
        self.y_eta =  H*sinT

        ## Second Spatial Derivatives ##
        self.x_xixi = -H_xixi*Eta*cosT + 2*H_xi*Eta*sinT*beta + R*cosT*beta**2
        self.x_xieta = -H_xi*cosT + H*sinT*beta
        self.x_etaeta = 0*Eta

        self.y_xixi = H_xixi*Eta*sinT + 2*H_xi*Eta*cosT*beta - R*sinT*beta**2
        self.y_xieta = H_xi*sinT + H*cosT*beta
        self.y_etaeta = 0*Eta

        ## Jacobians ##
        self.g = self.x_xi*self.y_eta - self.x_eta*self.y_xi
        self.g_xi = (self.x_xixi*self.y_eta + self.x_xi*self.y_xieta)\
            - (self.x_xieta*self.y_xi + self.x_eta*self.y_xixi)
        self.g_eta = (self.x_xieta*self.y_eta + self.x_xi*self.y_etaeta)\
            - (self.x_etaeta*self.y_xi + self.x_eta*self.y_xieta)

        self.J = 1/self.g
        self.J_xi = -self.g_xi/self.g**2
        self.J_eta = -self.g_eta/self.g**2

        ## Time Dependents ##
        Ht_xi = self.Ht_xi
        Ht_xixi = self.Ht_xixi

        self.geta_t = -beta*Ht*Eta*R
        self.geta_t_xi = -(Ht_xi*Eta*R + Ht*H_xi*Eta**2)
        self.geta_t_eta = -beta*Ht*(R + H*Eta)

        # Zeroth Derivatives #
        x_t = -Ht*Eta*cosT
        y_t = Ht*Eta*sinT

        # First Derivatives #
        x_xit = -Ht_xi*Eta*cosT + Ht*Eta*sinT*beta
        y_xit = Ht_xi*Eta*sinT + Ht*Eta*cosT*beta
        x_etat = -Ht*cosT
        y_etat = Ht*sinT

        # Second Derivatives #
        x_xixit = -Ht_xixi*Eta*cosT + 2*Ht_xi*Eta*sinT*beta + Ht*Eta*cosT*beta**2
        x_xietat = -Ht_xi*cosT + Ht*sinT*beta
        x_etaetat = 0*Eta

        y_xixit = Ht_xixi*Eta*sinT + 2*Ht_xi*Eta*cosT*beta - Ht*Eta*sinT*beta**2
        y_xietat = Ht_xi*sinT + Ht*cosT*beta
        y_etaetat = 0*Eta

        # Time-dependent Determinants #
        self.g_t = (x_xit*self.y_eta + self.x_xi*y_etat) - (x_etat*self.y_xi + self.x_eta*y_xit)
        self.g_xit = (x_xixit*self.y_eta + x_xit*self.y_xieta + self.x_xixi*y_etat + self.x_xi*y_xietat)\
            - (x_xietat*self.y_xi + x_etat*self.y_xixi + self.x_xieta*y_xit + self.x_eta*y_xixit)
        self.g_etat = (x_xietat*self.y_eta + x_xit*self.y_etaeta + self.x_xieta*y_etat + self.x_xi*y_etaetat)\
            - (x_etaetat*self.y_xi + x_etat*self.y_xieta + self.x_etaeta*y_xit + self.x_eta*y_xietat)

        # Surface term #
        self.surface_n, self.surface_v_n = self.get_surface()
        self.wall_n_n, self.wall_n_t = self.get_wall()

    def update_FDM_H_xi(self):
        self.H_xi = self.get_FDM_xi(self.H, self.coeffs_xi).reshape((1, -1))
        self.Ht_xi = self.get_FDM_xi(self.Ht, self.coeffs_xi).reshape((1, -1))

        self.H_xixi = self.get_FDM_xi(self.H, self.coeffs_xixi).reshape((1, -1))
        self.Ht_xixi = self.get_FDM_xi(self.Ht, self.coeffs_xixi).reshape((1, -1))

    # pseudo symmetric condition
    def get_FDM_xi(self, u, coeffs):
        u_xi = np.zeros(u.shape)

        N_m = int((self.stencil - 1)/2)
    

        ## left boundary (used for full)
        u_l = u[:self.stencil]
        for idx in range(N_m):
            a = coeffs[idx]
            u_xi[idx] = np.dot(a, u_l)

        # interior
        a = np.flip(coeffs[N_m])
        u_xi[N_m:-N_m] = np.convolve(a, u, mode="valid")
        
        # right boundary
        u_r = u[-self.stencil:]
        for i in range(N_m):
            idx = -(i + 1)
            a = coeffs[idx]
            u_xi[idx] = np.dot(a, u_r)

        return u_xi

    def get_dx_ref(self):
        dL_xi = self.dxi*np.sqrt(self.x_xi**2 + self.y_xi**2)
        dL_eta = self.deta*np.sqrt(self.x_eta**2 + self.y_eta**2)
        
        dx_ref = np.where(dL_xi < dL_eta, dL_xi, dL_eta)

        return dx_ref

    # NEED TO BE VERIFIED #
    def get_u_dx(self, u, u_xi, u_eta):
        J = 1/self.g
        u_x = J*(self.y_eta*u_xi - self.y_xi*u_eta)
        u_y = J*(-self.x_eta*u_xi + self.x_xi*u_eta)

        return u_x, u_y

    # NEED TO BE VERIFIED #
    def get_u_dx_xi(self, u, u_xi, u_eta, u_xixi, u_xieta, u_etaeta):
        u_x_xi = self.J_xi*(self.y_eta*u_xi - self.y_xi*u_eta)\
            + self.J*(self.y_xieta*u_xi + self.y_eta*u_xixi 
            - self.y_xixi*u_eta - self.y_xi*u_xieta)
        
        u_x_eta = self.J_eta*(self.y_eta*u_xi - self.y_xi*u_eta)\
            + self.J*(self.y_etaeta*u_xi + self.y_eta*u_xieta
            - self.y_xieta*u_eta - self.y_xi*u_etaeta)
        
        u_y_xi = self.J_xi*(-self.x_eta*u_xi + self.x_xi*u_eta)\
            + self.J*(-self.x_xieta*u_xi - self.x_eta*u_xixi 
            + self.x_xixi*u_eta + self.x_xi*u_xieta)

        u_y_eta = self.J_eta*(-self.x_eta*u_xi + self.x_xi*u_eta)\
            + self.J*(-self.x_etaeta*u_xi - self.x_eta*u_xieta
            + self.x_xieta*u_eta + self.x_xi*u_etaeta)

        return u_x_xi, u_x_eta, u_y_xi, u_y_eta

    def get_surface(self):
        y_xi = self.y_xi[-1]
        x_xi = self.x_xi[-1]

        geta_t = self.geta_t[-1]

        gradEta = np.sqrt(y_xi**2 + x_xi**2)

        n_hat = (np.array([y_xi, -x_xi])/gradEta)

        v_n = -geta_t/gradEta

        return n_hat, v_n

    def get_wall(self):
        x_xi = self.x_xi[0]
        y_xi = self.y_xi[0]

        x_eta = self.x_eta[0]
        y_eta = self.y_eta[0]

        gradEta = np.sqrt(y_xi**2 + x_xi**2)
        gradXi = np.sqrt(y_eta**2 + x_eta**2)

        n_n = np.array([-y_xi, x_xi])/gradEta
        n_t = np.array([y_eta, -x_eta])/gradXi

        return n_n, n_t

    def get_wall_velocity(self, u, v):
        n_t = self.wall_n_t
        n_n = self.wall_n_n

        u_t = n_t[0]*u + n_t[1]*v
        u_r = n_n[0]*u + n_n[1]*v

        return u_t, u_r

    # n_t
    # H_tt = (v_nt - c0)/c1
    def get_surface_shock_move(self):
        theta = self.beta*self.Xi[-1]

        y_xi = self.y_xi[-1]
        x_xi = self.x_xi[-1]

        sinT = np.sin(theta)
        cosT = np.cos(theta)

        x_t = -self.Ht*self.Eta[-1]*cosT
        y_t = self.Ht*self.Eta[-1]*sinT

        x_xit = -self.Ht_xi*self.Eta[-1]*cosT\
            + self.Ht*self.Eta[-1]*sinT*self.beta
        y_xit = self.Ht_xi*self.Eta[-1]*sinT\
            + self.Ht*self.Eta[-1]*cosT*self.beta

        a = 1/np.sqrt(y_xi**2 + x_xi**2)
        a_t = -(y_xi*y_xit + x_xi*x_xit)*a**3

        nt = np.zeros(self.surface_n.shape)
        nt[0] = y_xit*a + y_xi*a_t
        nt[1] = -(x_xit*a + x_xi*a_t)

        nx = theta.shape[0]
        c = np.zeros((2, nx))
        c[0] = (x_xit*y_t - y_xit*x_t)*a + (x_xi*y_t - y_xi*x_t)*a_t
        c[1] = (x_xi*sinT + y_xi*cosT)*a

        return nt, c


    def plot_mesh(self, style="-k", linewidth=0.6):
        r_c = self.r_c
        if self.is_H_init:
            # Xi line 
            for idx_j in range(self.N[0]):
                x = self.X[idx_j, :]/r_c
                y = self.Y[idx_j, :]/r_c
                plt.plot(x, y, style, linewidth=linewidth)

            # Xi line
            for idx_i in range(self.N[1]):
                x = self.X[:, idx_i]/r_c
                y = self.Y[:, idx_i]/r_c
                plt.plot(x, y, style, linewidth=linewidth)


if __name__ == "__main__":
    n = np.array([21, 11]) # eta, xi

    R = 0.5

    mesh = Mesh(n, R, stencil=5)
    mesh.init_H_polynomial(a_0=1, a_1=1, p=2)
    
    Ht = np.array(mesh.Ht + 10)
    mesh.update_H(mesh.H, Ht)

    #mesh.get_surface_shock_move()

    mesh.force_symmetric(mesh.H, mesh.Ht)

    #H0 = mesh.H
    #Ht0 = H0*2
    #print(H0)
    #mesh.update_H(H0, Ht0)

    #mesh.get_dx_ref()

    mesh.plot_mesh()
    
    #plt.contourf(mesh.X, mesh.Y, mesh.AAA, 20, cmap=cm.jet)
    #plt.colorbar()
    #plt.show()    
