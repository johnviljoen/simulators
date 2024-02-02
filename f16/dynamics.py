import torch
import numpy as np
from f16.parameters import x_lb, x_ub, u_lb, u_ub
import f16.parameters as parameters
from tables import CLookup, PyLookupTorch
import common.pytorch_utils as ptu

class NLPlant():
    def __init__(self, lookup_type='C'):
        self.lookup_type = lookup_type
        self.c_lookup = CLookup()
        self.py_lookup = PyLookupTorch()
        self.torchzero = ptu.from_numpy(np.zeros(1)) # we need a zero on the correct device in the py_lut_wrap method

    """
    This class encapsulates the full nonlinear dynamics model for the F16.
    Its states, inputs and observable states are as follows:
    x = {xe, ye, h, phi, theta, psi, V, alpha, beta, p, q, r, T, dh, da, dr, LF_state, dLEF}
    u = {T, dh, da, dr}
    y = {xe, ye, h, phi, theta, psi, V, alpha, beta, p, q, r, T, dh, da, dr, LF_state, dLEF}
    """

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        """
        args:
            x:
                {xe, ye, h,  phi, theta, psi, V,    alpha, beta, p,     q,     r,     T,  dh,  da,  dr,  dLEF, LF_state}
                {ft, ft, ft, rad, rad,   rad, ft/s, rad,   rad,  rad/s, rad/s, rad/s, lb, deg, deg, deg, deg,  deg     }
            u:
                {T,  dh,  da,  dr }
                {lb, deg, deg, deg}

        returns:
            xdot:
                time derivates of x, in same order
            accelerations:
                {anx_cg, any_cg, anz_cg}
                {g,      g,      g     }
            atmospherics:
                {mach,          qbar,          ps           }
                {dimensionless, dimensionless, dimensionless}
        """
        # Thrust Model
        T_dot = self.calc_thrust_dot(u[0], x[12]).squeeze(0)
        # Dstab Model
        dstab_dot = self.calc_dstab_dot(u[1], x[13]).squeeze(0)
        # aileron model
        ail_dot = self.calc_ail_dot(u[2], x[14]).squeeze(0)
        # rudder model
        rud_dot = self.calc_rud_dot(u[3], x[15]).squeeze(0)
        # leading edge flap model
        LF_state_dot, dLEF_dot = self.calc_lef_dot(x[2], x[6], x[7], x[16], x[17])
        # run nlplant for xdot
        actuator_xdot = torch.stack([T_dot, dstab_dot, ail_dot, rud_dot, dLEF_dot, LF_state_dot])
        xdot, accelerations, atmospherics = self.equations_of_motion(x)
        # assign actuator xdots
        xdot[12:18] = actuator_xdot
        return xdot, accelerations, atmospherics

    def obs(self, x, u):
        obs = ptu.from_numpy(parameters.C) @ x + ptu.from_numpy(parameters.D) @ u
        return obs

    def atmos(self, alt, vt):
        rho0 = 2.377e-3
        
        tfac =1 - .703e-5*(alt)
        temp = 519.0*tfac
        if alt >= 35000.0: temp=390

        rho=rho0*torch.pow(tfac,4.14)
        mach = vt/torch.sqrt(1.4*1716.3*temp)
        qbar = .5*rho*torch.pow(vt,2)
        ps   = 1715.0*rho*temp

        if ps == 0: ps = 1715
        
        return mach, qbar, ps

    """
    Actuator models:
    """

    def calc_lef_dot(self, h, V, alpha, dLEF, LF_state):
        """
        Leading edge flap (lef) actuator modelled as a 2nd order system with
        2 internal states

        args:
            {h (ft), V (ft/s), alpha (rad), LF_state (deg), dLEF (deg)}
        returns:
            {LF_state_dot (deg/s), dLEF (deg/s)}
        """
        coeff = self.atmos(h, V)
        atmos_out = coeff[1]/coeff[2] * 9.05
        alpha_deg = alpha*180/torch.pi

        # validated
        LF_state_dot = - 7.25*(alpha_deg + LF_state)
        
        # validated 
        LF_out = (LF_state + 2*alpha_deg) * 1.38

        # validated
        dLEF_cmd = LF_out + 1.45 - atmos_out
        
        # command saturation
        dLEF_cmd = torch.clip(dLEF_cmd,x_lb[16],x_ub[16])
        # rate saturation
        dLEF_dot = torch.clip((1/0.136) * (dLEF_cmd - dLEF),-25,25)

        return LF_state_dot, dLEF_dot

    def calc_thrust_dot(self, T_cmd, T_state):
        """
        Thrust is modelled as a 1st order system (a significant downgrade on
        the complex afterburner simulation of the original Nguyen work)

        args:
            {T_cmd (lbs), T_state (lbs)}
        returns:
            {T_state_dot (lbs/s)}
        """
        # command saturation
        T_cmd = torch.clip(T_cmd,u_lb[0],u_ub[0])
        # rate saturation
        return torch.clip(T_cmd - T_state, -10000, 10000)

    def calc_dstab_dot(self, dstab_cmd, dstab_state):
        """
        The horizontal stabilator (dstab), more commonly known as an elevator in
        ordinary aircraft, controls the pitch angle of the aircraft. It is
        modelled as a 1st order system here.

        args:
            {dstab_cmd (deg), dstab_state (deg)}
        returns:
            {dstab_state_dot (deg/s)}
        """
        # command saturation
        dstab_cmd = torch.clip(dstab_cmd,u_lb[1],u_ub[1])
        # rate saturation
        return torch.clip(20.2*(dstab_cmd - dstab_state), -60, 60)

    def calc_ail_dot(self, ail_cmd, ail_state):
        """
        The aileron is modelled as a 1st order system

        args:
            {ail_cmd (deg), ail_state (deg)}
        returns:
            {ail_state_dot (deg/s)}
        """
        # command saturation
        ail_cmd = torch.clip(ail_cmd,u_lb[2],u_ub[2])
        # rate saturation
        return torch.clip(20.2*(ail_cmd - ail_state), -80, 80)

    def calc_rud_dot(self, rud_cmd, rud_state):
        """
        The rudder is modelled as a 1st order system
        
        args:
            {rud_cmd (deg), rud_state (deg)}
        returns:
            {rud_state_dot (deg/s)}
        """
        # command saturation
        rud_cmd = torch.clip(rud_cmd,u_lb[3],u_ub[3])
        # rate saturation
        return torch.clip(20.2*(rud_cmd - rud_state), -120, 120)

    """
    The Python tables require some clipping preprocessing to operate as intended,
    therefore I have added an extra wrapper method here to take care of this, and
    to allow for external calls to the LUTs independantly to allow for unit testing
    of this crucial piece of the simulation,

    I shall also wrap the c tables in the exact same way to allow for easy unit testing

    This may seem incredibly redundant and look like obfuscation central especially with the
    c_lut_wrap, but it does allow me to call this stuff independently for testing. It
    is worth it I promise haha.
    """
    def py_lut_wrap(self, alpha, beta, el):
        
        inp = torch.stack([alpha,beta,el])

        # inp = np.array([
        #     ptu.to_numpy(alpha), 
        #     ptu.to_numpy(beta), 
        #     ptu.to_numpy(el)
        #     ])

        inp_clip = torch.clip(
            inp,
            min=torch.stack([self.py_lookup.axes['ALPHA2'][0],self.py_lookup.axes['BETA1'][0],self.py_lookup.axes['DH1'][0]]),
            max=torch.stack([self.py_lookup.axes['ALPHA2'][-1],self.py_lookup.axes['BETA1'][-1],self.py_lookup.axes['DH1'][-1]])
        )
        
        temp = torch.cat([inp[0:2], self.torchzero])

        temp_clip = torch.cat([inp_clip[0:2], self.torchzero])

        # hifi_C
        Cx = self.py_lookup.interp_3d(inp, 'Cx')
        Cz = self.py_lookup.interp_3d(inp, 'Cz')
        Cm = self.py_lookup.interp_3d(inp, 'Cm')
        Cy = self.py_lookup.interp_2d(inp[0:2], 'Cy')
        Cn = self.py_lookup.interp_3d(inp, 'Cn')
        Cl = self.py_lookup.interp_3d(inp, 'Cl')

        hifi_C = [Cx, Cz, Cm, Cy, Cn, Cl]
    
        # hifi_damping (clipped)
        Cxq = self.py_lookup.interp_1d(inp[0:1], 'CXq')
        Cyr = self.py_lookup.interp_1d(inp[0:1], 'CYr')
        Cyp = self.py_lookup.interp_1d(inp[0:1], 'CYp')
        Czq = self.py_lookup.interp_1d(inp[0:1], 'CZq')
        Clr = self.py_lookup.interp_1d(inp[0:1], 'CLr')
        Clp = self.py_lookup.interp_1d(inp[0:1], 'CLp')
        Cmq = self.py_lookup.interp_1d(inp[0:1], 'CMq')
        Cnr = self.py_lookup.interp_1d(inp[0:1], 'CNr')
        Cnp = self.py_lookup.interp_1d(inp[0:1], 'CNp')

        hifi_damping = [
            Cxq, 
            Cyr, 
            Cyp, 
            Czq, 
            Clr, 
            Clp, 
            Cmq, 
            Cnr, 
            Cnp
        ]

        # hifi_C_lef (clipped)
        # temp_clip = np.concatenate([inp_clip[0:2], np.array([0])])
        delta_Cx_lef = self.py_lookup.interp_2d(inp_clip[0:2], 'Cx_lef') - self.py_lookup.interp_3d(temp_clip, 'Cx')
        delta_Cz_lef = self.py_lookup.interp_2d(inp_clip[0:2], 'Cz_lef') - self.py_lookup.interp_3d(temp_clip, 'Cz')
        delta_Cm_lef = self.py_lookup.interp_2d(inp_clip[0:2], 'Cm_lef') - self.py_lookup.interp_3d(temp_clip, 'Cm')
        delta_Cy_lef = self.py_lookup.interp_2d(inp_clip[0:2], 'Cy_lef') - self.py_lookup.interp_2d(inp_clip[0:2], 'Cy')
        delta_Cn_lef = self.py_lookup.interp_2d(inp_clip[0:2], 'Cn_lef') - self.py_lookup.interp_3d(temp_clip, 'Cn')
        delta_Cl_lef = self.py_lookup.interp_2d(inp_clip[0:2], 'Cl_lef') - self.py_lookup.interp_3d(temp_clip, 'Cl')

        hifi_C_lef = [
            delta_Cx_lef,
            delta_Cz_lef,
            delta_Cm_lef,
            delta_Cy_lef,
            delta_Cn_lef,
            delta_Cl_lef
        ]

        # hifi_damping_lef (clipped)
        delta_Cxq_lef = self.py_lookup.interp_1d(inp_clip[0:1], 'delta_CXq_lef')
        delta_Cyr_lef = self.py_lookup.interp_1d(inp_clip[0:1], 'delta_CYr_lef')
        delta_Cyp_lef = self.py_lookup.interp_1d(inp_clip[0:1], 'delta_CYp_lef')
        delta_Czq_lef = self.py_lookup.interp_1d(inp_clip[0:1], 'delta_CZq_lef') # this being unused is not an error, it is as the original C was written, you can delete if you like.
        delta_Clr_lef = self.py_lookup.interp_1d(inp_clip[0:1], 'delta_CLr_lef')
        delta_Clp_lef = self.py_lookup.interp_1d(inp_clip[0:1], 'delta_CLp_lef')
        delta_Cmq_lef = self.py_lookup.interp_1d(inp_clip[0:1], 'delta_CMq_lef')
        delta_Cnr_lef = self.py_lookup.interp_1d(inp_clip[0:1], 'delta_CNr_lef')
        delta_Cnp_lef = self.py_lookup.interp_1d(inp_clip[0:1], 'delta_CNp_lef')

        hifi_damping_lef = [
            delta_Cxq_lef,
            delta_Cyr_lef,
            delta_Cyp_lef,
            delta_Czq_lef,
            delta_Clr_lef,
            delta_Clp_lef,
            delta_Cmq_lef,
            delta_Cnr_lef,
            delta_Cnp_lef
        ]

        # NOTE: The hifi rudder tables do not need to be clipped, however this is how the C has been coded,
        # and so I hypothesise that they have good reason to, perhaps the high alpha values were found to
        # be innacurate for Cy,n,l_r30?
        # hifi_rudder (clipped)
        delta_Cy_r30 = self.py_lookup.interp_2d(inp_clip[0:2], 'Cy_r30') - self.py_lookup.interp_2d(inp_clip[0:2], 'Cy')
        delta_Cn_r30 = self.py_lookup.interp_2d(inp_clip[0:2], 'Cn_r30') - self.py_lookup.interp_3d(temp_clip, 'Cn')
        delta_Cl_r30 = self.py_lookup.interp_2d(inp_clip[0:2], 'Cl_r30') - self.py_lookup.interp_3d(temp_clip, 'Cl')

        hifi_rudder = [
            delta_Cy_r30,
            delta_Cn_r30,
            delta_Cl_r30
        ]

        # hifi_ailerons (clipped)
        delta_Cy_a20     = self.py_lookup.interp_2d(inp_clip[0:2], 'Cy_a20') -      self.py_lookup.interp_2d(inp_clip[0:2], 'Cy')
        delta_Cy_a20_lef = self.py_lookup.interp_2d(inp_clip[0:2], 'Cy_a20_lef') -  self.py_lookup.interp_2d(inp_clip[0:2], 'Cy_lef') - delta_Cy_a20
        delta_Cn_a20     = self.py_lookup.interp_2d(inp_clip[0:2], 'Cn_a20') -      self.py_lookup.interp_3d(temp_clip, 'Cn')
        delta_Cn_a20_lef = self.py_lookup.interp_2d(inp_clip[0:2], 'Cn_a20_lef') -  self.py_lookup.interp_2d(inp_clip[0:2], 'Cn_lef') - delta_Cn_a20
        delta_Cl_a20     = self.py_lookup.interp_2d(inp_clip[0:2], 'Cl_a20') -      self.py_lookup.interp_3d(temp_clip, 'Cl')
        delta_Cl_a20_lef = self.py_lookup.interp_2d(inp_clip[0:2], 'Cl_a20_lef') -  self.py_lookup.interp_2d(inp_clip[0:2], 'Cl_lef') - delta_Cl_a20 
        
        hifi_ailerons = [
            delta_Cy_a20,
            delta_Cy_a20_lef,
            delta_Cn_a20,
            delta_Cn_a20_lef,
            delta_Cl_a20,
            delta_Cl_a20_lef
        ]

        # hifi_other_coeffs
        delta_Cnbeta = self.py_lookup.interp_1d(inp[0:1], 'delta_CNbeta')
        delta_Clbeta = self.py_lookup.interp_1d(inp[0:1], 'delta_CLbeta')
        delta_Cm = self.py_lookup.interp_1d(inp[0:1], 'delta_Cm')
        eta_el = self.py_lookup.interp_1d(inp[2:3], 'eta_el')
        delta_Cm_ds = 0.

        hifi_other_coeffs = [
            delta_Cnbeta,
            delta_Clbeta,
            delta_Cm,
            eta_el,            
            delta_Cm_ds                 
        ]
        
        return hifi_C, hifi_damping, hifi_C_lef, hifi_damping_lef, hifi_rudder, hifi_ailerons, hifi_other_coeffs

    def c_lut_wrap(self, alpha, beta, el):

        inp = np.array([
            ptu.to_numpy(alpha), 
            ptu.to_numpy(beta), 
            ptu.to_numpy(el)
            ])

        # pass alpha, beta, el
        hifi_C = self.c_lookup.hifi_C(inp)
        
        # pass alpha
        hifi_damping = self.c_lookup.hifi_damping(inp[0:1])
        
        # pass alpha, beta
        hifi_C_lef = self.c_lookup.hifi_C_lef(inp[0:2])
        
        # pass alpha
        hifi_damping_lef = self.c_lookup.hifi_damping_lef(inp[0:1])
        
        # pass alpha, beta
        hifi_rudder = self.c_lookup.hifi_rudder(inp[0:2])
        
        # pass alpha, beta
        hifi_ailerons = self.c_lookup.hifi_ailerons(inp[0:2])
        
        # pass alpha, el
        hifi_other_coeffs = self.c_lookup.hifi_other_coeffs(inp[::2])

        return hifi_C, hifi_damping, hifi_C_lef, hifi_damping_lef, hifi_rudder, hifi_ailerons, hifi_other_coeffs

    """
    The equations of motion describe how the motion of the aircraft evolves
    over time given state-input pairs. It relies on aerodynamic lookup tables
    for nonlinear dynamics.
    """

    def accels(self, state, xdot):
            
        grav = 32.174
        
        sina = torch.sin(state[7])
        cosa = torch.cos(state[7])
        sinb = torch.sin(state[8])
        cosb = torch.cos(state[8])
        vel_u = state[6]*cosb*cosa
        vel_v = state[6]*sinb
        vel_w = state[6]*cosb*sina
        u_dot =          cosb*cosa*xdot[6] \
                - state[6]*sinb*cosa*xdot[8] \
                - state[6]*cosb*sina*xdot[7]
        v_dot =          sinb*xdot[6] \
                + state[6]*cosb*xdot[8]
        w_dot =          cosb*sina*xdot[6] \
                - state[6]*sinb*sina*xdot[8] \
                + state[6]*cosb*cosa*xdot[7]
        nx_cg = 1.0/grav*(u_dot + state[10]*vel_w - state[11]*vel_v) \
                + torch.sin(state[4])
        ny_cg = 1.0/grav*(v_dot + state[11]*vel_u - state[9]*vel_w) \
                - torch.cos(state[4])*torch.sin(state[3])
        nz_cg = -1.0/grav*(w_dot + state[9]*vel_v - state[10]*vel_u) \
                + torch.cos(state[4])*torch.cos(state[3])
                
        return nx_cg, ny_cg, nz_cg 

    def equations_of_motion(self, xu):
        g    = 32.17                            # gravity, ft/s^2
        m    = 636.94                           # mass, slugs
        B    = 30.0                             # span, ft
        S    = 300.0                            # planform area, ft^2
        cbar = 11.32                            # mean aero chord, ft
        xcgr = 0.35                             # reference center of gravity as a fraction of cbar
        xcg  = 0.30                             # center of gravity as a fraction of cbar
        
        Heng = 0.0                              # turbine momentum along roll axis
        r2d  = torch.rad2deg(ptu.from_numpy(np.array(1.)))   # radians to degrees
        
        # NasaData translated via eq. 2.4-6 on pg 80 of Stevens and Lewis
        
        Jy  = 55814.0                           # slug-ft^2
        Jxz = 982.0                             # slug-ft^2   
        Jz  = 63100.0                           # slug-ft^2
        Jx  = 9496.0                            # slug-ft^2
        
        # instantiate xdot
        xdot = ptu.from_numpy(np.zeros(xu.shape))
        
        # In[states]
        
        npos  = xu[0]   # north position
        epos  = xu[1]   # east position
        alt   = xu[2]   # altitude
        phi   = xu[3]   # orientation angles in rad
        theta = xu[4]
        psi   = xu[5]
        
        vt    = xu[6]     # total velocity
        alpha = xu[7] * r2d # angle of attack in degrees
        beta  = xu[8] * r2d # sideslip angle in degrees
        P     = xu[9]    # Roll Rate --- rolling  moment is Lbar
        Q     = xu[10]    # Pitch Rate--- pitching moment is M
        R     = xu[11]    # Yaw Rate  --- yawing   moment is N
        
        sin = torch.sin
        cos = torch.cos
        tan = torch.tan
        
        sa    = sin(xu[7]) # sin(alpha)
        ca    = cos(xu[7]) # cos(alpha)
        sb    = sin(xu[8]) # sin(beta)
        cb    = cos(xu[8]) # cos(beta)
        tb    = tan(xu[8]) # tan(beta)
        
        st    = sin(theta)
        ct    = cos(theta)
        tt    = tan(theta)
        sphi  = sin(phi)
        cphi  = cos(phi)
        spsi  = sin(psi)
        cpsi  = cos(psi)
        
        if vt < 0.01: vt = 0.01
        
        # In[Control inputs]
        
        T     = xu[12]   # thrust
        el    = xu[13]   # Elevator setting in degrees
        ail   = xu[14]   # Ailerons mex setting in degrees
        rud   = xu[15]   # Rudder setting in degrees
        lef   = xu[16]   # Leading edge flap setting in degrees
        
        # dail  = ail/20.0;   aileron normalized against max angle
        # The aileron was normalized using 20.0 but the NASA report and
        # S&L both have 21.5 deg. as maximum deflection.
        # As a result...
        dail  = ail/21.5
        drud  = rud/30.0  # rudder normalized against max angle
        dlef  = (1 - lef/25.0)  # leading edge flap normalized against max angle
        
        # In[Atmospheric effects]
        mach, qbar, ps = self.atmos(alt, vt)
        
        # In[Navigation equations]
        U = vt*ca*cb  # directional velocities
        V = vt*sb
        W = vt*sa*cb
        
        # nposdot
        xdot[0] = U*(ct*cpsi) + \
                    V*(sphi*cpsi*st - cphi*spsi) + \
                    W*(cphi*st*cpsi + sphi*spsi)
                    
        # eposdot
        xdot[1] = U*(ct*spsi) + \
                    V*(sphi*spsi*st + cphi*cpsi) + \
                    W*(cphi*st*spsi - sphi*cpsi)
                    
        # altdot
        xdot[2] = U*st - V*(sphi*ct) - W*(cphi*ct)
        
        # In[Kinematic equations]
        
        # phidot
        xdot[3] = P + tt*(Q*sphi + R*cphi)


        # theta dot
        xdot[4] = Q*cphi - R*sphi

        # psidot
        xdot[5] = (Q*sphi + R*cphi)/ct
        
        # In[Table Lookup]
        if self.lookup_type == 'Py':
            
            hifi_C, hifi_damping, hifi_C_lef, hifi_damping_lef, hifi_rudder, \
                hifi_ailerons, hifi_other_coeffs = self.py_lut_wrap(alpha, beta, el)

        elif self.lookup_type == 'C':
            
            hifi_C, hifi_damping, hifi_C_lef, hifi_damping_lef, hifi_rudder, \
                    hifi_ailerons, hifi_other_coeffs = self.c_lut_wrap(alpha, beta, el) 

        # unwrap the returned lookup table coefficients

        Cx, Cz, Cm, Cy, Cn, Cl = hifi_C
        
        Cxq, Cyr, Cyp, Czq, Clr, Clp, Cmq, Cnr, Cnp = hifi_damping
        
        delta_Cx_lef, delta_Cz_lef, delta_Cm_lef, delta_Cy_lef, delta_Cn_lef, \
            delta_Cl_lef = hifi_C_lef
        
        # we expect delta_Czq_lef not to be used - this is not a bug
        delta_Cxq_lef, delta_Cyr_lef, delta_Cyp_lef, delta_Czq_lef, \
            delta_Clr_lef, delta_Clp_lef, delta_Cmq_lef, delta_Cnr_lef, \
                delta_Cnp_lef = hifi_damping_lef
        
        delta_Cy_r30, delta_Cn_r30, delta_Cl_r30 = hifi_rudder
        
        delta_Cy_a20, delta_Cy_a20_lef, delta_Cn_a20, delta_Cn_a20_lef, \
            delta_Cl_a20, delta_Cl_a20_lef = hifi_ailerons
        
        delta_Cnbeta, delta_Clbeta, delta_Cm, eta_el, delta_Cm_ds = hifi_other_coeffs
              
        # In[compute Cx_tot, Cz_tot, Cm_tot, Cy_tot, Cn_tot, and Cl_tot]
            # (as on NASA report p37-40)

        # Cx_tot
        dXdQ = (cbar/(2*vt))*(Cxq + delta_Cxq_lef*dlef)

        Cx_tot = Cx + delta_Cx_lef*dlef + dXdQ*Q

        # Cz_tot
        dZdQ = (cbar/(2*vt))*(Czq + delta_Cz_lef*dlef)

        Cz_tot = Cz + delta_Cz_lef*dlef + dZdQ*Q

        # Cm_tot
        dMdQ = (cbar/(2*vt))*(Cmq + delta_Cmq_lef*dlef)

        Cm_tot = Cm*eta_el + Cz_tot*(xcgr-xcg) + delta_Cm_lef*dlef + dMdQ*Q + \
            delta_Cm + delta_Cm_ds

        # Cy_tot
        dYdail = delta_Cy_a20 + delta_Cy_a20_lef*dlef

        dYdR = (B/(2*vt))*(Cyr + delta_Cyr_lef*dlef)

        dYdP = (B/(2*vt))*(Cyp + delta_Cyp_lef*dlef)

        Cy_tot = Cy + delta_Cy_lef*dlef + dYdail*dail + delta_Cy_r30*drud + \
            dYdR*R + dYdP*P

        # Cn_tot
        dNdail = delta_Cn_a20 + delta_Cn_a20_lef*dlef

        dNdR = (B/(2*vt))*(Cnr + delta_Cnr_lef*dlef)

        dNdP = (B/(2*vt))*(Cnp + delta_Cnp_lef*dlef)

        Cn_tot = Cn + delta_Cn_lef*dlef - Cy_tot*(xcgr-xcg)*(cbar/B) + \
            dNdail*dail + delta_Cn_r30*drud + dNdR*R + dNdP*P + \
                delta_Cnbeta*beta

        # Cl_tot
        dLdail = delta_Cl_a20 + delta_Cl_a20_lef*dlef

        dLdR = (B/(2*vt))*(Clr + delta_Clr_lef*dlef)

        dLdP = (B/(2*vt))*(Clp + delta_Clp_lef*dlef)

        Cl_tot = Cl + delta_Cl_lef*dlef + dLdail*dail + delta_Cl_r30*drud + \
            dLdR*R + dLdP*P + delta_Clbeta*beta

        # In[compute Udot,Vdot, Wdot,(as on NASA report p36)]
        
        Udot = R*V - Q*W - g*st + qbar*S*Cx_tot/m + T/m

        Vdot = P*W - R*U + g*ct*sphi + qbar*S*Cy_tot/m
        
        Wdot = Q*U - P*V + g*ct*cphi + qbar*S*Cz_tot/m
        
        # In[vt_dot equation (from S&L (Stevens and Lewis), p82)]
        
        xdot[6] = (U*Udot + V*Vdot + W*Wdot)/vt
        
        # In[alpha_dot equation]
        
        xdot[7] = (U*Wdot - W*Udot)/(U*U + W*W)
        
        # In[beta_dot equation]

        # we clone xdot[6] to allow torch.autograd.functional.jacobian to calculate the
        # linearisation using the graph.
        # this is VERIFIED to produce the same result as MATLAB for the A matrix [8,6]
        # (betadot (xdot[8]) w.r.t V (xdot[6])) position at trim steady wings level
        gradient_dummy_var = torch.clone(xdot[6])
        xdot[8] = (Vdot*vt - V*gradient_dummy_var)/(vt*vt*cb)
        
        # In[compute Pdot, Qdot, and Rdot (as in Stevens and Lewis p32)]
        
        L_tot = Cl_tot*qbar*S*B         # get moments from coefficients
        M_tot = Cm_tot*qbar*S*cbar 
        N_tot = Cn_tot*qbar*S*B
        
        denom = Jx*Jz - Jxz*Jxz
        
        # In[Pdot]
        
        xdot[9] =  (Jz*L_tot + Jxz*N_tot - (Jz*(Jz-Jy)+Jxz*Jxz)*Q*R + \
                    Jxz*(Jx-Jy+Jz)*P*Q + Jxz*Q*Heng)/denom
            
        # In[Qdot]
        
        xdot[10] = (M_tot + (Jz-Jx)*P*R - Jxz*(P*P-R*R) - R*Heng)/Jy

        # In[Rdot]
        
        xdot[11] = (Jx*N_tot + Jxz*L_tot + (Jx*(Jx-Jy)+Jxz*Jxz)*P*Q - \
                    Jxz*(Jx-Jy+Jz)*Q*R +  Jx*Q*Heng)/denom
            
        # In[Create accelerations anx_cg, any_cg, anz_cg as outputs]
        
        # anx_cg, any_cg, anz_cg = accels(xu, xdot)

        accelerations = self.accels(xu, xdot)
        # not the primary xdot values 
        #xdot[12] = anx_cg
        #xdot[13] = any_cg
        #xdot[14] = anz_cg
        #xdot[15] = mach
        #xdot[16] = qbar
        #xdot[17] = ps
        atmospherics = torch.stack([mach,qbar,ps])

        return xdot, accelerations, atmospherics

if __name__ == "__main__":

    """Example usage of code"""
    import common.pytorch_utils as ptu
    import numpy as np
    import torch
    import trim
    import copy
    import plot
    from tqdm import tqdm

    # gpu activation
    ptu.init_gpu(
        use_gpu=not False,
        gpu_id=0
    )

    def autonomous(nlplant, x0, u, t_end, dt):
        print(f"Using {nlplant.lookup_type} LUTs")
        x_traj = ptu.from_numpy(np.zeros([int(t_end/dt),len(x0)]))
        x = x0
        for idx, t in tqdm(enumerate(np.linspace(start=0,stop=t_end,num=int(t_end/dt)))):
            xdot = nlplant.forward(x0, u)[0]
            x += xdot*dt
            x_traj[idx,:] = x 
        return x_traj

    # instantiate a couple nlplants
    nlplant_c = NLPlant(lookup_type='C')
    nlplant_py = NLPlant(lookup_type='Py')

    # get an initial condition by trimming the aircraft at wings level flight
    # using a couple initial guesses

    # start at velocity of 700 ft/s at 10000 ft altitude
    altitude = 10000 # ft
    velocity = 700   # ft/s

    thrust = 5000;          # thrust, lbs
    elevator = -0.09;       # elevator, degrees
    alpha = 8.49;           # AOA, degrees
    rudder = 0.01;          # rudder angle, degrees
    aileron = -0.01;        # aileron, degrees

    trim_states = trim.wings_level(thrust, elevator, alpha, aileron, rudder, velocity, altitude, nlplant_c, verbose=False)

    # calculate dLEF
    rho0 = 2.377e-3
    tfac = 1 - 0.703e-5*altitude
    temp = 519*tfac
    if altitude >= 35000:
        temp = 390
    rho = rho0*tfac**4.14
    qbar = 0.5*rho*velocity**2
    ps = 1715*rho*temp
    dLEF = 1.38*trim_states[2]*180/np.pi - 9.05*qbar/ps + 1.45

    x0_np = np.array([
            0,                  # npos
            0,                  # epos
            altitude,                # altitude (ft)
            0*np.pi/180,      # phi (rad)
            trim_states[2],             # theta (rad)
            0*np.pi/180,      # psi (rad)
            velocity,                # velocity (ft/s)
            trim_states[2],             # alpha (rad)
            0,                  # beta (rad)
            0*np.pi/180,        # p (rad/s)
            0*np.pi/180,        # q ( rad/s)
            0*np.pi/180,        # r (rad/s)
            trim_states[0],             # thrust (lbs)
            trim_states[1],             # elevator (deg)
            trim_states[3],             # aileron (deg)
            trim_states[4],             # rudder (deg)
            dLEF,               # dLEF (deg)
            -trim_states[2]*180/np.pi   # LF_state (deg)
        ])

    x0 = ptu.from_numpy(copy.deepcopy(x0_np))
    u0 = ptu.from_numpy(copy.deepcopy(x0_np[12:16]))

    t_end = 10
    dt = 0.01

    x_seq_c = autonomous(nlplant_c, torch.clone(x0), torch.clone(u0), t_end, dt)
    x_seq_py = autonomous(nlplant_py, torch.clone(x0), torch.clone(u0), t_end, dt)

    t_seq = np.linspace(start=0, stop=t_end, num=int(t_end/dt))

    plot.states(ptu.to_numpy(x_seq_py), t_seq)

