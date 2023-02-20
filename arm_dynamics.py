import numpy as np
from casadi import *

class DoubleJointedArmDynamics(object):
    def __init__(self):
        self.constants = DoubleJointedArmConstants()
    
    def dynamics(self, state, input):
        c = self.constants
        theta_vec = state[:2]
        omega_vec = state[2:4]

        h = c.m2 * c.l1 * c.r2
        m_1 = c.m1*c.r1**2 + c.m2*c.l1**2 + c.I1
        m_2 = c.m2*c.r2**2 + c.I2
        c2 = cos(theta_vec[1])
        M = vertcat(
            horzcat(m_1 + 2*h*c2, m_2 + h*c2),
            horzcat(m_2 + h*c2, m_2))
        
        s2 = sin(theta_vec[1])
        C = vertcat(
            horzcat(-h*s2*omega_vec[1], -h*s2*(omega_vec[0]+omega_vec[1])),
            horzcat(h*s2*omega_vec[0], 0))

        tau_g = c.g * vertcat(c.m1*c.r1 + c.m2*c.l1 + c.m2*c.r2, c.m2*c.r2) * cos(vertcat(theta_vec[0], theta_vec[0] + theta_vec[1]))

        alpha_vec = solve(M, c.B @ input - (C + c.Kb) @ omega_vec - tau_g)

        return vertcat(omega_vec, alpha_vec)
    
    def feed_forward(self, pos, vel, accel):
        c = self.constants
        h = c.m2 * c.l1 * c.r2
        m_1 = c.m1*c.r1**2 + c.m2*c.l1**2 + c.I1
        m_2 = c.m2*c.r2**2 + c.I2
        c2 = cos(pos[1])
        M = vertcat(
            horzcat(m_1 + 2*h*c2, m_2 + h*c2),
            horzcat(m_2 + h*c2, m_2))
        
        s2 = sin(pos[1])
        C = vertcat(
            horzcat(-h*s2*vel[1], -h*s2*(vel[0]+vel[1])),
            horzcat(h*s2*vel[0], 0))

        tau_g = c.g * vertcat(c.m1*c.r1 + c.m2*c.l1 + c.m2*c.r2, c.m2*c.r2) * cos(vertcat(pos[0], pos[0] + pos[1]))

        return solve(c.B, M @ accel + (C + c.Kb) @ vel + tau_g)

    def current_draw(self, vels, input):
        stall_voltage = input - vels * np.array([self.constants.G1, self.constants.G2]) / self.constants.Kv
        return stall_voltage / self.constants.Rm * np.array([self.constants.N1, self.constants.N2])
    
    def to_cartesian_ee(self, state):
        x = self.constants.l1*np.cos(state[0]) + self.constants.l2*np.cos(state[0] + state[1])
        y = self.constants.l1*np.sin(state[0]) + self.constants.l2*np.sin(state[0] + state[1])
        return (x,y)
    
    def to_cartesian_j1(self, state):
        x = self.constants.l1*np.cos(state[0])
        y = self.constants.l1*np.sin(state[0])
        return (x,y)
        
class DoubleJointedArmConstants(object):
    def __init__(self):
        # Length of segments
        self.l1 = 46.25 * .0254
        self.l2 = 41.80 * .0254

        # Mass of segments
        self.m1 = 9.34 * .4536
        self.m2 = 9.77 * .4536

        # Distance from pivot to CG for each segment
        self.r1 = 21.64 * .0254
        self.r2 = 26.70 * .0254

        # Moment of inertia about CG for each segment
        self.I1 = 2957.05 * .0254*.0254 * .4536
        self.I2 = 2824.70 * .0254*.0254 * .4536

        # Gearing of each segment
        self.G1 = 140.
        self.G2 = 90.

        # Number of motors in each gearbox
        self.N1 = 1
        self.N2 = 2

        # Gravity
        self.g = 9.81

        self.stall_torque = 3.36
        self.free_speed = 5880.0 * 2.0*np.pi/60.0
        self.stall_current = 166

        self.Rm = 12.0/self.stall_current

        self.Kv = self.free_speed / 12.0
        self.Kt = self.stall_torque / self.stall_current

        # K3*Voltage - K4*velocity = motor torque
        self.B = np.array([[self.N1*self.G1, 0], [0, self.N2*self.G2]])*self.Kt/self.Rm
        self.Kb = np.array([[self.G1*self.G1*self.N1, 0], [0, self.G2*self.G2*self.N2]])*self.Kt/self.Kv/self.Rm