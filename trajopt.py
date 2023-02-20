import numpy as np
from casadi import *
import arm_dynamics
import matplotlib.pyplot as plt
import json

arm = arm_dynamics.DoubleJointedArmDynamics()
N = 200
initial_state = np.array([2*np.pi/3.0, 2*np.pi/3.0,0.0,0.0])
final_state = np.array([np.pi/3.0, -2*np.pi/3.0, 0.0, 0.0])
max_u = 8
max_accel = np.array([10, 20])
max_current = np.array([40, 25])
alpha = 0
T0 = 2.0

def rect_zone(pt, topleft, bottomright):
    (x1, y1) = topleft
    (x2, y2) = bottomright
    (x,y) = pt
    return [x1 <= x, x <= x2, y2 <= y, y <= y1]

def rect_obstacle(pt, topleft, bottomright):
    (x1, y1) = topleft
    (x2, y2) = bottomright
    xc = (x1 + x2)/2
    yc = (y1 + y2)/2
    w = x1 - x2
    h = y1 - y2
    (x,y) = pt
    return fmax(fabs(2*(x - xc)/w), fabs(2*(y - yc)/h)) > 1

# rect_obstacle(pt, (1, .75), (5, .25)), 
constraints = lambda pt: [rect_zone(pt, (-60*.0254, 65*.0254), (60*.0254, -7*.0254))]

class DynamicsSolver:
    def solve():
        opti = Opti()

        X = opti.variable(4, N + 1)
        U = opti.variable(2, N)
        pos = X[:2,:]
        vel = X[2:,:]

        T = opti.variable()

        dt = T/(N-1)

        for k in range(N):
            k1 = arm.dynamics(X[:,k], U[:,k])
            k2 = arm.dynamics(X[:,k] + dt/2*k1, U[:,k])
            k3 = arm.dynamics(X[:,k] + dt/2*k2, U[:,k])
            k4 = arm.dynamics(X[:,k] + dt*k3, U[:,k])
            next = X[:,k] + dt/6*(k1 + 2*k2 + 2*k3 + k4)

            opti.subject_to(X[:,k+1] == next)
            opti.subject_to(opti.bounded(-max_accel, k1[2:], max_accel))
            opti.subject_to(opti.bounded(-max_current, arm.current_draw(X[2:4,k], U[:,k]), max_current))
            

        opti.subject_to(T >= .01)
        opti.subject_to(opti.bounded(-max_u, U, max_u))

        opti.subject_to(X[:,0] == initial_state)
        opti.subject_to(X[:,-1] == final_state)

        for k in range(N+1):
            (x0, y0) = arm.to_cartesian_j1(X[:,k])
            (x1, y1) = arm.to_cartesian_ee(X[:,k])
            n = 3
            for (x, y) in zip(np.linspace(x0, x1, n), np.linspace(y0, y1, n)):
                opti.subject_to(constraints((x,y)))

        opti.set_initial(T, T0)
        opti.set_initial(U, 0.0 * np.ones((2,N)))
        opti.set_initial(pos[0,:], np.linspace(initial_state[0], final_state[0], N+1))
        opti.set_initial(pos[1,:], np.linspace(initial_state[1], final_state[1], N+1))
        opti.set_initial(vel[0,:], (final_state[0] - initial_state[0]) / T0 * np.ones(N+1))
        opti.set_initial(vel[1,:], (final_state[1] - initial_state[1]) / T0 * np.ones(N+1))

        opti.solver("ipopt")

        opti.minimize(T + alpha*dt*sumsqr(U))

        opti.solve()

        X_opt = opti.value(X)
        U_opt = opti.value(U)
        T_opt = opti.value(T)

        tvec = np.linspace(0, T_opt, N + 1)

        return (tvec, X_opt[:2,:], U_opt)

class VoltageSolver:
    def solve():
        opti = Opti()

        X = opti.variable(2, N + 1)
        U = opti.variable(2, N)

        T = opti.variable()

        dt = T/(N-1)

        for k in range(N):
            last_pos = X[:, k - 1 if k > 0 else k]
            pos = X[:, k]
            next_pos = X[:, k + 1]

            last_vel = (pos - last_pos) / dt
            next_vel = (next_pos - pos) / dt
            avg_vel = (next_pos - last_pos) / 2 / dt

            accel = (next_vel - last_vel) / dt

            voltage = arm.feed_forward(pos, avg_vel, accel)

            opti.subject_to(opti.bounded(-max_u, voltage, max_u))
            opti.subject_to(opti.bounded(-max_accel, accel, max_accel))
            opti.subject_to(opti.bounded(-max_current, arm.current_draw(avg_vel, U[:,k]), max_current))
            opti.subject_to(U[:,k] == voltage)

        opti.subject_to(X[:,0] == X[:,1])
        opti.subject_to(X[:,-1] == X[:,-2])
        opti.subject_to(opti.bounded(-np.pi, X[1,:], np.pi))

        opti.subject_to(T >= .01)
        opti.subject_to(T <= 10)

        for k in range(2):
            opti.subject_to(cos(X[k,0]) == cos(initial_state[k]))
            opti.subject_to(sin(X[k,0]) == sin(initial_state[k]))
            opti.subject_to(cos(X[k,-1]) == cos(final_state[k]))
            opti.subject_to(sin(X[k,-1]) == sin(final_state[k]))

        for k in range(N+1):
            (x0, y0) = arm.to_cartesian_j1(X[:,k])
            (x1, y1) = arm.to_cartesian_ee(X[:,k])
            n = 7
            for j in range(n):
                for c in constraints((x0 + j*(x1-x0)/(n-1), y0 + j*(y1-y0)/(n-1))):
                    opti.subject_to(c)

        opti.set_initial(T, T0)
        opti.set_initial(U, 0.0 * np.ones((2,N)))
        opti.set_initial(X[0,:], np.linspace(initial_state[0], final_state[0], N+1))
        opti.set_initial(X[1,:], np.linspace(initial_state[1], final_state[1], N+1))

        opti.solver("ipopt")

        opti.minimize(T + alpha*dt*sumsqr(U))

        opti.solve()

        X_opt = opti.value(X)
        U_opt = opti.value(U)
        T_opt = opti.value(T)

        vels = np.zeros_like(X_opt)
        for k in range(N+1):
            last_pos = X_opt[:, k - 1 if k > 0 else k]
            next_pos = X_opt[:, k + 1 if k < N else k]
            vels[:,k] = (next_pos - last_pos) / 2 / (T_opt/(N-1))
        X_opt = np.concatenate((X_opt, vels))

        tvec = np.linspace(0, T_opt, N + 1)

        return (tvec, X_opt, U_opt)

(tvec, X_opt, U_opt) = VoltageSolver.solve()

log = []
for k in range(N+1):
    d = {'t': tvec[k], 'q1': X_opt[0,k], 'q2': X_opt[1,k], 'q1d': X_opt[2,k], 'q2d': X_opt[3,k]}
    log.append(d)

with open("traj.json", "w") as outfile:
    json.dump(log, outfile)

fig, ax = plt.subplots(1,2)
ax[0].plot(tvec, X_opt[0,:])
ax[0].plot(tvec, X_opt[1,:])

ax[1].plot(tvec[:-1], U_opt[0,:])
ax[1].plot(tvec[:-1], U_opt[1,:])

plt.show()