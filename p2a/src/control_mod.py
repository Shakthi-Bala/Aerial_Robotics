import numpy as np
import math
from pyquaternion import Quaternion
from numpy.linalg import norm
import scipy

class pid:
    def __init__(self, kp, ki, kd, filter_tau, dt, dim=1, minVal=-1, maxVal=1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.filter_tau = filter_tau
        self.dt = dt
        self.minVal = minVal
        self.maxVal = maxVal

        if dim == 1:
            self.prev_filter_val = 0.0
            self.prev_err = 0.0
            self.prev_integral = 0.0
        else:
            self.prev_err = np.zeros(dim, dtype=float)
            self.prev_filter_val = np.zeros(dim, dtype=float)
            self.prev_integral = np.zeros(dim, dtype=float)
    
    def step(self, dsOrErr, current_state=None):
        # Error
        if current_state is None:
            err = dsOrErr
        else:
            desired_state = dsOrErr
            err = desired_state - current_state

        # Derivative (filtered, numerically stable)
        err_der = (err - self.prev_err) / self.dt
        alpha = self.dt / (self.filter_tau + self.dt)   # better than dt/tau
        err_der_filtered = alpha * err_der + (1 - alpha) * self.prev_filter_val

        # Integral (with simple anti-windup later)
        err_integral = err * self.dt + self.prev_integral

        # PID
        out = self.kp * err + self.kd * err_der_filtered + self.ki * err_integral

        # NaN/Inf guard
        if np.any(np.isnan(out)) or np.any(np.isinf(out)):
            raise Exception('PID blew up: output NaN/Inf')

        # Update states
        self.prev_err = err
        self.prev_filter_val = err_der_filtered
        # Clamp integral
        self.prev_integral = np.clip(err_integral, self.minVal, self.maxVal)

        # Clamp output
        out = np.clip(out, self.minVal, self.maxVal)
        return out

class quad_control:
    def __init__(self):
        # IMPORTANT: match simulator dt (often 0.02). Change here if needed.
        dt = 0.01
        self.dt = dt
        filter_tau = 0.04

        # tello-ish params
        self.param_mass = 0.08
        self.linearThrustToU = self.param_mass * 9.81 * 2 / 4

        maxAcc = 5.0
        maxVel = 20.0
        maxAng = 30. * math.pi / 180.
        self.maxRate = 1.5
        maxAct = 1.0

        minAcc = -maxAcc
        minVel = -maxVel
        self.minRate = -self.maxRate

        # --- Outer (position -> velocity) OFF while tuning velocity ---
        self.x_pid = pid(2.0, 0.0, 1.0, 0.02, dt, minVal=minVel, maxVal=maxVel)
        self.y_pid = pid(2.0, 0.0, 1.0, 0.02, dt, minVal=minVel, maxVal=maxVel)
        self.z_pid = pid(2.0, 0.0, 1.0, 0.02, dt, minVal=minVel, maxVal=maxVel)

        # --- Inner (velocity -> accel) ---
        self.vx_pid = pid(1.1, 1.2, 0.3, filter_tau, dt, minVal=minAcc, maxVal=maxAcc)
        self.vy_pid = pid(1.3, 1.2, 0.3, filter_tau, dt, minVal=minAcc, maxVal=maxAcc)
        self.vz_pid = pid(1.3, 1.2, 0.3, filter_tau, dt, minVal=minAcc, maxVal=maxAcc)

        # Attitude / rate loops
        self.tau_angle = 0.3
        self.angle_sf = np.array((1.0, 1.0, 0.4))  # deprioritize yaw
        kp_angvel = 6.0
        self.p_pid = pid(kp_angvel, 0, kp_angvel/15., filter_tau, dt, minVal=-maxAct, maxVal=maxAct)
        self.q_pid = pid(kp_angvel, 0, kp_angvel/15., filter_tau, dt, minVal=-maxAct, maxVal=maxAct)
        self.r_pid = pid(kp_angvel, 0, kp_angvel/15., filter_tau, dt, minVal=-maxAct, maxVal=maxAct)

        # Logging
        self.current_time = 0.0
        self.timeArray = 0.0
        self.controlArray = np.array([0., 0., 0., 0.])

    def step(self, X, WP, VEL_SP, ACC_SP):
        """Quadrotor cascaded controller (ENU: z up)."""
        # State unpack
        xyz = X[0:3]
        vxyz = X[3:6]
        quat_list = X[6:10]
        pqr = X[10:13]
        quat = Quaternion(quat_list)

        # --- Outer loop: position -> velocity setpoint (currently P=0 while tuning velocity) ---
        vx_sp = VEL_SP[0] + self.x_pid.step(WP[0], xyz[0])
        vy_sp = VEL_SP[1] + self.y_pid.step(WP[1], xyz[1])
        vz_sp = VEL_SP[2] + self.z_pid.step(WP[2], xyz[2])
        vxyz_sp = np.array([vx_sp, vy_sp, vz_sp])

        # --- Inner loop: velocity -> acceleration setpoint ---
        acc_x_sp = ACC_SP[0] + self.vx_pid.step(vxyz_sp[0], vxyz[0])
        acc_y_sp = ACC_SP[1] + self.vy_pid.step(vxyz_sp[1], vxyz[1])
        acc_z_sp = ACC_SP[2] + self.vz_pid.step(vxyz_sp[2], vxyz[2])

        # --- ENU gravity feedforward ---
        g = 9.81
        a_sp = np.array([acc_x_sp, acc_y_sp, acc_z_sp])    # desired accel (no gravity)
        f_inertial = a_sp + np.array([0.0, 0.0, g])        # specific force to generate

        # --- SAFE attitude target: align world +Z to f_inertial ---
        z_world = np.array([0.0, 0.0, 1.0])
        nF = np.linalg.norm(f_inertial)
        if nF < 1e-8:
            quat_wo_yaw = Quaternion()  # identity
        else:
            b3d = f_inertial / nF
            cosA = float(np.clip(np.dot(z_world, b3d), -1.0, 1.0))
            rot_axis = np.cross(z_world, b3d)
            s = np.linalg.norm(rot_axis)
            if s < 1e-8:
                if cosA > 0.0:
                    quat_wo_yaw = Quaternion()
                else:
                    quat_wo_yaw = Quaternion(axis=[1.0, 0.0, 0.0], radians=np.pi)
            else:
                rot_axis /= s
                angle = math.atan2(s, cosA)
                quat_wo_yaw = Quaternion(axis=rot_axis, radians=angle)

        # --- Yaw setpoint quaternion (define it BEFORE use) ---
        yaw_sp = float(WP[3]) if len(WP) >= 4 else 0.0
        quat_yaw = Quaternion(axis=[0.0, 0.0, 1.0], radians=yaw_sp)

        quat_sp = quat_wo_yaw * quat_yaw

        # --- Quaternion P controller -> desired body rates ---
        err_quat = quat.inverse * quat_sp
        pqr_sp = 2.0 / self.tau_angle * np.sign(err_quat.w) * np.array([err_quat.x, err_quat.y, err_quat.z])
        pqr_sp = np.multiply(pqr_sp, self.angle_sf)
        pqr_sp = np.clip(pqr_sp, self.minRate, self.maxRate)

        # --- Angular-rate PID ---
        tau_x = self.p_pid.step(pqr_sp[0], pqr[0])
        tau_y = self.q_pid.step(pqr_sp[1], pqr[1])
        tau_z = self.r_pid.step(pqr_sp[2], pqr[2])

        # --- Thrust / mixer ---
        netSpecificThrustFromRotors = norm(f_inertial)   # N/kg
        netThrust = netSpecificThrustFromRotors * self.param_mass
        thrustPerRotor = netThrust / 4.0
        throttle = thrustPerRotor / self.linearThrustToU

        u1 = throttle - tau_x + tau_y + tau_z
        u2 = throttle + tau_x - tau_y + tau_z
        u3 = throttle + tau_x + tau_y - tau_z
        u4 = throttle - tau_x - tau_y - tau_z
        U = np.clip(np.array([u1, u2, u3, u4]), 0.0, 1.0)

        # Logger
        self.controlArray = np.vstack((self.controlArray, np.array((throttle, tau_x, tau_y, tau_z))))
        self.timeArray = np.append(self.timeArray, self.current_time)
        self.current_time += self.dt
        scipy.io.savemat('./log/control.mat', {'control_time': self.timeArray, 'control_premix': self.controlArray})

        return U, f_inertial,a_sp

class QuadrotorController:
    """Thin wrapper to track a time-parameterized trajectory."""
    def __init__(self, drone_params):
        self.params = drone_params
        self.controller = quad_control()
        self.trajectory_points = None
        self.trajectory_velocities = None
        self.trajectory_accelerations = None
        self.time_points = None
        self.position_errors = []
        self.velocity_errors = []
        
    def set_trajectory(self, trajectory_points, time_points, velocities, accelerations):
        self.trajectory_points = trajectory_points
        self.time_points = time_points
        self.trajectory_velocities = velocities
        self.trajectory_accelerations = accelerations

    def get_desired_state(self, t):
        if (self.trajectory_points is None or len(self.trajectory_points) == 0 or
            self.time_points is None):
            return np.zeros(3), np.zeros(3), np.zeros(3)
        if t <= self.time_points[0]:
            idx = 0
        elif t >= self.time_points[-1]:
            idx = len(self.trajectory_points) - 1
        else:
            idx = np.searchsorted(self.time_points, t)
            if idx > 0:
                t1, t2 = self.time_points[idx-1], self.time_points[idx]
                alpha = (t - t1) / (t2 - t1) if t2 != t1 else 0.0
                pos = (1 - alpha) * self.trajectory_points[idx-1] + alpha * self.trajectory_points[idx]
                vel = (1 - alpha) * self.trajectory_velocities[idx-1] + alpha * self.trajectory_velocities[idx]
                acc = (1 - alpha) * self.trajectory_accelerations[idx-1] + alpha * self.trajectory_accelerations[idx]
                return pos, vel, acc
            else:
                idx = 0
        return self.trajectory_points[idx], self.trajectory_velocities[idx], self.trajectory_accelerations[idx]
    
    def compute_control(self, current_state, t):
        pos_des, vel_des, acc_des = self.get_desired_state(t)
        waypoint = np.append(pos_des, 0.0)  # yaw = 0
        try:
            control_input = self.controller.step(current_state, waypoint, vel_des, acc_des)
        except Exception as e:
            print(f"Controller error: {e}")
            hover_thrust = self.params.mass * 9.81 / 4.0 / self.params.linearThrustToU
            control_input = np.array([hover_thrust] * 4)
        current_pos = current_state[0:3]
        current_vel = current_state[3:6]
        self.position_errors.append(np.linalg.norm(current_pos - pos_des))
        self.velocity_errors.append(np.linalg.norm(current_vel - vel_des))
        return control_input
    
    def reset_metrics(self):
        self.position_errors = []
        self.velocity_errors = []
    
    def get_performance_summary(self):
        if not self.position_errors:
            return "No performance data"
        return (
            f"\nPerformance Summary:\n"
            f"  Mean Position Error: {np.mean(self.position_errors):.3f} m\n"
            f"  Max Position Error:  {np.max(self.position_errors):.3f} m\n"
            f"  Mean Velocity Error: {np.mean(self.velocity_errors):.3f} m/s\n"
            f"  Max Velocity Error:  {np.max(self.velocity_errors):.3f} m/s\n"
        )
