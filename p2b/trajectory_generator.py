import numpy as np
from scipy.interpolate import splprep, splev, BSpline
import matplotlib.pyplot as plt

class TrajectoryGenerator:
    """
    Generate smooth trajectory from waypoints using  
splines
    Complete implementation with velocity and acceleration profiles
    """
    
    def __init__(self, waypoints):
        self.waypoints = np.array(waypoints)
        self.trajectory_duration = None
        self.max_velocity = None
        self.max_acceleration = None
        self.average_velocity = 0.6  

    ##############################################################
    #### TODO - Implement spline trajectory generation ###########
    #### TODO - Ensure velocity and acceleration constraints #####
    #### TODO - Add member functions as needed ###################
    ##############################################################

    def generate_bspline_trajectory(self, num_points=None):
        """
        Generate spline trajectory with complete velocity/acceleration profiles
        Returns:
        trajectory_points: (N,3)
        time_points:       (N,)
        velocities:        (N,3)  dP/dt
        accelerations:     (N,3)  d2P/dt2
        """
        print("Generating spline trajectory...")

        if self.waypoints is None or len(self.waypoints) < 2:
            return None, None, None, None

        W = np.asarray(self.waypoints, dtype=float)
        if W.ndim != 2 or W.shape[1] != 3:
            return None, None, None, None

        # chord-length distances
        seg = np.linalg.norm(np.diff(W, axis=0), axis=1)
        # avoid division by zero if consecutive waypoints are identical
        safe_seg = np.maximum(seg, 1e-9)

        # average speed controls nominal timing
        v_avg = float(self.average_velocity) if hasattr(self, "average_velocity") else 0.6
        v_avg = max(v_avg, 1e-6)
        segment_times = safe_seg / v_avg

        time_knots = np.zeros(len(W))
        time_knots[1:] = np.cumsum(segment_times)
        total_T = float(time_knots[-1])
        if total_T <= 0:
            # degenerate path: all points the same
            N = num_points or 100
            tp = np.linspace(0.0, 1.0, N)
            P = np.repeat(W[0][None, :], N, axis=0)
            V = np.zeros_like(P)
            A = np.zeros_like(P)
            self.trajectory_duration = 1.0
            return P, tp, V, A

        # fit spline with time (seconds) as parameter u
        k = int(min(3, len(W) - 1))  # valid degree with few points
        tck, _ = splprep(W.T, u=time_knots, k=k, s=0.0)

        # sampling
        dt = 0.02 if num_points is None else None
        if num_points is None:
            num_points = int(np.ceil(total_T / dt))
        time_points = np.linspace(0.0, total_T, num_points)

        # positions at times
        x, y, z = splev(time_points, tck)
        trajectory_points = np.vstack((x, y, z)).T

        # derivatives wrt time (since u is seconds here)
        dx, dy, dz = splev(time_points, tck, der=1)
        ddx, ddy, ddz = splev(time_points, tck, der=2)
        velocities = np.vstack((dx, dy, dz)).T
        accelerations = np.vstack((ddx, ddy, ddz)).T

        # expose duration
        self.trajectory_duration = total_T

        # ---- OPTIONAL: enforce global caps by time-scaling ----
        if self.max_velocity is not None or self.max_acceleration is not None:
            v_cap = self.max_velocity if self.max_velocity is not None else np.inf
            a_cap = self.max_acceleration if self.max_acceleration is not None else np.inf

            v_peak = float(np.max(np.linalg.norm(velocities, axis=1))) if len(velocities) else 0.0
            a_peak = float(np.max(np.linalg.norm(accelerations, axis=1))) if len(accelerations) else 0.0

            scale_v = v_peak / v_cap if np.isfinite(v_cap) and v_peak > 1e-9 else 1.0
            scale_a = np.sqrt(a_peak / a_cap) if np.isfinite(a_cap) and a_peak > 1e-9 else 1.0
            scale = max(1.0, scale_v, scale_a)

            if scale > 1.0:
                time_points = time_points * scale
                self.trajectory_duration = self.trajectory_duration * scale
                velocities = velocities / scale
                accelerations = accelerations / (scale ** 2)

        return trajectory_points, time_points, velocities, accelerations
    # def generate_bspline_trajectory(self, num_points=None):
    #     """
    #     Generate a time-parameterized B-spline trajectory through waypoints.

    #     Returns:
    #         trajectory_points: (N,3)
    #         time_points:       (N,)
    #         velocities:        (N,3)  dP/dt
    #         accelerations:     (N,3)  d2P/F2
    #     """
    #     # ---- guards ----
    #     if self.waypoints is None or len(self.waypoints) < 2:
    #         return (None, None, None, None)

    #     W = np.asarray(self.waypoints, dtype=float)
    #     if W.ndim != 2 or W.shape[1] != 3:
    #         return (None, None, None, None)

    #     # Degenerate (all points the same)
    #     if np.allclose(W, W[0], atol=1e-12):
    #         N = num_points or 100
    #         t = np.linspace(0.0, 1.0, N)
    #         P = np.repeat(W[0][None, :], N, axis=0)
    #         V = np.zeros_like(P)
    #         A = np.zeros_like(P)
    #         T = 1.0
    #         return P, t * T, V, A

    #     # ---- chord-length parameterization ----
    #     seg = np.linalg.norm(np.diff(W, axis=0), axis=1)
    #     arclen = np.concatenate([[0.0], np.cumsum(seg)])
    #     total_len = arclen[-1]
    #     if total_len <= 1e-9:  # extremely small
    #         total_len = 1.0

    #     # Nominal speed & duration
    #     v_nom = 0.6  # m/s (tune if you like)
    #     T = max(total_len / v_nom, 1.0)
    #     self.trajectory_duration = T

    #     # normalized parameter u in [0,1]
    #     u = arclen / arclen[-1]

    #     # ---- fit spline (k ≤ #points-1) ----
    #     # If you have very few points, reduce k to avoid errors
    #     k = int(min(3, len(W) - 1))

    #     # splprep returns (tck, u_fitted)
    #     tck, _u_fit = splprep(W.T, u=u, k=k, s=0.0)

    #     # ---- sampling ----
    #     if num_points is None:
    #         num_points = max(200, 20 * len(W))

    #     time_points = np.linspace(0.0, T, num_points)
    #     u_t = time_points / T  # map time → spline parameter

    #     # Clamp u_t numerically to [0,1]
    #     u_t = np.clip(u_t, 0.0, 1.0)

    #     # Positions
    #     x, y, z = splev(u_t, tck)
    #     trajectory_points = np.vstack([x, y, z]).T

    #     # Derivatives wrt u
    #     dx, dy, dz = splev(u_t, tck, der=1)
    #     ddx, ddy, ddz = splev(u_t, tck, der=2)

    #     # Chain rule:
    #     # u = t/T → du/dt = 1/T
    #     # dP/dt   = dP/du * (1/T)
    #     # d2P/dt2 = d2P/du2 * (1/T)^2
    #     invT = 1.0 / T
    #     velocities = np.vstack([dx, dy, dz]).T * invT
    #     accelerations = np.vstack([ddx, ddy, ddz]).T * (invT ** 2)

    #     return trajectory_points, time_points, velocities, accelerations


    
 
    def visualize_trajectory(self, trajectory_points=None, velocities=None, 
                           accelerations=None, ax=None):
        """Visualize the trajectory with velocity and acceleration vectors"""
        if ax is None:
            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            standalone = True
        else:
            ax1 = ax
            standalone = False
        
        if trajectory_points is not None:
            # Plot 3D trajectory
            ax1.plot(trajectory_points[:, 0], trajectory_points[:, 1], 
                    trajectory_points[:, 2], 'b-', linewidth=2, label='Spline Trajectory')
            
            # Plot waypoints
            ax1.plot(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2], 
                    'ro-', markersize=8, linewidth=2, label='Waypoints')
            
            # Plot velocity vectors (sampled)
            if velocities is not None:
                step = max(1, len(trajectory_points) // 20)  # Show ~20 vectors
                for i in range(0, len(trajectory_points), step):
                    pos = trajectory_points[i]
                    vel = velocities[i] * 0.5  # Scale for visualization
                    ax1.quiver(pos[0], pos[1], pos[2], 
                             vel[0], vel[1], vel[2], 
                             color='green', alpha=0.7, arrow_length_ratio=0.1)
            
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title('3D Trajectory')
            ax1.legend()
        
        if standalone and velocities is not None and accelerations is not None:
            # Plot velocity magnitude over time
            time_points = np.linspace(0, self.trajectory_duration, len(velocities))
            vel_magnitudes = np.linalg.norm(velocities, axis=1)
            ax2.plot(time_points, vel_magnitudes, 'g-', linewidth=2)
            ax2.axhline(y=self.max_velocity, color='r', linestyle='--', 
                       label=f'Max Vel: {self.max_velocity} m/s')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Velocity (m/s)')
            ax2.set_title('Velocity Profile')
            ax2.grid(True)
            ax2.legend()
            
            # Plot acceleration magnitude over time
            acc_magnitudes = np.linalg.norm(accelerations, axis=1)
            ax3.plot(time_points, acc_magnitudes, 'm-', linewidth=2)
            ax3.axhline(y=self.max_acceleration, color='r', linestyle='--', 
                       label=f'Max Acc: {self.max_acceleration} m/s²')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Acceleration (m/s²)')
            ax3.set_title('Acceleration Profile')
            ax3.grid(True)
            ax3.legend()
            
            plt.tight_layout()
            plt.show()
        
        return ax1 if not standalone else None