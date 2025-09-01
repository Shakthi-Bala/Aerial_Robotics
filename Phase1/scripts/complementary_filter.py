#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import sys, pathlib, os
import numpy as np
from scipy import io
from scipy.spatial.transform import Rotation as R

# --- Headless backend BEFORE pyplot ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Local import, have rotplot in same directory 
THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from rotplot import rotplot

#paths
imu_path   = "/home/alien/MyDirectoryID_p0/Phase1/Data/Train/IMU/imuRaw5.mat"
vic_path   = "/home/alien/MyDirectoryID_p0/Phase1/Data/Train/Vicon/viconRot5.mat"
param_path = "/home/alien/MyDirectoryID_p0/Phase1/IMUParams.mat"



OUT_DIR = (THIS_DIR / "outputs").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

#Fetch data
def get_data(d, key): return d.get(key)

m = io.loadmat(vic_path)
vic_ts = np.asarray(get_data(m, "ts")).ravel()

#Load imu data
def load_imu_data(path):
    m = io.loadmat(path)
    vals = np.asarray(get_data(m, "vals"))
    ts = np.asarray(get_data(m, "ts")).ravel()
    if vals.ndim != 2 or vals.shape[0] != 6:
        raise ValueError("Expected IMU 'vals' to be 6xN.")
    return vals, ts

#Load vicon data
def load_vicon_data(path):
    v = io.loadmat(path)
    rots = np.asarray(v["rots"])       # (3,3,N)
    ts   = np.asarray(get_data(v, "ts")).ravel()
    if rots.ndim != 3 or rots.shape[0:2] != (3,3):
        raise ValueError("Expected Vicon 'rots' to be 3x3xN.")
    return rots, ts

#Load param data
def load_params(path):
    p = io.loadmat(path)
    IMUParams = np.asarray(p["IMUParams"])
    if IMUParams.shape != (2,3):
        raise ValueError("IMUParams must be 2x3 [[sx,sy,sz],[bax,bay,baz]].")
    (sx,sy,sz) = IMUParams[0]; (bax,bay,baz) = IMUParams[1]
    return (sx,sy,sz), (bax,bay,baz)

#Get matrix transpose
def rmat_stack_to_R(rots_3x3xN):
    mats = np.transpose(rots_3x3xN, (2,0,1))  # (N,3,3)
    return R.from_matrix(mats)

#Software time sync
def nearest_indices(t_ref, t_query):
    t_ref = np.asarray(t_ref).ravel(); t_query = np.asarray(t_query).ravel()
    idx = np.searchsorted(t_ref, t_query, side="left")
    idx = np.clip(idx, 0, len(t_ref)-1)
    left = np.maximum(idx - 1, 0)
    choose_left = (idx > 0) & (np.abs(t_query - t_ref[left]) <= np.abs(t_query - t_ref[idx]))
    idx[choose_left] = left[choose_left]
    return idx

#Gyro calib
def gyro_counts_to_rads(counts, n_bias=200):
    counts = np.asarray(counts).ravel()
    bg = counts[:min(int(n_bias), counts.size)].mean()
    scale = (3300.0/1023.0)*(np.pi/180.0)*0.3
    return scale*(counts - bg)

#Integrate gyro
def integrate_gyro(ts, wx, wy, wz, R0_3x3):
    ts = np.asarray(ts).ravel()
    wx = np.asarray(wx).ravel(); wy = np.asarray(wy).ravel(); wz = np.asarray(wz).ravel()
    N = ts.size
    if any(arr.size != N for arr in [wx,wy,wz]): raise ValueError("ts,wx,wy,wz mismatch")
    R_seq = [R.from_matrix(R0_3x3)]
    for k in range(N-1):
        dt = ts[k+1]-ts[k];  dt = 1e-6 if dt<=0 else dt
        dR = R.from_rotvec(np.array([wx[k],wy[k],wz[k]])*dt)
        R_seq.append(R_seq[-1]*dR)
    return R.concatenate(R_seq)

#Acc calib
def accel_tilt_from_calibrated(ax, ay, az):
    g = np.vstack([ax,ay,az]).T
    g /= np.clip(np.linalg.norm(g, axis=1, keepdims=True), 1e-9, None)
    roll  = np.arctan2(g[:,1], g[:,2])
    pitch = np.arctan2(-g[:,0], np.sqrt(g[:,1]**2 + g[:,2]**2))
    return roll, pitch


#Retun Rot matrix from calib values
def accel_to_tilt_orient(ts, ax, ay, az, scales, biases, yaw0_rad=0.0):
    sx,sy,sz = scales; bax,bay,baz = biases
    ax_p=((ax*sx)+bax)*9.81; ay_p=((ay*sy)+bay)*9.81; az_p=((az*sz)+baz)*9.81
    print(ax_p,ay_p,az_p)
    acc = np.vstack([ax_p,ay_p,az_p]).T
    acc_norm = np.linalg.norm(acc, axis=1, keepdims=True)
    acc_unit = acc/np.clip(acc_norm,1e-9,None)
    f_w = np.array([0.0,0.0,-1.0])
    R_list=[]; R_yaw = R.from_euler('Z', yaw0_rad)
    for u in acc_unit:
        v = np.cross(f_w,u); s=np.linalg.norm(v); c=float(np.dot(f_w,u))
        if s<1e-9: dR = R.identity() if c>0 else R.from_rotvec(np.pi*np.array([1.0,0,0]))
        else:
            axis=v/s; angle=np.arctan2(s,c); dR = R.from_rotvec(axis*angle)
        R_list.append(R_yaw*dR)
    return R.concatenate(R_list)

#Compute alpha
def compute_alpha(ts, alpha=None, tau=None, fc=None):
    if alpha is not None:
        return float(np.clip(alpha, 0.0, 0.999999)), None, None
    dts=np.diff(ts); dts=dts[dts>0]; Ts=float(np.median(dts)) if dts.size else 1e-2
    if tau is None and fc is not None and fc>0: tau = 1.0/(2.0*np.pi*fc)
    if tau is None or tau<=0: tau=0.5
    a = tau/(tau+Ts); return float(np.clip(a,0.0,0.999999)), Ts, tau

#Low pass filter acc
def accel_lowpass_filter(ts, ax, ay, az, scales, biases, alpha):
    sx,sy,sz=scales; bax,bay,baz=biases
    ax_p=(ax+bax)/sx; ay_p=(ay+bay)/sy; az_p=(az+baz)/sz
    roll_a,pitch_a = accel_tilt_from_calibrated(ax_p,ay_p,az_p)
    N=len(ts); roll_lp=np.zeros(N); pitch_lp=np.zeros(N); yaw_lp=np.zeros(N)
    roll_lp[0]=roll_a[0]; pitch_lp[0]=pitch_a[0]
    for k in range(N-1):
        roll_lp[k+1]=(1-alpha)*roll_a[k+1]+alpha*roll_lp[k]
        pitch_lp[k+1]=(1-alpha)*pitch_a[k+1]+alpha*pitch_lp[k]
    return roll_lp,pitch_lp,yaw_lp


def complementary_filter_angles(ts, wx, wy, wz, roll_lp, pitch_lp, yaw_lp, alpha):
    N=len(ts); roll=np.zeros(N); pitch=np.zeros(N); yaw=np.zeros(N)
    roll[0],pitch[0],yaw[0]=roll_lp[0],pitch_lp[0],0.0
    for k in range(N-1):
        dt=ts[k+1]-ts[k]; dt=1e-6 if dt<=0 else dt
        roll_g=roll[k]+wx[k+1]*dt; pitch_g=pitch[k]+wy[k+1]*dt; yaw_g=yaw[k]+wz[k+1]*dt
        roll[k+1]=(1-alpha)*roll_g+alpha*roll_lp[k+1]
        pitch[k+1]=(1-alpha)*pitch_g+alpha*pitch_lp[k+1]
        yaw[k+1]=yaw_g
    return roll,pitch,yaw

def complementary_filter(ts, vals, vicon_rots_3x3xN, vicon_ts,
                         alpha=None, tau=None, fc=None, imu_params_path=None):
    ax,ay,az = vals[0],vals[1],vals[2]
    wz_c,wx_c,wy_c = vals[3],vals[4],vals[5]
    wx=gyro_counts_to_rads(wx_c); wy=gyro_counts_to_rads(wy_c); wz=gyro_counts_to_rads(wz_c)
    alpha_val,Ts_est,tau_used = compute_alpha(ts, alpha=alpha, tau=tau, fc=fc)
    if imu_params_path is None: raise ValueError("Provide IMUParams.mat path.")
    scales,biases = load_params(imu_params_path)
    roll_lp,pitch_lp,yaw_lp = accel_lowpass_filter(ts, ax,ay,az, scales,biases, alpha_val)
    roll,pitch,yaw = complementary_filter_angles(ts, wx,wy,wz, roll_lp,pitch_lp,yaw_lp, alpha_val)
    eul_fused=np.column_stack([yaw,pitch,roll])
    eul_acc  =np.column_stack([np.zeros_like(yaw), pitch_lp, roll_lp])
    yaw_g   = np.cumsum(np.r_[0, np.diff(ts)] * np.r_[wz[:1], wz[1:]])
    pitch_g = np.cumsum(np.r_[0, np.diff(ts)] * np.r_[wy[:1], wy[1:]])
    roll_g  = np.cumsum(np.r_[0, np.diff(ts)] * np.r_[wx[:1], wx[1:]])
    eul_gyro=np.column_stack([yaw_g,pitch_g,roll_g])
    R_fused=R.from_euler('ZYX', eul_fused)
    R_acc  =R.from_euler('ZYX', eul_acc)
    R_gyro =R.from_euler('ZYX', eul_gyro)
    # Align Vicon to IMU timestamps for fair side-by-side
    idx_match = nearest_indices(vicon_ts, ts)
    R_vic_aligned = rmat_stack_to_R(vicon_rots_3x3xN[:,:,idx_match])
    return {"R_fused":R_fused,"R_acc":R_acc,"R_gyro_euler":R_gyro,"R_vic":R_vic_aligned,
            "alpha":alpha_val,"Ts":Ts_est,"tau":tau_used,"wx":wx,"wy":wy,"wz":wz,
            "scales":scales,"biases":biases}

#Video-4 frame side by side
def _pick_writer():
    try:
        if animation.writers.is_available("ffmpeg"): return "ffmpeg","mp4"
    except Exception: pass
    try:
        if animation.writers.is_available("pillow"): return "pillow","gif"
    except Exception: pass
    return None,None

def make_side_by_side_video_rotplot_4(
    R_gyro, R_acc, R_fused, R_gt, t_seq,
    out_prefix="orientations_rotplot_4",
    n_frames=60, fps=15, dpi=150,
    lim=2.0, elev=20, azim=35,
    titles=("Gyro", "Acc", "CF", "Ground Truth")  # <— NEW: override-able titles
):
    """
    Animate with the exact rotplot look by calling rotplot() each frame.
    Shows 4 panels: Gyro, Acc, CF, Ground Truth (Vicon aligned).
    Frames are evenly spaced over the entire run so motion is visible.
    """
    M_g  = R_gyro.as_matrix()
    M_a  = R_acc.as_matrix()
    M_f  = R_fused.as_matrix()
    M_gt = R_gt.as_matrix()
    N = min(M_g.shape[0], M_a.shape[0], M_f.shape[0], M_gt.shape[0], len(t_seq))
    if N < 2: raise ValueError(f"Not enough frames: N={N}")

    idx = np.linspace(0, N-1, min(n_frames, N), dtype=int)
    F = len(idx)

    fig, axes = plt.subplots(1, 4, subplot_kw={'projection':'3d'}, figsize=(16,4), constrained_layout=True)

    def _setup(ax, title):
        ax.set_title(title)
        ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([-lim, lim])
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=elev, azim=azim)

    def init():
        for ax, title in zip(axes, titles):
            ax.cla(); _setup(ax, title)
        return []

    def update(f):
        k = idx[f]
        for ax, title, M in zip(axes, titles, [M_g[k], M_a[k], M_f[k], M_gt[k]]):
            ax.cla()
            _setup(ax, f"{title}\nt={t_seq[k]:.2f}s, idx={k}")
            rotplot(M, currentAxes=ax)
        if f % 10 == 0:
            print(f"[INFO] frame {f+1}/{F} (global idx {k})")
        return []

    ani = animation.FuncAnimation(fig, update, frames=F, init_func=init, interval=1000/fps, blit=False)

    writer, ext = _pick_writer()
    out_path = OUT_DIR / f"{out_prefix}.{ext if ext else 'mp4'}"
    print(f"[INFO] Writer: {writer or 'NONE'}  →  {out_path}")
    try:
        if writer == "ffmpeg":
            Writer = animation.writers['ffmpeg']
            w = Writer(fps=fps, extra_args=['-vcodec','libx264','-pix_fmt','yuv420p','-preset','ultrafast'])
            ani.save(str(out_path), writer=w, dpi=dpi)
        elif writer == "pillow":
            ani.save(str(out_path.with_suffix('.gif')), writer='pillow', fps=fps, dpi=dpi)
            out_path = out_path.with_suffix('.gif')
        else:
            raise RuntimeError("No writer available. Install ffmpeg or pillow.")
        print(f"[OK] Saved: {out_path.resolve()}")
    finally:
        plt.close(fig)

def madgwick_imu(ts, wx, wy, wz, ax, ay, az, beta=0.05, q0=None):
    """
    Madgwick's IMU filter for orientation estimation.

    Args:
        ts (array-like): Timestamps in seconds. Shape (N,).
        wx, wy, wz (array-like): Gyroscope data in rad/s. Shape (N,).
        ax, ay, az (array-like): Accelerometer data in m/s^2. Shape (N,).
        beta (float, optional): Filter gain. Defaults to 0.1.
        q0 (array-like, optional): Initial orientation as a quaternion [w, x, y, z].
                                   Defaults to [1, 0, 0, 0].

    Returns:
        np.ndarray: Array of Euler angles [roll, pitch, yaw] in radians for each
                    timestep. Shape (N, 3).
    """
    # --- 1. Input Validation and Initialization ---
    ts = np.asarray(ts).ravel()
    wx = np.asarray(wx).ravel()
    wy = np.asarray(wy).ravel()
    wz = np.asarray(wz).ravel()
    ax = np.asarray(ax).ravel()
    ay = np.asarray(ay).ravel()
    az = np.asarray(az).ravel()

    N = ts.size
    if not all(arr.size == N for arr in [wx, wy, wz, ax, ay, az]):
        raise ValueError("madgwick_imu: All input arrays must have the same length.")

    # Quaternion state q = [w, x, y, z]
    if q0 is None:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        q = np.asarray(q0, dtype=float)
        q /= np.linalg.norm(q) # Ensure unit quaternion

    # Array to store all quaternion results
    Q = np.zeros((N, 4), dtype=float)
    Q[0] = q

    # --- 2. Main Filter Loop ---
    for k in range(1, N):
        dt = ts[k] - ts[k-1]
        if dt <= 0:
            dt = 1e-6 # Use a small positive timestep if timestamps are non-increasing

        # --- Gyroscope rate of change of quaternion ---
        w, x, y, z = Q[k-1] # Use the previous quaternion
        gx, gy, gz = wx[k-1], wy[k-1], wz[k-1]

        qDot_gyro = 0.5 * np.array([
            -x*gx - y*gy - z*gz,
             w*gx + y*gz - z*gy,
             w*gy - x*gz + z*gx,
             w*gz + x*gy - y*gx
        ])

        # --- Accelerometer Correction ---
        a = np.array([ax[k-1], ay[k-1], az[k-1]], dtype=float)
        an = np.linalg.norm(a)

        # Only apply correction if acceleration is valid (non-zero and not NaN)
        if an > 1e-9 and np.isfinite(an):
            a /= an # Normalize accelerometer vector

            # Objective function (error between estimated and measured gravity)
            f = np.array([
                2.0*(x*z - w*y) - a[0],
                2.0*(y*z + w*x) - a[1],
                w*w - x*x - y*y + z*z - a[2]
            ])

            # Jacobian of the objective function
            J = np.array([
                [-2.0*y,  2.0*z, -2.0*w,  2.0*x],
                [ 2.0*x,  2.0*w,  2.0*z,  2.0*y],
                [ 2.0*w, -2.0*x, -2.0*y,  2.0*z]
            ])

            # Gradient descent step
            gradient = J.T @ f
            gradient /= np.linalg.norm(gradient) # Normalize gradient

            # Update quaternion derivative with correction
            qDot = qDot_gyro - beta * gradient
        else:
            # No correction if accelerometer is unreliable
            qDot = qDot_gyro

        # Integrate to get the new quaternion and normalize
        q = Q[k-1] + qDot * dt
        Q[k] = q / np.linalg.norm(q)


    # --- 3. Convert Quaternions to Euler Angles ---
    roll = np.zeros(N)
    pitch = np.zeros(N)
    yaw = np.zeros(N)

    for i in range(N):
        q_w, q_x, q_y, q_z = Q[i]

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q_w * q_x + q_y * q_z)
        cosr_cosp = 1 - 2 * (q_x * q_x + q_y * q_y)
        roll[i] = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (q_w * q_y - q_z * q_x)
        if abs(sinp) >= 1:
            pitch[i] = math.copysign(math.pi / 2, sinp) # Use 90 degrees if out of range
        else:
            pitch[i] = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q_w * q_z + q_x * q_y)
        cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)
        yaw[i] = math.atan2(siny_cosp, cosy_cosp)

    # Stack into an (N, 3) array
    return np.column_stack([roll, pitch, yaw])
    

#Main
def main():
    print(f"[INFO] Running from: {pathlib.Path.cwd()}")
    print(f"[INFO] Outputs → {OUT_DIR}")

    vals, t_imu = load_imu_data(imu_path)
    rots_vic_3x3xN, t_vic = load_vicon_data(vic_path)

    # sort times just in case
    if not np.all(np.diff(t_imu) >= 0):
        order = np.argsort(t_imu); t_imu = t_imu[order]; vals = vals[:,order]
    if not np.all(np.diff(t_vic) >= 0):
        order = np.argsort(t_vic); t_vic = t_vic[order]; rots_vic_3x3xN = rots_vic_3x3xN[:,:,order]

    # initial orientation/yaw from Vicon
    idx0 = nearest_indices(t_vic, np.array([t_imu[0]]) )[0]
    R0   = rots_vic_3x3xN[:,:,idx0]
    yaw0_deg = rmat_stack_to_R(rots_vic_3x3xN)[idx0].as_euler('ZYX', degrees=True)[0]
    yaw0_rad = np.deg2rad(yaw0_deg)

    # complementary filter
    out = complementary_filter(t_imu, vals, rots_vic_3x3xN, t_vic, fc=0.3, imu_params_path=param_path)
    R_fused = out["R_fused"]; R_acc = out["R_acc"]; R_gyroE = out["R_gyro_euler"]
    R_vic   = out["R_vic"]     # already aligned to IMU timestamps
    scales  = out["scales"]; biases = out["biases"]
    wx,wy,wz = out["wx"], out["wy"], out["wz"]

    # replace R_gyroE with proper SO(3) integration from R0 (smoother)
    R_gyro = integrate_gyro(t_imu, wx, wy, wz, R0)

    # Accel-only tilt with fixed initial yaw (kept if you want)
    # ax,ay,az = vals[0], vals[1], vals[2]
    # R_acc = accel_to_tilt_orient(t_imu, ax, ay, az, scales, biases, yaw0_rad=yaw0_rad)

    # ---- 4-panel rotplot video: evenly spaced frames across sequence ----
    R_fused = out["R_fused"]
    scales  = out["scales"]; biases = out["biases"]
    wx, wy, wz = out["wx"], out["wy"], out["wz"]   # rad/s (already calibrated)

    # Accel calibrated for tilt (use same calibration as CF)
    ax_raw, ay_raw, az_raw = vals[0], vals[1], vals[2]
    sx, sy, sz = scales
    bax, bay, baz = biases
    ax_cal = (ax_raw + bax) / sx
    ay_cal = (ay_raw + bay) / sy
    az_cal = (az_raw + baz) / sz

    # Optional: seed q0 from Vicon at t0 for faster convergence
    # R0 is 3x3; convert to quaternion (scipy XYZ+W order)
    q0_xyzw = R.from_matrix(R0).as_quat()         # (x,y,z,w)
    q0_wxyz = np.r_[q0_xyzw[3], q0_xyzw[:3]]      # to [w,x,y,z]

    # Run Madgwick
    # R_mad,mad_angles = madgwick_imu(
    #     t_imu, wx, wy, wz, ax_cal, ay_cal, az_cal,
    #     beta=0.05,    # tune 0.02–0.2 typically
    #     q0=q0_wxyz    # or None to start from identity
    # )
    angle = madgwick_imu(
        t_imu, wx, wy, wz, ax_cal, ay_cal, az_cal,
        beta=0.05,    # tune 0.02–0.2 typically
        q0=q0_wxyz    # or None to start from identity
    )
    x = np.vstack(angle)
    x = np.transpose (x)
    # make_side_by_side_video_rotplot_4(
    #     R_gyro, R_acc, R_fused, R_vic, t_imu,
    #     out_prefix="orientations_rotplot_4",
    #     n_frames=60, fps=15, dpi=150
    # )
    # Four-panel: Gyro, Acc, Madgwick, Ground Truth
    # Four-panel: Gyro, Acc, Madgwick, Ground Truth
    # make_side_by_side_video_rotplot_4(
    #     R_gyro, R_acc, R_mad, R_vic, t_imu,
    #     out_prefix="orientations_rotplot_0.9_madgwick",
    #     n_frames=60, fps=15, dpi=150,
    #     titles=("Gyro", "Acc", "Madgwick", "Ground Truth")  # <— HERE
    # )
    labels = ['Roll', 'Pitch', 'Yaw']
    for k in range(3):
        plt.subplot(3, 1, k+1)
        plt.plot(t_imu, x[k], label='Gyroscope')
        # plt.plot(vic_ts, acc_new[k], label='Accelerometer')
        plt.plot(t_imu, vic[k], label='Vicon')
        # plt.plot(vic_ts, cf_new[k], label='Fused', linestyle='--')
        plt.title(labels[k])
        plt.ylabel('Angle (rad)')
        plt.legend()
    plt.xlabel('Time (s)')
    plt.suptitle("IMU Attitude (Madgwick)")   # avoid using undefined i
    out_path = OUT_DIR / "madgwick_angles.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Figure saved to: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
