#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import sys, pathlib, os
import numpy as np
from scipy import io
from scipy.spatial.transform import Rotation as R
import argparse
import os
# --- Headless backend BEFORE pyplot ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from rotplot import rotplot

THIS_DIR = pathlib.Path(__file__).resolve().parent

# imu_path   = "/home/adityapat/RBE 595 Aerial Robotics/p1_1/Aerial_Robotics/Phase1/YourDirectoryID_p1aTest/IMU/imuRaw10.mat"
# vic_path   = "/home/adityapat/RBE 595 Aerial Robotics/p1_1/Aerial_Robotics/Phase1/Data/Train/Vicon/viconRot6.mat"
# param_path = "/home/adityapat/RBE 595 Aerial Robotics/p1_1/Aerial_Robotics/Phase1/IMUParams.mat"

def process_data(imu_path, vic_path, param_path):
    for path in [imu_path, vic_path, param_path]:
            if not os.path.exists(path):
                print(f"Error: The file '{path}' was not found.")
                return
            


OUT_DIR = (THIS_DIR / "outputs").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_data(d, key):
    return d.get(key)

def load_imu_data(path):
    m = io.loadmat(path)
    vals = np.asarray(get_data(m, "vals"))
    ts   = np.asarray(get_data(m, "ts")).ravel()
    if vals.ndim != 2 or vals.shape[0] != 6:
        raise ValueError("Expected IMU 'vals' to be 6xN.")
    return vals, ts

def load_vicon_data(path):
    v = io.loadmat(path)
    rots = np.asarray(v["rots"])       # shape (3,3,N)
    ts   = np.asarray(get_data(v, "ts")).ravel()
    if rots.ndim != 3 or rots.shape[0:2] != (3,3):
        raise ValueError("Expected Vicon 'rots' to be 3x3xN.")
    return rots, ts

def load_params(path):
    p = io.loadmat(path)
    IMUParams = np.asarray(p["IMUParams"])
    if IMUParams.shape != (2,3):
        raise ValueError("IMUParams must be 2x3 [[sx,sy,sz],[bax,bay,baz]].")
    (sx,sy,sz) = IMUParams[0]; (bax,bay,baz) = IMUParams[1]
    return (sx,sy,sz), (bax,bay,baz)

# --------------------------------------------------------------------------------------
# Time alignment + conversions
# --------------------------------------------------------------------------------------
def nearest_indices(t_ref, t_query):
    """For each t in t_query, return index of nearest time in t_ref."""
    t_ref = np.asarray(t_ref).ravel(); t_query = np.asarray(t_query).ravel()
    idx = np.searchsorted(t_ref, t_query, side="left")
    idx = np.clip(idx, 0, len(t_ref)-1)
    left = np.maximum(idx - 1, 0)
    choose_left = (idx > 0) & (np.abs(t_query - t_ref[left]) <= np.abs(t_query - t_ref[idx]))
    idx[choose_left] = left[choose_left]
    return idx

def _project_to_SO3(M, make_proper=True):
    """Project a 3x3 matrix to the closest rotation matrix via SVD."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    Rm = U @ Vt
    if make_proper and np.linalg.det(Rm) < 0:
        U[:, -1] *= -1.0
        Rm = U @ Vt
    return Rm

def rmat_stack_to_R_safe(rots_3x3xN, repair=True, report=True):
    """
    Convert rots (3,3,N) to scipy Rotation, robust to NaNs / bad matrices.
    - Replaces non-finite with previous valid (or identity)
    - Projects each slice to SO(3) using SVD; fixes det<0
    """
    mats = np.transpose(rots_3x3xN, (2,0,1)).astype(float, copy=True)  # (N,3,3)
    N = mats.shape[0]
    repaired = 0
    replaced = 0
    last_good = np.eye(3)

    for k in range(N):
        M = mats[k]
        if not np.isfinite(M).all():
            # replace non-finite frame
            mats[k] = last_good.copy()
            replaced += 1
            continue
        detM = np.linalg.det(M)
        # also check near-singular / not-orthonormal
        if repair:
            # project to SO(3)
            Rm = _project_to_SO3(M, make_proper=True)
            # measure change
            diff = np.linalg.norm(M - Rm, ord='fro')
            if diff > 1e-6 or abs(np.linalg.det(Rm) - 1.0) > 5e-3:
                repaired += 1
            mats[k] = Rm
        # track last_good
        if np.isfinite(mats[k]).all() and abs(np.linalg.det(mats[k]) - 1.0) < 5e-2:
            last_good = mats[k]

    if report:
        print(f"[VICON] Frames: {N}, repaired (proj to SO(3)): {repaired}, non-finite replaced: {replaced}")

    return R.from_matrix(mats)

# --------------------------------------------------------------------------------------
# Sensor models / filters
# --------------------------------------------------------------------------------------
def gyro_counts_to_rads(counts, n_bias=200):
    """Convert gyro ADC counts to rad/s with simple bias estimate."""
    counts = np.asarray(counts).ravel()
    bg = counts[:min(int(n_bias), counts.size)].mean()
    # TODO: adjust per your gyro datasheet sensitivity
    scale = (3300.0/1023.0)*(np.pi/180.0)*0.3
    return scale*(counts - bg)

def integrate_gyro(ts, wx, wy, wz, R0_3x3):
    """SO(3) integration of body rates starting from R0 (gyro-only baseline)."""
    ts = np.asarray(ts).ravel()
    wx = np.asarray(wx).ravel(); wy = np.asarray(wy).ravel(); wz = np.asarray(wz).ravel()
    N = ts.size
    if any(arr.size != N for arr in [wx,wy,wz]):
        raise ValueError("ts,wx,wy,wz mismatch")
    R_seq = [R.from_matrix(R0_3x3)]
    for k in range(N-1):
        dt = ts[k+1]-ts[k];  dt = 1e-6 if dt<=0 else dt
        dR = R.from_rotvec(np.array([wx[k],wy[k],wz[k]])*dt)
        R_seq.append(R_seq[-1]*dR)
    return R.concatenate(R_seq)

def accel_tilt_from_calibrated(ax, ay, az):
    """Roll/pitch from accelerometer (gravity) only; yaw = 0."""
    g = np.vstack([ax,ay,az]).T
    g /= np.clip(np.linalg.norm(g, axis=1, keepdims=True), 1e-9, None)
    roll  = np.arctan2(g[:,1], g[:,2])
    pitch = np.arctan2(-g[:,0], np.sqrt(g[:,1]**2 + g[:,2]**2))
    return roll, pitch

def compute_alpha(ts, alpha=None, tau=None, fc=None):
    """One-pole IIR smoothing coefficient selection."""
    if alpha is not None:
        return float(np.clip(alpha, 0.0, 0.999999)), None, None
    dts=np.diff(ts); dts=dts[dts>0]; Ts=float(np.median(dts)) if dts.size else 1e-2
    if tau is None and fc is not None and fc>0:
        tau = 1.0/(2.0*np.pi*fc)
    if tau is None or tau<=0: tau=0.5
    a = tau/(tau+Ts)
    return float(np.clip(a,0.0,0.999999)), Ts, tau

def accel_lowpass_filter(ts, ax, ay, az, scales, biases, alpha):
    """LPF roll/pitch from accelerometer."""
    sx,sy,sz=scales; bax,bay,baz=biases
    # Use the SAME calibration convention everywhere:
    # counts -> g : x_cal_g = ax*sx + bax; then *9.81 -> m/s^2
    ax_p=(ax*sx + bax)*9.81
    ay_p=(ay*sy + bay)*9.81
    az_p=(az*sz + baz)*9.81

    roll_a,pitch_a = accel_tilt_from_calibrated(ax_p,ay_p,az_p)
    N=len(ts); roll_lp=np.zeros(N); pitch_lp=np.zeros(N); yaw_lp=np.zeros(N)
    roll_lp[0]=roll_a[0]; pitch_lp[0]=pitch_a[0]
    for k in range(N-1):
        roll_lp[k+1]=(1-alpha)*roll_a[k+1]+alpha*roll_lp[k]
        pitch_lp[k+1]=(1-alpha)*pitch_a[k+1]+alpha*pitch_lp[k]
    return roll_lp,pitch_lp,yaw_lp

# --------- helpers for Madgwick robustness ----------
def estimate_bias(series, ts, window_s=1.5):
    """Mean over the first ~window_s seconds, assuming near-static start."""
    ts = np.asarray(ts).ravel()
    series = np.asarray(series).ravel()
    if ts.size == 0:
        return 0.0
    t0 = ts[0]
    mask = ts - t0 <= window_s
    if not np.any(mask):
        mask = np.arange(min(200, series.size))  # fallback
    return float(np.mean(series[mask]))

def remap_axes(vecs, M):
    return (M @ vecs.T).T

def _coerce_map_or_eye(M):
    if M is None or M is Ellipsis:
        return np.eye(3, dtype=float)
    A = np.asarray(M, dtype=float)
    if A.shape != (3,3):
        raise ValueError("Mapping matrix must be 3x3.")
    return A

def madwick_fusion(
    ts,
    wx, wy, wz,
    ax, ay, az,
    beta=0.06,
    q0=None,
    gyro_map=None,
    accel_map=None,
    estimate_gyro_bias=True,
    return_debug=False
):
    """
    Madgwick AHRS (acc + gyro only) with optional axis maps and initial quaternion.
    Returns:
      Q: (N,4) quaternions [w,x,y,z]
      angles: (N,3) [roll, pitch, yaw] (rad)
    """
    ts = np.asarray(ts).ravel()
    wx = np.asarray(wx).ravel(); wy = np.asarray(wy).ravel(); wz = np.asarray(wz).ravel()
    ax = np.asarray(ax).ravel(); ay = np.asarray(ay).ravel(); az = np.asarray(az).ravel()

    N = ts.size
    if not (wx.size == wy.size == wz.size == ax.size == ay.size == az.size == N):
        raise ValueError("madwick_fusion: input lengths must match ts.")

    gyro_map  = _coerce_map_or_eye(gyro_map)
    accel_map = _coerce_map_or_eye(accel_map)

    gyro_stack = np.column_stack([wx, wy, wz])
    acc_stack  = np.column_stack([ax, ay, az])
    g_vec = (gyro_map  @ gyro_stack.T).T
    a_vec = (accel_map @ acc_stack.T ).T

    if estimate_gyro_bias:
        bx = estimate_bias(g_vec[:, 0], ts)
        by = estimate_bias(g_vec[:, 1], ts)
        bz = estimate_bias(g_vec[:, 2], ts)
        g_vec = g_vec - np.array([bx, by, bz])[None, :]

    if q0 is None or q0 is Ellipsis:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        q = np.array(q0, dtype=float)
        n = np.linalg.norm(q); q = q/(n if n>0 else 1.0)

    Q = np.zeros((N, 4), dtype=float)
    Q[0] = q

    for k in range(1, N):
        dt = ts[k] - ts[k - 1]
        if dt <= 0: dt = 1e-6

        qw, qx, qy, qz = Q[k - 1]
        gx, gy, gz = g_vec[k]

        # qdot from gyro: 0.5 * q ⊗ [0, gx, gy, gz]
        q_gyro = 0.5 * np.array([
            -qx*gx - qy*gy - qz*gz,
             qw*gx + qy*gz - qz*gy,
             qw*gy - qx*gz + qz*gx,
             qw*gz + qx*gy - qy*gx
        ])

        # Accelerometer correction
        a = a_vec[k].copy()
        # Keep this flip if your sensor/body axis convention requires it
        a[1] = -a[1]

        an = np.linalg.norm(a)
        if an > 0:
            a /= an
            f = np.array([
                2.0*(qx*qz - qw*qy)          - a[0],
                2.0*(qw*qx + qy*qz)          - a[1],
                2.0*(0.5 - qx*qx - qy*qy)    - a[2]
            ])
            J = np.array([
                [-2.0*qy,   2.0*qz,  -2.0*qw,  2.0*qx],
                [ 2.0*qx,   2.0*qw,   2.0*qz,  2.0*qy],
                [ 0.0,     -4.0*qx,  -4.0*qy,  0.0   ]
            ])
            grad = J.T @ f
            gn = np.linalg.norm(grad)
            q_dot = q_gyro - beta * (grad/gn if gn>0 else grad)
        else:
            q_dot = q_gyro

        q = Q[k - 1] + q_dot * dt
        n = np.linalg.norm(q); q = q/(n if n>0 else 1.0)
        Q[k] = q

    # Convert final quaternions [w,x,y,z] to scipy internal [x,y,z,w] for Rotation object
    Q_xyzw = Q[:, [1,2,3,0]]

    # Let scipy handle the conversion to Euler angles for consistency
    R_obj = R.from_quat(Q_xyzw)
    eul_angles_zyx = R_obj.as_euler('zyx') # returns [yaw, pitch, roll]
    # Reorder to [roll, pitch, yaw] for function output
    angles_rpy = eul_angles_zyx[:, ::-1]

    return Q_xyzw, angles_rpy


# ---------------- Complementary filter angles (add CF curves) ----------------
def complementary_filter_angles(ts, wx, wy, wz, roll_lp, pitch_lp, yaw_lp, alpha):
    ts = np.asarray(ts).ravel()
    wx = np.asarray(wx).ravel(); wy = np.asarray(wy).ravel(); wz = np.asarray(wz).ravel()
    N = ts.size
    roll  = np.zeros(N); pitch = np.zeros(N); yaw = np.zeros(N)
    roll[0], pitch[0], yaw[0] = roll_lp[0], pitch_lp[0], 0.0
    for k in range(N-1):
        dt = ts[k+1] - ts[k]
        if dt <= 0: dt = 1e-6
        # integrate gyro angles
        roll_g  = roll[k]  + wx[k+1] * dt
        pitch_g = pitch[k] + wy[k+1] * dt
        yaw_g   = yaw[k]   + wz[k+1] * dt
        # fuse (low-pass accel tilt; gyro provides high-frequency)
        roll[k+1]  = (1 - alpha) * roll_g  + alpha * roll_lp[k+1]
        pitch[k+1] = (1 - alpha) * pitch_g + alpha * pitch_lp[k+1]
        yaw[k+1]   = yaw_g  # accel cannot observe yaw
    return roll, pitch, yaw

#Video-5 frame side by side
def _pick_writer():
    try:
        if animation.writers.is_available("ffmpeg"): return "ffmpeg","mp4"
    except Exception: pass
    try:
        if animation.writers.is_available("pillow"): return "pillow","gif"
    except Exception: pass
    return None,None

def make_side_by_side_video_rotplot_5(
    R_gyro, R_acc, R_cf, R_mad, t_seq,
    out_prefix="orientations_rotplot_5_panel",
    n_frames=60, fps=15, dpi=150,
    lim=2.0, elev=20, azim=35,
    titles=("Gyro", "Accel", "CF", "Madgwick")
):
    """
    Animate 5 orientation estimates side-by-side using rotplot.
    """
    M_g   = R_gyro.as_matrix()
    M_a   = R_acc.as_matrix()
    M_cf  = R_cf.as_matrix()
    M_mad = R_mad.as_matrix()
    # M_gt  = R_gt.as_matrix()

    N = min(len(t_seq), *[m.shape[0] for m in [M_g, M_a, M_cf, M_mad]])
    if N < 2: raise ValueError(f"Not enough frames for all inputs: N={N}")

    idx = np.linspace(0, N-1, min(n_frames, N), dtype=int)
    F = len(idx)

    fig, axes = plt.subplots(1, 4, subplot_kw={'projection':'3d'}, figsize=(20, 4), constrained_layout=True)

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
        all_matrices = [M_g[k], M_a[k], M_cf[k], M_mad[k]]
        for ax, title, M in zip(axes, titles, all_matrices):
            ax.cla()
            _setup(ax, f"{title}\nt={t_seq[k]:.2f}s, idx={k}")
            rotplot(M, currentAxes=ax)
        if f % 10 == 0:
            print(f"[INFO] Animating frame {f+1}/{F} (data index {k})")
        return []

    ani = animation.FuncAnimation(fig, update, frames=F, init_func=init, interval=1000/fps, blit=False)

    writer, ext = _pick_writer()
    out_path = OUT_DIR / f"{out_prefix}.{ext if ext else 'mp4'}"
    print(f"[INFO] Using writer: '{writer}' to save to: {out_path}")
    try:
        if writer == "ffmpeg":
            w = animation.writers['ffmpeg'](fps=fps, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'ultrafast'])
            ani.save(str(out_path), writer=w, dpi=dpi)
        elif writer == "pillow":
            ani.save(str(out_path.with_suffix('.gif')), writer='pillow', fps=fps, dpi=dpi)
            out_path = out_path.with_suffix('.gif')
        else:
            raise RuntimeError("No animation writer available. Please install ffmpeg or pillow.")
        print(f"[OK] Animation saved: {out_path.resolve()}")
    finally:
        plt.close(fig)

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main(args):
    print(f"[INFO] Running from: {pathlib.Path.cwd()}")
    print(f"[INFO] Outputs will be saved to: {OUT_DIR}")

    # Load data
    vals, t_imu = load_imu_data(args.imu_path)
    rots_vic_3x3xN, t_vic = load_vicon_data(args.vic_path)
    scales, biases = load_params(args.param_path)

    # Ensure monotonic times
    if not np.all(np.diff(t_imu) >= 0):
        order = np.argsort(t_imu); t_imu = t_imu[order]; vals = vals[:, order]
    # if not np.all(np.diff(t_vic) >= 0):
        order = np.argsort(t_vic); t_vic = t_vic[order]; rots_vic_3x3xN = rots_vic_3x3xN[:, :, order]

    # Split IMU channels (dataset order: [ax,ay,az,wz,wx,wy])
    ax_raw, ay_raw, az_raw = vals[0], vals[1], vals[2]
    wz_c,    wx_c,  wy_c   = vals[3], vals[4], vals[5]

    # Convert gyro counts -> rad/s
    wx = gyro_counts_to_rads(wx_c)
    wy = gyro_counts_to_rads(wy_c)
    wz = gyro_counts_to_rads(wz_c)

    # Calibrate accelerometer data to m/s^2
    (sx, sy, sz) = scales
    (bax, bay, baz) = biases
    ax = (ax_raw*sx + bax) * 9.81
    ay = (ay_raw*sy + bay) * 9.81
    az = (az_raw*sz + baz) * 9.81

    # Get initial orientation from Vicon at t_imu[0]
    idx0 = nearest_indices(t_vic, np.array([t_imu[0]]))[0]
    R0   = rmat_stack_to_R_safe(rots_vic_3x3xN[:, :, idx0:idx0+1], report=False).as_matrix()[0]

    # --- 1. Gyro-only orientation (SO(3) integration) ---
    R_gyro = integrate_gyro(t_imu, wx, wy, wz, R0)
    eul_gyro = R_gyro.as_euler('ZYX') # [yaw, pitch, roll]
    gyro_roll, gyro_pitch, gyro_yaw = eul_gyro[:, 2], eul_gyro[:, 1], eul_gyro[:, 0]

    # --- 2. Accel-only (low-pass filtered tilt) ---
    alpha_val, _, _ = compute_alpha(t_imu, fc=0.3)
    acc_roll_lp, acc_pitch_lp, acc_yaw_lp = accel_lowpass_filter(
        t_imu, ax_raw, ay_raw, az_raw, scales, biases, alpha_val
    )
    # Convert accel Euler angles to a Rotation object
    # NOTE: from_euler('zyx', ...) expects angles in [yaw, pitch, roll] order
    acc_eul_stack = np.column_stack([acc_yaw_lp, acc_pitch_lp, acc_roll_lp])
    R_acc = R.from_euler('zyx', acc_eul_stack)

    # --- 3. Complementary Filter (CF) ---
    cf_roll, cf_pitch, cf_yaw = complementary_filter_angles(
        t_imu, wx, wy, wz, acc_roll_lp, acc_pitch_lp, acc_yaw_lp, alpha_val
    )
    # Convert CF Euler angles to a Rotation object
    cf_eul_stack = np.column_stack([cf_yaw, cf_pitch, cf_roll])
    R_cf = R.from_euler('zyx', cf_eul_stack)

    # --- 4. Madgwick Filter ---
    q0_xyzw = R.from_matrix(R0).as_quat()
    Qmad_xyzw, mad_angles_rpy = madwick_fusion(
        ts=t_imu, wx=wx, wy=wy, wz=wz, ax=ax, ay=ay, az=az,
        beta=0.06,
        q0=np.r_[q0_xyzw[3], q0_xyzw[:3]], # Madgwick expects [w,x,y,z]
        estimate_gyro_bias=True
    )
    R_mad = R.from_quat(Qmad_xyzw)
    mad_roll, mad_pitch, mad_yaw = mad_angles_rpy.T

    # --- 5. Vicon (Ground Truth, aligned to IMU time) ---
    idx_match = nearest_indices(t_vic, t_imu)
    R_vic_aligned = rmat_stack_to_R_safe(rots_vic_3x3xN[:, :, idx_match])
    eul_vic = R_vic_aligned.as_euler('ZYX') # [yaw, pitch, roll]
    vic_roll, vic_pitch, vic_yaw = eul_vic[:, 2], eul_vic[:, 1], eul_vic[:, 0]

    # --- Generate 5-Panel Animation ---
    make_side_by_side_video_rotplot_5(
        R_gyro, R_acc, R_cf, R_mad
        , t_imu,
        out_prefix="orientation_comparison_5_panel_10",
        n_frames=120, fps=20, dpi=120
    )

    # --- Generate 2D Comparison Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Attitude Estimation Comparison", fontsize=16, y=0.99)
    
    # Roll
    axes[0].plot(t_imu, np.rad2deg(vic_roll), 'k-', label='Vicon', linewidth=2)
    axes[0].plot(t_imu, np.rad2deg(gyro_roll), label='Gyro-only')
    axes[0].plot(t_imu, np.rad2deg(acc_roll_lp), label='Accel-only (LPF)')
    axes[0].plot(t_imu, np.rad2deg(cf_roll), label='Complementary')
    axes[0].plot(t_imu, np.rad2deg(mad_roll), label='Madgwick')
    axes[0].set_ylabel('Roll (deg)')
    axes[0].legend(loc='best'); axes[0].grid(True, alpha=0.4)

    # Pitch
    axes[1].plot(t_imu, np.rad2deg(vic_pitch), 'k-', label='Vicon', linewidth=2)
    axes[1].plot(t_imu, np.rad2deg(gyro_pitch), label='Gyro-only')
    axes[1].plot(t_imu, np.rad2deg(acc_pitch_lp), label='Accel-only (LPF)')
    axes[1].plot(t_imu, np.rad2deg(cf_pitch), label='Complementary')
    axes[1].plot(t_imu, np.rad2deg(mad_pitch), label='Madgwick')
    axes[1].set_ylabel('Pitch (deg)')
    axes[1].legend(loc='best'); axes[1].grid(True, alpha=0.4)

    # Yaw
    axes[2].plot(t_imu, np.rad2deg(vic_yaw), 'k-', label='Vicon', linewidth=2)
    axes[2].plot(t_imu, np.rad2deg(gyro_yaw), label='Gyro-only')
    axes[2].plot(t_imu, np.rad2deg(cf_yaw), label='Complementary')
    axes[2].plot(t_imu, np.rad2deg(mad_yaw), label='Madgwick')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Yaw (deg)')
    axes[2].legend(loc='best'); axes[2].grid(True, alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = OUT_DIR / "attitude_comparison_plot.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 2D plot saved to: {out_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process and compare IMU orientation estimates."
    )
    
    parser.add_argument(
        '--imu_path', 
        type=str, 
        required=True, 
        help="Path to the IMU .mat file"
    )
    
    parser.add_argument(
        '--vic_path', 
        type=str, 
        required=True, 
        help="Path to the Vicon .mat file"
    )
    
    parser.add_argument(
        '--param_path', 
        type=str, 
        required=True, 
        help="Path to the IMU parameters .mat file"
    )

    args = parser.parse_args()
    main(args)