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

# --------------------------------------------------------------------------------------
# Paths (edit these three if needed)
# --------------------------------------------------------------------------------------
THIS_DIR = pathlib.Path(__file__).resolve().parent
imu_path   = "/home/alien/MyDirectoryID_p0/Phase1/Data/Train/IMU/imuRaw1.mat"
vic_path   = "/home/alien/MyDirectoryID_p0/Phase1/Data/Train/Vicon/viconRot1.mat"
param_path = "/home/alien/MyDirectoryID_p0/Phase1/IMUParams.mat"

OUT_DIR = (THIS_DIR / "outputs").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------
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

def rmat_stack_to_R(rots_3x3xN):
    """rots_3x3xN -> scipy Rotation (N,3,3)."""
    mats = np.transpose(rots_3x3xN, (2,0,1))
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
    """SO(3) integration of body rates starting from R0."""
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
    ax_p=((ax*sx)+bax)*9.81; ay_p=((ay*sy)+bay)*9.81; az_p=((az*sz)+baz)*9.81
    print(ax_p,ay_p,az_p)
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
    """
    vecs: (N,3) in sensor frame S
    M: 3x3 mapping from S -> B (body frame)
    returns (N,3) in body frame
    """
    return (M @ vecs.T).T

def madgwick_imu_fixed(
    ts, wx, wy, wz, ax, ay, az,
    beta=0.1,
    q0=None,
    gyro_map=np.eye(3),
    accel_map=np.eye(3),
    estimate_gyro_bias=True
):
    """
    Madgwick IMU (no mag). Returns (Q Nx4, angles Nx3 [roll,pitch,yaw]).
    Axis maps put both sensors into the SAME body frame.
    """
    ts = np.asarray(ts).ravel()
    wx = np.asarray(wx).ravel(); wy = np.asarray(wy).ravel(); wz = np.asarray(wz).ravel()
    ax = np.asarray(ax).ravel(); ay = np.asarray(ay).ravel(); az = np.asarray(az).ravel()
    N = ts.size
    if not all(a.size == N for a in [wx,wy,wz,ax,ay,az]):
        raise ValueError("madgwick_imu_fixed: all inputs must share the same length")

    # 1) remap to a common body frame
    Gs = np.column_stack([wx, wy, wz])   # gyro in sensor frame
    As = np.column_stack([ax, ay, az])   # accel in sensor frame
    G  = remap_axes(Gs, gyro_map)        # -> body
    A  = remap_axes(As, accel_map)       # -> body

    # 2) optional gyro bias estimate at start
    if estimate_gyro_bias:
        bx = estimate_bias(G[:,0], ts)
        by = estimate_bias(G[:,1], ts)
        bz = estimate_bias(G[:,2], ts)
        G  = G - np.array([bx, by, bz])[None,:]

    # 3) init quaternion [w,x,y,z]
    if q0 is None:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        q = np.asarray(q0, dtype=float); n=np.linalg.norm(q); q = q/(n if n>0 else 1.0)

    Q = np.zeros((N,4), dtype=float); Q[0] = q

    # 4) main loop
    for k in range(1, N):
        dt = ts[k] - ts[k-1]
        if dt <= 0: dt = 1e-6

        qw, qx, qy, qz = Q[k-1]
        gx, gy, gz = G[k]  # use current gyro sample

        # gyro-only
        qDot_gyro = 0.5 * np.array([
            -qx*gx - qy*gy - qz*gz,
             qw*gx + qy*gz - qz*gy,
             qw*gy - qx*gz + qz*gx,
             qw*gz + qx*gy - qy*gx
        ])

        # accel correction
        a = A[k]
        an = np.linalg.norm(a)
        if np.isfinite(an) and an > 1e-9:
            a = a / an
            # Madgwick IMU equations
            f = np.array([
                2.0*(qx*qz - qw*qy) - a[0],
                2.0*(qw*qx + qy*qz) - a[1],
                qw*qw - qx*qx - qy*qy + qz*qz - a[2]
            ])
            J = np.array([
                [-2.0*qy,   2.0*qz,  -2.0*qw,   2.0*qx],
                [ 2.0*qx,   2.0*qw,   2.0*qz,   2.0*qy],
                [ 2.0*qw,  -2.0*qx,  -2.0*qy,   2.0*qz]
            ])
            grad = J.T @ f
            gnorm = np.linalg.norm(grad)
            if np.isfinite(gnorm) and gnorm > 1e-12:
                grad = grad / gnorm
                qDot = qDot_gyro - beta * grad
            else:
                qDot = qDot_gyro
        else:
            qDot = qDot_gyro

        q = Q[k-1] + qDot * dt
        n = np.linalg.norm(q); Q[k] = q / (n if n>0 else 1.0)

    # 5) to Euler (roll, pitch, yaw)
    roll = np.zeros(N); pitch = np.zeros(N); yaw = np.zeros(N)
    for i in range(N):
        qw, qx, qy, qz = Q[i]
        # roll (x)
        sinr_cosp = 2*(qw*qx + qy*qz)
        cosr_cosp = 1 - 2*(qx*qx + qy*qy)
        roll[i] = math.atan2(sinr_cosp, cosr_cosp)
        # pitch (y)
        sinp = 2*(qw*qy - qz*qx)
        pitch[i] = (math.copysign(math.pi/2, sinp)
                    if abs(sinp) >= 1 else math.asin(sinp))
        # yaw (z)
        siny_cosp = 2*(qw*qz + qx*qy)
        cosy_cosp = 1 - 2*(qy*qy + qz*qz)
        yaw[i] = math.atan2(siny_cosp, cosy_cosp)

    return Q, np.column_stack([roll, pitch, yaw])

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    print(f"[INFO] Running from: {pathlib.Path.cwd()}")
    print(f"[INFO] Outputs → {OUT_DIR}")

    # Load data
    vals, t_imu = load_imu_data(imu_path)                 # vals: (6,N)
    rots_vic_3x3xN, t_vic = load_vicon_data(vic_path)
    scales, biases = load_params(param_path)

    # Ensure monotonic times and reorder arrays if needed
    if not np.all(np.diff(t_imu) >= 0):
        order = np.argsort(t_imu); t_imu = t_imu[order]; vals = vals[:, order]
    if not np.all(np.diff(t_vic) >= 0):
        order = np.argsort(t_vic); t_vic = t_vic[order]; rots_vic_3x3xN = rots_vic_3x3xN[:, :, order]

    # Split IMU channels (dataset order: [ax,ay,az,wz,wx,wy])
    ax_raw, ay_raw, az_raw = vals[0], vals[1], vals[2]
    wz_c,    wx_c,  wy_c   = vals[3], vals[4], vals[5]

    # Convert gyro counts -> rad/s
    wx = gyro_counts_to_rads(wx_c)
    wy = gyro_counts_to_rads(wy_c)
    wz = gyro_counts_to_rads(wz_c)

    # Calibrate accelerometer
    (sx, sy, sz) = scales
    (bax, bay, baz) = biases
    ax = (ax_raw + bax) / sx
    ay = (ay_raw + bay) / sy
    az = (az_raw + baz) / sz

    # Initial orientation from Vicon at t_imu[0]
    idx0 = nearest_indices(t_vic, np.array([t_imu[0]]) )[0]
    R0   = rots_vic_3x3xN[:, :, idx0]

    # ---------------- Gyro-only orientation (SO(3) integrate) ----------------
    R_gyro = integrate_gyro(t_imu, wx, wy, wz, R0)
    eul_gyro = R_gyro.as_euler('ZYX')            # [yaw, pitch, roll]
    gyro_roll  = eul_gyro[:, 2]
    gyro_pitch = eul_gyro[:, 1]
    gyro_yaw   = eul_gyro[:, 0]

    # ---------------- Accel-only (tilt) ----------------
    alpha_val, _, _ = compute_alpha(t_imu, fc=0.3)   # e.g., 0.3 Hz cutoff
    acc_roll_lp, acc_pitch_lp, acc_yaw_lp = accel_lowpass_filter(
        t_imu, ax_raw, ay_raw, az_raw, scales, biases, alpha_val
    )  # yaw ~ 0

    # ---------------- Madgwick (robust) ----------------
    # Map both sensors into the SAME body frame.
    # If your board axes already match your chosen body frame, keep identity.
    GyroMap  = np.eye(3)
    AccelMap = np.eye(3)
    # Example of a possible swap/flip (uncomment and adjust if needed):
    # AccelMap = np.array([[0, 1, 0],
    #                      [1, 0, 0],
    #                      [0, 0,-1]])
    # GyroMap = AccelMap.copy()

    # Optional: seed initial quaternion from Vicon R0
    q0_xyzw = R.from_matrix(R0).as_quat()       # (x,y,z,w)
    q0_wxyz = np.r_[q0_xyzw[3], q0_xyzw[:3]]    # -> [w,x,y,z]

    Qmad, mad_angles = madgwick_imu_fixed(
        t_imu, wx, wy, wz, ax, ay, az,
        beta=0.05,
        q0=q0_wxyz,
        gyro_map=GyroMap,
        accel_map=AccelMap,
        estimate_gyro_bias=True
    )
    mad_roll, mad_pitch, mad_yaw = mad_angles.T

    # ---------------- Vicon (aligned to IMU time base) ----------------
    idx_match = nearest_indices(t_vic, t_imu)
    R_vic_aligned = rmat_stack_to_R(rots_vic_3x3xN[:, :, idx_match])
    eul_vic = R_vic_aligned.as_euler('ZYX')                 # [yaw, pitch, roll]
    vic_roll  = eul_vic[:, 2]
    vic_pitch = eul_vic[:, 1]
    vic_yaw   = eul_vic[:, 0]

    # ----------------------------------------------------------------------------------
    # Plot: 3 rows (Roll, Pitch, Yaw) — all four series on each subplot vs t_imu
    # ----------------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Roll
    axes[0].plot(t_imu, vic_roll,      label='Vicon')
    axes[0].plot(t_imu, gyro_roll,     label='Gyro-only')
    axes[0].plot(t_imu, acc_roll_lp,   label='Accel-only (LPF)')
    axes[0].plot(t_imu, mad_roll,      label='Madgwick')
    axes[0].set_ylabel('Roll (rad)')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # Pitch
    axes[1].plot(t_imu, vic_pitch,     label='Vicon')
    axes[1].plot(t_imu, gyro_pitch,    label='Gyro-only')
    axes[1].plot(t_imu, acc_pitch_lp,  label='Accel-only (LPF)')
    axes[1].plot(t_imu, mad_pitch,     label='Madgwick')
    axes[1].set_ylabel('Pitch (rad)')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    # Yaw
    axes[2].plot(t_imu, vic_yaw,       label='Vicon')
    axes[2].plot(t_imu, gyro_yaw,      label='Gyro-only')
    axes[2].plot(t_imu, acc_yaw_lp,    label='Accel-only (LPF)')  # ~zero line
    axes[2].plot(t_imu, mad_yaw,       label='Madgwick')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Yaw (rad)')
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Attitude Comparison: Vicon vs Gyro-only vs Accel-only vs Madgwick", y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_path = OUT_DIR / "attitude_all_2D_fixed.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figure saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
