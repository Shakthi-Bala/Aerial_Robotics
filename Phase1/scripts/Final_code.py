#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import sys, pathlib, os
import numpy as np
from scipy import io
from scipy.spatial.transform import Rotation as R

import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#To import a module within same folder
THIS_DIR = pathlib.Path(__file__).resolve().parent

imu_path   = "/home/alien/MyDirectoryID_p0/Phase1/Data/Train/IMU/imuRaw5.mat"
vic_path   = "/home/alien/MyDirectoryID_p0/Phase1/Data/Train/Vicon/viconRot5.mat"
param_path = "/home/alien/MyDirectoryID_p0/Phase1/IMUParams.mat"
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
    rots = np.asarray(v["rots"])
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

def nearest_indices(t_ref, t_query):
    t_ref = np.asarray(t_ref).ravel(); t_query = np.asarray(t_query).ravel()
    idx = np.searchsorted(t_ref, t_query, side="left")
    idx = np.clip(idx, 0, len(t_ref)-1)
    left = np.maximum(idx - 1, 0)
    choose_left = (idx > 0) & (np.abs(t_query - t_ref[left]) <= np.abs(t_query - t_ref[idx]))
    idx[choose_left] = left[choose_left]
    return idx

def _project_to_SO3(M, make_proper=True):
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    Rm = U @ Vt
    if make_proper and np.linalg.det(Rm) < 0:
        U[:, -1] *= -1.0
        Rm = U @ Vt
    return Rm

def rmat_stack_to_R_safe(rots_3x3xN, repair=True, report=True):
    mats = np.transpose(rots_3x3xN, (2,0,1)).astype(float, copy=True)  # (N,3,3)
    N = mats.shape[0]
    repaired = 0
    replaced = 0
    last_good = np.eye(3)
    for k in range(N):
        M = mats[k]
        if not np.isfinite(M).all():
            mats[k] = last_good.copy()
            replaced += 1
            continue
        if repair:
            Rm = _project_to_SO3(M, make_proper=True)
            diff = np.linalg.norm(M - Rm, ord='fro')
            if diff > 1e-6 or abs(np.linalg.det(Rm) - 1.0) > 5e-3:
                repaired += 1
            mats[k] = Rm
        if np.isfinite(mats[k]).all() and abs(np.linalg.det(mats[k]) - 1.0) < 5e-2:
            last_good = mats[k]
    if report:
        print(f"[VICON] Frames: {N}, repaired (proj to SO(3)): {repaired}, non-finite replaced: {replaced}")
    return R.from_matrix(mats)

# --- Gyro helpers ---
def gyro_counts_to_rads(counts, n_bias=200):
    counts = np.asarray(counts).ravel()
    bg = counts[:min(int(n_bias), counts.size)].mean()
    scale = (3300.0/1023.0)*(np.pi/180.0)*0.3
    return scale*(counts - bg)

def integrate_gyro(ts, wx, wy, wz, R0_3x3):
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
    g = np.vstack([ax,ay,az]).T
    g /= np.clip(np.linalg.norm(g, axis=1, keepdims=True), 1e-9, None)
    roll  = np.arctan2(g[:,1], g[:,2])
    pitch = np.arctan2(-g[:,0], np.sqrt(g[:,1]**2 + g[:,2]**2))
    return roll, pitch

def compute_alpha(ts, alpha=None, tau=None, fc=None):
    if alpha is not None:
        return float(np.clip(alpha, 0.0, 0.999999)), None, None
    dts=np.diff(ts); dts=dts[dts>0]; Ts=float(np.median(dts)) if dts.size else 1e-2
    if tau is None and fc is not None and fc>0:
        tau = 1.0/(2.0*np.pi*fc)
    if tau is None or tau<=0: tau=0.5
    a = tau/(tau+Ts)
    return float(np.clip(a,0.0,0.999999)), Ts, tau

def accel_lowpass_filter(ts, ax, ay, az, scales, biases, alpha):
    sx,sy,sz=scales; bax,bay,baz=biases
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

def estimate_bias(series, ts, window_s=1.5):
    ts = np.asarray(ts).ravel()
    series = np.asarray(series).ravel()
    if ts.size == 0:
        return 0.0
    t0 = ts[0]
    mask = ts - t0 <= window_s
    if not np.any(mask):
        mask = np.arange(min(200, series.size))
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

# --- Madgwick (unchanged) ---
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
        q = np.array(q0, dtype=float); n = np.linalg.norm(q); q = q/(n if n>0 else 1.0)
    Q = np.zeros((N, 4), dtype=float); Q[0] = q

    for k in range(1, N):
        dt = ts[k] - ts[k - 1];  dt = 1e-6 if dt<=0 else dt
        qw, qx, qy, qz = Q[k - 1]
        gx, gy, gz = g_vec[k]
        q_gyro = 0.5 * np.array([
            -qx*gx - qy*gy - qz*gz,
             qw*gx + qy*gz - qz*gy,
             qw*gy - qx*gz + qz*gx,
             qw*gz + qx*gy - qy*gx
        ])
        a = a_vec[k].copy(); a[1] = -a[1]
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

    roll  = np.zeros(N); pitch = np.zeros(N); yaw = np.zeros(N)
    for i in range(N):
        qw, qx, qy, qz = Q[i]
        sinr = 2*(qw*qx + qy*qz); cosr = 1 - 2*(qx*qx + qy*qy)
        roll[i] = math.atan2(sinr, cosr)
        sinp = 2*(qw*qy - qz*qx)
        pitch[i] = (math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp))
        siny = 2*(qw*qz + qx*qy); cosy = 1 - 2*(qy*qy + qz*qz)
        yaw[i] = math.atan2(siny, cosy)
    return Q, np.column_stack([roll, pitch, yaw])

def complementary_filter_angles(ts, wx, wy, wz, roll_lp, pitch_lp, yaw_lp, alpha):
    ts = np.asarray(ts).ravel()
    wx = np.asarray(wx).ravel(); wy = np.asarray(wy).ravel(); wz = np.asarray(wz).ravel()
    N = ts.size
    roll  = np.zeros(N); pitch = np.zeros(N); yaw = np.zeros(N)
    roll[0], pitch[0], yaw[0] = roll_lp[0], pitch_lp[0], 0.0
    for k in range(N-1):
        dt = ts[k+1] - ts[k];  dt = 1e-6 if dt<=0 else dt
        roll_g  = roll[k]  + wx[k+1] * dt
        pitch_g = pitch[k] + wy[k+1] * dt
        yaw_g   = yaw[k]   + wz[k+1] * dt
        roll[k+1]  = (1 - alpha) * roll_g  + alpha * roll_lp[k+1]
        pitch[k+1] = (1 - alpha) * pitch_g + alpha * pitch_lp[k+1]
        yaw[k+1]   = yaw_g
    return roll, pitch, yaw

# ========= UKF logic (Kraft) =========
def q_normalize(q):
    q = np.asarray(q, float); n = np.linalg.norm(q)
    return q if n == 0 else q / n

def q_mul(q1, q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], float)

def q_from_rotvec(rv):
    r = R.from_rotvec(np.asarray(rv, float))
    x,y,z,w = r.as_quat()
    return np.array([w,x,y,z], float)

def rotvec_from_q(q):
    w,x,y,z = q_normalize(q)
    return R.from_quat([x,y,z,w]).as_rotvec()

def q_inv(q):
    w,x,y,z = q_normalize(q)
    return np.array([w,-x,-y,-z], float)

# --- box operators for state x = [qw,qx,qy,qz, ωx,ωy,ωz]
def boxplus_state(x, eps6):
    dq = q_from_rotvec(eps6[0:3])
    q_new = q_mul(x[0:4], dq)
    w_new = x[4:7] + eps6[3:6]
    out = np.zeros(7); out[0:4] = q_normalize(q_new); out[4:7] = w_new
    return out

def boxminus_state(x1, x2):
    dq = q_mul(x1[0:4], q_inv(x2[0:4]))
    dtheta = rotvec_from_q(dq)
    dw = x1[4:7] - x2[4:7]
    return np.r_[dtheta, dw]

#Step1 (fixed, keep name)
def sigma_points_from_cov(x_prev, P6, Q6, n=7):
    # We operate in 6-D error space; build ± columns of sqrt(6*(P+Q))
    n_err = 6
    S = np.linalg.cholesky(n_err * (P6 + Q6) + 1e-12*np.eye(n_err))
    Wi = np.zeros((2*n_err, n_err))
    for i in range(n_err):
        col = S[:, i]
        Wi[i,   :] =  col
        Wi[i+ n_err,:] = -col
    # Apply to previous state to get Xi (state sigma points)
    Xi = np.zeros((Wi.shape[0], 7))
    for i, w in enumerate(Wi):
        Xi[i] = boxplus_state(x_prev, w)
    return Xi  # (12,7)

#Step 2 (fixed, keep name)
def transform_sigma_points(x_i, omega_k, delta_t, n):
    # Yi = ( q_i * q_delta , ω_i ) ; q_delta from ω_k and Δt
    omega = np.asarray(omega_k, float)
    normw = np.linalg.norm(omega)
    if normw < 1e-12:
        q_delta = np.array([1.0,0.0,0.0,0.0])
    else:
        axis = omega / normw
        half = 0.5 * normw * float(delta_t)
        q_delta = np.array([np.cos(half), *(axis*np.sin(half))])
    Yi = np.zeros_like(x_i)
    for i in range(x_i.shape[0]):
        q_i = x_i[i,0:4]
        Yi[i,0:4] = q_normalize(q_mul(q_i, q_delta))
        Yi[i,4:7] = x_i[i,4:7]  # ω unchanged here (perturbation already applied in Xi)
    return Yi

#Step 3 (fixed, keep name)
def compute_mean(y_i, max_iter=15):
    q_t = y_i[0,0:4].copy()
    for _ in range(max_iter):
        e_list=[]
        for i in range(y_i.shape[0]):
            q_i = y_i[i,0:4]
            dq  = q_mul(q_i, q_inv(q_t))
            e_list.append(rotvec_from_q(dq))
        e_list = np.asarray(e_list)
        e_bar = np.mean(e_list, axis=0)
        if np.linalg.norm(e_bar) < 1e-9:
            break
        dq_bar = q_from_rotvec(e_bar)
        q_t = q_mul(dq_bar, q_t)
        q_t = q_normalize(q_t)
    w_bar = np.mean(y_i[:,4:7], axis=0)
    x_bar = np.zeros(7); x_bar[0:4] = q_t; x_bar[4:7] = w_bar
    return x_bar

#Step4 (unchanged)
# (updated to also return Pvv, Pxz when z_i is provided)
def computing_covariance(x_bar, y_i, z_i=None):
    M = y_i.shape[0]
    q_bar = x_bar[0:4]; w_bar = x_bar[4:7]
    E = np.empty((M, 6), dtype=float)
    for i in range(M):
        qi = y_i[i, 0:4]; wi = y_i[i, 4:7]
        dq = q_mul(qi, q_inv(q_bar))
        r_w = rotvec_from_q(dq)
        omega_w = wi - w_bar
        E[i,:] = np.r_[r_w, omega_w]
    Px = (E.T @ E) / float(M)
    Px = 0.5 * (Px + Px.T)
    Pvv = None; Pxz = None
    if z_i is not None:
        z_bar = np.mean(z_i, axis=0)
        Zc = z_i - z_bar[None,:]
        Pvv = (Zc.T @ Zc) / float(M); Pvv = 0.5*(Pvv + Pvv.T)
        Pxz = (E.T @ Zc) / float(M)
    return Px, Pvv, Pxz

# Keep the name; we won’t use inside main, but leave for completeness
def kalman_update(X_state,P_xz,P_vv,z_measure, z_pred):
    K_k = P_xz @ np.linalg.inv(P_vv)
    v_k = z_measure - z_pred
    # Apply correction in error space for [q,ω]
    eps = K_k @ v_k
    X_state = boxplus_state(X_state, eps)
    # No covariance update here to avoid needing previous P; main performs it.
    return X_state, K_k, v_k

# --- measurement projection: accel gravity unit vector ---
def project_acc_measurement(Yi, g_ref=np.array([0,0,9.81], float)):
    Zi = np.zeros((Yi.shape[0], 3))
    for i in range(Yi.shape[0]):
        q = Yi[i,0:4]
        Rb = R.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
        zb = Rb.T @ g_ref
        n = np.linalg.norm(zb); Zi[i] = zb/(n if n>0 else 1.0)
    return Zi

# -------------------- MAIN --------------------
def main():
    print(f"[INFO] Running from: {pathlib.Path.cwd()}")
    print(f"[INFO] Outputs → {OUT_DIR}")

    vals, t_imu = load_imu_data(imu_path)
    rots_vic_3x3xN, t_vic = load_vicon_data(vic_path)
    scales, biases = load_params(param_path)

    if not np.all(np.diff(t_imu) >= 0):
        order = np.argsort(t_imu); t_imu = t_imu[order]; vals = vals[:, order]
    if not np.all(np.diff(t_vic) >= 0):
        order = np.argsort(t_vic); t_vic = t_vic[order]; rots_vic_3x3xN = rots_vic_3x3xN[:, :, order]

    ax_raw, ay_raw, az_raw = vals[0], vals[1], vals[2]
    wz_c,    wx_c,  wy_c   = vals[3], vals[4], vals[5]

    wx = gyro_counts_to_rads(wx_c)
    wy = gyro_counts_to_rads(wy_c)
    wz = gyro_counts_to_rads(wz_c)

    (sx, sy, sz) = scales
    (bax, bay, baz) = biases
    ax = (ax_raw*sx + bax) * 9.81
    ay = (ay_raw*sy + bay) * 9.81
    az = (az_raw*sz + baz) * 9.81

    idx0 = nearest_indices(t_vic, np.array([t_imu[0]]) )[0]
    R0   = rmat_stack_to_R_safe(rots_vic_3x3xN[:, :, idx0:idx0+1]).as_matrix()[0]

    R_gyro = integrate_gyro(t_imu, wx, wy, wz, R0)
    eul_gyro = R_gyro.as_euler('ZYX')
    gyro_roll  = eul_gyro[:, 2]; gyro_pitch = eul_gyro[:, 1]; gyro_yaw   = eul_gyro[:, 0]

    alpha_val, _, _ = compute_alpha(t_imu, fc=0.3)
    acc_roll_lp, acc_pitch_lp, acc_yaw_lp = accel_lowpass_filter(t_imu, ax_raw, ay_raw, az_raw, scales, biases, alpha_val)

    cf_roll, cf_pitch, cf_yaw = complementary_filter_angles(t_imu, wx, wy, wz, acc_roll_lp, acc_pitch_lp, acc_yaw_lp, alpha_val)

    GyroMap  = np.eye(3); AccelMap = np.eye(3)
    q0_xyzw = R.from_matrix(R0).as_quat()
    q0_wxyz = np.r_[q0_xyzw[3], q0_xyzw[:3]]

    Qmad, mad_angles = madwick_fusion(t_imu, wx, wy, wz, ax, ay, az,
        beta=0.06, q0=q0_wxyz, gyro_map=GyroMap, accel_map=AccelMap, estimate_gyro_bias=True, return_debug=True)
    mad_roll, mad_pitch, mad_yaw = mad_angles.T

    idx_match = nearest_indices(t_vic, t_imu)
    R_vic_aligned = rmat_stack_to_R_safe(rots_vic_3x3xN[:, :, idx_match])
    eul_vic = R_vic_aligned.as_euler('ZYX')
    vic_roll  = eul_vic[:, 2]; vic_pitch = eul_vic[:, 1]; vic_yaw   = eul_vic[:, 0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # --- UKF init (Q,R provided here) ---
    Q = np.diag([105.0, 105.0, 105.0, 0.1, 0.1, 0.1])          # error-space process
    Rm = np.diag([11.2, 11.2, 11.2])                            # accel unit-vector noise
    x_prev = np.zeros(7); x_prev[0:4] = q0_wxyz; x_prev[4:7] = np.array([wx[0],wy[0],wz[0]])
    P_prev = np.eye(6)*1e-3

    N = t_imu.size
    ukf_q = np.zeros((N,4)); ukf_q[0] = x_prev[0:4]
    ukf_angles = np.zeros((N,3))
    g_ref = np.array([0,0,9.81], float)

    for k in range(1, N):
        dt = max(t_imu[k]-t_imu[k-1], 1e-6)
        omega_km1 = np.array([wx[k-1], wy[k-1], wz[k-1]])

        # Step 1: sigma from P_prev+Q around x_prev
        Xi = sigma_points_from_cov(x_prev, P_prev, Q, n=7)

        # Step 2: propagate sigma points with q_delta(ω_k-1, dt)
        Yi = transform_sigma_points(Xi, omega_km1, dt, n=7)

        # Step 3: mean (a priori)
        x_bar = compute_mean(Yi)

        # A priori covariance from Yi (and set up measurement cross-covs)
        # Project Yi -> Zi via gravity model
        Zi = project_acc_measurement(Yi, g_ref=g_ref)
        Px, Pvv, Pxz = computing_covariance(x_bar, Yi, z_i=Zi)  # Px is P_k^-, Pvv is cov of Zi about its mean, Pxz cross-cov

        # z_pred and z_meas
        z_pred = Zi.mean(axis=0)
        z_meas = np.array([ax[k], ay[k], az[k]])
        nrm = np.linalg.norm(z_meas); z_meas = z_meas/(nrm if nrm>0 else 1.0)

        # Innovation and gain
        S = Pvv + Rm
        K = Pxz @ np.linalg.inv(S)
        v = z_meas - z_pred

        # State update on manifold
        eps = K @ v
        x_post = boxplus_state(x_bar, eps)
        P_post = Px - K @ S @ K.T
        P_post = 0.5*(P_post + P_post.T)

        # Store & roll
        x_prev = x_post; P_prev = P_post
        ukf_q[k] = x_post[0:4]
        # Euler (ZYX → yaw,pitch,roll)
        r = R.from_quat([ukf_q[k,1], ukf_q[k,2], ukf_q[k,3], ukf_q[k,0]]).as_euler('ZYX')
        ukf_angles[k] = np.array([r[2], r[1], r[0]])[[0,1,2]]  # we'll unpack properly below

    ukf_roll  = ukf_angles[:,0]
    ukf_pitch = ukf_angles[:,1]
    # compute yaw directly from quaternion for consistency
    eul_ukf = R.from_quat(np.column_stack([ukf_q[:,1],ukf_q[:,2],ukf_q[:,3],ukf_q[:,0]])).as_euler('ZYX')
    ukf_yaw = eul_ukf[:,0]

    # ---------- Plots ----------
    axes[0].plot(t_imu, vic_roll,      label='Vicon')
    axes[0].plot(t_imu, gyro_roll,     label='Gyro-only')
    axes[0].plot(t_imu, acc_roll_lp,   label='Accel-only (LPF)')
    axes[0].plot(t_imu, cf_roll,       label='Complementary')
    axes[0].plot(t_imu, mad_roll,      label='Madgwick')
    axes[0].plot(t_imu, ukf_roll,      label='UKF')
    axes[0].set_ylabel('Roll (rad)'); axes[0].legend(loc='best'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_imu, vic_pitch,     label='Vicon')
    axes[1].plot(t_imu, gyro_pitch,    label='Gyro-only')
    axes[1].plot(t_imu, acc_pitch_lp,  label='Accel-only (LPF)')
    axes[1].plot(t_imu, cf_pitch,      label='Complementary')
    axes[1].plot(t_imu, mad_pitch,     label='Madgwick')
    axes[1].plot(t_imu, ukf_pitch,     label='UKF')
    axes[1].set_ylabel('Pitch (rad)'); axes[1].legend(loc='best'); axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_imu, vic_yaw,       label='Vicon')
    axes[2].plot(t_imu, gyro_yaw,      label='Gyro-only')
    axes[2].plot(t_imu, acc_yaw_lp,    label='Accel-only (LPF)')
    axes[2].plot(t_imu, cf_yaw,        label='Complementary')
    axes[2].plot(t_imu, mad_yaw,       label='Madgwick')
    axes[2].plot(t_imu, ukf_yaw,       label='UKF')
    axes[2].set_xlabel('Time (s)'); axes[2].set_ylabel('Yaw (rad)'); axes[2].legend(loc='best'); axes[2].grid(True, alpha=0.3)

    fig.suptitle("Attitude Comparison: Gyro | Acc(LPF) | Complementary | Madgwick | UKF", y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_path = OUT_DIR / "attitude_all_2D_with_CF_and_UKF.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figure saved to: {out_path.resolve()}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Process and compare IMU orientation estimates.")
    # parser.add_argument('--imu_path',   type=str, required=True, help="Path to the IMU .mat file")
    # parser.add_argument('--vic_path',   type=str, required=True, help="Path to the Vicon .mat file")
    # parser.add_argument('--param_path', type=str, required=True, help="Path to the IMU parameters .mat file")
    # args = parser.parse_args()
    main()
