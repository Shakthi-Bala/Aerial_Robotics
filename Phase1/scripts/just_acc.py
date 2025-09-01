import sys, pathlib
import numpy as np
from scipy import io
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from rotplot import rotplot 

# ----------------- Utilities -----------------
def get_data(d, main, alt=None):
    return d.get(main, d.get(alt)) if alt else d.get(main)

def load_imu(path):
    m = io.loadmat(path)
    vals = np.asarray(get_data(m, "vals", "data"))       # shape (6, N)
    ts   = np.asarray(get_data(m, "ts", "time")).ravel() # shape (N,)
    if vals is None or ts is None:
        raise ValueError(f"Missing 'vals'/'ts' in {path}")
    return vals, ts

def load_vicon(path):
    v = io.loadmat(path)
    rots = np.asarray(v["rots"])                         # shape (3,3,N)
    ts   = np.asarray(get_data(v, "ts", "time")).ravel() # shape (N,)
    if rots is None or ts is None:
        raise ValueError(f"Missing 'rots'/'ts' in {path}")
    return rots, ts

def load_params(path):
    p = io.loadmat(path)
    IMUParams = np.asarray(p["IMUParams"])  # 2x3 [sx sy sz], [bax bay baz]
    sx, sy, sz = IMUParams[0]
    bax, bay, baz = IMUParams[1]
    return (sx, sy, sz), (bax, bay, baz)

def rmat_stack_to_R(rots_3x3xN):
    mats = np.transpose(rots_3x3xN, (2, 0, 1))  # (N,3,3)
    return R.from_matrix(mats)

def nearest_indices(t_ref, t_query): #Got this synchronization from gpt
    idx = np.searchsorted(t_ref, t_query, side="left")
    idx = np.clip(idx, 0, len(t_ref)-1)
    left = np.maximum(idx - 1, 0)
    choose_left = (idx == len(t_ref)) | (
        (idx > 0) & (np.abs(t_query - t_ref[left]) <= np.abs(t_query - t_ref[idx]))
    )
    idx[choose_left] = left[choose_left]
    return idx

#Accel-only orientation
def accel_to_tilt_orient(ts, ax, ay, az, scales, biases, yaw0_rad=0.0):
    sx, sy, sz = scales
    bax, bay, baz = biases

    #ã = (a + b) / s
    ax_p = (ax + bax) / sx
    ay_p = (ay + bay) / sy
    az_p = (az + baz) / sz

    acc = np.vstack([ax_p, ay_p, az_p]).T  # (N,3)
    acc_norm = np.linalg.norm(acc, axis=1, keepdims=True)
    acc_unit = acc / np.clip(acc_norm, 1e-9, None)  # direction only

    f_w = np.array([0.0, 0.0, -1.0])  # world gravity direction

    R_list = []
    for u in acc_unit:
        v = np.cross(f_w, u)
        s = np.linalg.norm(v)
        c = float(np.dot(f_w, u))

        if s < 1e-9:  # nearly parallel or antiparallel
            if c > 0:
                dR = R.identity()
            else:
                dR = R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0]))  # 180° about x
        else:
            axis = v / s
            angle = np.arctan2(s, c)
            dR = R.from_rotvec(axis * angle)

        R_yaw = R.from_euler('Z', yaw0_rad)  # constant yaw
        R_list.append(R_yaw * dR)            # world-yaw ∘ tilt

    return R.concatenate(R_list)

# 3D Orientation plotting helpers (using rotplot)
def plot_3d_frames_from_rotation_sequence(R_seq, t_seq, title, n_frames=3):
    """ 
    Show n_frames snapshots of the rotation sequence using rotplot (start/mid/end).
    """
    mats = R_seq.as_matrix()        # (N,3,3)
    N = mats.shape[0]
    if N < 1:
        raise ValueError("Empty rotation sequence.")
    idx = np.linspace(0, N - 1, n_frames, dtype=int)

    fig = plt.figure(figsize=(4 * n_frames, 4))
    fig.suptitle(title, fontsize=12)
    for i, k in enumerate(idx, 1):
        ax = fig.add_subplot(1, n_frames, i, projection='3d')
        rotplot(mats[k], currentAxes=ax)
        ax.set_title(f"t = {t_seq[k]:.3f}s\nidx {k}")
    return fig

def plot_3d_pair_at_index(RA, RB, t_seq, k, labels=("A","B"), title="3D comparison at k"):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    rotplot(RA.as_matrix()[k], currentAxes=ax)
    rotplot(RB.as_matrix()[k], currentAxes=ax)
    ax.set_title(f"{title} = {k} (t={t_seq[k]:.3f}s)\n{labels[0]} vs {labels[1]}")
    return fig

imu_path = "/home/alien/MyDirectoryID_p0/Phase1/Data/Train/IMU/imuRaw1.mat"
vic_path = "/home/alien/MyDirectoryID_p0/Phase1/Data/Train/Vicon/viconRot1.mat"
par_path = "/home/alien/MyDirectoryID_p0/Phase1/IMUParams.mat" 

vals, t_imu = load_imu(imu_path)
rots_vic_3x3xN, t_vic = load_vicon(vic_path)
(scales, biases) = load_params(par_path)

# [ax ay az ωz ωx ωy]
ax, ay, az = vals[0], vals[1], vals[2]

# initial yaw from Vicon
R_vic_full = rmat_stack_to_R(rots_vic_3x3xN)
idx0 = nearest_indices(t_vic, np.array([t_imu[0]]))[0]
yaw0_deg = R_vic_full[idx0].as_euler('ZYX', degrees=True)[0]
yaw0_rad = np.deg2rad(yaw0_deg)

# Accel-only orientation (tilt + constant yaw) 
R_acc = accel_to_tilt_orient(t_imu, ax, ay, az, scales, biases, yaw0_rad=yaw0_rad)

# Bring Vicon to IMU timeline for comparison 
idx_match = nearest_indices(t_vic, t_imu)
R_vic_aligned = rmat_stack_to_R(rots_vic_3x3xN[:, :, idx_match])

# 2D plots: roll/pitch (accel yaw is constant)
eul_acc = R_acc.as_euler('ZYX', degrees=True)        # [yaw, pitch, roll]
eul_vic = R_vic_aligned.as_euler('ZYX', degrees=True)

plt.figure(); plt.title("Pitch (ZYX) vs time")
plt.plot(t_imu, eul_vic[:,1], label="Vicon pitch")
plt.plot(t_imu, eul_acc[:,1], "--", label="Accel-only pitch")
plt.xlabel("time (s)"); plt.ylabel("deg"); plt.grid(True); plt.legend()

plt.figure(); plt.title("Roll (ZYX) vs time")
plt.plot(t_imu, eul_vic[:,2], label="Vicon roll")
plt.plot(t_imu, eul_acc[:,2], "--", label="Accel-only roll")
plt.xlabel("time (s)"); plt.ylabel("deg"); plt.grid(True); plt.legend()

plt.figure(); plt.title("Yaw (ZYX) vs time (accel-only)")
plt.plot(t_imu, eul_vic[:,0], label="Vicon yaw")
plt.plot(t_imu, eul_acc[:,0], "--", label="Accel-only yaw (constant)")
plt.xlabel("time (s)"); plt.ylabel("deg"); plt.grid(True); plt.legend()

# 3D orientation snapshots using rotplot 
plot_3d_frames_from_rotation_sequence(R_vic_aligned, t_imu, "Vicon 3D Orientation (snapshots)", n_frames=3)
plot_3d_frames_from_rotation_sequence(R_acc,         t_imu, "Accel-only 3D Orientation (snapshots)", n_frames=3)

plt.show()
