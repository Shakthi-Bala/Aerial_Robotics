import sys, pathlib
import numpy as np
from scipy import io
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from rotplot import rotplot

def get_data(d, main, alt=None):
    return d.get(main, d.get(alt)) if alt else d.get(main)

def load_imu(path):
    m = io.loadmat(path)
    vals = get_data(m, "vals", "data")         # (6,N)
    ts   = np.asarray(get_data(m, "ts", "time")).ravel()
    if vals is None or ts is None:
        raise ValueError(f"Missing 'vals'/'ts' in {path}")
    return np.asarray(vals), ts

def load_vicon(path):
    v = io.loadmat(path)
    rots = np.asarray(v["rots"])               # (3,3,N)
    ts   = np.asarray(get_data(v, "ts", "time")).ravel()
    if rots is None or ts is None:
        raise ValueError(f"Missing 'rots'/'ts' in {path}")
    return rots, ts

def gyro_counts_to_rads(counts, n_bias=200):
    """ω = s * (counts - bias). s from spec; first n_bias samples for bias."""
    n0 = min(n_bias, counts.size)
    bg = counts[:n0].mean()
    scale = (3300.0/1023.0) * (np.pi/180.0) * 0.3
    return scale * (counts - bg)

def nearest_indices(t_ref, t_query):
    idx = np.searchsorted(t_ref, t_query, side="left")
    idx = np.clip(idx, 0, len(t_ref)-1)
    left = np.maximum(idx - 1, 0)
    choose_left = (idx == len(t_ref)) | (
        (idx > 0) & (np.abs(t_query - t_ref[left]) <= np.abs(t_query - t_ref[idx]))
    )
    idx[choose_left] = left[choose_left]
    return idx

def integrate_gyro(ts, wx, wy, wz, R0_3x3):
    """
    Integrate body-frame angular rates to orientation:
      R_{k+1} = R_k * Exp(ω_k * dt_k)
    Returns scipy Rotation of length N.
    """
    N = ts.size
    R_seq = [R.from_matrix(R0_3x3)]
    for k in range(N - 1):
        dt = ts[k+1] - ts[k]
        if dt <= 0:
            dt = 1e-6
        rotvec = np.array([wx[k], wy[k], wz[k]]) * dt
        dR = R.from_rotvec(rotvec)
        R_seq.append(R_seq[-1] * dR)  # right-multiply for body rates
    return R.concatenate(R_seq)

def rmat_stack_to_R(rots_3x3xN):
    mats = np.transpose(rots_3x3xN, (2, 0, 1))  # (N,3,3)
    return R.from_matrix(mats)

def plot_3d_frames_from_rotation_sequence(R_seq, t_seq, title, n_frames=3):
    mats = R_seq.as_matrix()
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

# --- Paths (adjust as needed) ---
imu_path = "/home/alien/MyDirectoryID_p0/Phase1/Data/Train/IMU/imuRaw1.mat"
vic_path = "/home/alien/MyDirectoryID_p0/Phase1/Data/Train/Vicon/viconRot1.mat"

# --- Load data ---
vals, t_imu = load_imu(imu_path)
rots_vic_3x3xN, t_vic = load_vicon(vic_path)

# Unpack IMU channels: [ax ay az ωz ωx ωy]
ax, ay, az            = vals[0], vals[1], vals[2]
wz_counts, wx_counts, wy_counts = vals[3], vals[4], vals[5]

# Convert gyro counts to rad/s
wx = gyro_counts_to_rads(wx_counts)
wy = gyro_counts_to_rads(wy_counts)
wz = gyro_counts_to_rads(wz_counts)

# Initial orientation from nearest Vicon at t0
idx0 = nearest_indices(t_vic, np.array([t_imu[0]]))[0]
R0 = rots_vic_3x3xN[:, :, idx0]

# Integrate gyro-only orientation
R_gyro = integrate_gyro(t_imu, wx, wy, wz, R0)

# Vicon orientation aligned to IMU timestamps
idx_match = nearest_indices(t_vic, t_imu)
R_vic = rmat_stack_to_R(rots_vic_3x3xN[:, :, idx_match])

# --- 3D orientation snapshots only (no 2D plots) ---
plot_3d_frames_from_rotation_sequence(R_vic,  t_imu, "Vicon 3D Orientation (snapshots)", n_frames=3)
plot_3d_frames_from_rotation_sequence(R_gyro, t_imu, "Gyro-only 3D Orientation (snapshots)", n_frames=3)

plt.show()
