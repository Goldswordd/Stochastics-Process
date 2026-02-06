import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time

# Cấu hình hệ thống
ANCHORS = np.array([[0, 0], [5.0, 0], [5.0, 7.0], [0, 7.0]])  # mét
DT = 0.1  # 100 ms
SIM_TIME = 30  # giây

def generate_trajectory(num_points):
    """Tạo quỹ đạo hình số 8 nằm ngang"""
    t = np.linspace(0, 2*np.pi, num_points)

    # Hình số 8 với phương trình tham số
    scale_x = 1.5  # tỷ lệ theo trục x
    scale_y = 1.0  # tỷ lệ theo trục y
    center_x = 2.5  # trung tâm theo x
    center_y = 3.5  # trung tâm theo y

    x = center_x + scale_x * np.sin(t)
    y = center_y + scale_y * np.sin(t) * np.cos(t)

    # Tính vận tốc (đạo hàm)
    vx = scale_x * np.cos(t) * (2*np.pi/num_points/DT)
    vy = scale_y * (np.cos(t)**2 - np.sin(t)**2) * (2*np.pi/num_points/DT)

    return x, y, vx, vy

def simulate_measurements(true_x, true_y):
    """Mô phỏng phép đo khoảng cách với nhiễu Gauss"""
    measurements = []
    for i in range(len(true_x)):
        distances = []
        for anchor in ANCHORS:
            # Khoảng cách thực
            true_dist = np.sqrt((true_x[i] - anchor[0])**2 +
                               (true_y[i] - anchor[1])**2)

            # Nhiễu: 0.1 + 0.02*distance (mét)
            noise_std = 0.1 + 0.02 * true_dist
            noisy_dist = true_dist + np.random.normal(0, noise_std)

            # Đảm bảo khoảng cách không âm
            noisy_dist = max(noisy_dist, 0.1)
            distances.append(noisy_dist)
        measurements.append(distances)
    return np.array(measurements)

def lse_trilateration(distances):
    """Ước lượng vị trí bằng phương pháp bình phương tối thiểu"""
    if len(distances) < 4:
        return np.array([np.nan, np.nan])

    # Anchor tham chiếu (anchor đầu tiên)
    x1, y1 = ANCHORS[0]
    d1 = distances[0]

    A = []
    b = []

    for i in range(1, 4):
        xi, yi = ANCHORS[i]
        di = distances[i]

        A.append([xi - x1, yi - y1])
        b.append(0.5 * (xi**2 + yi**2 - di**2 - (x1**2 + y1**2 - d1**2)))

    try:
        A = np.array(A)
        b = np.array(b)
        return np.linalg.lstsq(A, b, rcond=None)[0]
    except:
        return np.array([np.nan, np.nan])

class ExtendedKalmanFilter:
    def __init__(self, initial_pos):
        # Kích thước trạng thái: [x, y, vx, vy]
        self.dim_x = 4
        # Kích thước đo: [d1, d2, d3, d4]
        self.dim_z = 4

        # Khởi tạo trạng thái
        self.x = np.array([initial_pos[0], initial_pos[1], 0, 0])

        # Ma trận hiệp phương sai trạng thái
        self.P = np.eye(self.dim_x) * 100

        # Ma trận chuyển trạng thái
        self.F = np.eye(self.dim_x)
        self.F[0, 2] = DT
        self.F[1, 3] = DT

        # Ma trận nhiễu quá trình (gia tốc trắng)
        dt = DT
        sigma_a = 0.1
        self.Q = sigma_a**2 * np.array([
            [dt**3/3, 0, dt**2/2, 0],
            [0, dt**3/3, 0, dt**2/2],
            [dt**2/2, 0, dt, 0],
            [0, dt**2/2, 0, dt]
        ])

        # Ma trận nhiễu đo (sẽ được cập nhật động)
        self.R_base = np.eye(self.dim_z) * 0.01

    def measurement_function(self, x):
        """Hàm đo: khoảng cách từ trạng thái đến các anchor"""
        distances = []
        for anchor in ANCHORS:
            dx = x[0] - anchor[0]
            dy = x[1] - anchor[1]
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)
        return np.array(distances)

    def measurement_jacobian(self, x):
        """Ma trận Jacobian của hàm đo"""
        H = np.zeros((self.dim_z, self.dim_x))

        for i, anchor in enumerate(ANCHORS):
            dx = x[0] - anchor[0]
            dy = x[1] - anchor[1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist > 0.001:
                H[i, 0] = dx / dist  # ∂h/∂x
                H[i, 1] = dy / dist  # ∂h/∂y
                # ∂h/∂vx và ∂h/∂vy = 0
        return H

    def predict(self):
        """Bước dự đoán"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z, measured_dists):
        """Bước cập nhật với phép đo z"""
        # Cập nhật R dựa trên khoảng cách đo
        for i in range(self.dim_z):
            self.R_base[i, i] = (0.1 + 0.02 * measured_dists[i])**2

        # Tính Jacobian tại trạng thái dự đoán
        H = self.measurement_jacobian(self.x)

        # Dự đoán phép đo
        z_pred = self.measurement_function(self.x)

        # Độ lệch
        y = z - z_pred

        # Hiệp phương sai đổi mới
        S = H @ self.P @ H.T + self.R_base

        # Hệ số Kalman
        K = self.P @ H.T @ np.linalg.inv(S)

        # Cập nhật trạng thái
        self.x = self.x + K @ y

        # Cập nhật hiệp phương sai
        self.P = (np.eye(self.dim_x) - K @ H) @ self.P

        return self.x

def main_simulation():
    # Tạo quỹ đạo thực
    num_points = int(SIM_TIME / DT)
    true_x, true_y, true_vx, true_vy = generate_trajectory(num_points)

    # Mô phỏng phép đo
    measurements = simulate_measurements(true_x, true_y)

    # Ước lượng với LSE
    lse_positions = []
    for i in range(num_points):
        pos = lse_trilateration(measurements[i])
        lse_positions.append(pos)
    lse_positions = np.array(lse_positions)

    # Ước lượng với EKF
    ekf_positions = []
    ekf = None

    for i in range(num_points):
        # Khởi tạo EKF ở bước đầu tiên
        if ekf is None and not np.any(np.isnan(lse_positions[i])):
            ekf = ExtendedKalmanFilter(lse_positions[i])
            ekf_positions.append(lse_positions[i])
            continue

        if ekf is not None:
            # Dự đoán
            ekf.predict()

            # Cập nhật với phép đo
            pos = ekf.update(measurements[i], measurements[i])[:2]
            ekf_positions.append(pos)
        else:
            ekf_positions.append([np.nan, np.nan])

    ekf_positions = np.array(ekf_positions)

    # Tính sai số
    lse_errors = []
    ekf_errors = []

    for i in range(num_points):
        if not np.any(np.isnan(lse_positions[i])):
            lse_error = np.sqrt((lse_positions[i,0] - true_x[i])**2 +
                               (lse_positions[i,1] - true_y[i])**2)
            lse_errors.append(lse_error)

        if i < len(ekf_positions) and not np.any(np.isnan(ekf_positions[i])):
            ekf_error = np.sqrt((ekf_positions[i,0] - true_x[i])**2 +
                               (ekf_positions[i,1] - true_y[i])**2)
            ekf_errors.append(ekf_error)

    # Tính RMSE
    lse_rmse = np.sqrt(np.mean(np.array(lse_errors)**2))
    ekf_rmse = np.sqrt(np.mean(np.array(ekf_errors)**2))

    # Tính RMSE theo từng trục
    lse_rmse_x = np.sqrt(np.mean((lse_positions[:,0][~np.isnan(lse_positions[:,0])] -
                                 true_x[:len(lse_positions[:,0][~np.isnan(lse_positions[:,0])])])**2))
    lse_rmse_y = np.sqrt(np.mean((lse_positions[:,1][~np.isnan(lse_positions[:,1])] -
                                 true_y[:len(lse_positions[:,1][~np.isnan(lse_positions[:,1])])])**2))

    ekf_rmse_x = np.sqrt(np.mean((ekf_positions[:,0][~np.isnan(ekf_positions[:,0])] -
                                 true_x[:len(ekf_positions[:,0][~np.isnan(ekf_positions[:,0])])])**2))
    ekf_rmse_y = np.sqrt(np.mean((ekf_positions[:,1][~np.isnan(ekf_positions[:,1])] -
                                 true_y[:len(ekf_positions[:,1][~np.isnan(ekf_positions[:,1])])])**2))

    return {
        'true_traj': (true_x, true_y),
        'lse_traj': (lse_positions[:,0], lse_positions[:,1]),
        'ekf_traj': (ekf_positions[:,0], ekf_positions[:,1]),
        'lse_errors': lse_errors,
        'ekf_errors': ekf_errors,
        'rmse': {
            'lse_total': lse_rmse,
            'ekf_total': ekf_rmse,
            'lse_x': lse_rmse_x,
            'lse_y': lse_rmse_y,
            'ekf_x': ekf_rmse_x,
            'ekf_y': ekf_rmse_y
        }
    }

def parameter_sensitivity_analysis():
    """Khảo sát ảnh hưởng của tham số R đến hiệu năng EKF"""
    R_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    results = {}

    for R_val in R_values:
        # Chạy mô phỏng với R cố định
        # ... (tương tự main_simulation nhưng với R cố định)

        # Lưu kết quả RMSE
        results[R_val] = {
            'rmse': ekf_rmse,
            'smoothness': calculate_smoothness(ekf_positions)
        }

    return results

def calculate_smoothness(trajectory):
    """Tính độ mượt của quỹ đạo (dựa trên đạo hàm bậc 2)"""
    if len(trajectory) < 3:
        return np.nan

    # Loại bỏ giá trị NaN
    traj = trajectory[~np.isnan(trajectory).any(axis=1)]

    if len(traj) < 3:
        return np.nan

    # Tính gia tốc (đạo hàm bậc 2 của vị trí)
    acc = np.diff(traj, n=2, axis=0)

    # Độ mượt tỷ lệ nghịch với gia tốc trung bình
    smoothness = 1.0 / (np.mean(np.abs(acc)) + 1e-6)
    return smoothness

def parameter_sensitivity_analysis():
    """Khảo sát ảnh hưởng của tham số R đến hiệu năng EKF"""
    R_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    results = {}

    for R_val in R_values:
        # Chạy mô phỏng với R cố định
        # ... (tương tự main_simulation nhưng với R cố định)

        # Lưu kết quả RMSE
        results[R_val] = {
            'rmse': ekf_rmse,
            'smoothness': calculate_smoothness(ekf_positions)
        }

    return results

def calculate_smoothness(trajectory):
    """Tính độ mượt của quỹ đạo (dựa trên đạo hàm bậc 2)"""
    if len(trajectory) < 3:
        return np.nan

    # Loại bỏ giá trị NaN
    traj = trajectory[~np.isnan(trajectory).any(axis=1)]

    if len(traj) < 3:
        return np.nan

    # Tính gia tốc (đạo hàm bậc 2 của vị trí)
    acc = np.diff(traj, n=2, axis=0)

    # Độ mượt tỷ lệ nghịch với gia tốc trung bình
    smoothness = 1.0 / (np.mean(np.abs(acc)) + 1e-6)
    return smoothness

def visualize_results(results):
    """Vẽ các đồ thị kết quả"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Quỹ đạo so sánh
    ax = axes[0, 0]
    ax.plot(results['true_traj'][0], results['true_traj'][1],
            'g-', label='Thực tế', linewidth=2)
    ax.plot(results['lse_traj'][0], results['lse_traj'][1],
            'r--', label='LSE', alpha=0.7)
    ax.plot(results['ekf_traj'][0], results['ekf_traj'][1],
            'b-', label='EKF', alpha=0.7)
    ax.scatter(ANCHORS[:,0], ANCHORS[:,1], c='k', marker='^',
               s=100, label='Anchor')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('So sánh quỹ đạo ước lượng')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    # 2. CDF của sai số
    ax = axes[0, 1]
    if len(results['lse_errors']) > 0:
        lse_sorted = np.sort(results['lse_errors'])
        lse_cdf = np.arange(1, len(lse_sorted)+1) / len(lse_sorted)
        ax.plot(lse_sorted, lse_cdf, 'r-', label=f'LSE (RMSE={results["rmse"]["lse_total"]:.3f}m)')

    if len(results['ekf_errors']) > 0:
        ekf_sorted = np.sort(results['ekf_errors'])
        ekf_cdf = np.arange(1, len(ekf_sorted)+1) / len(ekf_sorted)
        ax.plot(ekf_sorted, ekf_cdf, 'b-', label=f'EKF (RMSE={results["rmse"]["ekf_total"]:.3f}m)')

    ax.set_xlabel('Sai số vị trí (m)')
    ax.set_ylabel('CDF')
    ax.set_title('Hàm phân phối tích lũy sai số')
    ax.legend()
    ax.grid(True)

    # 3. RMSE theo từng trục
    ax = axes[0, 2]
    methods = ['LSE', 'EKF']
    rmse_x = [results['rmse']['lse_x'], results['rmse']['ekf_x']]
    rmse_y = [results['rmse']['lse_y'], results['rmse']['ekf_y']]

    x_pos = np.arange(len(methods))
    width = 0.35

    ax.bar(x_pos - width/2, rmse_x, width, label='RMSE X', color='skyblue')
    ax.bar(x_pos + width/2, rmse_y, width, label='RMSE Y', color='lightcoral')

    ax.set_xlabel('Phương pháp')
    ax.set_ylabel('RMSE (m)')
    ax.set_title('RMSE theo trục X và Y')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, axis='y')

    # 4. So sánh sai số theo thời gian
    ax = axes[1, 0]
    time_axis = np.arange(len(results['lse_errors'])) * DT
    ax.plot(time_axis, results['lse_errors'], 'r-', alpha=0.7, label='LSE')
    ax.plot(time_axis[:len(results['ekf_errors'])], results['ekf_errors'],
            'b-', alpha=0.7, label='EKF')
    ax.set_xlabel('Thời gian (s)')
    ax.set_ylabel('Sai số vị trí (m)')
    ax.set_title('Sai số theo thời gian')
    ax.legend()
    ax.grid(True)

    # 5. Phân bố sai số
    ax = axes[1, 1]
    if len(results['lse_errors']) > 0:
        ax.hist(results['lse_errors'], bins=30, alpha=0.5, label='LSE',
                density=True, color='red')
    if len(results['ekf_errors']) > 0:
        ax.hist(results['ekf_errors'], bins=30, alpha=0.5, label='EKF',
                density=True, color='blue')
    ax.set_xlabel('Sai số vị trí (m)')
    ax.set_ylabel('Mật độ xác suất')
    ax.set_title('Phân bố sai số vị trí')
    ax.legend()
    ax.grid(True)

    # 6. Bảng kết quả số
    ax = axes[1, 2]
    ax.axis('tight')
    ax.axis('off')

    table_data = [
        ['Phương pháp', 'RMSE tổng (m)', 'RMSE X (m)', 'RMSE Y (m)'],
        ['LSE', f"{results['rmse']['lse_total']:.3f}",
         f"{results['rmse']['lse_x']:.3f}", f"{results['rmse']['lse_y']:.3f}"],
        ['EKF', f"{results['rmse']['ekf_total']:.3f}",
         f"{results['rmse']['ekf_x']:.3f}", f"{results['rmse']['ekf_y']:.3f}"]
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.tight_layout()
    plt.savefig('ekf_uwb_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Chạy mô phỏng chính
    results = main_simulation()

    # Trực quan hóa kết quả
    visualize_results(results)

    # Khảo sát ảnh hưởng của tham số R
    sensitivity_results = parameter_sensitivity_analysis()

    print("=== KẾT QUẢ CHÍNH ===")
    print(f"RMSE LSE: {results['rmse']['lse_total']:.3f} m")
    print(f"RMSE EKF: {results['rmse']['ekf_total']:.3f} m")
    print(f"Cải thiện: {((results['rmse']['lse_total'] - results['rmse']['ekf_total'])/results['rmse']['lse_total']*100):.1f}%")

    print("\n=== ẢNH HƯỞNG CỦA THAM SỐ R ===")
    for R_val, res in sensitivity_results.items():
        print(f"R = {R_val}: RMSE = {res['rmse']:.3f} m, Độ mượt = {res['smoothness']:.3f}")
