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
    R_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    results = {}

    # Tạo quỹ đạo thực và phép đo (dùng chung cho tất cả các thử nghiệm)
    num_points = int(SIM_TIME / DT)
    true_x, true_y, true_vx, true_vy = generate_trajectory(num_points)
    measurements = simulate_measurements(true_x, true_y)

    for R_val in R_values:
        print(f"Đang chạy mô phỏng với R = {R_val}...")

        # Khởi tạo EKF với R cố định
        ekf_positions = []
        ekf = None

        for i in range(num_points):
            # Lấy ước lượng ban đầu từ LSE
            initial_pos = lse_trilateration(measurements[i])

            # Khởi tạo EKF ở bước đầu tiên
            if ekf is None and not np.any(np.isnan(initial_pos)):
                ekf = SimpleEKFWithFixedR(initial_pos, R_val)
                ekf_positions.append(initial_pos)
                continue

            if ekf is not None:
                # Dự đoán
                ekf.predict()

                # Cập nhật với phép đo
                z = measurements[i]
                pos = ekf.update(z)[:2]
                ekf_positions.append(pos)
            else:
                ekf_positions.append([np.nan, np.nan])

        ekf_positions = np.array(ekf_positions)

        # Tính sai số
        ekf_errors = []
        for i in range(min(len(ekf_positions), len(true_x))):
            if not np.any(np.isnan(ekf_positions[i])):
                error = np.sqrt((ekf_positions[i, 0] - true_x[i])**2 +
                               (ekf_positions[i, 1] - true_y[i])**2)
                ekf_errors.append(error)

        # Tính RMSE
        ekf_rmse = np.sqrt(np.mean(np.array(ekf_errors)**2)) if ekf_errors else np.nan

        # Tính độ mượt
        smoothness = calculate_smoothness(ekf_positions)

        # Lưu kết quả
        results[R_val] = {
            'rmse': ekf_rmse,
            'smoothness': smoothness,
            'trajectory': ekf_positions.copy(),
            'errors': ekf_errors.copy()
        }

    return results, true_x, true_y, measurements


class SimpleEKFWithFixedR:
    """Phiên bản EKF với R cố định để khảo sát tham số"""
    def __init__(self, initial_pos, R_value):
        self.dim_x = 4  # [x, y, vx, vy]
        self.dim_z = 4  # [d1, d2, d3, d4]

        # Khởi tạo trạng thái
        self.x = np.array([initial_pos[0], initial_pos[1], 0, 0])

        # Ma trận hiệp phương sai trạng thái
        self.P = np.eye(self.dim_x) * 100

        # Ma trận chuyển trạng thái
        self.F = np.eye(self.dim_x)
        self.F[0, 2] = DT
        self.F[1, 3] = DT

        # Ma trận nhiễu quá trình (giữ cố định)
        dt = DT
        sigma_a = 0.1
        self.Q = sigma_a**2 * np.array([
            [dt**3/3, 0, dt**2/2, 0],
            [0, dt**3/3, 0, dt**2/2],
            [dt**2/2, 0, dt, 0],
            [0, dt**2/2, 0, dt]
        ])

        # Ma trận nhiễu đo CỐ ĐỊNH (không phụ thuộc khoảng cách)
        self.R = np.eye(self.dim_z) * R_value

        # Lịch sử để tính độ mượt
        self.position_history = []

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
                H[i, 0] = dx / dist
                H[i, 1] = dy / dist
        return H

    def predict(self):
        """Bước dự đoán"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        """Bước cập nhật với phép đo z"""
        # Tính Jacobian tại trạng thái dự đoán
        H = self.measurement_jacobian(self.x)

        # Dự đoán phép đo
        z_pred = self.measurement_function(self.x)

        # Độ lệch
        y = z - z_pred

        # Hiệp phương sai đổi mới
        S = H @ self.P @ H.T + self.R

        # Hệ số Kalman
        K = self.P @ H.T @ np.linalg.inv(S)

        # Cập nhật trạng thái
        self.x = self.x + K @ y

        # Cập nhật hiệp phương sai
        self.P = (np.eye(self.dim_x) - K @ H) @ self.P

        # Lưu vị trí vào lịch sử
        self.position_history.append(self.x[:2].copy())

        return self.x


def visualize_parameter_sensitivity(results, true_x, true_y):
    """Trực quan hóa kết quả khảo sát tham số"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. RMSE theo R
    ax = axes[0, 0]
    R_vals = sorted(results.keys())
    rmse_vals = [results[R]['rmse'] for R in R_vals]

    ax.plot(R_vals, rmse_vals, 'bo-', linewidth=2, markersize=8)
    ax.set_xscale('log')
    ax.set_xlabel('Giá trị R (log scale)')
    ax.set_ylabel('RMSE (m)')
    ax.set_title('Ảnh hưởng của R đến RMSE')
    ax.grid(True, which="both", ls="--")

    # Đánh dấu điểm tối ưu
    min_rmse_idx = np.argmin(rmse_vals)
    ax.plot(R_vals[min_rmse_idx], rmse_vals[min_rmse_idx], 'r*',
            markersize=15, label=f'Tối ưu: R={R_vals[min_rmse_idx]}, RMSE={rmse_vals[min_rmse_idx]:.3f}m')
    ax.legend()

    # 2. Độ mượt theo R
    ax = axes[0, 1]
    smoothness_vals = [results[R]['smoothness'] for R in R_vals]

    ax.plot(R_vals, smoothness_vals, 'go-', linewidth=2, markersize=8)
    ax.set_xscale('log')
    ax.set_xlabel('Giá trị R (log scale)')
    ax.set_ylabel('Độ mượt (tỷ lệ nghịch với gia tốc)')
    ax.set_title('Ảnh hưởng của R đến độ mượt quỹ đạo')
    ax.grid(True, which="both", ls="--")

    # 3. So sánh quỹ đạo với các giá trị R khác nhau
    ax = axes[0, 2]
    # Chọn 3 giá trị R để so sánh: nhỏ, tối ưu, lớn
    sample_Rs = [R_vals[0], R_vals[min_rmse_idx], R_vals[-1]]
    colors = ['r', 'g', 'b']
    # SỬA LỖI Ở ĐÂY: sử dụng sample_Rs thay vì biến R chưa định nghĩa
    labels = [f'R={sample_Rs[0]} (nhỏ)', f'R={sample_Rs[1]} (tối ưu)', f'R={sample_Rs[2]} (lớn)']

    # Vẽ quỹ đạo thực
    ax.plot(true_x, true_y, 'k--', linewidth=3, alpha=0.5, label='Thực tế')

    for i, R_val in enumerate(sample_Rs):
        traj = results[R_val]['trajectory']
        valid_idx = ~np.isnan(traj[:, 0])
        ax.plot(traj[valid_idx, 0], traj[valid_idx, 1],
                color=colors[i], linewidth=1.5, alpha=0.8, label=labels[i])

    ax.scatter(ANCHORS[:, 0], ANCHORS[:, 1], c='k', marker='^',
               s=100, label='Anchor')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('So sánh quỹ đạo với các giá trị R khác nhau')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    # 4. CDF sai số với các giá trị R khác nhau
    ax = axes[1, 0]
    for i, R_val in enumerate(sample_Rs):
        errors = results[R_val]['errors']
        if len(errors) > 0:
            sorted_errors = np.sort(errors)
            cdf = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
            ax.plot(sorted_errors, cdf, color=colors[i], linewidth=2,
                   label=f'R={R_val} (RMSE={results[R_val]["rmse"]:.3f}m)')

    ax.set_xlabel('Sai số vị trí (m)')
    ax.set_ylabel('CDF')
    ax.set_title('CDF sai số với các giá trị R khác nhau')
    ax.legend()
    ax.grid(True)

    # 5. Phân bố sai số với các giá trị R khác nhau
    ax = axes[1, 1]
    bins = np.linspace(0, 2.0, 50)  # Sai số từ 0 đến 2m

    for i, R_val in enumerate(sample_Rs):
        errors = results[R_val]['errors']
        if len(errors) > 0:
            ax.hist(errors, bins=bins, alpha=0.5, color=colors[i],
                   density=True, label=f'R={R_val}')

    ax.set_xlabel('Sai số vị trí (m)')
    ax.set_ylabel('Mật độ xác suất')
    ax.set_title('Phân bố sai số với các giá trị R khác nhau')
    ax.legend()
    ax.grid(True)

    # 6. Bảng kết quả chi tiết
    ax = axes[1, 2]
    ax.axis('tight')
    ax.axis('off')

    # Tạo dữ liệu cho bảng
    table_data = [['R', 'RMSE (m)', 'Độ mượt', 'Sai số trung vị (m)', 'Sai số max (m)']]

    for R_val in R_vals:
        errors = results[R_val]['errors']
        if len(errors) > 0:
            median_error = np.median(errors)
            max_error = np.max(errors)
        else:
            median_error = np.nan
            max_error = np.nan

        table_data.append([
            f'{R_val}',
            f'{results[R_val]["rmse"]:.3f}',
            f'{results[R_val]["smoothness"]:.3f}',
            f'{median_error:.3f}',
            f'{max_error:.3f}'
        ])

    # Tìm hàng có RMSE tốt nhất để highlight
    highlight_row = min_rmse_idx + 1  # +1 vì hàng đầu là header

    # Tạo bảng
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Highlight hàng tốt nhất
    for j in range(len(table_data[0])):
        table.get_celld()[(highlight_row, j)].set_facecolor('#90EE90')  # Màu xanh nhạt

    plt.suptitle('KHẢO SÁT ẢNH HƯỞNG CỦA THAM SỐ R TRONG EKF', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('parameter_sensitivity_R.png', dpi=300, bbox_inches='tight')
    plt.show()

    return sample_Rs


# Hàm main để chạy khảo sát
if __name__ == "__main__":
    # Chạy khảo sát tham số R
    results, true_x, true_y, measurements = parameter_sensitivity_analysis()

    # Trực quan hóa kết quả
    sample_Rs = visualize_parameter_sensitivity(results, true_x, true_y)

    # In kết quả tóm tắt
    print("\n=== KẾT QUẢ KHẢO SÁT THAM SỐ R ===")
    print("Giá trị R được khảo sát:", sorted(results.keys()))

    # Tìm giá trị R tối ưu
    R_vals = sorted(results.keys())
    rmse_vals = [results[R]['rmse'] for R in R_vals]
    min_rmse_idx = np.argmin(rmse_vals)

    print(f"\nGiá trị R tối ưu: {R_vals[min_rmse_idx]}")
    print(f"RMSE tối ưu: {rmse_vals[min_rmse_idx]:.3f} m")

    # Phân tích xu hướng
    print("\n=== PHÂN TÍCH XU HƯỚNG ===")
    print("1. R quá nhỏ (< 0.01):")
    print("   - EKF tin tưởng quá nhiều vào phép đo nhiễu")
    print("   - Quỹ đạo dao động mạnh, độ mượt thấp")
    print("   - RMSE cao do bám theo nhiễu")

    print("\n2. R tối ưu (0.1 - 1.0):")
    print("   - Cân bằng giữa tin tưởng mô hình và phép đo")
    print("   - Quỹ đạo mượt và chính xác")
    print("   - RMSE thấp nhất")

    print("\n3. R quá lớn (> 10.0):")
    print("   - EKF bỏ qua phép đo, chỉ dựa vào mô hình động")
    print("   - Quỹ đạo quá mượt nhưng không bám sát thực tế")
    print("   - Sai số tích lũy theo thời gian do không hiệu chỉnh")

    # So sánh với LSE
    # Tính RMSE của LSE
    lse_errors = []
    for i in range(len(true_x)):
        pos = lse_trilateration(measurements[i])
        if not np.any(np.isnan(pos)):
            error = np.sqrt((pos[0] - true_x[i])**2 + (pos[1] - true_y[i])**2)
            lse_errors.append(error)

    lse_rmse = np.sqrt(np.mean(np.array(lse_errors)**2))

    print(f"\n=== SO SÁNH VỚI LSE ===")
    print(f"RMSE của LSE: {lse_rmse:.3f} m")
    print(f"RMSE tốt nhất của EKF: {rmse_vals[min_rmse_idx]:.3f} m")
    improvement = ((lse_rmse - rmse_vals[min_rmse_idx]) / lse_rmse) * 100
    print(f"Cải thiện: {improvement:.1f}%")
