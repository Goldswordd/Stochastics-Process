import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

# Thiết lập style nâng cao
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.2)

class PoissonBinomialConvergence:
    def __init__(self, lambda_param=3, n_values=None):
        self.lambda_param = lambda_param

        if n_values is None:
            self.n_values = [5, 10, 20, 50, 100, 200]
        else:
            self.n_values = n_values

        self.k_values = np.arange(0, 15)

    def calculate_probabilities(self):
        results = {}
        poisson_probs = poisson.pmf(self.k_values, self.lambda_param)
        results['Poisson'] = poisson_probs

        for n in self.n_values:
            p = self.lambda_param / n
            binomial_probs = binom.pmf(self.k_values, n, p)
            results[f'Binomial_n={n}'] = binomial_probs

        return results

    def plot_comparison(self, results):
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
        fig.suptitle(f'So sánh phân phối Nhị thức và Poisson với λ = {self.lambda_param}',
                    fontsize=18, fontweight='bold', y=1.05)

        axes = axes.ravel()
        poisson_probs = results['Poisson']

        for idx, n in enumerate(self.n_values):
            if idx >= len(axes):
                break

            ax = axes[idx]
            binomial_key = f'Binomial_n={n}'
            binomial_probs = results[binomial_key]
            p = self.lambda_param / n

            width = 0.35
            x = np.arange(len(self.k_values))

            bars1 = ax.bar(x - width/2, binomial_probs, width,
                          label=f'Binomial (n={n}, p={p:.3f})',
                          alpha=0.8, color='#3498db', edgecolor='darkblue', linewidth=1.5)

            bars2 = ax.bar(x + width/2, poisson_probs, width,
                          label=f'Poisson (λ={self.lambda_param})',
                          alpha=0.8, color='#e74c3c', edgecolor='darkred', linewidth=1.5)

            ax.set_xlabel('Số sự kiện (k)', fontsize=12)
            ax.set_ylabel('Xác suất P(X=k)', fontsize=12)
            ax.set_title(f'n = {n}\np = {p:.4f}, np = {self.lambda_param:.3f}',
                        fontsize=13, fontweight='bold', pad=10)
            ax.legend(fontsize=10, loc='best')
            ax.set_xticks(x[::2])
            ax.set_xticklabels(self.k_values[::2])
            ax.grid(True, alpha=0.3, linestyle='--')

            mae = np.mean(np.abs(binomial_probs - poisson_probs))
            info_text = f'MAE: {mae:.4f}'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow',
                            alpha=0.9, edgecolor='gold'))

        plt.savefig('poisson_binomial_comparison_enhanced.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def plot_convergence(self, results):
        fig = plt.figure(figsize=(14, 6), constrained_layout=True)

        poisson_probs = results['Poisson']
        mae_values = []

        for n in self.n_values:
            binomial_key = f'Binomial_n={n}'
            binomial_probs = results[binomial_key]
            mae = np.mean(np.abs(binomial_probs - poisson_probs))
            mae_values.append(mae)

        plt.subplot(1, 2, 1)
        plt.plot(self.n_values, mae_values, 'o-', linewidth=3, markersize=10,
                label='MAE', color='#2c3e50', markerfacecolor='#e74c3c',
                markeredgewidth=2, markeredgecolor='black')

        plt.xlabel('Số phép thử (n)', fontsize=13)
        plt.ylabel('Sai số tuyệt đối trung bình (MAE)', fontsize=13)
        plt.title('Sự hội tụ của sai số khi n tăng', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        log_n = np.log(self.n_values)
        log_mae = np.log(mae_values)

        plt.loglog(self.n_values, mae_values, 'o-', linewidth=3, markersize=10,
                  label='Dữ liệu', color='#9b59b6', markerfacecolor='#3498db',
                  markeredgewidth=2, markeredgecolor='black')

        plt.xlabel('n (thang log)', fontsize=13)
        plt.ylabel('MAE (thang log)', fontsize=13)
        plt.title('Tốc độ hội tụ (thang log-log)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, which='both')

        plt.suptitle(f'Sự hội tụ về phân phối Poisson (λ = {self.lambda_param})',
                    fontsize=16, fontweight='bold', y=1.05)

        plt.savefig('convergence_rate_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        return mae_values

    def plot_multiple_lambda_comparison(self, lambda_list=[1, 3, 5, 7], n_values_plot=[5, 10, 20, 50]):
        """
        Vẽ so sánh phân phối cho nhiều giá trị lambda (chỉ 1, 3, 5, 7)
        """
        fig, axes = plt.subplots(len(lambda_list), len(n_values_plot),
                                figsize=(4*len(n_values_plot), 3*len(lambda_list)),
                                constrained_layout=True)
        fig.suptitle('So sánh phân phối Nhị thức và Poisson với các giá trị λ khác nhau',
                    fontsize=20, fontweight='bold', y=1.05)

        for i, lam in enumerate(lambda_list):
            for j, n in enumerate(n_values_plot):
                ax = axes[i, j] if len(lambda_list) > 1 else axes[j]
                p = lam / n

                max_k = min(int(lam * 3) + 2, 25)
                k_values = np.arange(0, max_k + 1)

                binom_probs = binom.pmf(k_values, n, p)
                poisson_probs = poisson.pmf(k_values, lam)

                width = 0.35
                x = np.arange(len(k_values))

                ax.bar(x - width/2, binom_probs, width,
                      alpha=0.7, label=f'B(n={n})',
                      color='#3498db', edgecolor='darkblue', linewidth=1)
                ax.bar(x + width/2, poisson_probs, width,
                      alpha=0.7, label=f'Pois(λ={lam})',
                      color='#e74c3c', edgecolor='darkred', linewidth=1)

                mae = np.mean(np.abs(binom_probs - poisson_probs))

                ax.set_xlabel('Số sự kiện (k)', fontsize=10)
                if j == 0:
                    ax.set_ylabel(f'λ={lam}\nP(X=k)', fontsize=11)
                else:
                    ax.set_ylabel('P(X=k)', fontsize=10)

                ax.set_title(f'n={n}, p={p:.3f}', fontsize=11, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_xticks(x[::max(1, len(x)//5)])

                ax.text(0.05, 0.95, f'MAE: {mae:.4f}',
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.savefig('multiple_lambda_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        self.create_multi_lambda_summary(lambda_list, n_values_plot)

    def create_multi_lambda_summary(self, lambda_list, n_values_plot):
        summary_data = []

        for lam in lambda_list:
            for n in n_values_plot:
                p = lam / n
                k_values = np.arange(0, min(int(lam * 3) + 3, 30))

                binom_probs = binom.pmf(k_values, n, p)
                poisson_probs = poisson.pmf(k_values, lam)

                mae = np.mean(np.abs(binom_probs - poisson_probs))
                mse = np.mean((binom_probs - poisson_probs)**2)
                max_error = np.max(np.abs(binom_probs - poisson_probs))

                summary_data.append({
                    'λ': lam,
                    'n': n,
                    'p': f'{p:.4f}',
                    'MAE': f'{mae:.6f}',
                    'MSE': f'{mse:.8f}',
                    'Max Error': f'{max_error:.6f}',
                    'Chất lượng xấp xỉ': 'Tốt' if mae < 0.01 else 'Khá' if mae < 0.05 else 'Trung bình'
                })

        df_summary = pd.DataFrame(summary_data)

        print("\n" + "═" * 100)
        print(f"{'BẢNG TỔNG HỢP SAI SỐ':^100}")
        print("═" * 100)
        print(df_summary.to_string(index=False))
        print("═" * 100)

        df_summary.to_csv('multi_lambda_summary.csv', index=False)
        print(f"\nĐã lưu bảng tổng hợp vào 'multi_lambda_summary.csv'")

        return df_summary

    def plot_3d_error_surface(self):
        """
        Vẽ bề mặt sai số 3D đã sửa lỗi
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Tạo dữ liệu với các giá trị hợp lý
        lambda_vals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # 1 đến 10
        n_vals = np.array([5, 10, 15, 20, 30, 50, 100, 200, 500])  # Các giá trị n

        X, Y = np.meshgrid(lambda_vals, n_vals)
        Z = np.zeros_like(X, dtype=float)

        # Tính sai số cho mỗi cặp (λ, n)
        for i in range(len(n_vals)):
            for j in range(len(lambda_vals)):
                lam = lambda_vals[j]
                n = n_vals[i]

                if n > lam:  # Đảm bảo p <= 1
                    p = lam / n
                    k_range = np.arange(0, min(int(lam * 3) + 3, 30))

                    binom_probs = binom.pmf(k_range, n, p)
                    poisson_probs = poisson.pmf(k_range, lam)

                    Z[i, j] = np.mean(np.abs(binom_probs - poisson_probs))
                else:
                    Z[i, j] = np.nan  # Giá trị không hợp lệ

        # Vẽ bề mặt
        surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                              alpha=0.8, edgecolor='none', linewidth=0.1)

        ax.set_xlabel('λ (Tham số Poisson)', fontsize=12, labelpad=10)
        ax.set_ylabel('n (Số phép thử)', fontsize=12, labelpad=10)
        ax.set_zlabel('Sai số trung bình (MAE)', fontsize=12, labelpad=10)
        ax.set_title('Bề mặt sai số: Nhị thức → Poisson', fontsize=16, fontweight='bold', pad=20)

        # Thêm thanh màu
        fig.colorbar(surf, shrink=0.5, aspect=10, label='MAE')

        # Điều chỉnh góc nhìn
        ax.view_init(elev=30, azim=45)

        # Thêm grid
        ax.xaxis._axinfo['grid']['color'] = (0.5, 0.5, 0.5, 0.2)
        ax.yaxis._axinfo['grid']['color'] = (0.5, 0.5, 0.5, 0.2)
        ax.zaxis._axinfo['grid']['color'] = (0.5, 0.5, 0.5, 0.2)

        plt.savefig('3d_error_surface_corrected.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        # Vẽ thêm contour plot 2D
        self.plot_2d_contour(lambda_vals, n_vals, Z)

    def plot_2d_contour(self, lambda_vals, n_vals, Z):
        """
        Vẽ contour plot 2D của sai số
        """
        plt.figure(figsize=(12, 8))

        # Lọc bỏ các giá trị NaN
        Z_clean = np.where(np.isnan(Z), 0, Z)

        # Vẽ contour
        contour = plt.contourf(lambda_vals, n_vals, Z_clean, 20, cmap='viridis')
        plt.colorbar(contour, label='Sai số trung bình (MAE)')

        # Thêm contour lines
        CS = plt.contour(lambda_vals, n_vals, Z_clean, colors='black', linewidths=0.5, alpha=0.7)
        plt.clabel(CS, inline=True, fontsize=8, fmt='%.3f')

        plt.xlabel('λ (Tham số Poisson)', fontsize=12)
        plt.ylabel('n (Số phép thử)', fontsize=12)
        plt.title('Bản đồ sai số: Nhị thức → Poisson', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xscale('linear')
        plt.yscale('log')  # Dùng thang log cho n để dễ nhìn

        # Đánh dấu vùng sai số thấp
        plt.fill_between(lambda_vals, 20, 500, alpha=0.1, color='green', label='Vùng sai số thấp (n ≥ 20)')
        plt.legend()

        plt.savefig('2d_error_contour.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def run_comprehensive_simulation(self):
        print("=" * 100)
        print("MÔ PHỎNG TOÀN DIỆN: PHÂN PHỐI NHỊ THỨC → POISSON".center(100))
        print("=" * 100)
        print(f"Tham số chính: λ = {self.lambda_param}")
        print(f"Các giá trị n: {self.n_values}")
        print("=" * 100)

        # 1. Tính toán xác suất
        print("\n1. Đang tính toán xác suất...")
        results = self.calculate_probabilities()

        # 2. Vẽ so sánh phân phối
        print("2. Vẽ đồ thị so sánh phân phối...")
        self.plot_comparison(results)

        # 3. Phân tích tốc độ hội tụ
        print("3. Phân tích tốc độ hội tụ...")
        mae_values = self.plot_convergence(results)

        # 4. Tạo bảng tổng hợp
        print("4. Tạo bảng tổng hợp kết quả...")
        data = []
        for n in self.n_values:
            binomial_key = f'Binomial_n={n}'
            binomial_probs = results[binomial_key]
            poisson_probs = results['Poisson']

            mae = np.mean(np.abs(binomial_probs - poisson_probs))
            mse = np.mean((binomial_probs - poisson_probs)**2)
            max_error = np.max(np.abs(binomial_probs - poisson_probs))

            data.append({
                'n': n,
                'p = λ/n': f'{self.lambda_param/n:.6f}',
                'MAE': f'{mae:.6f}',
                'MSE': f'{mse:.8f}',
                'Sai số lớn nhất': f'{max_error:.6f}',
                'np = λ': f'{self.lambda_param:.3f}'
            })

        df = pd.DataFrame(data)
        print("\n" + "═" * 90)
        print(f"{'BẢNG TỔNG HỢP KẾT QUẢ (λ = ' + str(self.lambda_param) + ')':^90}")
        print("═" * 90)
        print(df.to_string(index=False))
        print("═" * 90)

        # 5. Vẽ so sánh nhiều lambda (chỉ 1, 3, 5, 7)
        print("\n5. Vẽ so sánh với các giá trị λ = [1, 3, 5, 7]...")
        self.plot_multiple_lambda_comparison(lambda_list=[1, 3, 5, 7], n_values_plot=[5, 10, 20, 50])

        # 6. Vẽ bề mặt sai số 3D
        print("6. Vẽ bề mặt sai số 3D...")
        self.plot_3d_error_surface()

        # Phân tích xu hướng
        print("\n" + "─" * 80)
        print("PHÂN TÍCH XU HƯỚNG HỘI TỤ")
        print("─" * 80)

        if len(mae_values) >= 2:
            improvement = mae_values[0] / mae_values[-1]
            print(f"• Khi n tăng từ {self.n_values[0]} đến {self.n_values[-1]}:")
            print(f"  - Sai số giảm từ {mae_values[0]:.6f} xuống {mae_values[-1]:.6f}")
            print(f"  - Tỉ lệ cải thiện: {improvement:.2f} lần")
            print(f"  - Sai số giảm {((mae_values[0] - mae_values[-1])/mae_values[0]*100):.1f}%")

        print("\n" + "=" * 100)
        print("KẾT THÚC MÔ PHỎNG".center(100))
        print("=" * 100)

# Thực thi mô phỏng
if __name__ == "__main__":
    # Khởi tạo và chạy mô phỏng
    simulator = PoissonBinomialConvergence(lambda_param=3, n_values=[5, 10, 20, 50, 100, 200])
    simulator.run_comprehensive_simulation()

    # In thông tin thêm
    print("\nCác file đã tạo:")
    print("1. poisson_binomial_comparison_enhanced.png - So sánh phân phối")
    print("2. convergence_rate_enhanced.png - Tốc độ hội tụ")
    print("3. multiple_lambda_comparison.png - So sánh 4 lambda (1,3,5,7)")
    print("4. 3d_error_surface_corrected.png - Bề mặt sai số 3D (đã sửa)")
    print("5. 2d_error_contour.png - Bản đồ sai số 2D")
    print("6. multi_lambda_summary.csv - Bảng tổng hợp nhiều lambda")
