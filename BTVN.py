import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
import pandas as pd
from math import factorial
import seaborn as sns

# Thiết lập style cho đồ thị
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PoissonBinomialConvergence:
    def __init__(self, lambda_param=3, n_values=None):
        """
        Khởi tạo tham số mô phỏng

        Parameters:
        lambda_param: tham số lambda của phân phối Poisson
        n_values: danh sách các giá trị n để mô phỏng
        """
        self.lambda_param = lambda_param

        if n_values is None:
            self.n_values = [5, 10, 20, 50, 100, 200]
        else:
            self.n_values = n_values

        self.k_values = np.arange(0, 15)  # Xét từ 0 đến 14 sự kiện

    def calculate_probabilities(self):
        """
        Tính xác suất cho cả phân phối nhị thức và Poisson
        """
        results = {}

        # Tính phân phối Poisson (chính xác)
        poisson_probs = poisson.pmf(self.k_values, self.lambda_param)
        results['Poisson'] = poisson_probs

        # Tính phân phối nhị thức với các n khác nhau
        for n in self.n_values:
            p = self.lambda_param / n
            binomial_probs = binom.pmf(self.k_values, n, p)
            results[f'Binomial_n={n}'] = binomial_probs

        return results

    def plot_comparison(self, results):
        """
        Vẽ đồ thị so sánh các phân phối
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        poisson_probs = results['Poisson']

        for idx, n in enumerate(self.n_values):
            if idx >= len(axes):
                break

            binomial_key = f'Binomial_n={n}'
            binomial_probs = results[binomial_key]

            # Vẽ biểu đồ cột
            width = 0.35
            x = np.arange(len(self.k_values))

            axes[idx].bar(x - width/2, binomial_probs, width,
                         label=f'Binomial (n={n})', alpha=0.8)
            axes[idx].bar(x + width/2, poisson_probs, width,
                         label=f'Poisson (λ={self.lambda_param})', alpha=0.8)

            axes[idx].set_xlabel('Số sự kiện (k)')
            axes[idx].set_ylabel('Xác suất P(X=k)')
            axes[idx].set_title(f'So sánh với n = {n}')
            axes[idx].legend()
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(self.k_values)

            # Tính và hiển thị sai số
            mae = np.mean(np.abs(binomial_probs - poisson_probs))
            axes[idx].text(0.05, 0.95, f'MAE: {mae:.4f}',
                          transform=axes[idx].transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('poisson_binomial_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_convergence(self, results):
        """
        Vẽ đồ thị thể hiện sự hội tụ
        """
        plt.figure(figsize=(12, 6))

        poisson_probs = results['Poisson']

        # Tính sai số tuyệt đối trung bình (MAE) cho từng n
        mae_values = []

        for n in self.n_values:
            binomial_key = f'Binomial_n={n}'
            binomial_probs = results[binomial_key]
            mae = np.mean(np.abs(binomial_probs - poisson_probs))
            mae_values.append(mae)

        # Vẽ đồ thị sai số theo n
        plt.subplot(1, 2, 1)
        plt.plot(self.n_values, mae_values, 'o-', linewidth=2, markersize=8)
        plt.xlabel('n (số phép thử)')
        plt.ylabel('Sai số tuyệt đối trung bình (MAE)')
        plt.title('Sự hội tụ của sai số')
        plt.grid(True, alpha=0.3)

        # Vẽ đồ thị log-log để thấy tốc độ hội tụ
        plt.subplot(1, 2, 2)
        plt.loglog(self.n_values, mae_values, 's-', linewidth=2, markersize=8)
        plt.xlabel('n (log scale)')
        plt.ylabel('MAE (log scale)')
        plt.title('Tốc độ hội tụ (log-log scale)')
        plt.grid(True, alpha=0.3)

        # Thêm đường hồi quy
        if len(self.n_values) > 1:
            coeffs = np.polyfit(np.log(self.n_values), np.log(mae_values), 1)
            slope = coeffs[0]
            plt.text(0.05, 0.95, f'Độ dốc: {slope:.3f}',
                    transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('convergence_rate.png', dpi=300, bbox_inches='tight')
        plt.show()

        return mae_values

    def create_summary_table(self, results):
        """
        Tạo bảng tổng hợp kết quả
        """
        data = []

        for n in self.n_values:
            binomial_key = f'Binomial_n={n}'
            binomial_probs = results[binomial_key]
            poisson_probs = results['Poisson']

            # Tính các độ đo sai số
            mae = np.mean(np.abs(binomial_probs - poisson_probs))
            mse = np.mean((binomial_probs - poisson_probs)**2)
            max_error = np.max(np.abs(binomial_probs - poisson_probs))

            data.append({
                'n': n,
                'p': self.lambda_param / n,
                'MAE': f'{mae:.6f}',
                'MSE': f'{mse:.6f}',
                'Max Error': f'{max_error:.6f}',
                'np': f'{self.lambda_param:.3f}'
            })

        df = pd.DataFrame(data)
        print("\n" + "="*80)
        print("BẢNG TỔNG HỢP KẾT QUẢ MÔ PHỎNG")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)

        return df

    def theoretical_derivation_plot(self):
        """
        Minh họa quá trình chứng minh toán học
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        n_range = np.logspace(1, 3, 100)  # từ 10 đến 1000

        # Plot 1: p = λ/n
        ax1 = axes[0, 0]
        p_values = self.lambda_param / n_range
        ax1.plot(n_range, p_values, linewidth=2)
        ax1.set_xscale('log')
        ax1.set_xlabel('n')
        ax1.set_ylabel('p = λ/n')
        ax1.set_title('Xác suất p tiến về 0 khi n → ∞')
        ax1.grid(True, alpha=0.3)

        # Plot 2: np = λ (constant)
        ax2 = axes[0, 1]
        np_values = n_range * (self.lambda_param / n_range)
        ax2.plot(n_range, np_values, linewidth=2)
        ax2.set_xscale('log')
        ax2.set_xlabel('n')
        ax2.set_ylabel('n × p')
        ax2.set_title('np = λ (giữ không đổi)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=self.lambda_param, color='r', linestyle='--', alpha=0.5)

        # Plot 3: (1 - λ/n)^n → e^{-λ}
        ax3 = axes[1, 0]
        limit_values = (1 - self.lambda_param / n_range) ** n_range
        exact_value = np.exp(-self.lambda_param)
        ax3.plot(n_range, limit_values, label='(1 - λ/n)^n', linewidth=2)
        ax3.axhline(y=exact_value, color='r', linestyle='--',
                   label=f'e^{-self.lambda_param} ≈ {exact_value:.3f}', alpha=0.7)
        ax3.set_xscale('log')
        ax3.set_xlabel('n')
        ax3.set_ylabel('Giá trị')
        ax3.set_title('Giới hạn quan trọng: (1 - λ/n)^n → e^{-λ}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: n!/((n-k)! n^k) → 1
        ax4 = axes[1, 1]
        k = 3  # chọn một giá trị k cụ thể
        ratio_values = []
        for n in n_range:
            n_int = int(n)
            if n_int > k:
                # Tính gần đúng để tránh overflow
                ratio = 1.0
                for i in range(k):
                    ratio *= (n_int - i) / n_int
                ratio_values.append(ratio)
            else:
                ratio_values.append(np.nan)

        ax4.plot(n_range[:len(ratio_values)], ratio_values, linewidth=2)
        ax4.set_xscale('log')
        ax4.set_xlabel('n')
        ax4.set_ylabel('n!/((n-k)! n^k)')
        ax4.set_title(f'Giới hạn: n!/((n-k)! n^k) → 1 (với k={k})')
        ax4.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('theoretical_derivation.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_simulation(self):
        """
        Chạy toàn bộ quá trình mô phỏng
        """
        print("="*80)
        print("MÔ PHỎNG SỰ HỘI TỤ CỦA PHÂN PHỐI NHỊ THỨC VỀ PHÂN PHỐI POISSON")
        print("="*80)
        print(f"Tham số: λ = {self.lambda_param}")
        print(f"Giá trị n: {self.n_values}")
        print("="*80)

        # Tính toán xác suất
        results = self.calculate_probabilities()

        # Minh họa chứng minh toán học
        print("\n1. Minh họa quá trình chứng minh toán học...")
        self.theoretical_derivation_plot()

        # Vẽ đồ thị so sánh
        print("\n2. Vẽ đồ thị so sánh phân phối...")
        self.plot_comparison(results)

        # Vẽ đồ thị hội tụ
        print("\n3. Phân tích tốc độ hội tụ...")
        mae_values = self.plot_convergence(results)

        # Tạo bảng tổng hợp
        print("\n4. Tạo bảng tổng hợp kết quả...")
        df = self.create_summary_table(results)

        # Phân tích bổ sung
        print("\n5. Phân tích bổ sung:")
        print(f"   - Khi n tăng từ {self.n_values[0]} đến {self.n_values[-1]}:")
        print(f"   - Sai số giảm từ {mae_values[0]:.6f} xuống {mae_values[-1]:.6f}")
        print(f"   - Tỉ lệ giảm: {mae_values[0]/mae_values[-1]:.2f} lần")

        return results, df

# Thực thi mô phỏng
if __name__ == "__main__":
    # Khởi tạo và chạy mô phỏng
    simulator = PoissonBinomialConvergence(lambda_param=3, n_values=[5, 10, 20, 50, 100, 200])
    results, summary_df = simulator.run_simulation()

    # Lưu kết quả ra file
    summary_df.to_csv('simulation_results.csv', index=False)
    print("\nKết quả đã được lưu vào 'simulation_results.csv'")

    # In thông báo kết thúc
    print("\n" + "="*80)
    print("KẾT THÚC MÔ PHỎNG")
    print("="*80)
