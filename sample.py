import numpy as np
import matplotlib.pyplot as plt

# パラメータの生成（ミュー、アルファ、ベータがベクトル形式で512個ずつ）
num_samples = 1000
num_parameter_sets = 512    

# パラメータセット生成
mu_values = np.random.uniform(-1, 1, num_parameter_sets)
alpha_values = np.random.uniform(1, 3, num_parameter_sets)
beta_values = np.random.uniform(0.5, 2, num_parameter_sets)

# パラメータセットごとにサンプリング
z_values = np.random.normal(size=(num_parameter_sets, num_samples))
samples = mu_values[:, np.newaxis] + alpha_values[:, np.newaxis] * np.random.standard_normal(size=(num_parameter_sets, num_samples)) * beta_values[:, np.newaxis]

# サンプルの可視化
plt.figure(figsize=(10, 6))
plt.hist(samples.T, bins=30, density=True, alpha=0.5)

plt.title('Generalized Normal Distribution Sampling with Multiple Parameter Sets (Vectorized)')
plt.xlabel('Value')
plt.ylabel('Probability Density')
# plt.legend()
plt.savefig('generalized_normal_sampling_vectorized.png')