# 在 PCVAE 中使用 QBM-VAE 玻尔兹曼先验的实现说明

## 理论替换要点（PCVAE → QBM-VAE）

PCVAE 原始实现使用高斯先验，编码器输出均值与方差，通过重参数化采样连续隐变量，并在训练时加入标准正态 KL 散度正则项。对应逻辑集中在 `PILP.encode` 和训练步骤中的 KL 项计算。PCVAE 当前的训练损失中仍然直接使用标准高斯 KL（`-0.5 * sum(1 + logvar - mu^2 - exp(logvar))`）。

QBM-VAE（QVAE）将潜变量的先验替换为玻尔兹曼机（RBM/QBM）能量模型。其关键差异为：

1. **潜变量分布**：将编码器输出解释为二值潜变量的 logits，并通过重叠分布（MixtureGeneric）进行可微采样（DVAE++ 思路）；
2. **先验能量**：使用 RBM/QBM 的线性偏置与二次耦合定义能量函数作为先验；
3. **KL 计算**：采用“交叉熵 - 熵”的方式估计 KL（其中交叉熵包含对 QBM 负相的估计样本），以替代原有高斯 KL。

以上思想在 QVAE 的实现中已被编码：QVAE 使用 `RestrictedBoltzmannMachine` 作为先验，并通过 `MixtureGeneric` 与重叠分布实现可微采样和 KL 估计。

## 具体代码改造位置

### 1) 编码器输出与后验采样

将 PCVAE 的 `encode` 输出从 `(mu, logvar)` 改为 `q_logits`，并引入 `MixtureGeneric` 作为可微采样分布。采样结果 `zeta` 用于解码器输入。

* `PILP.encode` 改为输出 `q_logits`。
* `PILP.posterior` 使用 `MixtureGeneric` 产生 `zeta`。

### 2) QBM/RBM 先验与 KL 估计

在 `PILP` 中新增 `RestrictedBoltzmannMachine` 作为先验模型，新增 `kl_divergence` 以实现 “交叉熵 - 熵” 的 KL 估计，并用 QVAE 示例一致的模拟退火采样器（`SimulatedAnnealingOptimizer`）进行负相估计。

### 3) 训练与预测逻辑

训练步骤中使用 `kl_divergence` 返回的 KL 替代高斯 KL；预测逻辑中也改为从 QBM 后验采样。

## 代码实现（已完成）

### 模型层（`PCVAE/src/model.py`）

关键修改：

* 新增 QBM 相关依赖：`RestrictedBoltzmannMachine` 与 `MixtureGeneric`。
* 使用 `q_logits` 代替 `(mu, logvar)`；新增 `posterior`、`kl_divergence` 以及基于模拟退火的负相采样 `_bm_negative_sample`。
* `forward` 输出包含 `kl`，供训练使用。

### 训练与预测（`PCVAE/src/steps.py`）

关键修改：

* 使用 `kl_loss` 取代原高斯 KL。
* 预测时从后验分布 `posterior` 采样 `zeta`，替代 Gaussian 采样。

## 使用方式建议

### 实例化

```python
model = PILP(
    ft_dim=feature_dim,
    latent_size=64,          # 总 latent 维度
    qbm_visible=32,          # QBM 可见层维度（默认 latent_size//2）
    dist_beta=10.0,          # MixtureGeneric 的平滑参数
    qbm_sampler_alpha=0.95,  # 模拟退火采样器参数
)
```

### 关键注意事项

* `latent_size` 必须等于 `qbm_visible + qbm_hidden`（其中 `qbm_hidden = latent_size - qbm_visible`）。
* 使用 QBM 先验会引入二值潜变量与能量模型训练，建议调小 KL 权重或采用 KL annealing。
* 负相采样默认使用模拟退火采样器，可替换为更高质量的采样器（如量子采样器或自定义优化器）。
