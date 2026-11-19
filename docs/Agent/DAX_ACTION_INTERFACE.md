# DAX physical-v3 动作契约

活动配置为 [dax_supermimic_robot_v3.json](../../config/dax_supermimic_robot_v3.json)。机器人 profile 为 `dax_physical_joint_range_action_v3`，内部动作接口版本名为 `dax_joint_range_action_v2`；后者是动作 schema 标识，不是已删除的 v2 机器人配置。完整契约由接口标识与配置文件 SHA256 共同确定。

位置范围、具名顺序与逐关节 motor 参数来自活动机器人 profile。v3 MJCF/USD 静态一致性通过，物理与硬件准入仍须按具体模型完成。

```text
alpha_old[j] = 0.25 * effort[j] / kp[j]
radius[j] = max(abs(q_lo[j]-default[j]), abs(q_hi[j]-default[j]))
alpha[j] = max(alpha_old[j], radius[j]/10)
noise_ratio[j] = alpha_old[j]/alpha[j]
a_applied = clamp(a_raw, -20, 20)
q_target = default_q + joint_offset + alpha * a_applied
```

合法位置范围对应动作绝对值不超过10。公共 `project_action` 用于collect、想象、evaluate与TrackingSim，Replay和动作历史保存applied action。探索噪声乘noise_ratio保持未裁剪区的物理位置扰动幅度；Actor末层无tanh。±20是动作数值范围，不能代替物理位置检查。

区分训练来源绑定与运行时动作语义：训练 checkpoint 的 `action_contract` 仍记录完整 profile hash，训练续跑和历史评估按该 hash 精确校验。新 checkpoint 另写 `runtime_action`，包含动作 schema、具名关节顺序、clip、训练时 `default_joint_position_rad` / `scale_rad` 与来源 contract。推理使用 checkpoint 自带变换，因此替换机器人几何或执行 profile 时，不必改写 checkpoint 的 profile hash；运行时仍按当前机器人 profile 的位置/速度界检查目标。

没有 `runtime_action` 的旧 checkpoint 继续采用完整 profile hash 严格匹配，不能靠删除校验或改 metadata 实现跨 profile 加载。新 schema 只解除部署推理对 profile hash 的相等要求，不代表模型已适配新机器人的几何、动力学或精度。导出物仍绑定 checkpoint 文件 SHA256；替换 checkpoint 后必须重新导出 ONNX/TensorRT 并更新对应文件，TensorRT engine 还需在目标硬件重建。

新模型须从空normalizer、optimizer和Replay训练。旧数据须按目标 profile 的位置、逐关节速度与几何/物理规则重新准入。

执行保护统一在 [dax_full18_execution.json](../../config/dax_full18_execution.json)，与策略动作解码分开管理。执行速度上限逐关节取全局保护、URDF和motor上限的最小值。训练motor上限不是实测速度或硬件许可。

实现证据见预期资产路径`output/dax_physical_v3_geometry_20261113_symmetric_v2/asset.json`、[双臂惯量审计](../../verification/dax_left_arm_inertia_audit_20261113/report.md)和[速度契约](../../verification/dax_joint_velocity_contract_20261111/report.md)。2026-11-16资产审计确认asset.json存在，其SHA256 `69984625161f386aec5e4724960e0c504b50a274d75fb90945b2de05b5d7012f` 与current_baseline绑定值一致；该检查不替代依赖完整性或运行时几何验证。
