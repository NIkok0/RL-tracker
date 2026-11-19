# DAX SuperMimic 架构

更新：2026-11-06；按现存源码与入口整理，未重新验证服务或硬件运行状态。

## 活动算法与机器人适配

唯一活动基线为非官方SuperMimic `a1508d42`。`dax_supermimic/train.py`、`networks.py`、`buffer.py`、`mdp.py`、`evaluate.py`保留冻结算法核心；`robot.py`、`kinematics.py`、`env.py`负责固定基座18关节、机器人资源和仿真适配。训练链为真实仿真采集、WM更新、冻结WM上的Actor想象优化及critic/GAE，用户批准的差异与校验边界见[基线契约](SUPERMIMIC_BASELINE.md)。

Actor接收本体观测历史与未来完整运动参考，输出关节位置目标的归一化动作。共同动作投影与关节尺度见[动作接口](DAX_ACTION_INTERFACE.md)。旧加速度积分、task48和servo v2不属于当前算法。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `data_pipeline` | 参考生成、处理、约束验证、物理回放证据与训练发布 |
| `robot_sim` | 共享碰撞、肘分支等几何检查 |
| `common` | 资源定位、JSON与哈希证据I/O |
| `backend` / `web` | 工作台HTTP、任务管理、资料库、编辑及展示 |
| `runtime.session` / `policy_worker` | 独立进程的Actor仿真会话 |
| `runtime.trajectory` | 录制18关节参考、状态观测历史及离线Actor提案 |

第一轮模块化已迁移的范围与未迁移部分见[模块边界](MODULARIZATION.md)。训练、评估核心未被包装为空目录。

## 推理与交互边界

工作台由`backend.http`启动，代码入口仅8770，仿真按需加载。端口约定不代表服务此刻在线，见[工作台契约](UNIFIED_WORKBENCH.md)。

录制轨迹推理独立于HTTP及仿真服务。参考提供未来帧，状态提供观测历史，冻结Actor输出位置目标；本模块不负责硬件发送。reference-proxy只检查计算，recorded也只是固定记录回放；闭环精度必须有独立物理状态反馈证据。详见[轨迹推理](TRAJECTORY_INFERENCE.md)。

旧O6/BC/加速度WBC实现已移除，历史范围见[停用路线](LEGACY_ROUTES.md)。
