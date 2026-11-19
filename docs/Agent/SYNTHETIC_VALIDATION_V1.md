# DAX合成扩充验证设计 v1

状态：**待审阅，不冻结测试ID或阈值**。评估配置绑定[空间定义](SYNTHETIC_REACHABILITY_PROTOCOL.md)、[约束表](SYNTHETIC_CONSTRAINTS_V1.md)、[覆盖统计](SYNTHETIC_COVERAGE_V1.md)。

## 数据分区

训练候选全集包含用户已指定的39段遥操train及每一个通过准入的合成clip，不设来源配比、不复制以提高来源权重。两段既有遥操val继续作开发集，但录制家族曾用于mount/pose相关工作，不能作为唯一泛化集。合成val候选不得与train共享源episode、seed family、关键姿态列表或轨迹变换族。

最终盲测需在阈值/生成策略冻结后构建。优先采用从未读payload的遥操test录制家族和独立固定seed的空间/姿态/路径query bank；按来源录制家族整体切分。新测试一旦用于诊断或调参，降级为development并新建test版本。manifest先锁定metadata与hash，盲测payload由指定评估阶段读取；这次协议准备阶段不读取未使用test payload。

## 固定评估轴

1. **模型/几何**：资产hash、固定基座/固定腰姿态、关节顺序、FK对齐、碰撞shape与pair exclusion版本。
2. **数据可行性**：全帧及自适应路径采样的位置限位、时间戳、双臂完整姿态、碰撞/净空、路径中断和动力学饱和。
3. **Frozen policy tracking**：相同Actor、相同环境/噪声/复位，单列每片段奖励、左右腕/头/胸位置RMSE及p95/p99、姿态角误差、关节误差、饱和、接触与failure。
4. **泛化**：分别报空间区域、腕姿态、相对双臂pose、构型cluster、路径family、动态状态留出和录制来源家族，汇报最差区而非仅总体均值。
5. **训练贡献**：最终固定训练步数/模型/超参数/采样器版本，报告多seed；检查点选择只使用dev，test只运行一次或按预注册规则。

Root/fixed-base退化的failure指标单独标记。直接PD通过只说明参考在该名义物理设置下可跟踪；不能替代Actor训练结果。几何路径无碰撞不等于特定Actor rollout无碰撞。仿真不代表真机能力。

## 对照组

| 组别 | 数据 | 用途 |
| --- | --- | --- |
| B0：当前混合基线 | 固定下腰v3：39 teleop train + 900 synthetic，当前超参数 | 实验前baseline；最终与midpoint均保留并明确选择规则 |
| Q：数量对照 | 从旧合格轨迹增加相近的trajectory-seconds；维持任务family组成尽可能不变 | 估计重复/同区域扩量收益 |
| C：覆盖组 | 总秒数与Q匹配，集中新增冻结coverage bank稀疏区及新的方向/双臂/动态family | 估计同等量下的覆盖收益 |
| OOD：锁定集 | B0/Q/C皆不可访问的workspace/query/source families | 评估外推；独立报告不用于选模 |

用户已确定“所有通过数据全部用上”，因此Q/C均是完整合格全集，表中对照通过生成协议控制候选内容和总量，不对训练输入按teleop/synthetic设比例。若无法精确匹配轨迹秒数，报告真实数据总秒、clip长度分布、重置采样频率与有效sim transitions，不宣称严格等量。

## 预算与统计

当前项目已有4090 8GPU-hours数据点，但训练job已 completed_budget，不能把旧checkpoint或旧GPU计数算成本轮新实验。下一次训练预算、seed数、每seed硬件GPU时、是否恢复checkpoint，待用户批准后在实验清单中冻结。推荐最低三个独立种子用于观察训练方差；如果算力不足，结果必须标为单seed探索，不做稳健收益声称。

主指标建议为按val clip等权、左右腕等权的位置RMSE；须预先选长/短轨迹权重，另报告p95/p99与最差空间cell。收益门槛和安全/物理精度阈值目前**未定**，由用户结合任务精度裁定；不得事后挑最有利指标。Normalized return、failure、姿态误差和约束违规均为次要独立轴。

## 检查点与停止

- 首先运行固定query bank的几何/规划器对照，不立即训练：测解决率、unknown率、碰撞/动力学拒绝率和每新覆盖单元耗时。
- 若生成通过率或连续路径有效性低，先分析失败分类，不通过降低约束门槛凑数据。
- 进入学习对照前先做冻结Actor on-policy评价，确认新增库基本可直接物理跟踪；通过此门槛不代表训练收益成立。
- 训练比较预算/超参数/验证种子绑定，保存首始、中间和最终全状态checkpoint；按dev规则选权重并只在冻结test报告一次。
- 若C组仅训练集return改善，而OOD覆盖、腕误差或约束违规不改善，则不宣称覆盖扩充有效。

此设计仍需冻结空间包围盒、coverage分辨率、几何净空/连续碰撞方法、腕姿态误差门槛、训练seed/预算、统计阈值及盲测family ID，然后才开始生成器实现。
