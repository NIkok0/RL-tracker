# DAX SuperMimic 数据与参考契约

2026-11-16当前候选入口已切到[52.205117小时symmetric-v2清单](../../data/dax_physical_v3_symmetric_full_20261116/manifest.json)，8,944条、10个根来源；formal training_admitted=false。当前入口、证据保存范围及冻结评估见[统一报告](../../verification/dax_v3_baseline_alignment_20261116/report.md)。后续段落为11月13日及更早发布事实，不再代表默认候选指针。

更新：2026-11-13。全部旧训练数据保留；新旧合格空间增广已有[51.2719小时独立合版](../../data/dax_physical_v3_merged_spatial_50h_20261112/manifest.json)，8,886条、每条运动20.00–29.96秒，逐轨迹模型内验收及全量保存状态独立复核通过。全部为train来源，仍缺独立lineage评估及训练入口准入，不自动替换latest_training或打开training_ready；原117.78小时清洗库其余部分仍未整体通过v3几何/物理准入。见[最终报告](../../verification/dax_physical_v3_50h_final_20261113/report.md)。

## 参考与反馈

`requested_q`是请求关节位置，`state_q`是录制反馈。当前参考CSV由请求生成，reset、future goal和reward使用同一参考；反馈、源state/action和源时间/行映射在哈希绑定旁车中独立保留。请求不等于实际应用命令或已到达状态，参考FK不是实测动力学转移。

保留的原始来源参考为[2026-11-02发布](../../data/dax_supermimic_v1_20261102/manifest.json)：train39/val2片段、50Hz CSV，前三列根位置、随后xyzw根四元数、18个具名关节。单位与顺序见[接口](PROTOCOLS.md)。245行训练请求越位置限位原样报告；原加载器浮点floor末帧边界与CSV完整帧数分开，见[基线说明](SUPERMIMIC_BASELINE.md)。

## 后续独立发布

各运行显式绑定其数据与安装几何，不将早期基线包称为唯一数据来源。增强、合成与物理记录发布见[数据流水线](DATA_GENERATION_PIPELINE.md)，组装接口见[模块化说明](MODULARIZATION.md)。参考几何、direct-PD实际状态和Actor策略评价是三个不同证据层。

2026-11-05合并前方候选训练的开发验证与一次指定实验授权，不改变新增前方数据`training_admitted=false`的物理准入状态，见[当前缺项](../../.context/BLOCKERS.md)及[训练评价](../../verification/dax_combined_front_training_20261105/evaluation_summary.md)。

独立合成v2 test已由冻结候选一次评分，不能再用于调参或选模；其来源隔离、val选择与test消费情况见[评估集报告](../../verification/dax_evalsets_20261105/report.md)。原teleop锁定test的读取范围按各历史任务单独记录，不用本次文档审计改变数据身份。

## 数据用于离线推理

单条CSV可直接送入[轨迹推理模块](TRAJECTORY_INFERENCE.md)，但仅有参考时须显式启用代理观测。recorded模式要求另有对齐q、qdot和上次实际应用动作；历史teleop旁车不自动满足此契约。

旧BC重定时、教师监督、混合配比和servo v2导出命令已退出当前路线；旧原始数据未在本次文档清理中删除。

## 新增独立增广发布

[physical-v3首批空间增广](../../data/dax_teleop_spatial_v3_batch1_20261111/manifest.json)包含14条模型内合格轨迹；10条来自train遥操父窗口、4条来自本批已验收父轨迹，递归保留root recording lineage及train划分。它们为增广参考q和direct-PD仿真状态q，不能称为真实执行数据。该发布未并入训练入口，旧清洗集整体准入状态不变；详细证据见[报告](../../output/dax_teleop_spatial_v3_batch1_20261111/report.md)。

[连续长轨迹v3增广发布](../../data/dax_teleop_spatial_v3_long_20261111/manifest.json)新增73条模型内合格轨迹，单条运动33.9–59.92秒，净运动1.02459小时（首尾hold另计）；来自8个父clip/14个合格来源窗口/2个train recording。数据仅作独立发布，训练入口未更改；所有后代须保留root recording划分，不能按clip随机拆分。

## 旧117.78小时自动并入（2026-11-13）

用户要求已落实到[扩展合版](../../data/dax_physical_v3_mixed_legacy_cleaned_plus_spatial_20261113_v2/manifest.json)：28,660条、173.9891小时完整参考。19,774条旧参考仅位置/采样速度通过，v3几何/direct-PD/实际tracking仍pending；8,886条既有合格层保持51.2719小时净运动及全量证据。默认训练候选已指向新合版且training_ready=false。相同payload的32组594个额外条目保留，未将加入等同准入或去重。详见[并入报告](../../verification/dax_legacy_inclusive_merge_20261113/report.md)。
