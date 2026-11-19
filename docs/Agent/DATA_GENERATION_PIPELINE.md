# physical-v3 数据生成状态

2026-11-13扩量完成：旧/新合格train父来源混合，新批次8,699条/50.2055小时净运动；与旧187条合并为8,886条/51.2719小时，单条运动20.00–29.96秒，hold另计。[最终合版](../../data/dax_physical_v3_merged_spatial_50h_20261112/manifest.json)和[全量独立审计/覆盖报告](../../verification/dax_physical_v3_50h_final_20261113/report.md)已交付。冻结24个排除父hold的源窗/15,600-query bank，处理10,368候选达到停止条件；try4因计时疑问保留、不计入。训练入口和硬件保护未修改，training_ready仍false。命令与修订证据见[扩量记录](../../.context/HISTORY/2026-11-12-physical-v3-50h-augmentation.md)。

2026-11-11：新目标为原始DAX Dual遥操轨迹的扩分布，采用完整physical-v3位置范围和逐关节速度约束。旧生成器固定合成源，尚不能满足本轮要求；以下保留的是现存框架实现说明，其旧批次配方和默认值不作为新v3任务启动授权。具体缺口见[准备度报告](../../verification/dax_teleop_data_generation_readiness_20261111/report.md)。

## 父轨迹来源

用户已确认各种来源都可作父轨迹。当前父池见[data parent pool v12](../../data/dax_physical_v3_parent_pool_v12_20261112/manifest.json)。父clip必须先取得当前physical-v3逐轨迹位置/速度、几何、direct-PD实际状态和跟踪通过receipt。2026-11-12重验18个非遥操候选，13个通过（synthetic 1、augmentation 6、front 6）；由这些来源再生成187条20–30秒子代，均逐条通过v3准入，净运动1.06635小时。其他同来源legacy clip仍待重验，不因类型相同自动放行。数据发布见data/dax_spatial_v3_nonteleop_20_30s_batch_20261112_try9/manifest.json与报告output/dax_spatial_v3_nonteleop_20_30s_batch_20261112_try5/report.md。

# 数据生成框架与工作台流程

2026-11-04用户要求先搭好四阶段框架，具体生成方式先采用“已有900段＋扰动＋不同速度”。当前实现为任务编排、来源核验、候选配方、执行器接口、约束证据门槛及独立发布导出。随后用户授权2档扰动×2档速度×900，实际增强和名义物理执行器现已接通：3600候选已生成、528通过本次模型筛查独立保存。详见[本轮报告](../../verification/dax_augmentation_3600_20261104/report.md)。数值guard与工程净空尚需区分复核，不据此称真机安全。

## 原工作台入口

8770 → 第四标签“数据生成”。同源`/data-generation`现在包含“生成任务”和“已有900段·数据预览”，后者保留原预览于`/data-generation/library`；没有另开服务。

生成任务页可配置平滑关节扰动幅度、速度倍率、每档变体数和种子。本批采用0.01/0.03rad两档、0.5/1倍两档、每组合1变体；网页已同步显示。它们是生成参数，不是硬件能力限制。默认全部900源片段，不设遥操/合成来源比例，也不复制数据调权重。

“核验配置”读取源manifest及900个CSV、绑定模型文件验证来源，然后估算候选数量/速度换算前时长。“建立生成任务”只保存plan和配方，记录尚未完成的阈值，不启动仿真。早期2700候选是框架示例，保留但未执行；本批新计划3600候选已经实际执行，状态和独立发布在任务页可查看。合法初态仍须由几何执行器检查，计划阶段不宣称合法。

## 四阶段与产物

| 阶段 | 当前真实能力 | 产物/拒绝规则 |
| --- | --- | --- |
| prepare：来源、目标与初态配方 | 已接通来源核验和确定性候选规划 | plan.json、prepare.json；每candidate绑定源clip/hash、seed、扰动幅度、速度；generated_trajectories=0 |
| geometry：轨迹与几何约束 | 实际平滑关节扰动和硬几何执行器已接通 | 几何执行器返回所有候选的accepted/rejected/unknown；必须含合法seed、肘下路径、自碰撞及限位证据 |
| validate：时序与物理 | 实际半速/原速与名义直接PD执行器已接通 | full_quality口径要求全部阈值冻结；本批reference_plus_physics_diagnostics仅将速度/加速度作为参考硬上限，jerk与跟踪精度单列诊断；变速重采样后再检查几何，实际物理state也检查；不把参考当状态 |
| publish：独立发布与可视化 | 已实际独立导出528条本批筛查通过参考 | 50Hz CSV、full20 NPZ、来源清单、逐候选失败记录、证据文件，写job/release；不写原训练发布。页面支持选择原900或新发布，分别回放参考与保存的实际PD状态 |

## 后端接口

实现模块：`dax_supermimic/data_pipeline/pipeline.py`，实例由Workbench持有。任务保存于`output/data_generation_jobs/<uuid>/`，计划不可修改；来源、模型或流程实现改变时拒绝沿旧计划继续，要求创建新计划。服务重启后可读取原状态；阶段失败保留原因，不重写前置成功产物。

GET：`/api/generation/pipeline`、`/api/generation/jobs`、`/api/generation/job?id=...`。POST：`/api/generation/preview`、`/api/generation/create`。沿用同源JSON限制，不接受客户端上传“验收成功”证据或执行任意脚本。

本地执行器接口为`Pipeline.execute(job_id,stage,adapter)`，adapter签名`(plan, previous_artifact, job_directory)->dict`。没有adapter时geometry/validate只能blocked，不能completed。真实批量入口为scripts/run_dax_augmentation.py，按用户授权显式运行；HTTP仍不因建计划自动启动生成。

每个阶段候选ID必须与上游集合完全相同，不能静默丢掉失败候选或重复计算。同一候选来源链与配置、当前轨迹hash、模型hash、gate名称一致才接受证据receipt；receipt自身也绑定文件。Receipt是受信本地执行器输出的可追溯记录，框架校验不替代执行器算法的几何/物理正确性验收。

## 必经约束

`source_lineage`、`joint_limits`、`elbow_down_seed`、`elbow_down_path`、`self_collision_path`、`timing_limits`、`resampled_geometry`、`actual_state_geometry`、`physics_tracking`。

双肘分支和自碰撞不可由页面关闭。新执行器需实际调用已有约束实现，并输出证据；本批已实际检查全部扰动组与通过候选的记录物理状态。当前碰撞检查针对编译凸几何、原48结构排除及控制步线性路径；未证明未记录物理substep或真机安全。1e-6m为未校准数值guard，不应把零附近结果统称确认碰撞。变速之后，几何与物理阶段不能拿旧几何receipt冒充新CSV证明。未知不计入accepted；无合格候选则不产生发布。

CSV必须finite、25列、固定root convention；full20包含固定下腰0/1＋具名18活动参考，逐帧float32一致及50Hz时间；不写state_q冒充实测。发布重新核验验证产物和所有receipt；新v2发布还逐条复核物理轨迹完整性，并复制CSV/full20/物理轨迹/证据及plan到原子发布目录；不能自动替换训练集。输出当前是框架release schema，新发布三维播放已通过job参数适配，训练导入尚未执行。

## 本次验证

14项流程测试使用临时目录与明确标注的**测试夹具证据**验证状态机、不可跳步、来源漂移、冻结计划、证据错配、全部拒绝不发布、CSV/full20和原子导出，不是实际机器人/物理成功。原工作台6项测试通过。真实HTTP核验900源和创建2700配方任务，generated_trajectories=0，无release；浏览器在原8770标签验证新流程与旧900段预览，未新开页面系统。

见[验证记录](../../verification/dax_generation_framework_20261104/report.md)。后续实现第一种增强方式时，先小批接通geometry和validate，再启用启动/暂停/进度控制；本次不提前提供无后端执行能力的“开始生成”按钮。

## 实际执行追加验收

本批18项流程/增强测试、原6项工作台测试通过，3600参考/旁车/四组合/去重与528证据文件全量核查完成。名义物理共1008候选，528通过记录状态几何检查；全部失败候选留存。脚本入口目前执行全部900，仍不自动加入训练。选择新发布使用同源/api/generation/catalog?job=UUID及frame的job参数，页面通过发布下拉框切换。


## 物理记录完整性与未知分类（2026-11-04修复）

新任务的直接PD记录包含初态和每个控制步的18维`q`、每步左右腕`wrist_error_m`及50Hz `time_s`。预期状态帧数严格复用MotionLib的`floor(((CSV帧数−1)/50)/0.02)+1`，误差帧数和控制步数均应少1。validate及publish逐条检查文件存在、路径在任务内、形状、有限性、完整时间轴、步数和SHA256；`actual_state_geometry`和`physics_tracking`两份receipt同时绑定该物理记录的hash与预期/实际帧数，并继续绑定候选、参考、来源、模型和plan。缺失、截短、改动或receipt错配均禁止发布。

发布schema升级为`dax_generated_release_v2`，每条物理记录复制进release/physical_traces并检查复制hash，manifest采用发布目录内相对路径。工作台读取v2时校验发布内记录；历史v1仍读取原job内记录。旧计划和旧发布原样保留，不补写时间戳、receipt或重分类；流程hash变化后需建立新任务才能执行新规则。

新几何报告的未通过结果包含`status`与`failure_category`：

| status | failure_category | 含义 |
| --- | --- | --- |
| rejected | confirmed_violation | 当前数值模型的明确违规，例如距离小于−1e-6m、明确肘上或关节越限；不代表真机实测 |
| unknown | numerical_uncertainty | 距离在[−1e-6,+1e-6]m、肘下分界附近或退化构型，不能断言确认碰撞/肘上 |
| unknown | inspection_incomplete | 路径节点预算或细分深度耗尽，或距离计算非有限，尚未完成判定 |

参考与记录物理状态路径均使用上述分类。unknown始终不进入发布，原通过阈值不放宽；1e-6m仍只是原数值guard，并未校准为工程净空。批量runner分别汇总rejected、unknown和各failure_category。参考合格候选仍逐条完整物理回放，没有引入抽样。

本次28项生成框架/增强回归和6项工作台回归通过，含真实receipt函数→临时发布→工作台读取发布内物理记录的接口测试；均为本地软件/夹具验收，未新跑GPU物理批次。第3项质量阈值执行与第4项真实驱动力矩饱和诊断仍待处理。见[修复报告](../../verification/dax_generation_integrity_20261104/report.md)。


## 原3600条新版全量复核

2026-11-04用户授权复核后，新job `edab1b956e8843f0afd919efb5aef2ec`沿用原配方重新检查全部3600候选，7200参考/旁车文件与原批次hash逐一一致。参考通过1008条全部重新GPU物理回放，完整记录绑定与发布审计通过；最终528条（4.2752小时）符合当前diagnostics口径，1286条明确违规、1786条未知。通过集合与旧528条逐ID一致，新版纠正未知分类并加强物理证据，没有放宽准入。见[全量复核报告和逐条名单](../../verification/dax_augmentation_recheck_20261104/report.md)。没有自动并入训练。


## 当前新任务默认：离散几何筛查（2026-11-04）

用户选择MoveIt式明确碰撞对＋离散轨迹检查流程后，新任务已切换为`sampled_joint_step_v1`。这是在现有MuJoCo编译凸几何上实现的流程，不是安装或调用MoveIt/FCL。旧递归函数仅留作历史对照，旧job、规则与统计不重写；前文递归证据描述保留其历史范围。

配置位于`geometry_screening`，冻结到plan；流程及几何/增强/底层checker与肘下实现hash同时绑定，变化后需新建计划。

| 字段 | 当前默认 | 含义 |
| --- | --- | --- |
| method | sampled_joint_step_v1 | 全部控制帧＋按关节位移均匀补点 |
| max_joint_step_rad | 0.01 | 每两个检查点任一关节变化不超过该值；工程初始分辨率，非连续碰撞保证 |
| max_samples | 200000 | 单轨迹检查点上限；超出为unknown，不悄悄放粗步长 |
| collision_margin_m | 0 | 额外全局成对净空要求；默认不新增毫米级裕量 |
| pair_margins_m | [] | 可按具名几何体对覆盖净空值；均不得为负，未知/已排除/重复pair拒绝配置 |

距离必须大于`该对margin + 1e-6m数值guard`才通过。深于负guard的距离标记`confirmed_violation / collision_penetration`；未满足净空筛选线但不能称确定穿透的结果标记`clearance_filter / collision_clearance_filter`，status为rejected，意为本数据筛选规则未放行，**不是确认发生碰撞**。肘下边界/退化仍为数值unknown；非有限计算或检查预算不足仍为检查未完成unknown。新统计中的rejected不再全部等于确认违规，必须读取failure_category分项。

这套流程不会围绕近接触递归求分离证明；关节步长加密不等于自适应CCD，可能漏掉检查点间的窄碰撞。1µm仍非已校准工程安全净空。真实驱动力矩饱和、full_quality精度阈值的原缺项未在本次处理。

每个新job导出`collision_pairs.json`，包含946个实际检查pair、各自净空、48条继承模型排除及same-body抑制pair；不自动新增排除项。表hash绑定所有gate凭据，发布复核并复制到release，manifest记录文件/hash。`geometry_pair_diagnostics.json`和`validation_pair_diagnostics.json`按首个失败/最小净空pair聚合数量及少量原始样例，供人工审阅；不把统计直接转换成排除表。

筛查接入三个位置：扰动原路径；变速/末端hold后的完整参考CSV；全部参考合格候选完整物理回放后的状态NPZ。变速参考重新检查后才确定物理回放集合，不沿用旧几何证明；实际物理仍全量回放，未引入抽样。

示例配置为[config/dax_generation_sampled_v1.json](../../config/dax_generation_sampled_v1.json)。本地runner新增`--config`读取JSON，合并到原900×2×2配方；`--job_id`续跑不能同时覆盖配置或来源。以下命令会实际启动批量工作，本轮没有执行：

```bash
bash scripts/dax_supermimic_python.sh scripts/run_dax_augmentation.py --config config/dax_generation_sampled_v1.json --workers 6
```

本轮代码验收见[报告](../../verification/dax_sampled_geometry_20261104/report.md)：40项生成测试、6项工作台测试及小批真实模型几何检查通过。未重跑3600/物理/训练、未重启工作台，因此历史528不能称新离散规则下已复核的最终数量。


## 非整数速度与新增2000候选（2026-11-04）

用户要求增加不同扰动和速度数据后，`resample_candidate`现支持配置允许的0.1至3倍速度。按50Hz输出线性插值、保留源末端，时间向上取整到控制周期，必要时追加静止尾帧以兼容原加载bin。0.5/1倍旧行为回归保持；任意非整数倍的输出弦可能改变源折线拐角，必须对变速后实际参考重新做几何与时序检查，不能声称整数细分或沿用旧路径证明。超出原0.85rad/s、1.5rad/s²上限的候选筛除，不自动裁剪或改变声明速度。

本次job `bc273afaa25a41be8ed9ee6482ad37ae`对全部900源均衡选择2000新候选：扰动0.005/0.015/0.025/0.045rad，速度0.4/0.65/0.85/1.15/1.3倍，20组合各100，seed2026110421。2000整段float32参考批内唯一、对原900＋旧3600重复0；986参考合格者全部物理回放，886通过/1103未通过/11未知，新增通过6.8745小时。与既有1616增强通过名单合并为2502条、20.3501小时，保留逐条验证版本来源；未启动训练或替换数据集。41项生成及6项工作台回归、全文件/物理/receipt/发布审计通过，见[报告](../../verification/dax_additional_2000_20261104/report.md)。


## 当前前端自定义与训练数据整理（2026-11-04）

现有8770第四模块已支持前端输入扰动档位、速度档位、候选数量上限、变体数和种子。扰动0<幅度≤0.3rad（最多5档），速度0.1～3倍（最多10档）；支持中英文逗号。预览回显后端实际解析配置与度数，空项/重复/非法范围拒绝，生成参数范围不代表硬件能力。

新增可选`candidate_count`控制候选数量上限，不超过当前来源×档位×变体的可用组合；按种子打乱并轮转来源，避免只保留前几个源。配置冻结进plan，状态接口返回任务config，不能在运行时静默覆盖。

POST `/api/generation/start`只接受`{job_id}`；服务核验计划及实现绑定，使用固定入口和`--job_id`后台启动本地执行器，直接读取前端保存的plan。HTTP不接收命令/脚本路径/参数覆盖。页面“仅保存配方”仍不启动；“建立并开始生成”显式创建并启动，5秒刷新状态；同一工作台已有生成进程或策略仿真时拒绝再启动。页面打开不会自动执行。此项替代前文“页面尚无启动按钮”的历史状态。

本轮在原工作台真实输入0.06/0.10/0.16rad（约3.44/5.73/9.17°）、0.3/0.6/1/1.6/2.4倍、4500候选上限和seed2026110428，完整执行job `a8a916d1bdd24ea9969be963f4677cc0`。1641参考合格者全部物理回放，1479通过、2986未通过、35未知；4500新参考批内及对既有6500合成参考去重无重复。用户明确超过3600也全用，因此旧2502＋新1479全部纳入，增强训练候选3981条、32.8696小时，无截断。

已独立物化`data/dax_all_qualified_augmented_v4_20261104`：train4920条＝39遥操＋900继承原合成＋3981合格增强，约38.6678小时；val沿用原2条。4922个CSV复制hash全部验证，原900属于继承历史基线、未声称通过新版约束；增强条目保留各自验证版本、full20和物理证据来源。原数据集、训练默认目录未替换，未启动训练。44生成回归、6工作台回归、真实浏览器/HTTP/全文件及发布审计通过，见[本次报告](../../verification/dax_custom_generation_3600_20261104/report.md)。


## 源码归属更新（2026-11-04）

本框架实际实现已迁入data_pipeline，旧generation_*.py为兼容导出；生成/处理/质量/物理回放/验证/凭据/发布分离，几何基础位于robot_sim。实际实现文件纳入新计划hash绑定，旧发布保留、旧未完成计划需新建；数据与门槛不因本轮重构改变。正式训练包组装新增scripts/prepare_dax_training_release.py，见[模块边界与入口](MODULARIZATION.md)。

## physical-v3遥操任务空间增广首批（2026-11-11）

已实现独立入口`scripts/dax_teleop_spatial_augment.py`，配置`config/dax_teleop_spatial_v3_batch1.json`。读取train lineage，以源腕SE3轨迹共同刚性变换、多初值连续IK、公共时间轴重定时与最终参考/实际状态几何验收生成候选；该入口未替换旧合成源的HTTP生成器，也未接训练。通过当前v3几何和完整direct-PD验收的轨迹可继续作为父入口，子代递归继承root recording和split；本批限制为最大深度1、最多6条子代，合格父索引与证据hash独立发布。

批次24候选、14合格、5拒绝、5UNKNOWN；合格5592帧/111.56秒。覆盖以同一54query、5/10cm分箱和系统5Hz采样比较，单列遥操、旧合成、旧增强和front来源；联合query仍有缺口。严格边界、失败记录、命令和图表见[首批报告](../../output/dax_teleop_spatial_v3_batch1_20261111/report.md)，[发布manifest](../../data/dax_teleop_spatial_v3_batch1_20261111/manifest.json)。旧清洗集的未复验部分仍未获准入。

## 连续长轨迹数据增广（2026-11-11）

用户认为首批短片段不足，确定每条30–60秒连续运动、累计新增合格运动至少1小时（不含首尾保持）。新增`scripts/dax_long_spatial_augment.py`和配置`config/dax_teleop_spatial_v3_long_20261111.json`，以自然30秒train遥操窗口作父动作，施加双腕共同任务空间变换，用解析雅可比多初值IK连续求解，形状保持Hermite插值后统一时间轴重定时。速度cap、1.5rad/s²参考加速度和几何/direct-PD门槛沿用首批，真机保护不变。

冻结288候选，实际处理168后停止；73条、195449帧、净运动3688.52秒（1.02459小时），运动段33.9–59.92秒。另219秒hold单列。86条因约束重定时需超过60秒而拒绝，1条实际状态越界拒绝，8条IK未知；未循环短片段或只增加hold凑时长。全部73条有完整仿真q/qdot/time/error与hash，独立去重为0近重复，合格数据父索引独立提供。见[报告](../../output/dax_teleop_spatial_v3_long_20261111/report.md)和[发布](../../data/dax_teleop_spatial_v3_long_20261111/manifest.json)。

## 旧清洗库自动并入入口

`scripts/merge_dax_physical_v3_legacy_cleaned.py`将117.7806小时旧位置/速度清洗train层与51.2719小时净运动合格空间层按来源和准入层级合并。当前[扩展合版](../../data/dax_physical_v3_mixed_legacy_cleaned_plus_spatial_20261113_v2/manifest.json)包含28,660条、173.9891小时完整参考，默认候选入口已绑定该manifest hash。旧层不因此获得几何/direct-PD准入，详见[本次报告](../../verification/dax_legacy_inclusive_merge_20261113/report.md)。
