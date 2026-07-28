# 多源机器人数据入库架构设计

## 目标与边界

本系统处理已经下载到本地的公开机器人/具身数据集，将它们逐 episode 转换为可追溯、可质检、可增量导出的训练数据索引。数据获取（Hugging Face clone、网页下载、权限和凭据）由使用者负责；pipeline 不承担下载，也不修改原始数据目录。

当前仓库中的真实输入适合作为首批验证对象：

| 来源 | 原始格式 | 本体/模态特征 | 接入重点 |
| --- | --- | --- | --- |
| LeRobot ALOHA transfer cube | Parquet + MP4 | 双臂、14 维 state/action、50 FPS、顶视相机 | Parquet 行与 video 时间轴对齐 |
| OXE Bridge | WebDataset tar 内 pickle | 单臂、嵌套 action/observation、JPEG 字节 | 嵌套动作语义和图像字节不直接写 SQLite |
| robomimic | HDF5 | 单臂、`demo/actions/obs` 分组 | HDF5 组结构与可选图像模态 |

LeRobot 目录的现有切片中，Parquet 直接位于数据集根目录而非题目描述的 `data/` 子目录；适配器应以 `info.json` 和文件清单为依据，不能硬编码单一目录布局。

已对当前 robomimic 文件完成只读探测：`data` 下有 10 个 `demo_n`；每个 demo 含 7 维 `actions`、32 维 `states`、`dones`、`rewards`，以及 `obs/agentview_image`、`obs/robot0_eye_in_hand_image` 两路 84×84 RGB 图像。没有原生时间戳，因此该来源配置为 `time_basis: step_index`。

## 总体流程

```text
本地数据目录 + source manifest
              |
              v
       Source adapter（逐 episode 读取）
              |
              v
  原始证据保存 + 统一表示（无损优先）
              |
              v
       质量检查（不阻塞后续 episode）
              |
              v
 SQLite 元数据/低维帧索引 + 原始媒体引用
              |
              v
       分层抽样导出 JSON Lines
```

`source manifest` 是唯一需要使用者维护的接入配置。每个 source 声明：稳定 `source_id`、`source_revision`（上游 commit、release、导出批次号或人工冻结版本）、格式类型、根目录/`source_uri`、机器人类型、标称 fps、动作/状态语义映射和可选视频路径规则。根目录必须是只读输入；输出数据库、运行日志和导出文件放在独立的 `output/` 目录。`source_uri` 用于血缘追踪，不用作唯一键。标准 LeRobot 文件名、目录和字段约定属于 adapter 默认值；只有偏离标准的数据集才通过 `adapter_options` 覆盖。不同数据集会变化的字段路径（如 robomimic 的 demo/action/state/camera 路径、OXE 的嵌套图像路径）保留在配置中。

## 统一表示

核心原则是“统一容器，不伪造统一物理意义”。不同本体的 action 维度不能仅靠补零或截断变成同一向量：ALOHA 的 14 维双臂关节命令、Bridge 的末端位移/旋转增量、纯视频的人类动作没有可直接互换的控制含义。因此入库时保留 native 表示，并且只在有明确映射时额外生成 canonical 表示。

### 数据实体

| 实体 | 核心字段 | 说明 |
| --- | --- | --- |
| `source` | `source_id`、format、dataset_version、manifest、fingerprint | 数据集级血缘和配置快照 |
| `episode` | source、native_episode_id、robot_profile、task、native_fps、capabilities、原始定位信息 | 增量和恢复的最小工作单元 |
| `frame` | episode、frame_index、native_timestamp_sec、derived_timestamp_sec、time_basis、action、state、camera_refs、extra | 训练导出和质检的最小索引单元 |
| `quality_result` | episode、rule_id、severity、passed、evidence | 每条规则的可审计结论 |
| `checkpoint` | source、episode、stage、fingerprint、attempt、error、updated_at | 阶段恢复和幂等控制 |

`native_episode_id` 必须保留上游的原始标识，例如 LeRobot 的 `episode_index`、robomimic 的 `demo_0`、OXE 的 tar member 名。它们只在本来源内有意义；数据库中真正的 episode 身份是 `(source_id, source_revision, native_episode_id)`。

`action` 和 `state` 为可空的结构化字段，而不是数组占位符。每个 signal 只能使用一种形态：数值向量（`values.length == dimension`，`components` 为 null）或命名组件（`components` 非空，`values/dimension` 为 null）。前者用于关节 action/state，后者用于 OXE 的位移、旋转、夹爪等混合控制量：

```json
{
  "representation": "native_joint_position",
  "values": [0.0, -0.96],
  "dimension": 14,
  "groups": {"left_arm": [0, 1, 2, 3, 4, 5, 6], "right_arm": [7, 8, 9, 10, 11, 12, 13]},
  "units": "rad_or_dataset_native",
  "coordinate_frame": null,
  "semantic_version": "source-declared-v1"
}
```

`capabilities` 以布尔声明描述 `action`、`state`、`proprioception`、`images`、`language_instruction`、`robot_identity` 是否可用。纯视频数据会有 `action: null`、`state: null`，这是合法记录，不应填零或判为坏数据。图像/视频默认只存可解析引用、校验和和尺寸等元数据；大二进制不复制进 SQLite。

时间字段严格遵守以下约束：

| `time_basis` | `native_timestamp_sec` | `derived_timestamp_sec` | 适用来源 |
| --- | --- | --- | --- |
| `native_timestamp` | 必填 | null | LeRobot Parquet 主时间轴 |
| `derived_from_fps` | null | 必填 | 上游无时间字段，但 manifest 中有经验证的 fps |
| `step_index` | null | null | 无时间字段且 fps 不可靠的序列 |

不在入库阶段重采样。LeRobot 的 Parquet `timestamp` 是 episode 主时间轴；MP4 PTS 是相机局部时间轴，必须通过 episode video 索引映射，不能直接替代主时间轴。导出层若按 fps 生成时间，会单独填 `derived_timestamp_sec`，避免把派生时间误当原始事实。

`camera_refs` 是受约束的引用数组，不放媒体内容。所有媒体路径都相对该 source 的 `root`，不是相对工作目录、SQLite 或配置文件；路径必须为不含 `..` 的相对 POSIX 路径。所有引用都应有 `kind` 与 `encoding`；LeRobot 每一帧的相机引用必须保存 `video_frame_index` 或 `pts + pts_time_base`，首版同时保存两者，并记录从 Parquet 主时间轴到相机 PTS 的映射方法。OXE 内嵌图像必须至少包含：

```json
{
  "kind": "embedded_tar_pickle",
  "container_relative_path": "bridge_00000.tar",
  "member_name": "sample_000000000000.data.pickle",
  "step_index": 0,
  "field_path": "steps[0].observation.image",
  "encoding": "jpeg"
}
```

LeRobot 视频引用保存相对 source root 的 `video_relative_path`、camera key、`video_frame_index`、PTS/time base、关联 episode 的 timestamp 范围和映射方法；HDF5 相机引用保存相对 source root 的 `container_relative_path`、dataset path 和 frame index。这样导出器能够不依赖猜测地定位媒体。

`extra` 不是原始对象仓库，只允许白名单 JSON 元数据：`reward`、`termination`（first/last/terminal/done）、任务文本/任务索引、语言标注、采集环境标签和 `unknown_field_names`。图像字节、任意 ndarray、嵌套 observation/action 原始对象以及大文本 embedding 都不得进入 `extra`；原字段若需要保留，应存为受约束的一等字段或媒体引用。

## 格式适配层

每种格式实现同一接口：`discover()` 产出 episode 描述，`read_episode()` 逐帧产出原始字段，`fingerprint()` 返回源文件列表及版本哈希。适配器不得自行写数据库或决定质量结论。

- LeRobot adapter：读取 `info.json`、帧 Parquet、episode 索引和视频元数据。`observation.state`、`action`、`timestamp`、`episode_index` 是核心字段；相机从 feature/video 描述和 episode video 索引生成引用。
- RLDS/OXE adapter：保持 episode → steps 结构，显式映射 observation、action、reward、language 和 image。对 OXE tar 内 pickle 仅接受可信的本地公开数据，避免将不受信任 pickle 当作安全输入。
- HDF5 adapter：扫描 robomimic 的 `data/<demo>`，读取 actions、obs/state 与可选 camera dataset；不同键路径配置化，而非写死单一种类。
- Video-only adapter：这是 schema 已预留但当前尚未实现的扩展点；当前运行时只接受 `lerobot`、`oxe_tar`、`robomimic_hdf5` 三种 format。

## 存储与 checkpoint

本题规模（约十几万帧）使用 SQLite。关系表保存 source、episode、quality、运行报告和 checkpoint；帧表保存低维信号 JSON、时间戳和媒体引用。episode 表建立唯一约束：

```sql
UNIQUE(source_id, source_revision, native_episode_id)
```

frame、quality 和 checkpoint 均通过该 episode 记录关联。该约束只解决同一上游版本的幂等入库和断点重续；首版不做昂贵且可能误判的跨来源内容自动去重。源文件的轻量 fingerprint（路径、大小、mtime、可选文件 hash）只用于判断是否需要重新扫描输入。

建议的每 episode 状态机：

```text
discovered -> normalized -> quality_checked -> stored
                                  |                    
                                  +-> accepted / needs_review / rejected
任一阶段异常 -> checkpoint 记录 last_error（保留最后成功阶段和错误详情）
```

`discovered` 代表原始文件存在且 fingerprint 已登记，不表示 pipeline 下载了数据。每次阶段写入都在一个 SQLite 事务中完成：帧替换和 `normalized` 一起提交；质量结果和 `quality_checked` 一起提交；最终状态单独原子提交。异常不会覆盖该阶段，而是在 checkpoint 写入 `last_error`；重启时对同一唯一键跳过已 `stored` 的 episode，从最后一个已提交阶段继续。新的 `source_revision` 创建新的 episode 版本，旧证据不被覆盖。

## 质量检查

规则产生 `accepted`、`rejected` 或 `needs_review`，不因一个 episode 失败而停止批次。第一版至少包含：

1. 语义完整性：已声明具备 action/state 的 source 不得在任一帧产生空映射；这能发现配置字段路径错误。
2. 时间完整性：有原生时间戳时必须严格单调，并检查相邻间隔与标称 fps 的偏差；step-index 来源显式标记为不适用。
3. 动作有效性：检查有限值、绝对值范围和相邻帧突变。
4. 示教有效性：episode 最小长度，以及 state/action 长期完全相同的采集异常。
5. 媒体引用完整性：已声明 images 的 episode 每帧必须有可解析 camera ref；LeRobot 还必须能定位 video frame 或 PTS。

视频内容质量（抽样解码、黑帧/白帧、尺寸突变）属于下一阶段规则，当前不将其误报为已经实现。

阈值与规则版本均需要写入结果证据，确保后续能够解释“为什么某条 episode 被拒绝”。

## 训练子集导出

导出只读取 `accepted`（是否允许 `needs_review` 由参数显式决定）。给定帧预算后，先按 `source × robot_profile × task` 分层分配保底配额，再按该层可用帧数和任务多样性分配剩余预算。采样的基本单位是连续 clip，而不是散帧：这保证时序训练可用，同时在 JSON Lines 中保存 `episode_id`、`frame_start`、`frame_end`、原始时间轴、关键 action/state、媒体引用和质量结论。

若某层数据不足，将未使用预算再分配给相邻机器人组或同任务层，并在导出 manifest 中记录分配与回填过程。这样避免最大来源吞没预算，也避免小来源因强制均分而被过采样。

## 运行报告与扩展

每次运行输出本轮发现、跳过、归一化成功/失败、各规则命中数；累计输出 episode/帧总数、按来源/本体/任务分布、质量命中率及待复核量。报告写入数据库并额外输出 JSON，方便与历史运行对比。

## 已验证结果

使用真实 LeRobot ALOHA 数据执行 quality checkpoint 恢复演练：第一次运行在第一个 episode 的 `quality_checked` 事务提交后中断，报告为 `normalized=1, quality_checked=1, stored=0`；重启后该 episode 直接进入 `stored`，其余 49 个 episode 正常处理；第三次运行发现 50 个已完成 episode，全部 `skipped`。SQLite 中帧主键没有重复，最终为 50 episode、20,000 帧。

三条真实来源完整运行的累计结果为 576 episode、38,106 帧：LeRobot 50/20,000，OXE Bridge 516/17,575，robomimic 10/531。重复运行时 adapter 只枚举已完成 episode 的身份，不再解码 LeRobot 视频、unpickle OXE 或读取 HDF5 数组；本机为数秒级运行且 576 条全部跳过。以 100 帧预算、16 帧 clip 导出 JSONL 时，分层轮询结果为 LeRobot 36 帧、OXE 32 帧、robomimic 32 帧。

扩展到 500 个数据集、5 亿帧时，原始媒体迁移对象存储，SQLite 替换为 PostgreSQL（元数据/checkpoint）+ Parquet/Arrow/Iceberg（帧数据），并通过队列和租约并行处理。训练随机帧读取使用分片索引而非扫描数据库；数据集版本、schema migration、可观测性、访问控制、下载重试和成本治理成为独立组件。

## 建议的实施顺序

1. 固化 source manifest 和统一 schema，包括三个真实来源的动作语义映射。
2. 先完成 LeRobot adapter 与 SQLite 状态机，使用真实小切片验证恢复。
3. 接入 OXE tar 和 robomimic HDF5，记录各自的字段映射和缺模态策略。
4. 实现质量规则、报告和 JSONL 分层 clip 导出。
5. 做 kill/restart、重复运行、fingerprint 变化和坏样本注入测试，再补齐文档中的实际结果。

## 需要共同确认的决策

- 第一版是否把 action/state 统一为“native 保留 + 有映射才 canonical”的双层表示？我建议是。
- 训练导出是否以连续 clip 为单位，而不是任意独立帧？我建议是，默认 clip 长度可配置。
- 视频首版是否仅保存引用并做抽样解码质检，而不复制或转码？我建议是，能控制本地存储和运行时间。
- 以现有的 LeRobot、OXE、robomimic 作为三条正式来源，还是保留纯第一视角视频源来专门验证缺模态降级？前者格式更完整，后者更能展示 schema 的降级能力；我倾向于正式三源之外再增加一个很小的视频-only 切片。
