# Robot Data Pipeline

多来源机器人示教数据的增量入库、质量检查和训练子集导出工具。项目只处理已经落盘的数据目录，不负责下载上游数据。

详细设计见 [ARCHITECTURE.md](ARCHITECTURE.md)，统一记录字段契约见带中文注释的 [schema.jsonc](schema.jsonc)。

## 当前支持的数据格式

管线目前只支持以下三类本地输入；`config.yaml` 中的 `format` 也只接受这三个值：

| `format` | 输入组织 | 关键配置 |
| --- | --- | --- |
| `lerobot` | LeRobot v2/v3 的 Parquet 帧数据与可选 MP4。支持标准 `data/**/*.parquet`，也支持当前下载切片的根目录 `file-*.parquet`。 | 通常无需 `adapter_options`；adapter 自动读取 `info.json`、`episodes.parquet` 和视频。 |
| `oxe_tar` | 本地 OXE/WebDataset tar，成员为 `*.data.pickle`，内部为 episode → steps。 | `adapter_options.image_field_path` 必须与实际嵌套图像字段一致。仅接受可信的本地 pickle 数据。 |
| `robomimic_hdf5` | robomimic HDF5，demo 位于指定根组下。 | 必须核对 HDF5 文件名、demo/action/state/done/reward/camera dataset 路径。 |

当前**不支持**直接读取 RLDS TFRecord、任意 CSV/JSON、纯视频目录或未知 HDF5 布局。配置阶段会拒绝不支持的 `format`；即使格式名称正确，实际文件布局或字段路径不匹配时，处理会记录该 episode 的错误并继续其它 episode。新增格式需要在 `core/adapters/` 实现 adapter，并同时扩展 `SourceConfig.format` 的允许值。

## 新增来源的最小配置

大多数来源不需要逐项填写 action、state 和 adapter 字段。`profile` 会提供已验证的字段路径、时间依据和语义映射；使用者通常只填写身份、版本和输入目录。

例如新增一个结构相同的 OXE Bridge tar 数据：

```yaml
- source_id: oxe_bridge_new
  source_revision: <上游不可变_commit>
  source_uri: https://huggingface.co/datasets/jxu124/OpenX-Embodiment/tree/<commit>
  profile: oxe_bridge
  root: oxe_bridge_new
```

`root` 相对 `input_data/`。当前内置 profile 只有以下三个，使用 `profile` 字段时必须从中选择：

| profile | 已验证的适用布局 |
| --- | --- |
| `lerobot_aloha_dual_arm` | LeRobot ALOHA 双臂 14 维关节 action/state，原生 Parquet 时间戳。 |
| `oxe_bridge` | 当前 OXE Bridge tar/pickle 导出，含 world-vector、rotation、gripper 等嵌套 action。 |
| `robomimic_panda_low_dim` | 当前 robomimic Panda 单臂 low-dimensional HDF5 布局。 |

profile 是可选的：省略时，使用者必须手动填写 `format`、`time_basis`、`action_mapping`、`state_mapping` 及必要的 `adapter_options`。只有数据布局或语义偏离 profile 时，才在该 source 下追加覆盖字段，例如 `adapter_options.image_field_path` 或 `action_mapping`；不要将结构不同的数据强行套用现有 profile。

## 环境安装

项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境和依赖。请先安装 uv，然后在项目根目录执行：

```bash
uv sync --locked
```

该命令会根据 `uv.lock` 在根目录创建 `.venv`，并安装 Parquet、HDF5、视频解析、配置校验和测试依赖。

激活项目环境：

```bash
source .venv/bin/activate
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

激活后可确认解释器来自项目环境：

```bash
python --version
python -c "import pyarrow, h5py, av; print('dependencies ready')"
```

## 配置

运行参数都在根目录的 [config.yaml](config.yaml)：

- `runtime.batch_size`：低维帧读取的批大小。
- `runtime.worker_count`：并行处理 worker 数。
- `runtime.max_in_flight_episodes`：同时处于处理中的 episode 上限。
- `paths.input_root`：已下载数据的根目录；所有相对 `sources[].root` 均以它为基准。
- `paths.output_root`、`paths.database_filename`：运行产物目录和数据库文件名；SQLite 固定写入 `output_root / database_filename`。
- `sources`：每个来源的 `source_id`、不可变 `source_revision`、格式、路径、时间依据，以及 action/state 语义映射。

根据本地数据实际位置修改各 source 的 `root`。`source_revision` 应填写下载时记录的上游 commit、release 或不可变快照标识；不要使用文件修改时间。

当前配置对应仓库中的三类真实数据：LeRobot Parquet/MP4、OXE tar/pickle、robomimic HDF5。

配置时必须重点核对：

- `source_id`、`source_revision`：与 `native_episode_id` 共同组成 episode 唯一约束；同一数据集的新版本必须修改 revision，不能覆盖旧版本。
- `root`：相对 `input_root`，必须指向只读的原始数据目录或文件。
- `format` 与 `adapter_options`：只为该数据集确实变化的字段路径配置覆盖项。LeRobot 标准文件名不需要重复配置；robomimic 的 HDF5 键路径和 OXE 的嵌套图像路径必须核对。
- `time_basis`：有上游秒时间戳用 `native_timestamp`；只有可靠 fps 才用 `derived_from_fps`；两者都没有则用 `step_index`。`step_index` 不会生成或伪造秒时间。
- `action_mapping`、`state_mapping`：必须确认 `representation`、`units`、`coordinate_frame` 和分量/关节分组。未知语义应写为 `dataset_native` 或 `unspecified`，不能猜测。
- 媒体路径：`camera_refs` 中的 video、tar、HDF5 路径一律相对该 source 的 `root`；LeRobot 每帧需要保存可定位视频帧的 `video_frame_index` 或 PTS 信息。

## 处理单位

pipeline 的处理、checkpoint 和断点恢复单位是 **episode**，不是原始文件或单帧：一个 LeRobot Parquet 可包含多个 episode；一个 robomimic `demo_n` 和一个 OXE tar member 分别代表一个 episode。数据库通过以下约束保证同一上游版本不重复入库：

```text
(source_id, source_revision, native_episode_id)
```

同一 episode 内，`frame_index` 是低维信号、时间与媒体引用的存储和导出索引。视频、图像和 HDF5 数组不会复制进 SQLite；frame 只保存能够重新定位原始媒体的 `camera_refs`。更多字段约束、时间字段规则和 `extra` 白名单见 [schema.jsonc](schema.jsonc)。

## 使用

激活环境并完成配置后，先校验配置和路径解析：

```bash
python run.py --config config.yaml --check-config
```

也可以不激活环境，直接让 uv 运行：

```bash
uv run python run.py --config config.yaml --check-config
```

根目录的 `run.py` 是唯一启动入口。执行增量入库、归一化、质检和报告：

```bash
python run.py --config config.yaml
```

已入库的合格数据可按 `config.yaml` 的预算和连续 clip 长度导出 JSON Lines：

```bash
python run.py --config config.yaml --export-jsonl
```

临时覆盖导出预算与 clip 长度：

```bash
python run.py --config config.yaml --export-jsonl --frame-budget 50000 --clip-length 16
```

为验证断点恢复，可在阶段事务提交后模拟中断；重启时会从最后成功阶段继续：

```bash
python run.py --config config.yaml --interrupt-after-stage quality_checked
python run.py --config config.yaml
```

所有 adapter、SQLite checkpoint、质量检查和导出逻辑均位于 `core/`。

## 目录说明

```text
core/           管线核心模块
config.yaml     运行配置
run.py          启动入口
input_data/     本地下载的原始数据，不纳入 Git
output/         SQLite、报告和导出结果，不纳入 Git
```
