# WISEConv LaTeX-Derived Outline

本文件只映射 `RAL/main.tex` 与 `RAL/src/*.tex` 的当前结构，不承载新的 story proposal。
若与 LaTeX 冲突，以 LaTeX 为准；未决定的 challenge/reuse/scheduling 讨论见仓库根目录的
`discussion.md`。

## Title

**WISEConv: Worklist-driven Masked Convolution for Onboard High-Speed Robotic
Perception**

## Abstract

1. Onboard robotic perception 需要在 tight compute and power budgets 下维持高更新率。
2. Event increments、temporal frame differences 和 learned gates 暴露 spatial masks，
   但现有 execution paths 没有把 sparsity 充分转化为 latency reduction。
3. Tile skipping 保留规则执行，却在 active tiles 内计算 inactive positions；
   gather-scatter 精确选择 positions，却付出 feature materialization、irregular access 和
   writeback 成本。
4. WISEConv 将 active-coordinate worklist 作为 tiled convolution 的 coordinate source，
   结合 position-level selectivity 与规则的 convolution reduction。
5. Construction 在 mask space 内生成 tile-local worklist segments；compute 直接消费这些
   coordinates，无需 input-patch materialization 或 global sorting。
6. Evaluation claim 暂留数值 placeholder，最终按 fastest competing path、fixed weights 和
   fixed upstream mask policies 报告。

## 1. Introduction

当前段落顺序：

1. **Broad robotics scope.** CNNs 支撑 onboard motion/geometry estimation、detection、
   segmentation 和 pose estimation；其 latency 限制 downstream navigation、planning 与
   manipulation 使用新 observations 的频率。
2. **Common operator problem.** Event-camera increments、temporal differences 和 learned
   gates 来源不同，却都要求在 dense 2D grids 上更新 selected positions。
3. **Dense coordinate-driven execution.** Dense GPU convolution 以 contiguous output
   coordinates 驱动 reads/writes，并保留规则的 channel/kernel reduction。
4. **Existing paths.** Tile skipping 以 tile 为选择粒度；gather-scatter 精确选择 positions，
   但离开规则的 dense execution path。
5. **Two efficiency axes.** Throughput 与 useful-compute ratio 共同决定 latency，因而
   theoretical FLOP reduction 不等于 realized latency reduction。
6. **Insight.** 用 active-position worklist 替换 contiguous coordinate source，可以改变
   computed outputs 而不物化 receptive-field patches。
7. **当前临时 challenge.** Coordinate discovery 必须在 compute 前完成；worklist ordering
   又要保留 spatial structure，不能依赖 arbitrary order 或 global sorting。
8. **Solution.** Construction 融合 output-mask propagation 与 per-tile compaction；compute
   直接消费 tile-local segments，保留规则 reduction。
9. **Contributions.** 提出 two-axis framing、worklist-driven design，以及跨三类 masks、
   四个模型和多类 GPU 的 full-model evaluation。

第 7 点是安全基线，不代表 challenge 已最终确定。任何 construction amortization、
conservative reuse 或 short-worklist scheduling 扩展都必须先在 `discussion.md` 中完成证据与
Design 义务审查。

## 2. Background and Motivation

### 2.1 Latency-Critical Onboard Perception

- Robotic perception 位于 online sensing-to-action loop。
- Latency 同时影响 observation age 与 effective feedback frequency。
- Onboard processor 还需与 sensing、planning 和 control 共享 power/thermal budget。
- 因此 deployment objective 是 target device 上的 full-model latency，而非孤立 FLOPs。

### 2.2 Dense-Grid Convolution

- 定义 dense 2D convolution、output coordinate grid 与 computed coordinate set。
- GPU kernel 将 output coordinates/channels 分块，并利用规则 reduction 和 overlapping
  receptive fields 获得 throughput。
- Dense path 每次 invocation 都计算 full output grid。

### 2.3 Masked Convolution

- 输入仍是 dense tensor，upstream policy 提供 spatial active mask。
- 对传播型 mask，output active position 由 receptive-field intersection 决定。
- Mask generation 与 accuracy tradeoff 属于 upstream policy，不属于 WISEConv contribution。

Mask sources 由 prior work 支撑：event-camera increments、temporal differences、learned
spatial gating。

### 2.4 Existing Execution Paradigms

- **Tile skipping:** regular execution，较低 useful-compute ratio。
- **Gather-scatter:** exact active set，较低 throughput 与较高 non-convolution overhead。

### 2.5 Sparsity-to-Latency Gap

- 定义 required work、executed work、useful-compute ratio 与 sustained throughput。
- 用 `t approximately W_req / (eta * Theta) + t_oh` 解释 latency。
- 用 MAC-weighted required-work ratio 代替无权 layer-average activity。

### 2.6 Relation to 3D Sparse Convolution

SpConv/TorchSparse++ 面向 irregular point-cloud coordinates；WISEConv 面向保留 dense
2D representation、由 changing spatial masks 选择 outputs 的执行问题。

## 3. WISEConv Design

### 3.1 Overview

- Input：dense feature tensor 与 input mask。
- Output：requested dense output values、propagated output mask 与 active-coordinate worklist。
- Construction 将 coordinate discovery 移出 convolution critical path。
- Compute 保留 dense feature/weight representation 与规则 reduction。

### 3.2 Tiled Worklist Construction

- 按 contiguous construction tiles 扫描 output grid。
- 每个 position 计算 output mask；每个 tile compact active positions。
- Per-tile segment 保持该 tile 的 local traversal order。
- Parallel reservation 可改变 segment 间顺序，因此不承诺 global raster order。
- Construction work 为 `O(H_out * W_out * k^2)`，并计入 measured operator latency。

### 3.3 Worklist-Driven Convolution

- 在 implicit-GEMM interpretation 中，以 worklist coordinates 替换 dense row-coordinate
  enumeration；output-channel 与 reduction dimensions 仍规则。
- Compute tiles 读取 coordinate blocks，直接从 dense input 加载对应 neighborhoods，并将
  results 写回相同 output coordinates。
- 当前 Design 中 worklist 是 exact mask support，不包含 conservative candidate semantics。

## 4. Implementation

- CUDA C++ extension for PyTorch，FP32 channels-last tensors。
- Model-conversion pass 替换 compatible mask-aware operators，不修改 graph topology、
  convolution parameters 或 learned weights。
- Construction kernel 以 warp ballot、population counts 和一次 segment reservation 完成
  tile-local compaction，不做 device-wide prefix scan 或 sort。
- Identity `1x1` convolution 可在 compatible metadata 已存在时复用 worklist，但当前没有
  将该实现事实提升为主 Design contribution。
- Compute 使用 persistent tiled implicit GEMM、register accumulators、shared-memory staging、
  split-K 与 device/shape/activity-aware autotuning。
- Correctness 对 active outputs 使用 numerical tolerance，不宣称 bitwise identity。

## 5. Evaluation

### 5.1 Experimental Setup（已写，当前冻结）

- **Platforms:** Jetson Xavier NX、Jetson AGX Orin、RTX 4070 Laptop、RTX 3080。
- **Models/tasks:** FireFlowNet optical flow、YOLOv8n/m detection、DynConv human pose。
- **Mask sources:** Ev-Conv、DeltaCNN、DynConv learned gates。
- **Datasets:** MVSEC、MOT16、MPII。
- **Baselines:** native dense PyTorch/cuDNN、prior-work-backed tile skipping、
  gather-scatter。
- **Primary metric:** complete held-out sequences 上的 full-model GPU latency，timed region
  包含 mask propagation、construction、all network operators 和 output update。
- **Analysis metrics:** required-work ratio、useful-compute ratio、sustained convolution
  throughput、construction fraction、task quality 与 operator correctness。

### 5.2 Result Sections（结构已建，正文待填）

1. Full-Model Latency
2. Efficiency Analysis
3. Construction Overhead
4. Task Accuracy

这些小节只从正式 logs 与已冻结 protocol 填数。最终 challenge 产生的 ablation 是否进入
Section 5，取决于 `discussion.md` 中的决策实验。

## Remaining Placeholders

- Abstract、Introduction 与 contributions 中的 speedup/range 数字。
- Background efficiency figure。
- Design overview figure。
- Platform software/power/clock details。
- Evaluation result figures、tables 与 prose。
- Related Work、Conclusion 与 Acknowledgments。
