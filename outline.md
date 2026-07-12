# WISEConv LaTeX-Derived Outline

This outline is synchronized with the current LaTeX draft in `RAL/`. It is not
an independent source of truth. If this file conflicts with `main.tex` or
`src/*.tex`, follow the LaTeX.

## Title

**WISEConv: Worklist-driven Masked Convolution for Onboard High-Speed Robotic
Perception**

## Abstract-Level Claim

Onboard robotic perception relies on 2D convolutional networks that must meet
tight latency constraints on embedded platforms. Spatial active masks are common
in high-speed vision systems, but existing masked-convolution execution paths
lose the sparsity benefit: tile skipping preserves dense-pipeline throughput but
computes many unneeded outputs, while gather-scatter computes only needed
outputs but sacrifices throughput. WISEConv uses an ordered worklist of active
positions to drive the dense convolution datapath, aiming for both high
useful-compute ratio and near-dense throughput.

## Introduction Flow

1. **Robotic deployment motivation**：onboard robots run convolutional
   perception in the control loop for navigation, manipulation, and closed-loop
   control. Embedded platforms have limited compute, while perception latency
   directly affects the control path.
2. **Masked convolution setting**：many systems concentrate compute on active
   regions. The input remains a dense image-grid tensor, but only active output
   positions are updated.
3. **Dense pipeline baseline**：dense convolution partitions the output grid into
   contiguous tiles. Coordinate-driven, tile-based execution is fast because it
   accesses spatially contiguous data with regular memory patterns.
4. **Existing execution paths**：tile skipping stays inside the dense pipeline
   and skips only fully inactive tiles; gather-scatter extracts active positions
   exactly but abandons the regular dense pipeline.
5. **Trade-off characterization**：the bottleneck is useful-compute ratio vs.
   throughput. Tile skipping loses useful-compute ratio, gather-scatter loses
   throughput, and neither yields proportional latency reduction from sparsity.
6. **WISEConv insight**：replace the dense tile's contiguous coordinate source
   with active-position coordinates from a worklist. This gives position-level
   selectivity while keeping the per-position dense convolution datapath.
7. **Practical challenges**：active coordinates must be discovered before
   compute, and worklist order must preserve spatial locality. Kernel-internal
   scanning and global sorting are both too expensive.
8. **WISEConv design**：a construction stage scans the mask and emits an ordered
   worklist; a compute stage drives dense convolution with that worklist. Tiled
   construction discovers coordinates and compacts active positions within each
   tile into contiguous worklist segments.
9. **Contributions**：identify the trade-off, present the worklist-driven
   operator, and evaluate on Jetson Xavier NX and Orin across three mask sources
   and robotic perception tasks.

## Background Structure

### 2D Dense Convolution on Embedded Platforms

CNN-based depth estimation, optical flow, detection, segmentation, and pose
estimation remain central to real-time onboard perception. The section defines a
single 2D convolution layer, output positions, receptive fields, and a tiled
output index grid. Dense convolution computes all output positions
unconditionally, which can exceed embedded latency budgets.

### Masked Convolution for Onboard Perception

The paper defines an input active mask `M_in` and an output active mask `M`. An
output position is active when its receptive field contains at least one active
input position. In multi-layer networks, masks propagate layer by layer.

Mask sources in the current draft:

- **Event-camera increments** for high-speed event-camera inference.
- **Temporal frame differences** for continuous video perception.
- **Learned spatial gating** for adaptive perception.

### Execution Paradigms

Tile skipping executes a tile if any position inside it is active. It preserves
throughput but computes inactive positions inside active tiles.

Gather-scatter extracts active positions exactly and computes only those
outputs. It avoids wasted computation but pays gather, scatter, indexing, and
irregular-memory overhead.

### Sparsity-to-Latency Gap

The draft defines the layer active ratio and the network-level average active
ratio. It then uses throughput and useful-compute ratio to explain why high
upstream sparsity does not automatically yield proportional latency speedup.

### 3D Sparse Convolution Positioning

3D sparse convolution targets irregular point-cloud workloads. WISEConv targets
regular dense 2D grids and skips inactive outputs without leaving the dense
pipeline. This distinction belongs in background or evaluation, not the main
intro story.

## Design Structure

### Overview

WISEConv has two stages:

- **Construction stage**：scan `M_in`, evaluate the output active mask, and emit
  the worklist `q` of active output indices.
- **Compute stage**：consume `q` as the coordinate source, read receptive fields
  from the dense input grid, accumulate convolution identically to the dense
  datapath, and write active outputs back.

The computed set contains only active positions, while the per-position datapath
remains dense.

### Worklist Construction

The construction stage operates on the tile partition defined in the background.
For each output position, it evaluates receptive-field intersection with the
input mask. Within each tile, active indices are gathered in row-major order.
The final worklist concatenates per-tile active subsequences. The draft allows
inter-tile order to depend on parallel block scheduling, while preserving
tile-local order.

Construction cost is independent of channel dimensions. The paper models this
cost as mask reads and integer writes, then compares it against active-output
convolution cost.

### Worklist-Driven Convolution

The compute stage divides the worklist into compute tiles and output channels
into channel segments. Each compute tile loads a chunk of worklist entries,
uses those coordinates to gather the corresponding dense-grid input slices,
loads weights, accumulates, and writes results to the dense output addresses.

Locality comes from the tiled construction: consecutive entries within each
per-tile worklist segment are spatial neighbors with overlapping receptive
fields. The claim is tile-local spatial locality, not global sorted order.

## Implementation Structure

The implementation section currently sketches a CUDA C++ extension exposed to
PyTorch through Pybind11. It uses NHWC tensors, a construction kernel that maps
threads to output positions, warp-level ballot and prefix counting to compact
active lanes, and a Cutlass-style implicit-GEMM compute kernel with offline
autotuning.

## Evaluation Plan

The LaTeX currently references evaluation but the full evaluation section is not
yet written. The intended evaluation should cover:

- Jetson Xavier NX and Jetson Orin as primary robotic embedded platforms.
- Dense cuDNN, tile skipping, and gather-scatter as primary baselines.
- Event-camera increments, temporal frame differences, and learned spatial
  gating as representative mask sources.
- On-device latency, active-ratio to latency scaling, perception frequency,
  task accuracy, and construction vs. compute breakdown.

Numeric placeholders such as `XX--YY%`, `AA--BB%`, and `DD--EE%` remain in the
current LaTeX and should be filled only from measured results.
