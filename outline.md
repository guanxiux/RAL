# WMConv: Worklist-driven Masked Convolution for Onboard High-Speed Robotic Perception

## Introduction 大纲

### Paragraph 1: Motivating scenario - masked convolution as a deployment paradigm

1. 端侧机器人需要高性能视觉 CNN 感知来支撑 onboard perception、closed-loop control、navigation、obstacle avoidance 和 manipulation。
2. 但机器人/edge 平台受功耗、散热、重量和成本约束，板载算力远低于服务器级 GPU；同时与物理环境的闭环交互（避障、抓取、导航）把感知延迟直接送进控制环，构成 **hard real-time** 约束。在受限算力上，dense convolution 对所有 spatial positions 一视同仁地计算，往往跑不出闭环所需的更新频率。
3. 正是 **受限算力 ∩ 硬实时** 这个交集，使"跳过与当前感知无关的空间计算"从优化变成 enabling technology。许多部署在 UAV、mobile robot 和 embedded edge device 上的高速视觉系统因此先生成一个 spatial active mask，再只更新与 active regions 相关的卷积输出。
4. 我们称之为 masked convolution：输入仍是 dense image-grid tensor，但执行计划由 mask 决定。它正成为 onboard high-speed 视觉 CNN 的常见执行范式，把有限算力集中在与当前感知相关的空间区域。

### Paragraph 2: Existing methods use masked convolution to accelerate CNNs

1. 许多已有工作已经在不同视觉场景中采用了这种 masked convolution 思路，只是它们利用的稀疏来源不同。
2. 在 event-camera inference 中，Ev-Conv 利用相邻事件时间窗之间的 increment sparsity：事件流被编码成 voxel grid 或 event image 后，系统对相邻窗口做差并 sparsify increments，再只对 active increment regions 触发 CNN 更新，用于高速 depth estimation、optical flow 和 object recognition。
3. 在 temporal change inference 中，DeltaCNN-style 方法利用视频帧间冗余：系统比较当前帧与历史帧或 motion-compensated history，生成 change/residual masks，并只更新被变化区域影响的卷积输出，适合移动机器人或车载相机中的连续 detection、segmentation 和 tracking。
4. 在 learned spatial gating 中，dynamic networks 利用任务相关性稀疏：轻量 gating modules 预测当前输入中值得计算的 spatial regions，让主干 CNN 跳过背景、低价值区域或与当前决策无关的区域，以适配 edge AI 设备上的 adaptive perception。
5. 这些工作虽然面向不同应用，但共同点是用某种 active mask 来驱动 CNN 的空间选择性执行，希望 masked convolution 将领域稀疏性转化为实际推理加速。

### Paragraph 3: Existing execution paths for masked convolution

1. 现有系统沿两条路径实现 mask-driven selective execution，而它们其实卡在同一权衡的两端：**有效计算占比**（跑的计算里有多少是真正需要的）与 **throughput**（硬件单位时间能跑多少计算）。latency 同时取决于两者，两条路径各拿一头、丢另一头。
2. **Tile skipping**（tile 指 GPU 一次卷积所处理的一组 output positions）几乎原样沿用 dense pipeline，只加一个 tile 级判断：整个 tile 全 inactive 才跳过。它因此保住 dense pipeline 的 **throughput**，但跳过粒度是 tile——一个 tile 内有少量 active 就要整块计算，**有效计算占比低**。Ev-Conv 和 DeltaCNN-style 方法接近这一路线。
3. **Gather-scatter** 则根据 mask 精确抽出 active positions、gather 邻域、计算、再 scatter 回 dense output，跳过做到 position 级，**几乎全是有效计算**；但 feature 抽取、receptive-field gather、scatter writeback、indexing 和不规则访存的 overhead 会把 **throughput 打崩**，省下的计算被这些开销吃掉。Learned spatial gating 等接近这一路线。

### Paragraph 4: Problem - sparsity does not automatically become speedup

1. 这一权衡的直接后果是：上游产生的高 sparsity 或 theoretical FLOP reduction，并不自动转化为相应的 GPU latency speedup——没有一条现有路径能同时拿到两头，所以省下的 FLOPs 大多漏在了有效计算占比或 throughput 上。
2. 在受限的 **embedded robot compute（Jetson Xavier NX / Orin）** 上这一落差尤其明显。In our evaluation, event-camera inference, temporal change inference, and learned spatial gating reduce the active output ratio to roughly `XX--YY%`, but existing masked-convolution execution paths only reduce convolution latency by `AA--BB%`, leaving perception frequency below the `CC` Hz needed to keep up with closed-loop control.
3. 因此 onboard high-speed perception 的 masked convolution 核心问题是：如何**同时**拿到 position-level 的高有效计算占比与 dense pipeline 的高 throughput，让 sparsity 真正转成可用的 perception frequency。

### Paragraph 5: Insight - dense pipeline 加坐标偏移即可只算有效计算

1. 我们的核心洞察很直观：给 dense convolution pipeline 加上适当的坐标偏移，就能让它在原有的执行框架内只计算 active output——既不改 pipeline 本身，也不新增独立的数据搬运阶段。
2. 这是因为 dense conv kernel 本来就在做 gather：它对每个 output position 用网格坐标从 dense grid 取出 receptive field（边算边取数，并不先把 patch 拷成独立矩阵），再做累加。我们要做的，只是把"遍历所有 output position"换成"按坐标偏移跳到 active output"，其余取数与累加和 dense kernel 完全一样。
3. 好处一，**gather 几乎免费**：我们没有新增任何 gather/scatter 阶段，只是给 dense pipeline 已有的隐式 gather 换了个起点坐标，唯一的增量是定位每个 active output 的一次坐标查找。
4. 好处二，**dense pipeline 的优点全部保留**：每个 output position 内部的卷积计算（receptive field 与所有 channel）依旧稠密，访存依旧连续、复用 spatial locality，cuDNN 级硬件效率不受影响。
5. 这正好回应 Para 3/4 的双轴张力——按坐标偏移只算 active output 带来高**有效计算占比**，复用隐式 gather 保住接近 dense 的 **throughput**，两头同时拿到。

### Paragraph 6: Challenge - 把相对 dense pipeline 的 overhead 压到最小

1. insight 把 feature 搬运压到了每个 active output 的一个坐标查找，但要真正逼近 dense pipeline 的 throughput，就要把"为做 masked conv 而相对 dense pipeline 多出来的 overhead"降到最小。这部分 overhead 主要来自两处。
2. **得知道偏移量。** kernel 要按 active output 的坐标做 gather，但这些坐标事先并不已知；naive 做法（比如在计算 kernel 内部扫描去找 active output）会严重拖慢主计算。
3. **得保持 locality。** 坐标喂入 kernel 的顺序决定能否复用 dense pipeline 的连续访存与 spatial locality；任意顺序会退化成不规则、相距很远的访存，throughput 塌回 gather-scatter 的水平。但 naive 做法（比如对 active output 做全局排序）本身的开销同样很糟。
4. 而且这两处 overhead 都发生在主卷积计算之外、不随计算量摊薄，因此挑战是用足够轻的方式同时解决它们。

### Paragraph 7: Proposal - WMConv operator

1. 基于上述 insight，本文提出 WMConv（Worklist-driven Masked Convolution）：一个 **worklist-driven** 的 dense-grid masked convolution operator，把 feature gather fuse 进 conv kernel，只稀疏化要计算的 output 位置、保留 dense pipeline 的执行方式。
2. 算子分两阶段：**worklist 构建** 与 **worklist-driven convolution**。worklist 即需要计算的 active output 及其在 dense grid 上的坐标。
3. 构建阶段在 mask 空间工作，从 active mask 产出 worklist——只处理 boolean mask 和坐标 metadata，不触碰 feature 与 weights。
4. convolution 阶段由 worklist 驱动：只遍历其中的 active output，按坐标从 dense grid 隐式 gather receptive field、做完整稠密卷积、写回 dense output grid，每个 position 内部计算与 dense kernel 完全一样。
5. 由此 mask scheduling 与 feature computation 彻底解耦：构建阶段廉价地决定"算哪些、按什么坐标和顺序算"，convolution 阶段就是一个只跑 active output 的 dense conv kernel。

### Paragraph 8: Solution - tile 化的 worklist 构建同时压住两处 overhead

1. WMConv 的关键，是用一个 **tile 化** 的 worklist 构建机制同时解决 Para 6 的两处 overhead，且自身成本远小于它保护的计算。tile 沿用 Para 3 的定义（GPU 一次处理的一组连续 output positions）。
2. **针对"得知道偏移量"**：算子把寻找 active output 从计算 kernel 内部，挪到一个独立的、在 output grid 上并行的 tile 扫描阶段——每个 output position 检查其 receptive field 是否与 active mask 相交，相交即为 active。这避免了在 convolution kernel 内扫描，主计算因此保持 dense pipeline 的执行方式。
3. **针对"得保持 locality"**：扫描以**连续的 tile** 为单位进行，并把每个 tile 内的 active positions **compact 成一整块**写入 worklist。由于 tile 在 output grid 上连续、tile 内又紧凑排列，相邻 worklist 条目在 dense grid 上高度邻近，receptive field 大量重叠，从而复用 dense pipeline 的连续访存与 spatial locality。这是 tile 内局部紧凑，而非全局排序，避免了全局排序的高开销。
4. 同一个 tile 结构因此一举两得：既是并行扫描的单位（解决偏移量），又是局部紧凑的单位（解决 locality），无需额外的全局排序或索引重映射。
5. 这一阶段只在 2D mask 空间操作，成本只与 spatial resolution、kernel size、tile 大小和 active pattern 有关，不随 channel 数增长；而它保护的卷积计算随 channel 数增长。因此当 output active ratio 较低时，构建 overhead 相对主计算可忽略，跳过 inactive output 直接转成 latency reduction——比 tile skipping 更细粒度，比 gather-scatter 更轻量。

### Paragraph 10: Contributions

1. 我们指出 onboard high-speed perception 中的 masked convolution bottleneck，并把它刻画为有效计算占比与 throughput 的双轴权衡——event-camera increments、frame differences 和 spatial gating 共享同一 active-mask-driven pattern，却都只能拿到其中一头。
2. 我们提出 WMConv，一个 worklist-driven 的 dense-grid masked convolution operator：用 tile 化的轻量 worklist 构建同时解决"知道偏移量"与"保持 locality"两处 overhead，让 dense pipeline 只算 active output。
3. 我们在 **Jetson Xavier NX 和 Orin** 上，于三类 mask 来源、三个机器人感知任务上评估 WMConv，对比 dense cuDNN、tile skipping 和 gather-scatter，证明它在受限板载算力下比现有路径更有效地把 sparsity 转成 on-device latency reduction 与更高 perception frequency，并保持任务精度。

## Experimental Design 大纲

### 评测平台

- **Jetson Xavier NX** — 低功耗 embedded 级，代表受限板载感知平台。
- **Jetson Orin** — 高端 embedded 级，代表现代板载机器人算力。
- （可选）desktop/server GPU 作为次要参考，刻画算力充裕时算子的行为；robotic 主张以两块 Jetson 为准。

### 三个机器人应用 × mask 来源（平权）

| 应用 | 输出形态 | Mask 来源 | 依据/baseline 来源 | 备注 |
|---|---|---|---|---|
| **Depth estimation** | dense 逐像素回归 | event-camera increments | Ev-Conv (RAL'23) | high-speed 感知；DENSE / MVSEC 类事件数据 |
| **Object detection** | bbox | learned spatial gating | DGNet / Dual Gating | gating 原生产生 tiled spatial mask；COCO + RetinaNet/Faster-RCNN |
| **Semantic segmentation** | 逐像素分类 | temporal change | DeltaCNN-style frame difference | video 连续分割；视觉效果直观 |

三个应用覆盖三种不同输出形态（回归 / 检测 / 逐像素分类），证明算子对 output 任务类型不敏感；三种 mask 来源作为上游输入被消费，共同验证算子在机器人感知中的通用性。

### Baselines

- Dense cuDNN / 平台原生 dense convolution。
- Tile skipping（DeltaCNN-style block skipping）。
- Gather-scatter / coordinate-list sparse convolution（如 submanifold sparse conv、TorchSparse++）。

### Metrics

1. **On-device latency**：每个 Jetson 平台上的 conv / 端到端推理延迟，跨 mask 来源和 active output ratio `rho_out`。
2. **Sparsity-to-latency 转化**：给定 input/output active ratio 实测 latency reduction 对比 theoretical FLOP reduction，与各 baseline 对照（呼应 DGNet 自述的 theoretical-vs-actual speedup gap）。
3. **Perception frequency (Hz)**：把 latency 换算成可达感知/更新频率，并对照代表性 closed-loop control-rate 需求，证明算子让 dense 路径在板载上达不到的频率成为可能。
4. **Accuracy preservation**：在 depth / detection / segmentation 上，在近无损 mask 来源下确认算子不引入额外任务精度损失（robotic 硬门槛）。
5. **Operator cost breakdown**：拆分 mask-scan / index-construction overhead 与 active-output convolution compute，证明 scheduling overhead 与 channel dimensions 解耦、且相对节省的计算足够小。

### 关键论点呈现

- 主结果：三个机器人应用 × 两块 Jetson，证明在受限算力 + 实时约束下，算子比 tile skipping 和 gather-scatter 更有效地把 sparsity 转成 on-device 加速和更高 perception frequency。
- 泛化：跨三种输出形态和三种 mask 来源均有效，说明算子是 mask-source / task agnostic 的通用后端。
- 行为刻画：报告算子收益随 `rho_out` 变化的曲线（如 segmentation decoder 上采样使 mask 变密时的表现），给出算子的适用区间。
