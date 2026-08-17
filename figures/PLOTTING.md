# 论文图表绘制范例

以 `fig:gap`(`gap_stats.py` + `plot_gap.py` → `gap.pdf`)为模板。新图沿用同一套约定,
保证数据可信、字体与正文一致、尺寸 1:1、配色克制。

## 1. 两脚本模式(不手工填数)

手工填 CSV 极易产生幻觉。固定拆成两个脚本:

- **统计脚本**(`gap_stats.py`):顶部一个 `CONFIG` 块(易改的 `LOGS` / run 路径 / workload),
  读 `logs/` 下的正式日志,推导所有量,写出一个多侧面的 CSV。**没有一个数字是手填的**;
  关键推导带 `assert` 交叉校验(如 `A_run/D_run` 复现日志里的 `A_over_D`)。
- **绘图脚本**(`plot_gap.py`):**只读 CSV**,不碰 `logs/`(logs 被 gitignore)。改图不改数。

新增测量后先跑统计脚本刷新 CSV,再跑绘图脚本。

## 2. 字体(关键,最易错)

IEEEtran 正文是 **Nimbus Roman No9 L**(Times 克隆)。matplotlib 请求 `"Times New Roman"`
在本机**不存在**,会静默回退到 **DejaVu Serif**(更粗更宽),与正文明显不搭。

正确做法:注册 **TeX Gyre Termes**(Nimbus Roman 的 OTF 后裔,已随 texlive 安装),
让图内字体与正文同源:

```python
import matplotlib.font_manager as fm, glob, os
for p in glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf"):
    fm.fontManager.addfont(p)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",         # 数学符号配 Computer Modern,与正文一致
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
```

验证:`pdffonts gap.pdf` 应显示 `TeXGyreTermes`,**不应**出现 `DejaVuSerif`。
(poppler 对 CID Type 0C OTF 会打印 "Mismatch between font type..." 警告,是已知无害噪声。)

## 3. 尺寸:按 \columnwidth 1:1 出图

**不要**让 LaTeX 缩放图片(缩放会连字号一起缩,导致图内字与正文对不上)。
先量出目标宽度,再按该宽度作图,LaTeX 用 `width=\columnwidth` 原样放入:

```
% 探针:在正式 documentclass 下打印列宽
\documentclass[letterpaper,journal]{IEEEtran}
\begin{document}\showthe\columnwidth\end{document}
```

本刊 `\columnwidth = 252pt = 3.49in`。于是 `FIG_W = 3.49`,高度按视觉比例定
(gap 图用 `FIG_H = 1.82`)。字号在 6.5–7.5pt 之间(见 `plot_gap.py` 的 `FS_*`),
放入后与正文脚注级字号相当。

## 4. 子图在 Python 内组合(不用 LaTeX subfloat)

多子图 + 共享 legend + (a)/(b) 标签**全在一个 matplotlib figure 内**完成,
输出单个 PDF。这样两个面板面积严格一致、legend 精确居中、字号不被二次缩放。

- 用 `fig.add_axes([x, y, w, h])`(figure 分数坐标)显式排两个等面积面板。
- 共享 legend:`fig.legend(handles, labels, loc="upper center", ncol=N, frameon=False, bbox_to_anchor=(0.55, 1.01))`。legend 的 `Patch` 描边 `linewidth` 要 ≤ bar 的描边,否则显得比 bar 框还粗。
- (a)/(b) 用 `fig.text(...)` 直接烙在图上。
- 存图**不要**用 `bbox_inches="tight"`(会破坏你精心算好的 \columnwidth 宽度)。

## 5. 配色与描边

跨图复用 `bar_patterns.py` 中的分类调色板与纹理,当前颜色采用 ColorBrewer
Set2 的青/珊瑚/长春花,WISEConv 使用 muted gold。要点:

- 深≠高级;选**柔和有区分度**的成套配色,别自己调深浅。
- 每个 bar 加近黑描边 `#2b2b2b`(`linewidth≈0.8`),灰底上更清晰。
- 使用常规密度的实线斜线 hatch 保证黑白打印和色觉受限时仍可辨: Dense 无纹理,
  Tile skipping 为 `//`,Gather-scatter 为 `\\`,WISEConv 为 `xx`。纹理由
  `bar_patterns.py` 统一维护。

## 6. 断轴(某个 bar 特别高时)

一根 bar 远高于其余(如 gather-scatter 194 vs 其余 <70)会压扁刻度。
拆成上下两个子轴(`ax_top` / `ax_bot`)共享 x,各自 `set_ylim` 到不同区间,
断口画斜线标记(`dashes=(2,1.4)`),只在下轴留 x 轴。

## 7. 数值归正文,setup 归 caption

**正文与 caption 不重复数字。**

- **正文**:所有量化结论(84%、η=0.49、36.6 ms/3.8×、194 ms、5%/9% 等)只此一处。
- **caption**:只讲 setup 与定性含义 — 测的是什么(端到端/逐帧 vs 跨层聚合)、
  哪个模型、每条路径对应哪份实现、虚线代表什么。不抄正文里的具体数字。

单位:吞吐用 **GFLOPS**(FLOPS = 2 × MAC,统计脚本里 `MAC_TO_FLOP = 2`)。

## 8. 收尾核对

1. `python figures/gap_stats.py` 刷新 CSV → `python figures/plot_gap.py` 出 PDF。
2. `pdffonts` 确认嵌入 TeXGyreTermes、无 DejaVu。
3. `RAL/` 下 `latexmk -pdf main.tex`,确认无新 undefined reference。
4. 高 DPI 渲染该页(`pdftoppm -png -r 300`)肉眼比对:图内字与正文同字体、同粗细、字号相称。
