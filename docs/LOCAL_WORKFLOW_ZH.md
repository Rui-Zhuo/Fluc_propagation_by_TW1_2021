# Tianwen-1 本地分析代码

> 本文保留自发布整理前的本地 README，记录既有操作方法与历史检查结果。末尾的旧投稿题目和投稿状态属于历史记录；正式发表信息以仓库首页为准，本次发布验证以 `RELEASE_VALIDATION.md` 为准。

本项目用于天问一号多地面站射电观测中的密度扰动传播分析。现在按“路径、观测配置、公共函数、分析流程”组织；旧脚本文件名继续可用。

## 目录与职责

```text
CODE/
  run_analysis.py               统一运行入口
  project_paths.py              兼容原路径导入
  utils_signal_analyse.py       兼容原公共函数导入
  multibaseline_fit.py          兼容原多基线拟合导入
  tw1/
    paths.py                   本地路径与年份目录映射
    cases.json                 59 组案例：文件、UTC 时间段、时间格式、站点极性
    cases.py                   配置加载、临时覆盖、输入检查
    time.py                    时间转换与时间窗口
    io.py                      观测序列读取、PLL MAT 转换
    signal.py                  去异常、去趋势、插值、谱拟合
    density.py                 密度扰动计算与原诊断绘图函数
    propagation.py             多基线速度拟合
    cli.py                     命令行参数
    workflows/                 11 个可调用的分析/绘图流程
  Sun_Mars_Earth/               几何计算、台站坐标、SPICE 内核、MATLAB 工具
  analyse_multiple_baseline.py  多基线分析入口及绘图函数
  read_PLL_data.py              PLL 转换入口
```

输入数据为 `E:/Research/Data/Tianwen-1/{2021,2023}/...`。`outdated` 和 `[phase]` 均位于 `2021` 下；保留原有处理版本，不自动替换为同一天的其他文件。输出保持在项目同级 `RESULT`；新的多基线结果使用 `RESULT/multiple_baseline_rmse`。

`CODE`、`RESULT` 从源文件位置推导，与启动时所在目录无关。换本地数据盘时，只需设置环境变量 `TW1_DATA_DIR`；外部小波工具目录可用 `TW1_UTILS_DIR` 覆盖，默认 `E:/Research/Program/[else]`。

## 本地运行

在 PowerShell 中进入 CODE：

```powershell
Set-Location E:\Research\Project\Fluc_propagation_by_TW1_2021\CODE
python run_analysis.py --list
python run_analysis.py analyse_wavelet_coherence --list
python run_analysis.py --check
python run_analysis.py analyse_wavelet_coherence --case 9 --start 08:00 --end 08:20 --no-save --plot 1
```

也可以继续使用原脚本名，并传入相同参数：

```powershell
python analyse_wavelet_coherence.py --case 9 --start 08:05 --end 08:15 --no-save
python reconstruct_wavelet.py --case 12 --no-save --no-show
python calc_power_index.py --case 3 --no-save --no-show
python plot_wavelet_spectrum.py --case 4 --no-show
```

案例编号保留原脚本的 `i_date`、`i_case`、`file_type` 含义。未传参时沿用各脚本原默认案例和保存/绘图选项。`--save/--no-save` 控制原流程中已有的保存分支；`--show/--no-show` 控制窗口显示，不等同于保存。`--plot 0/1` 仅用于三个小波相干流程，分别选择 Plotly/Matplotlib。

## 选择观测、站点与时间段

在 `tw1/cases.json` 找到对应流程及编号，修改：

- `data_relative`：含年份、处理版本的输入目录。
- `file1_name`、`file2_name`：当前选中的两个文件；其余 `file_*` 保留原脚本中可选站点的文件名。
- `time_beg`、`time_end`：原始 `YYYYDDDHHMMSS` UTC 编码。
- `time_formats`：每个文件明确标记为 `utc_code` 或 `sod`，不根据文件路径长度猜测。
- `frequency_signs`：特殊站点的符号修正，例如 2023-10-28 的 S6。
- `time_lb`、`time_ub`：谱指数拟合的子时间段，必须包含在分析时间段内。

一次运行可以用 `--start/--end` 覆盖时间，用 `--file1/--file2` 选择配置中已声明时间格式的文件，不修改保存的配置。时间支持 `HH:MM[:SS]` 或完整编码；短格式沿用案例日期。时间选择保留原代码的开区间约定，即不含两个端点，当前仅支持同一 UTC 日。

谱图和密度演示仍沿用原代码的样本范围，分别在对应 workflow 中集中设置；不支持 `--start/--end` 的流程不要传这两个参数。重构的周期范围仍在对应 workflow 开头的 `period_lb/period_ub` 设置。各科学算法和绘图细节保留在 workflow 中，尚未统一成一套通用算法。

## Python 调用

在 CODE 下启动 Python，或把 CODE 加入本地解释器搜索路径：

```python
from tw1.workflows.analyse_wavelet_coherence import main
result = main(case=9, start="08:00", end="08:20", save=False, show=False, plot=1)
coherence = result["coh"]
time_lag = result["time_lag"]

from tw1.cases import get_case
case = get_case("analyse_wavelet_coherence", 9)
print(case.selected_files())

from tw1.time import TimeWindow
from tw1.signal import detrend, interpolate
from tw1.propagation import fit_velocity
from tw1.density import cal_Ne, cal_sigma_Ne
```

相干流程返回相干对象、两条序列和时间延迟；重构流程返回两条序列及重构分量；谱指数流程返回两个斜率。其他绘图流程主要通过图像展示结果。导入 workflow 不读取观测数据、不执行分析。

PLL 转换可以指定观测、站点和输出位置：

```powershell
python read_PLL_data.py --date s1a26x --station ur --channel chan3 --output E:\Research\Project\Fluc_propagation_by_TW1_2021\RESULT\pll_ur.dat
```

不传 `--output` 时保留原行为，在对应观测目录生成文本数据。该命令会写文件，请使用明确的输出位置保存新的转换产品。

## 检查与环境

```powershell
python run_analysis.py --check
python -m unittest discover -s . -p "test_*.py" -v
```

检查覆盖 59 组配置的选中文件、年份及处理版本、时间格式与边界、配置覆盖、导入行为，以及原有多基线拟合测试。路径存在检查并不替代科学结果验证。

继续使用本机 Python 及现有科学计算库。当前环境运行时提示 NumPy 1.26.4 超出所装 SciPy 声明的支持范围；本次整理没有更改环境。小波重构/谱指数仍依赖本地 `python_utils_JSHEPT.py`。几何计算使用 `Sun_Mars_Earth/kernels` 中现有的 SPICE 内核；几何模块已改为在计算时加载内核。

已修复原代码中 Excel 文件路径被误当作目录、末尾调试变量 `db` 导致运行中断，标量时间转换不支持 NumPy 整数，以及 PLL 的 Matplotlib 分支忽略保存开关的问题。`KSfreq.dat` 的秒数时间格式也改为显式配置，避免放在第一个站点时被误转换。

本次验证：14 项回归检查通过，59 组配置选中的输入文件全部存在；11 个 workflow 均用本地数据运行通过，其中高分辨率 PLL 使用 30 秒区间。SPICE 地球位置计算及不同启动目录的路径解析也通过。验证发现并修复了密度谱拟合函数对工作流全局变量的隐式依赖。

密度演示保留原采样时间构造、固定半径等计算假设；原积分方法在当前数据上仍会产生积分细分次数警告。运行通过不表示这些科学假设及数值精度已重新评审。

## 原项目说明

# SuperiorSolarConjunction

This repository contains the codes and results for the analysis of compressional waves propagation based on the radio sounding of Tianwen-1 and multiple ground-based radio telescopes. It is maintained by Rui Zhuo (ruizhuo@pku.edu.cn)

the manuscript titled *'Radio Sounding of Compressional Waves Propagation and Reflection in Solar Wind Acceleration Region'* submitted to *Nature Astronomy* by Maoli Ma, Rui Zhuo, Ziqi Wu and Liangliang Yuan. The emails of corresponding authors are jshept@pku.edu.cn and mamaoli@shao.ac.cn.

- *analyse_wavelet_coherence.py*: given the radio signals from two stations, calculate the coherency spectrum and time-lag spectrum between them, and obtain the propagation speed along the baseline finnally.

- *reconstruct_wavelet.py*: given the wavelet coefficients and base functions, reconstruct the time series of the signal.

- *calc_solar_offset.py*: given the UTC time and two stations, calculate the projected points on the sky plane, define the baseline, and calculate the baseline length and the solar offset. 

- *calc_power_index.py*: given the radio signal from one station, calculate its PSD spectrum and power index.

- *calc_density_fluctuation.py*: given the power spectrum of one station, calculate the density fluctuation. 

- *statisticize_wavelet_coherence.py*: given the dataset of wavelet coherence, statisticize the dependency of propagation speed on the solar offset and temperal scale. 

- *analyse_multiple_baseline.py*: given the dataset of the *identical* density fluctuations and multiple baselines, show their relationships, and obtain the 2-D propagation velocity on the sky plane. 

- *wavelet_coherence.xlsx*: the dataset of wavelet coherence and propagation speeds along the baselines. 

- *multiple_baseline.xlsx*: the dataset of the *identical* density fluctuations observed by multiple baselines.
