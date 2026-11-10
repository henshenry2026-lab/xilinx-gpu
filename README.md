# xilinx-gpu

[![CI](https://github.com/henshenry2026-lab/xilinx-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/henshenry2026-lab/xilinx-gpu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

一个从小型 2D 图形核开始的 FPGA GPU 学习项目：**PC 通过 UART 改变矩形，FPGA 每个像素周期计算颜色并输出视频时序**。

核心使用可综合 Verilog，面向 Xilinx 7 系列。v1 支持一个实心矩形、黑色背景、RGB332、640×480 视频输出。项目已包含可重复运行的主机测试、RTL 仿真和综合检查，方便在此基础上继续实现精灵、帧缓冲或光栅化。

![真实 RTL 仿真生成的 640×480 画面](docs/demo.png)

上图来自集成测试第三帧的实际 RGB 输出：矩形位于 `(160, 120)`，尺寸 `320×240`，颜色 `0x1F`。检查脚本会比较仿真像素与这张图片。

## 快速运行

需要 Python **3.10+**、Icarus Verilog（`iverilog`、`vvp`）和 Yosys。仿真、协议测试与图片生成只使用 Python 标准库。

```sh
git clone https://github.com/henshenry2026-lab/xilinx-gpu.git
cd xilinx-gpu
python tools/check.py
```

Ubuntu / Debian：

```sh
sudo apt-get update
sudo apt-get install -y iverilog yosys python3
python3 tools/check.py
```

Windows：在 **MSYS2 UCRT64 终端**安装工具，然后回到 PowerShell 执行检查：

```sh
pacman -S --needed mingw-w64-ucrt-x86_64-iverilog mingw-w64-ucrt-x86_64-yosys
```

```powershell
python tools/check.py
```

脚本会自动查找 PATH，以及常见的 `C:\msys64\ucrt64\bin` 和 `C:\msys64\mingw64\bin`；其他安装位置请把工具目录加入 PATH。检查失败会返回非零退出码，并保留完整日志。

生成内容位于被 Git 忽略的 `build/`：仿真日志、综合报告、综合后 JSON 网表、`demo.ppm`、`demo.png` 和 `check-summary.json`。有意修改演示场景后，使用 `python tools/check.py --update-preview` 刷新 README 图片。

## 模块与“存、算”

```mermaid
flowchart LR
    PC[Python CLI] -->|UART RX| RX[uart_rx]
    RX --> PARSER[command_parser]
    PARSER --> SCENE[scene_regs]
    CLOCK[像素时钟] --> TIMING[video_timing]
    TIMING -->|x / y / active| RENDER[pixel_renderer]
    TIMING -->|垂直消隐开始| SCENE
    SCENE -->|当前矩形参数| RENDER
    RENDER --> OUT[video_out]
    TIMING -->|HS / VS / active| OUT
    OUT --> VIDEO[RGB332 / HS / VS / DE]
```

| 模块 | 职责 |
|---|---|
| `uart_rx` | 两级同步，8-N-1 中点采样，报告停止位错误 |
| `command_parser` | 包头、CRC、操作码、边界校验，接收超时恢复 |
| `scene_regs` | 当前与待提交两组参数，垂直消隐期间原子切换 |
| `video_timing` | 像素坐标、显示有效区、行场同步、提交脉冲 |
| `pixel_renderer` | 并行比较矩形边界，决定每个像素的颜色 |
| `video_out` | 把 RGB、DE、HS、VS 一起寄存，消隐区输出黑色 |
| `gpu_top` | 连接上述模块，供仿真和综合使用 |

**存储：**用触发器保存位置、宽高、颜色和待更新状态，v1 不需要帧缓冲。**计算：**对扫描坐标执行矩形覆盖判断，每个像素时钟输出一个颜色。矩形使用半开区间 `[x, x+width)`、`[y, y+height)`。

复位后画面全黑。收到有效命令只更新待提交参数；扫描到垂直消隐起点时整体提交，所以活动画面中不会混用两组参数。同一帧的多条有效命令采用最后一条。若新命令恰好与提交处于同一时钟，旧待提交参数先提交，新命令保留到下一次提交。

## PC 控制与 UART 协议

不连接硬件即可查看命令：

```sh
python tools/uart_rect.py --x 160 --y 120 --width 320 --height 240 --color 0x1f --dry-run
```

向板卡发送时才需要 `pyserial`：

```sh
python -m pip install -r requirements.txt
python tools/uart_rect.py --port COM3 --x 40 --y 30 --width 200 --height 120 --color 0xe0
# Linux 示例：--port /dev/ttyUSB0
```

串口默认 **115200 baud、8 数据位、无校验、1 停止位**。13 字节命令如下：

| 偏移 | 长度 | 字段 |
|---|---|---|
| 0 | 2 | 固定包头 `A5 5A` |
| 2 | 1 | 操作码 `01`：设置矩形 |
| 3 | 2 | x，无符号 16 位、小端 |
| 5 | 2 | y，无符号 16 位、小端 |
| 7 | 2 | width，无符号 16 位、小端 |
| 9 | 2 | height，无符号 16 位、小端 |
| 11 | 1 | RGB332：`RRRGGGBB` |
| 12 | 1 | CRC-8/SMBUS，覆盖偏移 2～11 |

CRC 多项式 `0x07`，初值 `0x00`，不反射，最终异或 `0x00`；标准校验向量 `123456789` 得到 `0xF4`。宽高须大于零，且整个矩形必须处于 640×480 内。硬件使用 17 位边界加法，避免 16 位溢出绕过检查。

坏 CRC、不支持的操作码、非法矩形会被丢弃，并保留当前画面。字节间超时 10 ms、UART 停止位错误或复位会清除未完成的命令。超时按 RTL 参数 `CLOCK_HZ` 计算。v1 单向接收，无 ACK；完整帧发送后，命令在下一次可用的垂直消隐提交。

## FPGA 接口与上板

`gpu_top` 的默认参数为 `CLOCK_HZ=25175000`、`BAUD=115200`。

| 端口 | 方向 | 含义 |
|---|---|---|
| `clk_pix` | 输入 | 25.175 MHz 像素时钟，使用 FPGA 专用时钟网络 |
| `rst` | 输入 | 与 `clk_pix` 同步的高有效复位，至少保持两个时钟 |
| `uart_rx_pin` | 输入 | 串口 RX，模块内部含两级同步 |
| `rgb[7:0]` | 输出 | RGB332，颜色与 DE、HS、VS 同拍对齐 |
| `de` | 输出 | 有效显示区标志 |
| `hsync` / `vsync` | 输出 | 低有效行/场同步 |

| 时序 | 有效区 | 前沿 | 同步脉宽 | 后沿 | 总计 |
|---|---:|---:|---:|---:|---:|
| 水平（像素） | 640 | 16 | 96 | 48 | 800 |
| 垂直（行） | 480 | 10 | 2 | 33 | 525 |

标称帧率为 `25,175,000 / (800 × 525) ≈ 59.9405 Hz`。输出级有一个像素周期的寄存延迟，RGB 和同步信号共同延迟。修改 `CLOCK_HZ` 只会改变 UART/超时的分频参数；实际视频速率由输入像素时钟决定。

上板时需要针对真实开发板完成以下接线：用 Clocking Wizard 产生像素时钟；把 PLL/MMCM 的稳定状态纳入同步复位；按原理图配置时钟、UART、视频引脚与 I/O 电平。VGA 需要板载 DAC 或合适的电阻网络；HDMI 还需要 TMDS 编码与串行发送模块，不能直接接这些 RGB 信号。

板卡型号、封装引脚和视频电路尚未确认，因此仓库目前提供可移植核心，没有宣称可直接下载的 bitstream 或已验证的 XDC。

## 验证状态

`python tools/check.py` 和 GitHub Actions 使用同一检查入口：

- Python：CRC 标准向量、字节序、合法边界、非法参数、无串口依赖的 dry-run。
- UART：连续字节、约 ±1.4% 位速率偏差、短毛刺、坏停止位、break 后恢复、中途复位。
- 命令解析：CRC/操作码错误、零尺寸、越界、加法溢出、包头噪声、超时和错误恢复。
- 场景：参数原子提交、后到覆盖、提交与新命令同拍、矩形四边及消隐。
- 集成：经真实串行比特流控制，逐像素检查连续 **3 帧、共 1,260,000 个扫描周期**；每帧 307,200 个活动像素、50,400 个 HS 低电平周期、1,600 个 VS 低电平周期。
- 综合：`synth_xilinx -family xc7`，随后执行 `check -assert`。

2026-11-10 本地验证使用 Icarus Verilog 13.0、Yosys 0.66 和 Python 3.13。该次映射约为 **261 个 LUT、398 个触发器、0 BRAM、0 DSP**；不同综合版本和布局结果可能变化。完整统计见本地 `build/synthesis.log` 或 CI 下载的验证产物。

**尚未完成：**Vivado 布局布线/时序收敛、实际 FPGA 下载、显示器和串口的硬件验证。当前结果证明功能仿真和 7 系列逻辑综合通过。

## 项目结构

```text
rtl/                  6 个功能模块与 gpu_top
tests/rtl/            4 个自检查 SystemVerilog 测试台
tests/test_uart_rect.py
tools/uart_rect.py    UART 编码与发送 CLI
tools/check.py        跨平台验证与预览生成
docs/demo.png         从真实 RTL 输出生成的演示图
.github/workflows/    同入口 CI
```

后续可在这个基础上逐步加入多矩形/精灵、BRAM 帧缓冲、画线和三角形光栅化。首版集中验证“控制命令 → 参数存储 → 像素计算 → 视频扫描”这条通路。

## English overview

A small synthesizable 2D graphics core for Xilinx 7-series FPGAs. A Python CLI sends CRC-protected UART commands; the FPGA stores one rectangle and computes RGB332 pixels at 640×480, approximately 60 Hz. Scene updates commit atomically at vertical blanking. No framebuffer, BRAM or DSP is required by v1.

Run `python tools/check.py` with Python 3.10+, Icarus Verilog and Yosys. It runs host tests, four self-checking RTL benches, three complete video frames and Xilinx 7-series synthesis. The README image is generated from the actual simulated pixel stream. Physical board integration, Vivado timing closure and hardware validation remain to be done once the board is identified.

## 许可证

[MIT](LICENSE)，Copyright © 2026 Slash。
