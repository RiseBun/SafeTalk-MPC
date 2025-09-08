# 🗣️ SafeTalk-MPC: Semantic-to-MPC with Safety Shield

**SafeTalk-MPC** 将**自然语言语义**直接编译为 **MPC 优化问题 (OCP)** 的 JSON 补丁，并由 **Safety Shield（安全盾）** 保底：即使 LLM 指令激进/含糊/异常，仍尽量保证轨迹**可行**与**安全**。  
项目支持**仿真验证**与**实车部署（RDK X5 移动平台）**，现在正在同步开发中

---

## ✨ 亮点与创新

- **语义编译器 (Semantic Compiler)**  
  自然语言 → DSL JSON Patch → 自动合成 MPC 问题。  
  支持修改：目标点、避障半径、状态/控制权重、预测步长 (horizon)、控制约束、速度上下限、平滑度权重等。  
  **无需手动改配置**，一句话即可控制。

- **安全盾 (Safety Shield)**  
  三模式避障：**Hard** / **Soft** / **Hybrid**。  
  当 LLM 生成不合理参数时自动兜底，始终以**安全可行**为第一目标。

- **反回退 (No-Backtrack)**  
  引入单调前进约束，抑制到点前“回拉/倒退”，让轨迹更符合直觉与实机需求。

- **自适应限速 (Speed Cap)**  
  远离目标时更快、接近目标自动减速，**平稳逼近**、避免过冲。

- **终端盒 (Terminal Box)**  
  在目标邻域内定义收敛盒，要求低速进入并停车，提升“到点且停住”的可靠性。

- **风险自适应 (Risk-Adaptive)**  
  `--risk={low,med,high}` 自动调节安全半径、预测步长、转角上限等，实现不同风险偏好的快速对比。

---

## 📦 目录结构

```text
SafeTalk-MPC/
├─ dsl/                       # 任务 DSL（JSON）
│  ├─ base.json
│  └─ example_task_curve_01.json
├─ sem2mpc/
│  ├─ compiler/               # DSL 解析 & OCP 构建
│  ├─ semantics/              # 语义编译器 & LLM Provider
│  ├─ sim/                    # 仿真/可视化（绘图、动画、指标）
│  └─ ...
├─ patch/                     # 示例 JSON 补丁（可选）
├─ exp/                       # 实验输出（建议忽略版本控制）
├─ requirements.txt
├─ run.py                     # 一键开始文件，运行之后输入自然语言
└─ README.md
```

> 运行生成物（`exp/`、`*_result.png`、`*_anim.mp4`、`*_metrics.json`、`last_patch.json`、`_tmp_task.json`、`llm_logs/`）建议通过 `.gitignore` 排除，保持仓库整洁。

---

## 🛠️ 安装与环境

> 推荐 **Conda + Python 3.10**；CasADi 依赖已在 `requirements.txt` 中。

```bash
git clone https://github.com/RiseBun/SafeTalk-MPC.git
cd SafeTalk-MPC

# 方式 A：进入 sem2mpc 后运行（常用）
cd sem2mpc
conda create -n mpcenv python=3.10 -y
conda activate mpcenv
pip install -r requirements.txt

# 方式 B：保持在仓库根目录运行
# conda create -n mpcenv python=3.10 -y
# conda activate mpcenv
# pip install -r sem2mpc/requirements.txt
```

> Windows + 本地 LLM（Ollama）用户：若需语义编译，请先安装并启动 Ollama；否则可先用 `--llm none` 跑基线。

---

## 🚀 快速开始

### 1) 基线仿真（不使用 LLM）
**在 `sem2mpc/` 目录执行：**
```bash
python -m sim.sim_runner dsl/example_task_curve_01.json --out exp/base --llm none
```
生成文件：
- `exp/base_result.png`（静态轨迹）
- `exp/base_anim.mp4`（轨迹动画）
- `exp/base_metrics.json`（指标，含末端误差/最近障碍距离/求解时间）

> 若在仓库根目录运行，命令为：  
> `python -m sem2mpc.sim.sim_runner dsl/example_task_curve_01.json --out exp/base --llm none`

---

### 2) 语义编译：用自然语言直接改 MPC
**第一种方法，PowerShell（Here-String，无需转义引号）**
```powershell
$instr = @'
绕障更保守，把安全半径加到 0.6 米
'@
python -m sim.sim_runner dsl/base.json $instr --llm ollama --model qwen2.5:7b-instruct --out exp/E1 --save-llm
```

**Bash**
```bash
python -m sim.sim_runner dsl/base.json "绕障更保守，把安全半径加到 0.6 米" \
  --llm ollama --model qwen2.5:7b-instruct --out exp/E1 --save-llm
```
**第二种方法，利用根目录下的run.py直接运行**
```powershell
python run.py
```
输入自然语言后
将生成：
- `last_patch.json`（LLM 产出的 JSON 补丁）
- `_tmp_task.json`（应用补丁后的 DSL）
- 对应轨迹/动画/指标

> 若 LLM 输出非严格 JSON，系统会尝试抽取；仍失败则自动触发**规则兜底**（fallback），保障可用性。

---

### 3) 离线补丁：不依赖 LLM（对照用）
```bash
echo '{ "obstacle": { "radius": 0.6 } }' > patch/E1.json
python -m sim.sim_runner dsl/base.json patch/E1.json --llm none --out exp/E1_manual
```

---

## 🧪 对照实验：验证“语义编译器是否发挥作用”

下表将**自然语言**与**手工补丁**一一对应；预期两者在 `metrics.json` 与动画上“等价或接近”。

| 实验 | 自然语言 (LLM) | 手工补丁 JSON | 预期变化 |
|---|---|---|---|
| E1 | 绕障更保守，把安全半径改为 0.6 米 | `{"obstacle":{"radius":0.6}}` | 最近障碍距离 ↑ |
| E2 | 更强调平滑控制，把 `u_rate_weight` 提高到 1.0 | `{"u_rate_weight":1.0}` | 控制更平滑，响应略慢 |
| E3 | 更稳地停车，把 `terminal_velocity` 提到 60 | `{"weights":{"terminal_velocity":60}}` | 末端更稳，过冲↓ |
| E4 | 缩短预测步长到 120 | `{"horizon":120}` | 求解更快 |
| E5 | 启用中点引导 | `{"insert_midpoint":true}` | 绕障姿态改变 |
| E6 | 远快近慢更明显：`v_far=1.3, v_near=0.22` | `{"speed_cap":{"v_far":1.3,"v_near":0.22}}` | 收敛更平顺 |

**判定 LLM 是否生效：**
1. 控制台会打印**补丁来源**（`llm` / `fallback`）；  
2. 查看 `last_patch.json` 是否为严格 JSON，且语义与指令一致；  
3. 对比“自然语言”与“手工补丁”两次运行的 `*_metrics.json` 与动画。

---

## 🧰 常见问题（尤其 Windows + Ollama）

- **LLM 卡住/返回 503**  
  - 确认 Ollama 服务已启动；测试：  
    ```bash
    curl http://127.0.0.1:11434/api/tags
    ```
  - 显存不足：改用 `qwen2.5:3b-instruct` 或启用 CPU 推理（慢但稳）。  

- **PowerShell 里 JSON 解析报错**  
  - 使用 Here-String（如上示例），或确保 JSON 的键和值均为**双引号**。

- **UTF-8 BOM 相关**  
  - 项目读写已兼容常见 BOM 场景；若仍遇到，保存为 “UTF-8 (无 BOM)”。

---


---

## 📊 指标说明（metrics.json）

- `end_position_error` —— 末端位置误差（越小越好）  
- `min_obstacle_distance` —— 最近障碍物距离（应 ≥ 安全半径）  
- `solve_time_sec` —— 单次求解时间（对比不同 horizon/约束配置）

---

## 📖 引用

```bibtex
@misc{SafeTalkMPC2025,
  author    = {RiseBun},
  title     = {SafeTalk-MPC: Semantic-to-MPC with Safety Shield},
  year      = {2025},
  howpublished = {\url{https://github.com/RiseBun/SafeTalk-MPC}}
}
```

---

## 🤝 贡献

欢迎 PR：  
- 语义映射与提示模板的改进  
- 新的约束/代价项与可视化方式  
- 更多 LLM Provider 适配（如多模型自一致性、在线 API 等）

---
