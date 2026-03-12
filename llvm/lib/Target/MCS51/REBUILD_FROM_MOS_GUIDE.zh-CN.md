# 基于 MOS 梳理并重建 51/251 后端指南

## 背景结论

当前目录下的 MCS51 实现是 AVR 复制替换而来，不适合作为 8051 正向实现基础。
建议保留它作为“搭建脚手架的样例”，但实现层面按 51 架构重新建模。

本指南目标：
1. 梳理 MOS 后端的关键建构思想。
2. 与 51 架构对照，划分“可复用”与“需新机制”。
3. 给出 51 先行、251 后续扩展的落地路线。

---

## 一、MOS 后端的关键建构点（按价值排序）

### 1) 指令分层：真实指令层 + 逻辑指令层 + 伪指令层

MOS 把“机器真实编码指令”和“代码生成逻辑指令”分离：
- 真实指令：用于 MC 层编码/反汇编。
- 逻辑指令：用于 CodeGen 选择、寄存器分配、优化。
- 伪指令：用于在较晚阶段再做不可提前的决策（尤其是 Post-RA）。

收益：
- 统一语义，减少重复 Pattern。
- 把“寄存器具体选型”延后到更合适阶段。
- 复杂目标可用伪指令桥接多阶段实现。

### 2) 特性与设备建模：Feature/Family/Device 三层

MOS 使用 Feature -> Family -> Device 的层级组合：
- Feature：指令能力位。
- Family：组合能力。
- Device：具体 CPU 名称绑定到 family。

收益：
- 新增子架构时只需要扩展特性集。
- 一份指令定义可通过 Predicates 针对不同设备生效。

### 3) 寄存器建模不是“纯硬寄存器”，而是可服务编译器决策

MOS 引入了 imaginary registers（映射到零页/软栈语义），并配套分配 pass。

收益：
- 让寄存器分配器处理“看起来像寄存器，实则可映射内存”的资源。
- 通过后端 pass 做目标相关成本优化（例如零页优先）。

### 4) 目标相关 Late Pass 体系

MOS 的 Pass 管线里有较多目标专用阶段（拷贝优化、零页分配、Post-RA scavenging、late optimization、static stack alloc 等）。

收益：
- 可以把架构特有约束拆分到多个可控阶段。
- 避免在 ISel 阶段过早做不可逆决策。

### 5) 测试策略：MC 与 CodeGen 双轨

MOS 测试覆盖：
- MC 汇编/反汇编/编码正确性。
- MIR/LL 的 CodeGen 选择与后优化行为。

收益：
- 每层可独立回归。
- 重构时能快速定位回归层级。

---

## 二、51 架构与 MOS 的“相似点/差异点”

## 相似点（可参考 MOS 思路）

1. 都是资源非常紧张的 8-bit 微架构：
- 指令不规则、寻址模式强约束、寄存器稀缺。

2. 都需要“延后决策”：
- 很多最优序列依赖寄存器位置、立即数范围、跳转距离。
- 适合用逻辑指令 + 伪指令 + Late Pass 的组合。

3. 都需要强设备特性建模：
- 51 与 251 是明显的能力分层，和 MOS 的 Family/Device 模式非常契合。

## 差异点（需要新机制，不建议照抄）

1. 地址空间模型差异巨大：
- 51 有 CODE/DATA/IDATA/XDATA/BDATA/SFR 等空间语义。
- MOS 的零页与 imaginary register 思路可借鉴，但不能直接映射为 51 的空间模型。

2. 寄存器体系差异：
- 51 是 A/B/DPTR/PSW/SP + R0-R7（且有 bank 语义），与 MOS/AVR 的大统一寄存器类不同。

3. 位寻址与布尔处理差异：
- 51 有 bit-addressable memory 和 bit 分支/置位清零指令族。
- 需要单独 Operand/Pattern/Legalization 机制。

4. 控制流与调用编码约束不同：
- 51 有短/绝对/长跳转、页约束、不同 call 变体。
- 需要单独分支放松和 call-lowering 策略。

5. 251 扩展方向不是“加几个 opcode”那么简单：
- 涉及更宽数据与更大地址空间语义、更多寻址组合。

---

## 三、可复用机制 vs 需新建机制

## 可复用（推荐直接借鉴 MOS 结构）

1. 顶层 td 组织方式
- 入口 td -> Features/Devices/RegisterInfo/InstrInfo/CallingConv 的分层。

2. 指令分层思想
- Real 指令给 MC。
- Logical 指令给 CodeGen。
- Pseudos 给 Post-RA / late expansion。

3. Feature/Family/Device 建模方法
- 51 基础族。
- 251 扩展族在 Feature 级别递增。

4. 测试布局
- 先建 MC 测试，再建 CodeGen 测试。
- 允许大量 xfail 作为早期能力边界记录。

## 需新建（51 专属）

1. 地址空间与指针模型
- 明确不同空间的 load/store/addr lowering。

2. 51 寄存器与 bank 语义
- PSW 与 bank 切换影响寄存器分配与调用约定。

3. bit-addressable 机制
- bit 操作与 bit 分支的 DAG/MI 语义建模。

4. 跳转/调用范围策略
- 短/中/长跳转选择与放松。

5. 251 扩展策略
- 通过新 Feature 逐步打开，而不是重写另一套后端。

---

## 四、建议的重建路径（51 -> 251）

### Phase 0：重建骨架（1-2 周）

目标：把“AVR 残留命名/语义”彻底剥离，保留可编译框架。

动作：
1. 先定义最小目标特征与一个 generic 51 device。
2. 建立最小寄存器模型（A/B/DPTR/PSW/SP/R0-R7）。
3. 建立最小 MC 指令格式（MOV/ADD/SJMP/LJMP/LCALL/RET 这类基本集合）。
4. 把测试框架先搭起来（MC + CodeGen 目录和 lit.local.cfg）。

验收：
- llc/llvm-mc 能识别目标并完成最小汇编往返。

### Phase 1：先打通 MC 层（2-3 周）

目标：编码/反汇编/汇编语法稳定。

动作：
1. 先覆盖核心寻址模式与控制流。
2. 每加一批指令就补 all-opcodes 风格测试（可分组）。
3. 明确 fixup/reloc 规则，先支持最常见重定位。

验收：
- 核心指令族有稳定 MC 回归。

### Phase 2：CodeGen 最小可用（3-5 周）

目标：C 到汇编可跑通基础程序。

动作：
1. 建立 logical 指令层（先覆盖 load/store/add/sub/branch/call/ret）。
2. 建立必要 pseudo 与 late expansion。
3. 调通 calling convention 与 frame lowering 最小路径。

验收：
- 小型 C 程序（算术、分支、函数调用）能编译并通过回归。

### Phase 3：51 特有机制补齐（持续）

目标：体现 51 真正特性。

动作：
1. bit-addressable 数据与分支。
2. bank 相关寄存器语义。
3. 不同地址空间访存与优化。

### Phase 4：251 扩展（在 51 稳定后）

目标：以 feature-gated 方式扩展，不复制分叉。

动作：
1. 先补最关键扩展指令与寻址。
2. 在同一后端中通过 Feature251 控制启用。
3. 补充 51/251 行为差异测试矩阵。

---

## 五、执行时的三个硬规则

1. 先测试后指令
- 每新增一组指令必须同步加 MC 回归。

2. 不把硬件细节提前塞进 ISel
- 能晚做的决策，尽量在 pseudo 扩展和 late pass 做。

3. 不在 51 和 251 之间复制代码
- 一套后端，多 feature 分层。

---

## 六、建议你立即开始的最小任务清单

1. 写出 51 的最小 Feature/Family/Device 定义（只留 1 个 generic）。
2. 重写最小寄存器 td（A/B/DPTR/PSW/SP/R0-R7）。
3. 先只实现 MOV + ADD + SJMP/LJMP + LCALL/RET 的 MC 编码。
4. 建立 llvm/test/MC/MCS51 与 llvm/test/CodeGen/MCS51 目录及最小用例。
5. 做第一版逻辑指令层与 pseudo 扩展，只覆盖 i8 基本运算和跳转。

完成以上 5 步后，再进入 251 扩展，不会返工太多。
