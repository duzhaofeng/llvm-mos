# MCS51 内存模型与 C 兼容扩展 RFC（草案）

状态：Draft

作者：MCS51 后端工作流

适用范围：Clang 前端、LLVM IR 地址空间、MCS51 后端 ABI/CodeGen

---

## 0. 背景与目标

MCS51 的关键难点不在单条指令，而在“多内存空间 + 多指针宽度 + ABI 一致性”。
传统工具链（如 SDCC/Keil C51）通过 `data/idata/xdata/code`、`small/large` 等机制解决该问题。

本 RFC 的目标：

1. 在 Clang/LLVM 中引入可落地的 MCS51 内存空间语义。
2. 提供可演进的内存模型（small/compact/large）框架。
3. 约束指针语义与调用约定，避免后续 ABI 返工。
4. 给出最小可交付子集与测试矩阵。

非目标（第一阶段不做）：

1. 一次性做到与 SDCC/Keil 100% 行为一致。
2. 一次性实现所有关键字、pragma、扩展库的兼容。
3. 在语义未稳定前进行大规模性能调优。

---

## 1. 设计原则

1. 先语义正确，再优化质量。
2. 前端语义必须完整保留到 IR（不允许“早丢失”地址空间信息）。
3. ABI 与内存模型先文档化，再编码实现。
4. “关键字兼容”与“核心语义实现”解耦：
   - 核心语义由地址空间属性驱动。
   - 关键字只作为语法糖/兼容入口。

---

## 2. 地址空间编号与语义映射（建议初稿）

说明：编号为建议值，真正落地前需统一到 DataLayout 与后端 lowering。

| 逻辑空间 | 建议 Clang 语法 | LLVM AddrSpace | 典型地址宽度（8051） | 可写 | 备注 |
|---|---|---:|---:|---|---|
| generic/default | 无限定（受内存模型影响） | 0 | 模型相关 | Y | 保留默认地址空间 |
| data | __attribute__((address_space(1))) / `__data` | 1 | 8-bit | Y | 低 128B 内部 RAM |
| idata | __attribute__((address_space(2))) / `__idata` | 2 | 8-bit | Y | 间接内部 RAM（含高 128B） |
| xdata | __attribute__((address_space(3))) / `__xdata` | 3 | 16-bit | Y | 外部 RAM |
| code | __attribute__((address_space(4))) / `__code` | 4 | 16-bit（251 可扩） | N（常规） | 程序存储器/常量 |
| pdata | __attribute__((address_space(5))) / `__pdata` | 5 | 8-bit page + page寄存器 | Y | 分页外部 RAM |
| bdata/bit区 | __attribute__((address_space(6))) / `__bdata` | 6 | 8-bit bit-address | Y | 位寻址区，需独立规则 |
| sfr/sbit | __attribute__((address_space(7))) / `__sfr` | 7 | 8-bit SFR 编址 | Y | 特殊功能寄存器映射 |

约束建议：

1. 不同地址空间指针默认不可隐式互转。
2. 仅允许白名单的 addrspace cast（并发出目标特定警告/错误）。
3. `code` 空间对象默认 `const`，写入行为默认诊断为错误。

---

## 3. 语言入口设计（Clang）

### 3.1 用户可见入口

建议同时支持两层入口：

1. 核心入口：属性
   - `__attribute__((address_space(N)))`
2. 兼容入口：关键字别名
   - `__data/__idata/__xdata/__code/__pdata/__sfr`

建议新增目标选项：

1. `-mmcs51-memory-model={small,compact,large}`
2. `-f[mcs51]-generic-ptr={on,off}`
3. `-f[mcs51]-keyword-compat={none,sdcc,keil,all}`

### 3.2 语义规则（建议）

1. 未限定对象的默认空间由 `-mmcs51-memory-model` 决定。
2. 未限定指针的默认指向空间由模型决定；若 `generic-ptr=off`，禁止不带空间的对象指针。
3. 函数指针与数据指针严格区分（禁止无诊断混用）。
4. 关键字入口最终统一降为地址空间限定类型。

---

## 4. small/compact/large 模型定义（建议版）

说明：本节为“工程可实施定义”，不是对现有工具链的逐字镜像。

| 模型 | 未限定对象默认空间 | 未限定数据指针默认类型 | 目标用途 |
|---|---|---|---|
| small | data/idata（内部 RAM 优先） | near/internal pointer（窄指针） | 追求代码尺寸与速度 |
| compact | pdata/xdata（分页外部优先） | paged pointer | 外部 RAM 分页场景 |
| large | xdata（外部统一） | far/external pointer（宽指针） | 大数据容量场景 |

建议策略：

1. Phase-1 先实现 `small` 与 `large`，`compact` 后补。
2. 明确“默认对象空间”和“默认指针类型”是两件事，文档必须分别定义。
3. 每个模型导出预定义宏，供兼容头做条件编译。

---

## 5. 指针与 ABI 矩阵（草案）

### 5.1 指针类型建议

| 指针种类 | 目标语义 | 建议宽度（8051） | IR 形态 |
|---|---|---:|---|
| ptr_data | 指向 data | 8 | ptr addrspace(1) |
| ptr_idata | 指向 idata | 8 | ptr addrspace(2) |
| ptr_pdata | 指向 pdata | 8(+page语义) | ptr addrspace(5) |
| ptr_xdata | 指向 xdata | 16 | ptr addrspace(3) |
| ptr_code | 指向 code | 16（251 可扩） | ptr addrspace(4) |
| ptr_generic | 通用指针（含空间标签） | 24（建议） | 结构化 lowering 或目标自定义表示 |

### 5.2 调用约定建议（第一版）

1. 先固定一个可实现 ABI，再扩展优化 ABI。
2. 窄指针与宽指针传参与返回规则必须显式写入文档。
3. `generic` 指针传递可先按“聚合对象”处理，避免早期寄存器分配复杂化。
4. 变参与函数指针跨空间调用先限制，再逐步放开。

---

## 6. Clang/LLVM 分层实现建议

### 6.1 Clang 前端

1. 解析关键字/属性并落到 QualType 地址空间。
2. 完成 Sema 约束：
   - 跨空间赋值/参数传递检查。
   - 未限定指针在不同模型下的默认规则。
3. 生成 IR 时保留 addrspace 指针，不做隐式抹平。

### 6.2 LLVM IR 与优化

1. 在 DataLayout 中写清各 addrspace 指针宽度。
2. 审核别名分析与内存优化对 addrspace 的假设。
3. 禁止会破坏空间语义的无约束 ptr cast。

### 6.3 MCS51 后端

1. 为各空间 load/store 建立明确 lowering 规则。
2. 在 SelectionDAG/GlobalISel 路径中对空间敏感指令选择。
3. 明确 code 常量池、函数地址、重定位模型。

---

## 7. 兼容策略（SDCC/Keil C51）

建议采用“三层兼容”：

1. 核心语义兼容：地址空间 + 指针宽度 + ABI。
2. 语法兼容：关键字、常见 pragma 的子集映射。
3. 生态兼容：提供 compatibility headers 和迁移诊断。

优先覆盖子集：

1. 全局变量空间限定。
2. 结构体字段与数组的空间限定。
3. 指针参数、返回值、函数指针基础用法。
4. 常见库接口（memcpy/memset/字符串）在空间语义下的版本化入口。

---

## 8. MVP 范围与测试清单

### 8.1 MVP（建议）

1. 支持 `data/xdata/code` 三空间。
2. 支持 `small/large` 两模型。
3. 支持限定对象与限定指针的基本语义检查。
4. 支持基本函数调用 + 指针参数传递。

### 8.2 测试矩阵（必须）

1. Clang 语义测试（Sema）：
   - 合法/非法跨空间赋值。
   - 默认模型下未限定对象与指针行为。
2. IR 测试：
   - 生成正确 addrspace 指针。
   - 禁止错误 addrspacecast。
3. CodeGen 测试：
   - data/xdata/code 的 load/store/call 序列。
   - small 与 large 模型差异。
4. MC 测试：
   - 与空间相关的重定位/符号路径。

---

## 9. 分阶段推进计划

1. Phase-A（语义打底）
   - 仅做前端 + IR 地址空间保持；CodeGen 可先保守。
2. Phase-B（可用编译链）
   - small/large 打通，跑通基础 C 用例。
3. Phase-C（兼容扩展）
   - 引入兼容关键字组与迁移诊断。
4. Phase-D（性能与完善）
   - 优化指针传递、常量访问、跨空间库调用开销。

---

## 10. 未决问题（需尽快拍板）

1. `generic pointer` 的精确表示与 ABI 编码。
2. `compact` 模型是否纳入第一里程碑。
3. `code` 空间函数指针在 251 扩展下的宽度策略。
4. 与现有 C 库（newlib/picolibc/自定义库）的接口边界。

---

## 11. 建议的第一批实现任务（两周内）

1. 确认地址空间编号并写入 DataLayout 设计说明。
2. 在 Clang 增加最小关键字映射（先 `__data/__xdata/__code`）。
3. 增加 `-mmcs51-memory-model={small,large}` 选项骨架与默认规则。
4. 增加 Sema 诊断：跨空间指针隐式转换报错。
5. 建立最小端到端测试：
   - 一个 small 用例。
   - 一个 large 用例。
   - 一个 code 常量读取用例。
