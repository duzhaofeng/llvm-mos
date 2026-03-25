# MCS51 Phase-A 可执行计划（Clang/LLVM）

状态：WIP

关联文档：

1. MEMORY_MODEL_RFC.zh-CN.md

本计划目标：把 RFC 的 Phase-A 变成可逐条提交的 patch 序列。

---

## A. 当前阻塞（已验证）

结论：Clang 目前不识别 mcs51 作为 AVR 目标 CPU。

验证命令：

1. build/bin/clang -cc1 -triple avr -target-cpu mcs51 -fsyntax-only -x c /dev/null

现状：报错 unknown target CPU 'mcs51'。

影响：

1. 无法用 Clang 前端承载后续 MCS51 地址空间语义测试。
2. RFC 中的内存模型选项和 Sema 规则缺少目标入口。

---

## B. Phase-A 总体里程碑

### Milestone A0：前端入口打通（必须先做）

目标：Clang 能接受 mcs51/mcs251 目标 CPU，并可跑最小 cc1 流程。

验收：

1. -target-cpu mcs51 不再报 unknown CPU。
2. -target-cpu mcs251 不再报 unknown CPU。

### Milestone A1：最小地址空间语义入口

目标：先支持 data/xdata/code 三空间关键词宏映射。

验收：

1. 预定义宏可见（__data/__xdata/__code）。
2. __attribute__((address_space(N))) IR 中保留 addrspace。

### Milestone A2：最小 Sema 约束

目标：跨空间指针隐式转换发出诊断（至少 warning，可升级 error）。

验收：

1. data* -> xdata* 的隐式赋值触发诊断。
2. 显式 cast 保持可用（第一阶段允许）。

---

## C. 首批 patch 序列（建议）

### Patch 1（P0）：CPU 识别与最小目标宏

修改文件：

1. clang/lib/Basic/Targets/AVR.cpp
2. clang/lib/Basic/Targets/AVR.h（如需声明扩展）
3. clang/test/Preprocessor/predefined-arch-macros.c（新增 mcs51 检查段）

动作：

1. 在 AVR target CPU 列表中加入 mcs51/mcs251。
2. 在 getTargetDefines 中为 mcs51 系列定义最小宏：
   - __MCS51__
   - __MCS251__（仅 mcs251）
3. 先不改 driver 的 avr-libc 链接策略。

验收：

1. clang -cc1 -triple avr -target-cpu mcs51 可运行。
2. 预处理输出包含 __MCS51__。

风险：

1. 现有 AVR 宏集合可能混入 mcs51 语义，需要后续清理。

### Patch 2（P0）：地址空间关键词宏入口（三空间）

修改文件：

1. clang/lib/Basic/Targets/AVR.cpp
2. clang/test/Sema（新增 mcs51-memory-spaces.c）
3. clang/test/CodeGen（新增 mcs51-address-space.c）

动作：

1. 为 mcs51 目标定义关键词别名宏：
   - __data  -> __attribute__((__address_space__(1)))
   - __xdata -> __attribute__((__address_space__(3)))
   - __code  -> __attribute__((__address_space__(4)))
2. 先不引入 idata/pdata/sfr，避免一次性扩大面。

验收：

1. Sema 通过最小样例。
2. IR 中出现 addrspace(1/3/4)。

风险：

1. 地址空间编号尚未与后端 DataLayout 绑定，仅为前端语义占位。

### Patch 3（P1）：最小内存模型选项骨架

修改文件：

1. clang/include/clang/Options/Options.td
2. clang/lib/Frontend/CompilerInvocation.cpp
3. clang/include/clang/Basic/LangOptions.def（如需）
4. clang/include/clang/Basic/TargetOptions.h（如需）

动作：

1. 新增选项：-mmcs51-memory-model=small|large。
2. 先只存储与传递到 cc1，不改变生成行为。

验收：

1. 选项可被解析并进入 invocation。
2. 非法值给出诊断。

风险：

1. 先不改变默认指针推断，避免影响现有 C 前端路径。

### Patch 4（P1）：跨空间隐式转换诊断（最小版）

修改文件：

1. clang/lib/Sema/SemaExpr.cpp（指针赋值/初始化路径）
2. clang/test/Sema（新增 mcs51-addrspace-conv.c）

动作：

1. 在 mcs51 目标下，对非同地址空间指针的隐式赋值发 warning。
2. 显式 cast 允许通过。

验收：

1. 覆盖赋值、参数传递、返回值三类样例。

风险：

1. 与现有 OpenCL/SYCL 地址空间规则交叉，需限定只在 mcs51 目标启用。

---

## D. 第一周执行顺序（建议）

1. 先做 Patch 1（CPU 识别）。
2. 紧接 Patch 2（三空间宏）。
3. 再做 Patch 3（选项骨架）。
4. 最后 Patch 4（Sema 诊断）。

每个 patch 要求：

1. 单独可编译。
2. 单独可测试。
3. 单独可回滚。

---

## E. 最小测试清单

1. clang/test/Preprocessor：
   - mcs51 预定义宏存在性。
2. clang/test/Sema：
   - 关键词宏可用于变量/指针声明。
   - 跨空间隐式转换诊断。
3. clang/test/CodeGen：
   - addrspace(1/3/4) IR 生成。

---

## F. 当前建议的下一步

1. 直接开始 Patch 1：先加 mcs51/mcs251 CPU 识别与基础宏测试。
