# MOS 后端架构分层说明

本文档面向 llvm/lib/Target/MOS 的实现维护，描述目录分层、TableGen 层次、C++ 组件关系，以及从 IR 到目标文件的数据流。

## 1. 目录分层

1. 根目录 `llvm/lib/Target/MOS`
- 目标机核心（TargetMachine/Subtarget）
- 指令选择与合法化（GlobalISel + 部分 SelectionDAG 支撑）
- 寄存器/栈帧/调用约定
- 后端机器级优化 Pass
- AsmPrinter 与 MCInst Lower

2. `llvm/lib/Target/MOS/TargetInfo`
- 目标注册入口，绑定 `Triple::mos`。

3. `llvm/lib/Target/MOS/MCTargetDesc`
- MC 层实现：AsmInfo、InstPrinter、CodeEmitter、AsmBackend、Fixup、ELF Writer、Streamer、MCExpr。

4. `llvm/lib/Target/MOS/AsmParser`
- 汇编语法解析。

5. `llvm/lib/Target/MOS/Disassembler`
- 反汇编。

## 2. TableGen 层次（.td）

## 2.1 顶层入口

`MOS.td` 是总入口，按如下顺序拼装：

1. `MOSFeatures.td`
2. `MOSDevices.td`
3. `MOSRegisterInfo.td`
4. `MOSRegisterBanks.td`
5. `MOSInstrInfo.td`
6. `MOSCombine.td`
7. `MOSCallingConv.td`

并定义 Target、AsmWriter、AsmParser 及 parser variants。

## 2.2 指令层次

`MOSInstrInfo.td` 继续拆分为：

1. `MOSInstrFormats.td`
- 指令格式、编码位、地址模式、Operand 类型、谓词基类。

2. `MOSInstrInfoTables.td`
- SearchableTable（如指令 relax 关系表）。

3. `MOSInstrInfoSPC700.td`
- SPC700 指令与模式扩展。

4. `MOSInstrPseudos.td`
- 伪指令层（延迟决策/后期展开）。

5. `MOSInstrLogical.td`
- 逻辑指令层（规则化语义，便于 RA/优化）。

6. `MOSInstrGISel.td`
- GlobalISel 扩展 generic 指令。

## 2.3 其它 .td 职责

1. `MOSFeatures.td`
- 子目标特性位。

2. `MOSDevices.td`
- 处理器家族/设备定义与 feature 组合。

3. `MOSRegisterInfo.td`
- 寄存器、子寄存器、寄存器类、Imaginary/ZP 寄存器。

4. `MOSRegisterBanks.td`
- GlobalISel register bank 定义。

5. `MOSCallingConv.td`
- 参数传递、返回值、callee/caller saved 规则。

6. `MOSCombine.td`
- GICombiner 规则。

## 3. TableGen 生成物映射（MOSGen*.inc）

由 `llvm/lib/Target/MOS/CMakeLists.txt` 触发：

1. `MOSGenInstrInfo.inc`
2. `MOSGenRegisterInfo.inc`
3. `MOSGenSubtargetInfo.inc`
4. `MOSGenCallingConv.inc`
5. `MOSGenGlobalISel.inc`
6. `MOSGenGICombiner.inc`
7. `MOSGenAsmMatcher.inc`
8. `MOSGenAsmWriter.inc`
9. `MOSGenDisassemblerTables.inc`
10. `MOSGenMCCodeEmitter.inc`
11. `MOSGenMCPseudoLowering.inc`
12. `MOSGenRegisterBank.inc`
13. `MOSGenSearchableTables.inc`

## 4. C++ 组件关系（.h/.cpp 与 .td）

## 4.1 目标机与子目标

1. `MOSTargetMachine.h/.cpp`
- 后端入口；注册 MOS pass；定义编译 pipeline。

2. `MOSSubtarget.h/.cpp`
- 聚合 InstrInfo/RegisterInfo/FrameLowering/Legalizer/CallLowering 等。
- 消费由 `.td` 生成的 Subtarget/Feature 信息。

## 4.2 指令相关

1. `MOSInstrInfo.h/.cpp`
- 目标指令语义接口实现（TargetInstrInfo）。

2. `MOSInstructionSelector.h/.cpp`
- GlobalISel 选择器，消费 `MOSGenGlobalISel.inc` 和指令/寄存器信息。

3. `MOSLegalizerInfo.h/.cpp`
- GlobalISel 合法化规则。

4. `MOSCombiner.h/.cpp`
- 消费 `MOSGenGICombiner.inc` 的 combine 规则实现。

## 4.3 寄存器与栈帧

1. `MOSRegisterInfo.h/.cpp`
- 目标寄存器接口实现。

2. `MOSRegisterBankInfo.h/.cpp`
- GlobalISel register bank 映射。

3. `MOSFrameLowering.h/.cpp`
- 栈帧布局、prologue/epilogue。

4. `MOSMachineFunctionInfo.h`
- 函数级 MOS 私有状态。

## 4.4 调用约定与调用 lowering

1. `MOSCallingConv.td` -> `MOSCallingConv.h/.cpp`
- C ABI 的寄存器分配/栈传递规则。

2. `MOSCallLowering.h/.cpp`
- GlobalISel 调用/返回 lowering。

## 4.5 MC/汇编/目标文件

1. `MOSAsmPrinter.cpp`
- CodeGen 与 MC 的边界；将 MachineInstr 输出为 MCInst/汇编。

2. `MOSMCInstLower.h/.cpp`
- MachineInstr -> MCInst lowering。

3. `MCTargetDesc/*`
- `MOSMCCodeEmitter`、`MOSAsmBackend`、`MOSFixupKinds`、`MOSELFObjectWriter`、`MOSInstPrinter`、`MOSMCExpr`、`MOSTargetStreamer`。

4. `AsmParser/MOSAsmParser.cpp`
- 汇编解析。

5. `Disassembler/MOSDisassembler.cpp`
- 反汇编。

6. `MOSTargetObjectFile.h/.cpp`
- 目标文件节区与对象布局策略。

## 4.6 机器级优化 Pass（完整清单）

1. IR/Loop 侧
- `MOSNonReentrant`
- `MOSIndexIV`（通过 PassBuilder 回调接入 loop pipeline）

2. GlobalISel 侧
- `MOSCombiner`
- `MOSShiftRotateChain`
- `MOSInternalize`
- `MOSLowerSelect`
- `MOSInsertCopies`

3. 机器后期侧
- `MOSCopyOpt`
- `MOSZeroPageAlloc`
- `MOSPostRAScavenging`
- `MOSLateOptimization`
- `MOSStaticStackAlloc`

以上由 `MOSTargetMachine.cpp` 中的 `MOSPassConfig` 与 `registerPassBuilderCallbacks` 串接。

## 5. 注释驱动的职责说明（关键文件）

以下为文件头注释可直接读出的职责（原文语义）：

1. `MOSTargetMachine.cpp`
- 定义 MOS 的 `TargetMachine` 子类，并配置后端 pass pipeline。

2. `MOSSubtarget.h`
- 声明 MOS 的 `TargetSubtargetInfo` 子类。

3. `MOSInstrInfo.cpp`
- MOS 的 `TargetInstrInfo` 实现。

4. `MOSRegisterInfo.cpp`
- MOS 的 `TargetRegisterInfo` 实现。

5. `MOSAsmPrinter.cpp`
- 将 MachineInstr/MCInst 路径输出为 MOS 汇编。

6. `MOSCallLowering.cpp`
- GlobalISel 调用/返回 lowering。

7. `MOSInstructionSelector.cpp`
- MOS 指令选择器（GlobalISel）。

8. `MOSLegalizerInfo.cpp`
- MOS 目标合法化规则（GlobalISel）。

9. `MOSCombine.td`
- MOS Combiner 规则集合。

10. `MOSCallingConv.td`
- MOS 调用约定规则。

说明：并非每个文件都写了同粒度头注释；在附录全量索引中，缺少明确头注释的条目采用“文件名语义 + 引用关系”给出职责。

## 6. 从 IR 到汇编/目标文件的数据流

1. LLVM IR（含常规中端优化）
2. `IRTranslator` 转为 Generic MIR
3. `MOSCombiner` / `MOSShiftRotateChain` 预合法化规范化
4. `Legalizer` + `MOSInternalize`
5. `MOSCombiner`（第二轮）+ `MOSLowerSelect`
6. `RegBankSelect`（`MOSRegisterBankInfo`）
7. `Localizer` + `InstructionSelect`（`MOSInstructionSelector`）
8. MachineSSA 优化 + `MOSInsertCopies`
9. RA 前后与后期优化（`MOSCopyOpt`、`MOSZeroPageAlloc`、`MOSPostRAScavenging`、`MOSLateOptimization`、`MOSStaticStackAlloc`）
10. `MOSAsmPrinter` + `MOSMCInstLower`
11. `MOSMCCodeEmitter` / `MOSAsmBackend` / `MOSELFObjectWriter`
12. 输出 `.s` 或 ELF 目标文件

## 7. Pass 时序表（以 MOSTargetMachine.cpp 为准）

## 7.1 Legacy CodeGen Pipeline（`MOSPassConfig`）

1. `addIRPasses`
- O1+：`MOSNonReentrant`
- Target 默认 IR passes
- O1+：`InstructionCombiningPass`

2. `addIRTranslator`
- `IRTranslator`

3. `addPreLegalizeMachineIR`
- O1+：`MOSCombiner`
- O1+：`MOSShiftRotateChain`

4. `addLegalizeMachineIR`
- `Legalizer`
- `MOSInternalize`

5. `addPreRegBankSelect`
- O1+：`MOSCombiner`
- `MOSLowerSelect`

6. `addRegBankSelect`
- `RegBankSelect`

7. `addPreGlobalInstructionSelect`
- `Localizer`

8. `addGlobalInstructionSelect`
- `InstructionSelect`

9. `addMachineSSAOptimization`
- Target 默认 MachineSSA passes
- O1+：`MOSInsertCopies`

10. `addOptimizedRegAlloc`
- O1+：在 `TwoAddressInstruction` 后插入一次 `RegisterCoalescer`
- O1+：在 `MachineScheduler` 后插入一次 `LiveIntervals`
- 再执行目标默认 optimized regalloc pipeline

11. `addMachineLateOptimization`
- Target 默认 late machine passes
- O1+：`MOSCopyOpt`

12. `addPrePEI`
- O1+：`MOSZeroPageAlloc`

13. `addPreSched2`
- `MOSPostRAScavenging`
- `FinalizeISel`
- `ExpandPostRAPseudos`
- `MOSPostRAScavenging`（第二次）
- `MOSLateOptimization`（当前为强制）
- O1+：`MOSStaticStackAlloc`

14. `addPreEmitPass`
- `BranchRelaxation`

## 7.2 NewPM 回调（`registerPassBuilderCallbacks`）

1. Loop pipeline 名称 `mos-indexiv`
- 加入 `MOSIndexIV`

2. Module pipeline 名称 `mos-nonreentrant`
- 加入 `MOSNonReentrantPass`

3. LateLoopOptimizations EP
- O1+：`MOSIndexIV`，随后 `IndVarSimplifyPass`

## 8. 全量文件索引（.h/.cpp/.td）与关系

本节覆盖 `llvm/lib/Target/MOS` 下全部 `.h/.cpp/.td` 文件，按子目录分组。

## 8.1 根目录：入口与基础设施

1. [MOS.h](MOS.h)
- 公共枚举/声明（目标地址空间、约定常量等）供全后端共享。

2. [MOSTargetMachine.h](MOSTargetMachine.h) / [MOSTargetMachine.cpp](MOSTargetMachine.cpp)
- 目标机入口；定义/注册 pass pipeline；连接 Subtarget 和 TTI。

3. [MOSSubtarget.h](MOSSubtarget.h) / [MOSSubtarget.cpp](MOSSubtarget.cpp)
- 子目标聚合层，持有 `InstrInfo`/`RegisterInfo`/`FrameLowering`/GISel 组件。
- 依赖 `MOSGenSubtargetInfo.inc`。

4. [MOSTargetTransformInfo.h](MOSTargetTransformInfo.h)
- 提供 MOS 的 TTI 实现入口。

5. [MOSTargetObjectFile.h](MOSTargetObjectFile.h) / [MOSTargetObjectFile.cpp](MOSTargetObjectFile.cpp)
- 定义节区/对象文件布局策略。

## 8.2 根目录：指令与寄存器核心

1. [MOSInstrInfo.h](MOSInstrInfo.h) / [MOSInstrInfo.cpp](MOSInstrInfo.cpp)
- 目标指令语义、copy/fold、分支与访存属性等。
- 依赖 `MOSGenInstrInfo.inc`。

2. [MOSRegisterInfo.h](MOSRegisterInfo.h) / [MOSRegisterInfo.cpp](MOSRegisterInfo.cpp)
- 物理寄存器分配约束、保留寄存器、frame index 消解。
- 依赖 `MOSGenRegisterInfo.inc`。

3. [MOSFrameLowering.h](MOSFrameLowering.h) / [MOSFrameLowering.cpp](MOSFrameLowering.cpp)
- Prologue/Epilogue 与栈帧布局。

4. [MOSMachineFunctionInfo.h](MOSMachineFunctionInfo.h)
- 函数级目标私有状态。

5. [MOSInstrBuilder.h](MOSInstrBuilder.h)
- 机器指令构建辅助接口。

6. [MOSInstrCost.h](MOSInstrCost.h) / [MOSInstrCost.cpp](MOSInstrCost.cpp)
- 指令代价模型与选择辅助。

## 8.3 根目录：GlobalISel 组件

1. [MOSCallLowering.h](MOSCallLowering.h) / [MOSCallLowering.cpp](MOSCallLowering.cpp)
- 调用/返回 lowering（与 `MOSCallingConv` 协同）。

2. [MOSLegalizerInfo.h](MOSLegalizerInfo.h) / [MOSLegalizerInfo.cpp](MOSLegalizerInfo.cpp)
- 操作合法化规则。

3. [MOSRegisterBankInfo.h](MOSRegisterBankInfo.h) / [MOSRegisterBankInfo.cpp](MOSRegisterBankInfo.cpp)
- RegBank 映射与 copy/legal bank 策略。
- 依赖 `MOSGenRegisterBank.inc`。

4. [MOSInstructionSelector.h](MOSInstructionSelector.h) / [MOSInstructionSelector.cpp](MOSInstructionSelector.cpp)
- 指令选择主逻辑。
- 依赖 `MOSGenGlobalISel.inc`。

5. [MOSCombiner.h](MOSCombiner.h) / [MOSCombiner.cpp](MOSCombiner.cpp)
- GICombiner C++ 胶水层。
- 依赖 `MOSGenGICombiner.inc`。

6. [MOSInlineAsmLowering.h](MOSInlineAsmLowering.h) / [MOSInlineAsmLowering.cpp](MOSInlineAsmLowering.cpp)
- GlobalISel 路径的 inline asm lowering。

7. [MOSISelLowering.h](MOSISelLowering.h) / [MOSISelLowering.cpp](MOSISelLowering.cpp)
- SelectionDAG 兼容层/辅助 lowering（与 GISel 并存）。

## 8.4 根目录：调用约定与调用图工具

1. [MOSCallingConv.h](MOSCallingConv.h) / [MOSCallingConv.cpp](MOSCallingConv.cpp)
- 调用约定执行逻辑；消费 TableGen 规则。
- 依赖 `MOSGenCallingConv.inc`。

2. [MOSCallGraphUtils.h](MOSCallGraphUtils.h) / [MOSCallGraphUtils.cpp](MOSCallGraphUtils.cpp)
- 调用图分析与工具函数，供后端优化/属性判定使用。

## 8.5 根目录：后端机器级优化与调度

1. [MOSCopyOpt.h](MOSCopyOpt.h) / [MOSCopyOpt.cpp](MOSCopyOpt.cpp)
- copy 消除/收缩。

2. [MOSInsertCopies.h](MOSInsertCopies.h) / [MOSInsertCopies.cpp](MOSInsertCopies.cpp)
- 插入必要 copy，改善后续分配与约束满足。

3. [MOSLowerSelect.h](MOSLowerSelect.h) / [MOSLowerSelect.cpp](MOSLowerSelect.cpp)
- lowering/规范化选择类伪操作。

4. [MOSLateOptimization.h](MOSLateOptimization.h) / [MOSLateOptimization.cpp](MOSLateOptimization.cpp)
- RA 后关键 peephole 与模式优化。

5. [MOSStaticStackAlloc.h](MOSStaticStackAlloc.h) / [MOSStaticStackAlloc.cpp](MOSStaticStackAlloc.cpp)
- 静态栈槽布局与固化。

6. [MOSZeroPageAlloc.h](MOSZeroPageAlloc.h) / [MOSZeroPageAlloc.cpp](MOSZeroPageAlloc.cpp)
- 零页对象分配策略。

7. [MOSPostRAScavenging.h](MOSPostRAScavenging.h) / [MOSPostRAScavenging.cpp](MOSPostRAScavenging.cpp)
- RA 后 scavenging 清理与修复。

8. [MOSShiftRotateChain.h](MOSShiftRotateChain.h) / [MOSShiftRotateChain.cpp](MOSShiftRotateChain.cpp)
- 移位/旋转链优化。

9. [MOSIndexIV.h](MOSIndexIV.h) / [MOSIndexIV.cpp](MOSIndexIV.cpp)
- 循环 IV 重写（偏向 8-bit offset）。

10. [MOSNonReentrant.h](MOSNonReentrant.h) / [MOSNonReentrant.cpp](MOSNonReentrant.cpp)
- 非重入属性与相关变换。

11. [MOSInternalize.h](MOSInternalize.h) / [MOSInternalize.cpp](MOSInternalize.cpp)
- 内部化与可见性/局部化相关变换。

12. [MOSMachineScheduler.h](MOSMachineScheduler.h) / [MOSMachineScheduler.cpp](MOSMachineScheduler.cpp)
- 机器调度策略实现（寄存器压力导向）。

## 8.6 根目录：汇编打印与 MC Lower

1. [MOSAsmPrinter.cpp](MOSAsmPrinter.cpp)
- MachineInstr 到汇编输出的桥接。
- 依赖 `MOSGenAsmWriter.inc`、`MOSGenMCPseudoLowering.inc`。

2. [MOSMCInstLower.h](MOSMCInstLower.h) / [MOSMCInstLower.cpp](MOSMCInstLower.cpp)
- `MachineOperand`/`MachineInstr` 到 `MCOperand`/`MCInst`。

3. [MOSModifierNames.h](MOSModifierNames.h)
- 修饰符名称声明（与 MCTargetDesc 中实现配合）。

## 8.7 根目录：TableGen 源（全部）

1. [MOS.td](MOS.td)
- 顶层装配入口。

2. [MOSFeatures.td](MOSFeatures.td)
- 特性位定义。

3. [MOSDevices.td](MOSDevices.td)
- 设备/CPU 族定义。

4. [MOSRegisterInfo.td](MOSRegisterInfo.td)
- 寄存器与寄存器类。

5. [MOSRegisterBanks.td](MOSRegisterBanks.td)
- GlobalISel register bank。

6. [MOSInstrInfo.td](MOSInstrInfo.td)
- 指令总入口。

7. [MOSInstrFormats.td](MOSInstrFormats.td)
- 指令格式和编码位布局。

8. [MOSInstrInfoTables.td](MOSInstrInfoTables.td)
- SearchableTable（如 relax 对照）。

9. [MOSInstrInfoSPC700.td](MOSInstrInfoSPC700.td)
- SPC700 变体。

10. [MOSInstrPseudos.td](MOSInstrPseudos.td)
- 伪指令。

11. [MOSInstrLogical.td](MOSInstrLogical.td)
- 逻辑层指令定义。

12. [MOSInstrGISel.td](MOSInstrGISel.td)
- GISel 扩展与匹配规则。

13. [MOSCombine.td](MOSCombine.td)
- GICombiner 规则。

14. [MOSCallingConv.td](MOSCallingConv.td)
- 调用约定规则。

## 8.8 AsmParser

1. [AsmParser/MOSAsmParser.cpp](AsmParser/MOSAsmParser.cpp)
- 汇编语法解析，驱动 `MOSGenAsmMatcher.inc`。

## 8.9 Disassembler

1. [Disassembler/MOSDisassembler.cpp](Disassembler/MOSDisassembler.cpp)
- 反汇编实现，消费 `MOSGenDisassemblerTables.inc`。

## 8.10 MCTargetDesc（全部）

1. [MCTargetDesc/MOSMCTargetDesc.h](MCTargetDesc/MOSMCTargetDesc.h) / [MCTargetDesc/MOSMCTargetDesc.cpp](MCTargetDesc/MOSMCTargetDesc.cpp)
- MC 目标描述入口与注册。

2. [MCTargetDesc/MOSMCAsmInfo.h](MCTargetDesc/MOSMCAsmInfo.h) / [MCTargetDesc/MOSMCAsmInfo.cpp](MCTargetDesc/MOSMCAsmInfo.cpp)
- 汇编语法/指令打印基础属性。

3. [MCTargetDesc/MOSInstPrinter.h](MCTargetDesc/MOSInstPrinter.h) / [MCTargetDesc/MOSInstPrinter.cpp](MCTargetDesc/MOSInstPrinter.cpp)
- MCInst 到文本汇编。

4. [MCTargetDesc/MOSMCCodeEmitter.h](MCTargetDesc/MOSMCCodeEmitter.h) / [MCTargetDesc/MOSMCCodeEmitter.cpp](MCTargetDesc/MOSMCCodeEmitter.cpp)
- MCInst 到机器码比特流。
- 依赖 `MOSGenMCCodeEmitter.inc`。

5. [MCTargetDesc/MOSAsmBackend.h](MCTargetDesc/MOSAsmBackend.h) / [MCTargetDesc/MOSAsmBackend.cpp](MCTargetDesc/MOSAsmBackend.cpp)
- fixup 应用、relax、目标 backend 规则。

6. [MCTargetDesc/MOSFixupKinds.h](MCTargetDesc/MOSFixupKinds.h) / [MCTargetDesc/MOSFixupKinds.cpp](MCTargetDesc/MOSFixupKinds.cpp)
- fixup kind 枚举与解释。

7. [MCTargetDesc/MOSELFObjectWriter.h](MCTargetDesc/MOSELFObjectWriter.h) / [MCTargetDesc/MOSELFObjectWriter.cpp](MCTargetDesc/MOSELFObjectWriter.cpp)
- ELF 重定位与对象写出策略。

8. [MCTargetDesc/MOSMCExpr.h](MCTargetDesc/MOSMCExpr.h) / [MCTargetDesc/MOSMCExpr.cpp](MCTargetDesc/MOSMCExpr.cpp)
- MOS 特化 `MCExpr`（目标表达式语义）。

9. [MCTargetDesc/MOSMCELFStreamer.h](MCTargetDesc/MOSMCELFStreamer.h) / [MCTargetDesc/MOSMCELFStreamer.cpp](MCTargetDesc/MOSMCELFStreamer.cpp)
- ELF streamer 扩展。

10. [MCTargetDesc/MOSTargetStreamer.h](MCTargetDesc/MOSTargetStreamer.h) / [MCTargetDesc/MOSTargetStreamer.cpp](MCTargetDesc/MOSTargetStreamer.cpp)
- 目标 streamer 钩子层。

11. [MCTargetDesc/MOSMCInstrAnalysis.h](MCTargetDesc/MOSMCInstrAnalysis.h) / [MCTargetDesc/MOSMCInstrAnalysis.cpp](MCTargetDesc/MOSMCInstrAnalysis.cpp)
- MC 指令级分析辅助（分支/长度等）。

12. [MCTargetDesc/MOSModifierNames.cpp](MCTargetDesc/MOSModifierNames.cpp)
- 修饰符名称字符串映射实现。

## 8.11 TargetInfo

1. [TargetInfo/MOSTargetInfo.h](TargetInfo/MOSTargetInfo.h) / [TargetInfo/MOSTargetInfo.cpp](TargetInfo/MOSTargetInfo.cpp)
- 目标注册入口，绑定 `Triple::mos` 到 `Target`。

## 9. 维护建议

1. 改 ISA 编码/寻址模式：优先改 `MOSInstrFormats.td` + `MOSInstrInfo.td`，再核对 `MOSMCCodeEmitter.cpp`/`MOSAsmBackend.cpp`。
2. 改调用约定：先改 `MOSCallingConv.td`，再对齐 `MOSCallingConv.cpp`/`MOSCallLowering.cpp`，最后补 `llvm/test/CodeGen/MOS`。
3. 改 GISel 选择：优先看 `MOSInstrGISel.td`、`MOSInstructionSelector.cpp`、`MOSLegalizerInfo.cpp`、`MOSCombiner.cpp`。
4. 改机器后期优化：从 `MOSTargetMachine.cpp` 的时序表定位插入点，再改对应 `MOS*` Pass。
5. 改 MC/汇编行为：入口通常在 `MOSAsmPrinter.cpp`、`MOSMCInstLower.cpp`、`MCTargetDesc/*` 三层联动。
