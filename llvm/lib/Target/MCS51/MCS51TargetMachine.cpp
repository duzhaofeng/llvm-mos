//===-- MCS51TargetMachine.cpp - Define TargetMachine for MCS51 ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MCS51 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#include "MCS51TargetMachine.h"

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"

#include "MCS51.h"
#include "MCS51MachineFunctionInfo.h"
#include "MCS51TargetObjectFile.h"
#include "MCTargetDesc/MCS51MCTargetDesc.h"
#include "TargetInfo/MCS51TargetInfo.h"

#include <optional>

namespace llvm {

static const char *MCS51DataLayout =
    "e-P1-p:16:8-i8:8-i16:8-i32:8-i64:8-f32:8-f64:8-n8-a:8";

/// Processes a CPU name.
static StringRef getCPU(StringRef CPU) {
  if (CPU.empty() || CPU == "generic") {
    return "avr2";
  }

  return CPU;
}

static Reloc::Model getEffectiveRelocModel(std::optional<Reloc::Model> RM) {
  return RM.value_or(Reloc::Static);
}

MCS51TargetMachine::MCS51TargetMachine(const Target &T, const Triple &TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   std::optional<Reloc::Model> RM,
                                   std::optional<CodeModel::Model> CM,
                                   CodeGenOptLevel OL, bool JIT)
    : LLVMTargetMachine(T, MCS51DataLayout, TT, getCPU(CPU), FS, Options,
                        getEffectiveRelocModel(RM),
                        getEffectiveCodeModel(CM, CodeModel::Small), OL),
      SubTarget(TT, std::string(getCPU(CPU)), std::string(FS), *this) {
  this->TLOF = std::make_unique<MCS51TargetObjectFile>();
  initAsmInfo();
}

namespace {
/// MCS51 Code Generator Pass Configuration Options.
class MCS51PassConfig : public TargetPassConfig {
public:
  MCS51PassConfig(MCS51TargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  MCS51TargetMachine &getMCS51TargetMachine() const {
    return getTM<MCS51TargetMachine>();
  }

  void addIRPasses() override;
  bool addInstSelector() override;
  void addPreSched2() override;
  void addPreEmitPass() override;
};
} // namespace

TargetPassConfig *MCS51TargetMachine::createPassConfig(PassManagerBase &PM) {
  return new MCS51PassConfig(*this, PM);
}

void MCS51PassConfig::addIRPasses() {
  // Expand instructions like
  //   %result = shl i32 %n, %amount
  // to a loop so that library calls are avoided.
  addPass(createMCS51ShiftExpandPass());

  TargetPassConfig::addIRPasses();
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeMCS51Target() {
  // Register the target.
  RegisterTargetMachine<MCS51TargetMachine> X(getTheMCS51Target());

  auto &PR = *PassRegistry::getPassRegistry();
  initializeMCS51ExpandPseudoPass(PR);
  initializeMCS51ShiftExpandPass(PR);
  initializeMCS51DAGToDAGISelPass(PR);
}

const MCS51Subtarget *MCS51TargetMachine::getSubtargetImpl() const {
  return &SubTarget;
}

const MCS51Subtarget *MCS51TargetMachine::getSubtargetImpl(const Function &) const {
  return &SubTarget;
}

MachineFunctionInfo *MCS51TargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return MCS51MachineFunctionInfo::create<MCS51MachineFunctionInfo>(Allocator, F,
                                                                STI);
}

//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

bool MCS51PassConfig::addInstSelector() {
  // Install an instruction selector.
  addPass(createMCS51ISelDag(getMCS51TargetMachine(), getOptLevel()));
  // Create the frame analyzer pass used by the PEI pass.
  addPass(createMCS51FrameAnalyzerPass());

  return false;
}

void MCS51PassConfig::addPreSched2() {
  addPass(createMCS51ExpandPseudoPass());
}

void MCS51PassConfig::addPreEmitPass() {
  // Must run branch selection immediately preceding the asm printer.
  addPass(&BranchRelaxationPassID);
}

} // end of namespace llvm
