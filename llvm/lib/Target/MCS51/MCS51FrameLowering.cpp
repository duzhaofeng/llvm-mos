//===-- MCS51FrameLowering.cpp - MCS51 Frame Information ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the MCS51 implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "MCS51FrameLowering.h"

#include "MCS51.h"
#include "MCS51InstrInfo.h"
#include "MCS51MachineFunctionInfo.h"
#include "MCS51TargetMachine.h"
#include "MCTargetDesc/MCS51MCTargetDesc.h"

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"

namespace llvm {

MCS51FrameLowering::MCS51FrameLowering()
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, Align(1), -2) {}

bool MCS51FrameLowering::canSimplifyCallFramePseudos(
    const MachineFunction &MF) const {
  // Always simplify call frame pseudo instructions, even when
  // hasReservedCallFrame is false.
  return true;
}

bool MCS51FrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  // Reserve call frame memory in function prologue under the following
  // conditions:
  // - Y pointer is reserved to be the frame pointer.
  // - The function does not contain variable sized objects.

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return hasFP(MF) && !MFI.hasVarSizedObjects();
}

void MCS51FrameLowering::emitPrologue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc DL = (MBBI != MBB.end()) ? MBBI->getDebugLoc() : DebugLoc();
  const MCS51Subtarget &STI = MF.getSubtarget<MCS51Subtarget>();
  const MCS51InstrInfo &TII = *STI.getInstrInfo();
  const MCS51MachineFunctionInfo *AFI = MF.getInfo<MCS51MachineFunctionInfo>();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  bool HasFP = hasFP(MF);

  // Interrupt handlers re-enable interrupts in function entry.
  if (AFI->isInterruptHandler()) {
    BuildMI(MBB, MBBI, DL, TII.get(MCS51::BSETs))
        .addImm(0x07)
        .setMIFlag(MachineInstr::FrameSetup);
  }

  // Emit special prologue code to save R1, R0 and SREG in interrupt/signal
  // handlers before saving any other registers.
  if (AFI->isInterruptOrSignalHandler()) {
    BuildMI(MBB, MBBI, DL, TII.get(MCS51::PUSHRr))
        .addReg(STI.getTmpRegister(), RegState::Kill)
        .setMIFlag(MachineInstr::FrameSetup);

    BuildMI(MBB, MBBI, DL, TII.get(MCS51::INRdA), STI.getTmpRegister())
        .addImm(STI.getIORegSREG())
        .setMIFlag(MachineInstr::FrameSetup);
    BuildMI(MBB, MBBI, DL, TII.get(MCS51::PUSHRr))
        .addReg(STI.getTmpRegister(), RegState::Kill)
        .setMIFlag(MachineInstr::FrameSetup);
    if (!MRI.reg_empty(STI.getZeroRegister())) {
      BuildMI(MBB, MBBI, DL, TII.get(MCS51::PUSHRr))
          .addReg(STI.getZeroRegister(), RegState::Kill)
          .setMIFlag(MachineInstr::FrameSetup);
      BuildMI(MBB, MBBI, DL, TII.get(MCS51::EORRdRr))
          .addReg(STI.getZeroRegister(), RegState::Define)
          .addReg(STI.getZeroRegister(), RegState::Kill)
          .addReg(STI.getZeroRegister(), RegState::Kill)
          .setMIFlag(MachineInstr::FrameSetup);
    }
  }

  // Early exit if the frame pointer is not needed in this function.
  if (!HasFP) {
    return;
  }

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  unsigned FrameSize = MFI.getStackSize() - AFI->getCalleeSavedFrameSize();

  // Skip the callee-saved push instructions.
  while (
      (MBBI != MBB.end()) && MBBI->getFlag(MachineInstr::FrameSetup) &&
      (MBBI->getOpcode() == MCS51::PUSHRr || MBBI->getOpcode() == MCS51::PUSHWRr)) {
    ++MBBI;
  }

  // Update Y with the new base value.
  BuildMI(MBB, MBBI, DL, TII.get(MCS51::SPREAD), MCS51::R29R28)
      .addReg(MCS51::SP)
      .setMIFlag(MachineInstr::FrameSetup);

  // Mark the FramePtr as live-in in every block except the entry.
  for (MachineBasicBlock &MBBJ : llvm::drop_begin(MF)) {
    MBBJ.addLiveIn(MCS51::R29R28);
  }

  if (!FrameSize) {
    return;
  }

  // Reserve the necessary frame memory by doing FP -= <size>.
  unsigned Opcode = (isUInt<6>(FrameSize) && STI.hasADDSUBIW()) ? MCS51::SBIWRdK
                                                                : MCS51::SUBIWRdK;

  MachineInstr *MI = BuildMI(MBB, MBBI, DL, TII.get(Opcode), MCS51::R29R28)
                         .addReg(MCS51::R29R28, RegState::Kill)
                         .addImm(FrameSize)
                         .setMIFlag(MachineInstr::FrameSetup);
  // The SREG implicit def is dead.
  MI->getOperand(3).setIsDead();

  // Write back R29R28 to SP and temporarily disable interrupts.
  BuildMI(MBB, MBBI, DL, TII.get(MCS51::SPWRITE), MCS51::SP)
      .addReg(MCS51::R29R28)
      .setMIFlag(MachineInstr::FrameSetup);
}

static void restoreStatusRegister(MachineFunction &MF, MachineBasicBlock &MBB) {
  const MCS51MachineFunctionInfo *AFI = MF.getInfo<MCS51MachineFunctionInfo>();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();

  DebugLoc DL = MBBI->getDebugLoc();
  const MCS51Subtarget &STI = MF.getSubtarget<MCS51Subtarget>();
  const MCS51InstrInfo &TII = *STI.getInstrInfo();

  // Emit special epilogue code to restore R1, R0 and SREG in interrupt/signal
  // handlers at the very end of the function, just before reti.
  if (AFI->isInterruptOrSignalHandler()) {
    if (!MRI.reg_empty(STI.getZeroRegister())) {
      BuildMI(MBB, MBBI, DL, TII.get(MCS51::POPRd), STI.getZeroRegister());
    }
    BuildMI(MBB, MBBI, DL, TII.get(MCS51::POPRd), STI.getTmpRegister());
    BuildMI(MBB, MBBI, DL, TII.get(MCS51::OUTARr))
        .addImm(STI.getIORegSREG())
        .addReg(STI.getTmpRegister(), RegState::Kill);
    BuildMI(MBB, MBBI, DL, TII.get(MCS51::POPRd), STI.getTmpRegister());
  }
}

void MCS51FrameLowering::emitEpilogue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  const MCS51MachineFunctionInfo *AFI = MF.getInfo<MCS51MachineFunctionInfo>();

  // Early exit if the frame pointer is not needed in this function except for
  // signal/interrupt handlers where special code generation is required.
  if (!hasFP(MF) && !AFI->isInterruptOrSignalHandler()) {
    return;
  }

  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  assert(MBBI->getDesc().isReturn() &&
         "Can only insert epilog into returning blocks");

  DebugLoc DL = MBBI->getDebugLoc();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  unsigned FrameSize = MFI.getStackSize() - AFI->getCalleeSavedFrameSize();
  const MCS51Subtarget &STI = MF.getSubtarget<MCS51Subtarget>();
  const MCS51InstrInfo &TII = *STI.getInstrInfo();

  // Early exit if there is no need to restore the frame pointer.
  if (!FrameSize && !MF.getFrameInfo().hasVarSizedObjects()) {
    restoreStatusRegister(MF, MBB);
    return;
  }

  // Skip the callee-saved pop instructions.
  while (MBBI != MBB.begin()) {
    MachineBasicBlock::iterator PI = std::prev(MBBI);
    int Opc = PI->getOpcode();

    if (Opc != MCS51::POPRd && Opc != MCS51::POPWRd && !PI->isTerminator()) {
      break;
    }

    --MBBI;
  }

  if (FrameSize) {
    unsigned Opcode;

    // Select the optimal opcode depending on how big it is.
    if (isUInt<6>(FrameSize) && STI.hasADDSUBIW()) {
      Opcode = MCS51::ADIWRdK;
    } else {
      Opcode = MCS51::SUBIWRdK;
      FrameSize = -FrameSize;
    }

    // Restore the frame pointer by doing FP += <size>.
    MachineInstr *MI = BuildMI(MBB, MBBI, DL, TII.get(Opcode), MCS51::R29R28)
                           .addReg(MCS51::R29R28, RegState::Kill)
                           .addImm(FrameSize);
    // The SREG implicit def is dead.
    MI->getOperand(3).setIsDead();
  }

  // Write back R29R28 to SP and temporarily disable interrupts.
  BuildMI(MBB, MBBI, DL, TII.get(MCS51::SPWRITE), MCS51::SP)
      .addReg(MCS51::R29R28, RegState::Kill);

  restoreStatusRegister(MF, MBB);
}

// Return true if the specified function should have a dedicated frame
// pointer register. This is true if the function meets any of the following
// conditions:
//  - a register has been spilled
//  - has allocas
//  - input arguments are passed using the stack
//
// Notice that strictly this is not a frame pointer because it contains SP after
// frame allocation instead of having the original SP in function entry.
bool MCS51FrameLowering::hasFP(const MachineFunction &MF) const {
  const MCS51MachineFunctionInfo *FuncInfo = MF.getInfo<MCS51MachineFunctionInfo>();

  return (FuncInfo->getHasSpills() || FuncInfo->getHasAllocas() ||
          FuncInfo->getHasStackArgs() ||
          MF.getFrameInfo().hasVarSizedObjects());
}

bool MCS51FrameLowering::spillCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  if (CSI.empty()) {
    return false;
  }

  unsigned CalleeFrameSize = 0;
  DebugLoc DL = MBB.findDebugLoc(MI);
  MachineFunction &MF = *MBB.getParent();
  const MCS51Subtarget &STI = MF.getSubtarget<MCS51Subtarget>();
  const TargetInstrInfo &TII = *STI.getInstrInfo();
  MCS51MachineFunctionInfo *MCS51FI = MF.getInfo<MCS51MachineFunctionInfo>();

  for (const CalleeSavedInfo &I : llvm::reverse(CSI)) {
    Register Reg = I.getReg();
    bool IsNotLiveIn = !MBB.isLiveIn(Reg);

    // Check if Reg is a sub register of a 16-bit livein register, and then
    // add it to the livein list.
    if (IsNotLiveIn)
      for (const auto &LiveIn : MBB.liveins())
        if (STI.getRegisterInfo()->isSubRegister(LiveIn.PhysReg, Reg)) {
          IsNotLiveIn = false;
          MBB.addLiveIn(Reg);
          break;
        }

    assert(TRI->getRegSizeInBits(*TRI->getMinimalPhysRegClass(Reg)) == 8 &&
           "Invalid register size");

    // Add the callee-saved register as live-in only if it is not already a
    // live-in register, this usually happens with arguments that are passed
    // through callee-saved registers.
    if (IsNotLiveIn) {
      MBB.addLiveIn(Reg);
    }

    // Do not kill the register when it is an input argument.
    BuildMI(MBB, MI, DL, TII.get(MCS51::PUSHRr))
        .addReg(Reg, getKillRegState(IsNotLiveIn))
        .setMIFlag(MachineInstr::FrameSetup);
    ++CalleeFrameSize;
  }

  MCS51FI->setCalleeSavedFrameSize(CalleeFrameSize);

  return true;
}

bool MCS51FrameLowering::restoreCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    MutableArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  if (CSI.empty()) {
    return false;
  }

  DebugLoc DL = MBB.findDebugLoc(MI);
  const MachineFunction &MF = *MBB.getParent();
  const MCS51Subtarget &STI = MF.getSubtarget<MCS51Subtarget>();
  const TargetInstrInfo &TII = *STI.getInstrInfo();

  for (const CalleeSavedInfo &CCSI : CSI) {
    Register Reg = CCSI.getReg();

    assert(TRI->getRegSizeInBits(*TRI->getMinimalPhysRegClass(Reg)) == 8 &&
           "Invalid register size");

    BuildMI(MBB, MI, DL, TII.get(MCS51::POPRd), Reg);
  }

  return true;
}

/// Replace pseudo store instructions that pass arguments through the stack with
/// real instructions.
static void fixStackStores(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator StartMI,
                           const TargetInstrInfo &TII) {
  // Iterate through the BB until we hit a call instruction or we reach the end.
  for (MachineInstr &MI :
       llvm::make_early_inc_range(llvm::make_range(StartMI, MBB.end()))) {
    if (MI.isCall())
      break;

    unsigned Opcode = MI.getOpcode();

    // Only care of pseudo store instructions where SP is the base pointer.
    if (Opcode != MCS51::STDSPQRr && Opcode != MCS51::STDWSPQRr)
      continue;

    assert(MI.getOperand(0).getReg() == MCS51::SP &&
           "SP is expected as base pointer");

    // Replace this instruction with a regular store. Use Y as the base
    // pointer since it is guaranteed to contain a copy of SP.
    unsigned STOpc =
        (Opcode == MCS51::STDWSPQRr) ? MCS51::STDWPtrQRr : MCS51::STDPtrQRr;

    MI.setDesc(TII.get(STOpc));
    MI.getOperand(0).setReg(MCS51::R31R30);
  }
}

MachineBasicBlock::iterator MCS51FrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MI) const {
  const MCS51Subtarget &STI = MF.getSubtarget<MCS51Subtarget>();
  const MCS51InstrInfo &TII = *STI.getInstrInfo();

  if (hasReservedCallFrame(MF)) {
    return MBB.erase(MI);
  }

  DebugLoc DL = MI->getDebugLoc();
  unsigned int Opcode = MI->getOpcode();
  int Amount = TII.getFrameSize(*MI);

  if (Amount == 0) {
    return MBB.erase(MI);
  }

  assert(getStackAlign() == Align(1) && "Unsupported stack alignment");

  if (Opcode == TII.getCallFrameSetupOpcode()) {
    // Update the stack pointer.
    // In many cases this can be done far more efficiently by pushing the
    // relevant values directly to the stack. However, doing that correctly
    // (in the right order, possibly skipping some empty space for undef
    // values, etc) is tricky and thus left to be optimized in the future.
    BuildMI(MBB, MI, DL, TII.get(MCS51::SPREAD), MCS51::R31R30).addReg(MCS51::SP);

    MachineInstr *New =
        BuildMI(MBB, MI, DL, TII.get(MCS51::SUBIWRdK), MCS51::R31R30)
            .addReg(MCS51::R31R30, RegState::Kill)
            .addImm(Amount);
    New->getOperand(3).setIsDead();

    BuildMI(MBB, MI, DL, TII.get(MCS51::SPWRITE), MCS51::SP).addReg(MCS51::R31R30);

    // Make sure the remaining stack stores are converted to real store
    // instructions.
    fixStackStores(MBB, MI, TII);
  } else {
    assert(Opcode == TII.getCallFrameDestroyOpcode());

    // Note that small stack changes could be implemented more efficiently
    // with a few pop instructions instead of the 8-9 instructions now
    // required.

    // Select the best opcode to adjust SP based on the offset size.
    unsigned AddOpcode;

    if (isUInt<6>(Amount) && STI.hasADDSUBIW()) {
      AddOpcode = MCS51::ADIWRdK;
    } else {
      AddOpcode = MCS51::SUBIWRdK;
      Amount = -Amount;
    }

    // Build the instruction sequence.
    BuildMI(MBB, MI, DL, TII.get(MCS51::SPREAD), MCS51::R31R30).addReg(MCS51::SP);

    MachineInstr *New = BuildMI(MBB, MI, DL, TII.get(AddOpcode), MCS51::R31R30)
                            .addReg(MCS51::R31R30, RegState::Kill)
                            .addImm(Amount);
    New->getOperand(3).setIsDead();

    BuildMI(MBB, MI, DL, TII.get(MCS51::SPWRITE), MCS51::SP)
        .addReg(MCS51::R31R30, RegState::Kill);
  }

  return MBB.erase(MI);
}

void MCS51FrameLowering::determineCalleeSaves(MachineFunction &MF,
                                            BitVector &SavedRegs,
                                            RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);

  // If we have a frame pointer, the Y register needs to be saved as well.
  if (hasFP(MF)) {
    SavedRegs.set(MCS51::R29);
    SavedRegs.set(MCS51::R28);
  }
}
/// The frame analyzer pass.
///
/// Scans the function for allocas and used arguments
/// that are passed through the stack.
struct MCS51FrameAnalyzer : public MachineFunctionPass {
  static char ID;
  MCS51FrameAnalyzer() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    const MachineFrameInfo &MFI = MF.getFrameInfo();
    MCS51MachineFunctionInfo *AFI = MF.getInfo<MCS51MachineFunctionInfo>();

    // If there are no fixed frame indexes during this stage it means there
    // are allocas present in the function.
    if (MFI.getNumObjects() != MFI.getNumFixedObjects()) {
      // Check for the type of allocas present in the function. We only care
      // about fixed size allocas so do not give false positives if only
      // variable sized allocas are present.
      for (unsigned i = 0, e = MFI.getObjectIndexEnd(); i != e; ++i) {
        // Variable sized objects have size 0.
        if (MFI.getObjectSize(i)) {
          AFI->setHasAllocas(true);
          break;
        }
      }
    }

    // If there are fixed frame indexes present, scan the function to see if
    // they are really being used.
    if (MFI.getNumFixedObjects() == 0) {
      return false;
    }

    // Ok fixed frame indexes present, now scan the function to see if they
    // are really being used, otherwise we can ignore them.
    for (const MachineBasicBlock &BB : MF) {
      for (const MachineInstr &MI : BB) {
        int Opcode = MI.getOpcode();

        if ((Opcode != MCS51::LDDRdPtrQ) && (Opcode != MCS51::LDDWRdPtrQ) &&
            (Opcode != MCS51::STDPtrQRr) && (Opcode != MCS51::STDWPtrQRr) &&
            (Opcode != MCS51::FRMIDX)) {
          continue;
        }

        for (const MachineOperand &MO : MI.operands()) {
          if (!MO.isFI()) {
            continue;
          }

          if (MFI.isFixedObjectIndex(MO.getIndex())) {
            AFI->setHasStackArgs(true);
            return false;
          }
        }
      }
    }

    return false;
  }

  StringRef getPassName() const override { return "MCS51 Frame Analyzer"; }
};

char MCS51FrameAnalyzer::ID = 0;

/// Creates instance of the frame analyzer pass.
FunctionPass *createMCS51FrameAnalyzerPass() { return new MCS51FrameAnalyzer(); }

} // end of namespace llvm
