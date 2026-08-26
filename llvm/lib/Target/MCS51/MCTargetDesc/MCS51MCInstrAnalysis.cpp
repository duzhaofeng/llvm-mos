//===-- MCS51MCInstrAnalysis.cpp - MCS51 instruction analysis -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MCS51-specific MC instruction analysis.
//
//===----------------------------------------------------------------------===//

#include "MCS51MCInstrAnalysis.h"

#include "MCS51MCTargetDesc.h"

#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"

namespace llvm {

bool MCS51MCInstrAnalysis::evaluateBranch(const MCInst &Inst, uint64_t Addr,
                                          uint64_t Size,
                                          uint64_t &Target) const {
  if ((!isBranch(Inst) && !isCall(Inst)) || isIndirectBranch(Inst))
    return false;

  unsigned NumOps = Inst.getNumOperands();
  if (NumOps == 0)
    return false;

  const auto &Op = Info->get(Inst.getOpcode()).operands()[NumOps - 1];
  switch (Op.OperandType) {
  case MCS51Op::OPERAND_ADDR16:
    // LJMP/LCALL: 16-bit absolute target.
    Target =
        (Addr & 0xFFFF0000) | (Inst.getOperand(NumOps - 1).getImm() & 0xFFFF);
    return true;
  case MCS51Op::OPERAND_ADDR11:
    // AJMP/ACALL: 11-bit target within the current 2 KiB page. The page is
    // taken from the address of the following instruction (Addr + Size).
    Target = ((Addr + Size) & 0xF800) |
             (Inst.getOperand(NumOps - 1).getImm() & 0x7FF);
    return true;
  case MCOI::OPERAND_PCREL:
    // SJMP/JZ/JNZ/JC/JNC/JB/JNB/JBC/CJNE/DJNZ: 8-bit signed displacement
    // relative to the following instruction.
    Target = Addr + Size + Inst.getOperand(NumOps - 1).getImm();
    return true;
  }
  return false;
}

} // namespace llvm
