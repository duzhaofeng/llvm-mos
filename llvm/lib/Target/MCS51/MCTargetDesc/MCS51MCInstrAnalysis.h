//===-- MCS51MCInstrAnalysis.h - MCS51 instruction analysis -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCS51-specific MC instruction analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MCS51_MC_INSTR_ANALYSIS_H
#define LLVM_MCS51_MC_INSTR_ANALYSIS_H

#include "llvm/MC/MCInstrAnalysis.h"

namespace llvm {

class MCS51MCInstrAnalysis : public MCInstrAnalysis {
public:
  explicit MCS51MCInstrAnalysis(const MCInstrInfo *Info)
      : MCInstrAnalysis(Info) {}

  bool evaluateBranch(const MCInst &Inst, uint64_t Addr, uint64_t Size,
                      uint64_t &Target) const override;
};

} // namespace llvm

#endif // LLVM_MCS51_MC_INSTR_ANALYSIS_H
