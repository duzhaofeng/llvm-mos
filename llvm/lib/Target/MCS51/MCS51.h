//===-- MCS51.h - Top-level interface for MCS51 representation ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// MCS51 back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MCS51_H
#define LLVM_MCS51_H

#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class MCS51TargetMachine;
class FunctionPass;
class PassRegistry;

Pass *createMCS51ShiftExpandPass();
FunctionPass *createMCS51ISelDag(MCS51TargetMachine &TM, CodeGenOptLevel OptLevel);
FunctionPass *createMCS51ExpandPseudoPass();
FunctionPass *createMCS51FrameAnalyzerPass();
FunctionPass *createMCS51BranchSelectionPass();

void initializeMCS51DAGToDAGISelPass(PassRegistry &);
void initializeMCS51ExpandPseudoPass(PassRegistry &);
void initializeMCS51ShiftExpandPass(PassRegistry &);

/// Contains the MCS51 backend.
namespace MCS51 {

/// An integer that identifies all of the supported MCS51 address spaces.
enum AddressSpace {
  DataMemory,
  ProgramMemory,
  ProgramMemory1,
  ProgramMemory2,
  ProgramMemory3,
  ProgramMemory4,
  ProgramMemory5,
  NumAddrSpaces,
};

/// Checks if a given type is a pointer to program memory.
template <typename T> bool isProgramMemoryAddress(T *V) {
  auto *PT = cast<PointerType>(V->getType());
  assert(PT != nullptr && "unexpected MemSDNode");
  return PT->getAddressSpace() == ProgramMemory ||
         PT->getAddressSpace() == ProgramMemory1 ||
         PT->getAddressSpace() == ProgramMemory2 ||
         PT->getAddressSpace() == ProgramMemory3 ||
         PT->getAddressSpace() == ProgramMemory4 ||
         PT->getAddressSpace() == ProgramMemory5;
}

template <typename T> AddressSpace getAddressSpace(T *V) {
  auto *PT = cast<PointerType>(V->getType());
  assert(PT != nullptr && "unexpected MemSDNode");
  unsigned AS = PT->getAddressSpace();
  if (AS < NumAddrSpaces)
    return static_cast<AddressSpace>(AS);
  return NumAddrSpaces;
}

inline bool isProgramMemoryAccess(MemSDNode const *N) {
  auto *V = N->getMemOperand()->getValue();
  if (V != nullptr && isProgramMemoryAddress(V))
    return true;
  return false;
}

// Get the index of the program memory bank.
//  -1: not program memory
//   0: ordinary program memory
// 1~5: extended program memory
inline int getProgramMemoryBank(MemSDNode const *N) {
  auto *V = N->getMemOperand()->getValue();
  if (V == nullptr || !isProgramMemoryAddress(V))
    return -1;
  AddressSpace AS = getAddressSpace(V);
  assert(ProgramMemory <= AS && AS <= ProgramMemory5);
  return static_cast<int>(AS - ProgramMemory);
}

} // end of namespace MCS51

} // end namespace llvm

#endif // LLVM_MCS51_H
