//===-- MCS51TargetObjectFile.cpp - MCS51 Object Files ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCS51TargetObjectFile.h"
#include "MCS51TargetMachine.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"

#include "MCS51.h"

namespace llvm {
void MCS51TargetObjectFile::Initialize(MCContext &Ctx, const TargetMachine &TM) {
  Base::Initialize(Ctx, TM);
  ProgmemDataSection =
      Ctx.getELFSection(".progmem.data", ELF::SHT_PROGBITS, ELF::SHF_ALLOC);
  Progmem1DataSection =
      Ctx.getELFSection(".progmem1.data", ELF::SHT_PROGBITS, ELF::SHF_ALLOC);
  Progmem2DataSection =
      Ctx.getELFSection(".progmem2.data", ELF::SHT_PROGBITS, ELF::SHF_ALLOC);
  Progmem3DataSection =
      Ctx.getELFSection(".progmem3.data", ELF::SHT_PROGBITS, ELF::SHF_ALLOC);
  Progmem4DataSection =
      Ctx.getELFSection(".progmem4.data", ELF::SHT_PROGBITS, ELF::SHF_ALLOC);
  Progmem5DataSection =
      Ctx.getELFSection(".progmem5.data", ELF::SHT_PROGBITS, ELF::SHF_ALLOC);
}

MCSection *MCS51TargetObjectFile::SelectSectionForGlobal(
    const GlobalObject *GO, SectionKind Kind, const TargetMachine &TM) const {
  // Global values in flash memory are placed in the progmem*.data section
  // unless they already have a user assigned section.
  const auto &MCS51TM = static_cast<const MCS51TargetMachine &>(TM);
  if (MCS51::isProgramMemoryAddress(GO) && !GO->hasSection() &&
      Kind.isReadOnly()) {
    // The MCS51 subtarget should support LPM to access section '.progmem*.data'.
    if (!MCS51TM.getSubtargetImpl()->hasLPM()) {
      // TODO: Get the global object's location in source file.
      getContext().reportError(
          SMLoc(),
          "Current MCS51 subtarget does not support accessing program memory");
      return Base::SelectSectionForGlobal(GO, Kind, TM);
    }
    // The MCS51 subtarget should support ELPM to access section
    // '.progmem[1|2|3|4|5].data'.
    if (!MCS51TM.getSubtargetImpl()->hasELPM() &&
        MCS51::getAddressSpace(GO) != MCS51::ProgramMemory) {
      // TODO: Get the global object's location in source file.
      getContext().reportError(SMLoc(),
                               "Current MCS51 subtarget does not support "
                               "accessing extended program memory");
      return ProgmemDataSection;
    }
    switch (MCS51::getAddressSpace(GO)) {
    case MCS51::ProgramMemory: // address space 1
      return ProgmemDataSection;
    case MCS51::ProgramMemory1: // address space 2
      return Progmem1DataSection;
    case MCS51::ProgramMemory2: // address space 3
      return Progmem2DataSection;
    case MCS51::ProgramMemory3: // address space 4
      return Progmem3DataSection;
    case MCS51::ProgramMemory4: // address space 5
      return Progmem4DataSection;
    case MCS51::ProgramMemory5: // address space 6
      return Progmem5DataSection;
    default:
      llvm_unreachable("unexpected program memory index");
    }
  }

  // Otherwise, we work the same way as ELF.
  return Base::SelectSectionForGlobal(GO, Kind, TM);
}
} // end of namespace llvm
