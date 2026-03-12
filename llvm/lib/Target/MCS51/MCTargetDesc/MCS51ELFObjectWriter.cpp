//===-- MCS51ELFObjectWriter.cpp - MCS51 ELF Writer ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/MCS51FixupKinds.h"
#include "MCTargetDesc/MCS51MCExpr.h"
#include "MCTargetDesc/MCS51MCTargetDesc.h"

#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

/// Writes MCS51 machine code into an ELF32 object file.
class MCS51ELFObjectWriter : public MCELFObjectTargetWriter {
public:
  MCS51ELFObjectWriter(uint8_t OSABI);

  ~MCS51ELFObjectWriter() override = default;

  unsigned getRelocType(const MCFixup &, const MCValue &,
                        bool IsPCRel) const override;
};

MCS51ELFObjectWriter::MCS51ELFObjectWriter(uint8_t OSABI)
    : MCELFObjectTargetWriter(false, OSABI, ELF::EM_AVR, true) {}

unsigned MCS51ELFObjectWriter::getRelocType(const MCFixup &Fixup,
                                            const MCValue &Target,
                                            bool IsPCRel) const {
  auto Spec = Target.getSpecifier();
  switch ((unsigned)Fixup.getKind()) {
  case FK_Data_1:
    (void)Spec;
    return ELF::R_AVR_8;
  case FK_Data_4:
    return ELF::R_AVR_32;
  case FK_Data_2:
    return Spec ? ELF::R_AVR_16_PM : ELF::R_AVR_16;
  case MCS51::fixup_32:
    return ELF::R_AVR_32;
  case MCS51::fixup_7_pcrel:
    return ELF::R_AVR_7_PCREL;
  case MCS51::fixup_13_pcrel:
    return ELF::R_AVR_13_PCREL;
  case MCS51::fixup_16:
    return ELF::R_AVR_16;
  case MCS51::fixup_16_pm:
    return ELF::R_AVR_16_PM;
  case MCS51::fixup_lo8_ldi:
    return ELF::R_AVR_LO8_LDI;
  case MCS51::fixup_hi8_ldi:
    return ELF::R_AVR_HI8_LDI;
  case MCS51::fixup_hh8_ldi:
    return ELF::R_AVR_HH8_LDI;
  case MCS51::fixup_lo8_ldi_neg:
    return ELF::R_AVR_LO8_LDI_NEG;
  case MCS51::fixup_hi8_ldi_neg:
    return ELF::R_AVR_HI8_LDI_NEG;
  case MCS51::fixup_hh8_ldi_neg:
    return ELF::R_AVR_HH8_LDI_NEG;
  case MCS51::fixup_lo8_ldi_pm:
    return ELF::R_AVR_LO8_LDI_PM;
  case MCS51::fixup_hi8_ldi_pm:
    return ELF::R_AVR_HI8_LDI_PM;
  case MCS51::fixup_hh8_ldi_pm:
    return ELF::R_AVR_HH8_LDI_PM;
  case MCS51::fixup_lo8_ldi_pm_neg:
    return ELF::R_AVR_LO8_LDI_PM_NEG;
  case MCS51::fixup_hi8_ldi_pm_neg:
    return ELF::R_AVR_HI8_LDI_PM_NEG;
  case MCS51::fixup_hh8_ldi_pm_neg:
    return ELF::R_AVR_HH8_LDI_PM_NEG;
  case MCS51::fixup_call:
    return ELF::R_AVR_CALL;
  case MCS51::fixup_ldi:
    return ELF::R_AVR_LDI;
  case MCS51::fixup_6:
    return ELF::R_AVR_6;
  case MCS51::fixup_6_adiw:
    return ELF::R_AVR_6_ADIW;
  case MCS51::fixup_ms8_ldi:
    return ELF::R_AVR_MS8_LDI;
  case MCS51::fixup_ms8_ldi_neg:
    return ELF::R_AVR_MS8_LDI_NEG;
  case MCS51::fixup_lo8_ldi_gs:
    return ELF::R_AVR_LO8_LDI_GS;
  case MCS51::fixup_hi8_ldi_gs:
    return ELF::R_AVR_HI8_LDI_GS;
  case MCS51::fixup_8:
    return ELF::R_AVR_8;
  case MCS51::fixup_8_lo8:
    return ELF::R_AVR_8_LO8;
  case MCS51::fixup_8_hi8:
    return ELF::R_AVR_8_HI8;
  case MCS51::fixup_8_hlo8:
    return ELF::R_AVR_8_HLO8;
  case MCS51::fixup_diff8:
    return ELF::R_AVR_DIFF8;
  case MCS51::fixup_diff16:
    return ELF::R_AVR_DIFF16;
  case MCS51::fixup_diff32:
    return ELF::R_AVR_DIFF32;
  case MCS51::fixup_lds_sts_16:
    return ELF::R_AVR_LDS_STS_16;
  case MCS51::fixup_port6:
    return ELF::R_AVR_PORT6;
  case MCS51::fixup_port5:
    return ELF::R_AVR_PORT5;
  default:
    llvm_unreachable("invalid fixup kind!");
  }
}

std::unique_ptr<MCObjectTargetWriter> createMCS51ELFObjectWriter(uint8_t OSABI) {
  return std::make_unique<MCS51ELFObjectWriter>(OSABI);
}

} // end of namespace llvm
