//===-- MCS51InstPrinter.cpp - Convert MCS51 MCInst to assembly syntax --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class prints an MCS51 MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "MCS51InstPrinter.h"

#include "MCTargetDesc/MCS51MCTargetDesc.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"

#include <cstring>

#define DEBUG_TYPE "asm-printer"

namespace llvm {

// Include the auto-generated portion of the assembly writer.
#define PRINT_ALIAS_INSTR
#include "MCS51GenAsmWriter.inc"

void MCS51InstPrinter::printInst(const MCInst *MI, uint64_t Address,
                               StringRef Annot, const MCSubtargetInfo &STI,
                               raw_ostream &O) {
  unsigned Opcode = MI->getOpcode();

  // First handle load and store instructions with postinc or predec
  // of the form "ld reg, X+".
  // TODO: We should be able to rewrite this using TableGen data.
  switch (Opcode) {
  case MCS51::LDRdPtr:
  case MCS51::LDRdPtrPi:
  case MCS51::LDRdPtrPd:
    O << "\tld\t";
    printOperand(MI, 0, O);
    O << ", ";

    if (Opcode == MCS51::LDRdPtrPd)
      O << '-';

    printOperand(MI, 1, O);

    if (Opcode == MCS51::LDRdPtrPi)
      O << '+';
    break;
  case MCS51::STPtrRr:
    O << "\tst\t";
    printOperand(MI, 0, O);
    O << ", ";
    printOperand(MI, 1, O);
    break;
  case MCS51::STPtrPiRr:
  case MCS51::STPtrPdRr:
    O << "\tst\t";

    if (Opcode == MCS51::STPtrPdRr)
      O << '-';

    printOperand(MI, 1, O);

    if (Opcode == MCS51::STPtrPiRr)
      O << '+';

    O << ", ";
    printOperand(MI, 2, O);
    break;
  default:
    if (!printAliasInstr(MI, Address, O))
      printInstruction(MI, Address, O);

    printAnnotation(O, Annot);
    break;
  }
}

const char *MCS51InstPrinter::getPrettyRegisterName(unsigned RegNum,
                                                  MCRegisterInfo const &MRI) {
  // GCC prints register pairs by just printing the lower register
  // If the register contains a subregister, print it instead
  if (MRI.getNumSubRegIndices() > 0) {
    unsigned RegLoNum = MRI.getSubReg(RegNum, MCS51::sub_lo);
    RegNum = (RegLoNum != MCS51::NoRegister) ? RegLoNum : RegNum;
  }

  return getRegisterName(RegNum);
}

void MCS51InstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                  raw_ostream &O) {
  const MCOperandInfo &MOI = this->MII.get(MI->getOpcode()).operands()[OpNo];
  if (MOI.RegClass == MCS51::ZREGRegClassID) {
    // Special case for the Z register, which sometimes doesn't have an operand
    // in the MCInst.
    O << "Z";
    return;
  }

  if (OpNo >= MI->size()) {
    // Not all operands are correctly disassembled at the moment. This means
    // that some machine instructions won't have all the necessary operands
    // set.
    // To avoid asserting, print <unknown> instead until the necessary support
    // has been implemented.
    O << "<unknown>";
    return;
  }

  const MCOperand &Op = MI->getOperand(OpNo);

  if (Op.isReg()) {
    bool isPtrReg = (MOI.RegClass == MCS51::PTRREGSRegClassID) ||
                    (MOI.RegClass == MCS51::PTRDISPREGSRegClassID) ||
                    (MOI.RegClass == MCS51::ZREGRegClassID);

    if (isPtrReg) {
      O << getRegisterName(Op.getReg(), MCS51::ptr);
    } else {
      O << getPrettyRegisterName(Op.getReg(), MRI);
    }
  } else if (Op.isImm()) {
    O << formatImm(Op.getImm());
  } else {
    assert(Op.isExpr() && "Unknown operand kind in printOperand");
    MAI.printExpr(O, *Op.getExpr());
  }
}

/// This is used to print an immediate value that ends up
/// being encoded as a pc-relative value.
void MCS51InstPrinter::printPCRelImm(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
  if (OpNo >= MI->size()) {
    // Not all operands are correctly disassembled at the moment. This means
    // that some machine instructions won't have all the necessary operands
    // set.
    // To avoid asserting, print <unknown> instead until the necessary support
    // has been implemented.
    O << "<unknown>";
    return;
  }

  const MCOperand &Op = MI->getOperand(OpNo);

  if (Op.isImm()) {
    int64_t Imm = Op.getImm();
    O << '.';

    // Print a position sign if needed.
    // Negative values have their sign printed automatically.
    if (Imm >= 0)
      O << '+';

    O << Imm;
  } else {
    assert(Op.isExpr() && "Unknown pcrel immediate operand");
    MAI.printExpr(O, *Op.getExpr());
  }
}

void MCS51InstPrinter::printMemri(const MCInst *MI, unsigned OpNo,
                                raw_ostream &O) {
  assert(MI->getOperand(OpNo).isReg() &&
         "Expected a register for the first operand");

  const MCOperand &OffsetOp = MI->getOperand(OpNo + 1);

  // Print the register.
  printOperand(MI, OpNo, O);

  // Print the {+,-}offset.
  if (OffsetOp.isImm()) {
    int64_t Offset = OffsetOp.getImm();

    if (Offset >= 0)
      O << '+';

    O << Offset;
  } else if (OffsetOp.isExpr()) {
    MAI.printExpr(O, *OffsetOp.getExpr());
  } else {
    llvm_unreachable("unknown type for offset");
  }
}

/// Return the name of the Special Function Register at byte address \p Addr,
/// or an empty StringRef if \p Addr is not a standard 8051 SFR.
static StringRef getSFRName(unsigned Addr) {
  switch (Addr) {
  case 0x80: return "P0";
  case 0x81: return "SP";
  case 0x82: return "DPL";
  case 0x83: return "DPH";
  case 0x87: return "PCON";
  case 0x88: return "TCON";
  case 0x89: return "TMOD";
  case 0x8A: return "TL0";
  case 0x8B: return "TL1";
  case 0x8C: return "TH0";
  case 0x8D: return "TH1";
  case 0x90: return "P1";
  case 0x98: return "SCON";
  case 0x99: return "SBUF";
  case 0xA0: return "P2";
  case 0xA8: return "IE";
  case 0xB0: return "P3";
  case 0xB8: return "IP";
  case 0xD0: return "PSW";
  case 0xE0: return "ACC";
  case 0xF0: return "B";
  default: return StringRef();
  }
}

/// Return the name of the bit-addressable SFR whose byte address is
/// \p ByteAddr, or an empty StringRef if that byte is not bit-addressable
/// on the standard 8051.
static StringRef getBitAddressableSFRName(unsigned ByteAddr) {
  switch (ByteAddr) {
  case 0x80: return "P0";
  case 0x88: return "TCON";
  case 0x90: return "P1";
  case 0x98: return "SCON";
  case 0xA0: return "P2";
  case 0xA8: return "IE";
  case 0xB0: return "P3";
  case 0xB8: return "IP";
  case 0xD0: return "PSW";
  case 0xE0: return "ACC";
  case 0xF0: return "B";
  default: return StringRef();
  }
}

void MCS51InstPrinter::printMCS51Direct(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (!Op.isImm()) {
    printOperand(MI, OpNo, O);
    return;
  }

  unsigned Addr = Op.getImm();
  StringRef Name = getSFRName(Addr);
  if (!Name.empty())
    O << Name;
  else
    O << formatImm(Addr);
}

void MCS51InstPrinter::printMCS51Bit(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (!Op.isImm()) {
    printOperand(MI, OpNo, O);
    return;
  }

  unsigned BitAddr = Op.getImm();
  unsigned BitNo = BitAddr & 0x7;

  if (BitAddr < 0x80) {
    // Internal RAM bit: byte address 0x20 + (BitAddr / 8).
    unsigned ByteAddr = 0x20 + (BitAddr >> 3);
    O << "0x" << utohexstr(ByteAddr, /*LowerCase=*/true) << '.' << BitNo;
  } else if (StringRef Name = getBitAddressableSFRName(BitAddr & 0xF8);
             !Name.empty()) {
    O << Name << '.' << BitNo;
  } else {
    O << formatImm(BitAddr);
  }
}

void MCS51InstPrinter::printMCS51Imm(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O) {
  // 8051 immediates are spelled with a leading '#'.
  O << '#';
  printOperand(MI, OpNo, O);
}

} // end of namespace llvm
