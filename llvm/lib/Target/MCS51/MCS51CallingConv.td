//===-- MCS51CallingConv.td - Calling Conventions for MCS51 ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This describes the calling conventions for the 8051 architecture (MCS51),
// modeled on the SDCC/Keil ABI: arguments and return values are passed in the
// working registers R0..R7 (the first i16 in R7:R6), with further arguments
// passed in fixed memory (data) locations. Normal functions use a special
// calling convention, solved in code.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MCS51 Return Value Calling Convention
//===----------------------------------------------------------------------===//

// Special return value calling convention for runtime functions.
def RetCC_MCS51_BUILTIN : CallingConv<[
  CCIfType<[i8], CCAssignToReg<[R24, R25]>>,
  CCIfType<[i16], CCAssignToReg<[R23R22, R25R24]>>
]>;

// Native 8051 return values, SDCC/Keil style: the first i8 in R7, the first
// i16 in R7:R6, then R5:R4, R3:R2 and R1:R0. Everything stays within the
// 8-byte R0..R7 window, so no data-RAM memory locations are needed for
// return values (they will be, only if a future ABI outgrows R0..R7).
def RetCC_MCS51 : CallingConv<[
  CCIfType<[i8], CCAssignToReg<[R7, R6, R5, R4, R3, R2, R1, R0]>>,
  CCIfType<[i16], CCAssignToReg<[R6R7, R4R5, R2R3, R0R1]>>
]>;

//===----------------------------------------------------------------------===//
// MCS51 Argument Calling Conventions
//===----------------------------------------------------------------------===//

// The ordinary calling convention is implemented in custom C++ code

// Calling convention for variadic functions. The fixed (named) arguments are
// handled by the custom C++ code; the variadic arguments are always passed on
// the stack (byte-aligned), as the 8051 register window cannot accommodate an
// arbitrary number of variable arguments.
def ArgCC_MCS51_Vararg : CallingConv<[
  // i8 are always passed through the stack with a byte slot and byte alignment.
  CCIfType<[i8], CCAssignToStack<1, 1>>,
  // i16 are always passed through the stack with an alignment of 1.
  CCAssignToStack<2, 1>
]>;

//===----------------------------------------------------------------------===//
// Callee-saved register lists.
//===----------------------------------------------------------------------===//

// Legacy AVR-derived callee-saved lists, retained only for the AVR-derived
// CodeGen path that still exists in this backend. The 8051 has no hardware
// register-window save/restore convention; the Phase-2 native frame handling
// will instead save/restore ACC, B, PSW and DPTR explicitly.
def CSR_Normal : CalleeSavedRegs<(add R29, R28, (sequence "R%u", 17, 2))>;
def CSR_NormalTiny : CalleeSavedRegs<(add R29, R28, R19, R18)>;
def CSR_Interrupts : CalleeSavedRegs<(add(sequence "R%u", 31, 2))>;
def CSR_InterruptsTiny : CalleeSavedRegs<(add(sequence "R%u", 31, 18))>;
