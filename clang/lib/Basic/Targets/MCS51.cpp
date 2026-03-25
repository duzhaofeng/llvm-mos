//===--- MCS51.cpp - Implement MCS51 target feature support --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCS51.h"
#include "clang/Basic/MacroBuilder.h"

using namespace clang;
using namespace clang::targets;

namespace {
constexpr llvm::StringLiteral MCS51CPUNames[] = {"mcs51", "mcs251"};
} // namespace

bool MCS51TargetInfo::isValidCPUName(StringRef Name) const {
  for (llvm::StringLiteral CPUName : MCS51CPUNames) {
    if (CPUName == Name)
      return true;
  }
  return false;
}

void MCS51TargetInfo::fillValidCPUList(
    SmallVectorImpl<StringRef> &Values) const {
  for (llvm::StringLiteral CPUName : MCS51CPUNames)
    Values.push_back(CPUName);
}

bool MCS51TargetInfo::setCPU(const std::string &Name) {
  if (!isValidCPUName(Name))
    return false;

  CPU = Name;
  ABI = "mcs51";
  DefineName = "";
  Arch = (Name == "mcs251") ? "251" : "51";
  NumFlashBanks = 0;
  return true;
}

void MCS51TargetInfo::getTargetDefines(const LangOptions &Opts,
                                       MacroBuilder &Builder) const {
  Builder.defineMacro("__MCS51__", "1");
  if (CPU == "mcs251")
    Builder.defineMacro("__MCS251__", "1");

  Builder.defineMacro("__MCS51_ARCH__", Arch);
  Builder.defineMacro("__data", "__attribute__((__address_space__(1)))");
  Builder.defineMacro("__idata", "__attribute__((__address_space__(2)))");
  Builder.defineMacro("__xdata", "__attribute__((__address_space__(3)))");
  Builder.defineMacro("__pdata", "__attribute__((__address_space__(5)))");
  Builder.defineMacro("__bdata", "__attribute__((__address_space__(6)))");
  Builder.defineMacro("__bit", "__bdata _Bool");
  Builder.defineMacro("__code", "__attribute__((__address_space__(4)))");
  Builder.defineMacro(
      "__sfr",
      "__attribute__((annotate(\"mcs51_sfr\"))) "
      "__attribute__((__address_space__(7))) volatile unsigned char");
  Builder.defineMacro(
      "__sfr16",
      "__attribute__((__address_space__(7))) volatile unsigned short");
  Builder.defineMacro(
      "__sbit",
      "__attribute__((annotate(\"mcs51_sbit\"))) "
      "__attribute__((__address_space__(7))) volatile unsigned char");
}
