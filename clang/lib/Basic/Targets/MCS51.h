//===--- MCS51.h - Declare MCS51 target feature support -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares MCS51 TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_MCS51_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_MCS51_H

#include "AVR.h"

namespace clang {
namespace targets {

class LLVM_LIBRARY_VISIBILITY MCS51TargetInfo : public AVRTargetInfo {
public:
  MCS51TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : AVRTargetInfo(Triple, Opts) {
    resetDataLayout("e-P1-p:16:8-i8:8-i16:8-i32:8-i64:8-f32:8-f64:8-n8-a:8");
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  bool isValidCPUName(StringRef Name) const override;
  void fillValidCPUList(SmallVectorImpl<StringRef> &Values) const override;
  bool setCPU(const std::string &Name) override;
};

} // namespace targets
} // namespace clang

#endif // LLVM_CLANG_LIB_BASIC_TARGETS_MCS51_H
