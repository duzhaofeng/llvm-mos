//===----- SemaMCS51.h ------ MCS51 target-specific routines ---*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis functions specific to MCS51.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMAMCS51_H
#define LLVM_CLANG_SEMA_SEMAMCS51_H

#include "clang/Sema/SemaBase.h"

namespace llvm {
class StringRef;
}

namespace clang {
class Expr;
enum class LangAS : unsigned;
class VarDecl;

class SemaMCS51 : public SemaBase {
public:
  SemaMCS51(Sema &S);

  bool isAddressSpaceAllowedInFunctionLocal(LangAS AS) const;
  bool isAddressSpaceAllowedInFunctionCompoundLiteral(LangAS AS) const;

  bool isFileScopeOnlyAnnotatedDecl(const VarDecl *VD) const;

  bool normalizeBitAddressInitializer(VarDecl *VDecl, Expr *&Init);
  bool checkBitAddressDeclHasInitializer(VarDecl *VDecl);

private:
  bool isMCS51CPU() const;
  bool hasAnnotation(const VarDecl *VD, llvm::StringRef Annotation) const;
  bool isBitAddressDecl(const VarDecl *VD) const;
  bool isBitAddressBaseDecl(const VarDecl *VD) const;
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAMCS51_H
