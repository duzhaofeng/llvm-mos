//===------ SemaMCS51.cpp -------- MCS51 target-specific routines --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements semantic analysis functions specific to MCS51.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaMCS51.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringRef.h"

using namespace clang;

SemaMCS51::SemaMCS51(Sema &S) : SemaBase(S) {}

bool SemaMCS51::isMCS51CPU() const {
  llvm::StringRef CPU = getASTContext().getTargetInfo().getTargetOpts().CPU;
  return CPU == "mcs51" || CPU == "mcs251";
}

bool SemaMCS51::hasAnnotation(const VarDecl *VD,
                              llvm::StringRef Annotation) const {
  if (!VD)
    return false;

  for (const auto *Attr : VD->specific_attrs<AnnotateAttr>()) {
    if (Attr->getAnnotation() == Annotation)
      return true;
  }
  return false;
}

bool SemaMCS51::isBitAddressDecl(const VarDecl *VD) const {
  return isMCS51CPU() && hasAnnotation(VD, "mcs51_sbit");
}

bool SemaMCS51::isBitAddressBaseDecl(const VarDecl *VD) const {
  return isMCS51CPU() && hasAnnotation(VD, "mcs51_sfr");
}

bool SemaMCS51::isAddressSpaceAllowedInFunctionLocal(LangAS AS) const {
  if (!isMCS51CPU() || AS == LangAS::Default)
    return false;

  unsigned TargetAS = getASTContext().getTargetAddressSpace(AS);
  return TargetAS == 1 || TargetAS == 2 || TargetAS == 3 || TargetAS == 5 ||
         TargetAS == 6;
}

bool SemaMCS51::isAddressSpaceAllowedInFunctionCompoundLiteral(LangAS AS) const {
  return isAddressSpaceAllowedInFunctionLocal(AS);
}

bool SemaMCS51::isFileScopeOnlyAnnotatedDecl(const VarDecl *VD) const {
  return isMCS51CPU() && hasAnnotation(VD, "mcs51_sbit");
}

bool SemaMCS51::normalizeBitAddressInitializer(VarDecl *VDecl, Expr *&Init) {
  if (!isBitAddressDecl(VDecl))
    return true;

  auto EvalConstInt = [&](const Expr *E, llvm::APSInt &Out) {
    Expr::EvalResult R;
    if (!E->EvaluateAsInt(R, getASTContext()))
      return false;
    Out = R.Val.getInt();
    return true;
  };
  auto CheckBitAddressRange = [&](llvm::APSInt Addr, SourceLocation Loc) {
    if (Addr.isSigned() && Addr.isNegative()) {
      Diag(Loc, diag::err_mcs51_sbit_address_range);
      return false;
    }
    llvm::APSInt Max(Addr.getBitWidth(), Addr.isUnsigned());
    Max = 255;
    if (Addr > Max) {
      Diag(Loc, diag::err_mcs51_sbit_address_range);
      return false;
    }
    return true;
  };

  Expr *CoreInit = Init->IgnoreParenImpCasts();
  if (const auto *BO = dyn_cast<BinaryOperator>(CoreInit);
      BO && BO->getOpcode() == BO_Xor) {
    const Expr *LHSE = BO->getLHS()->IgnoreParenImpCasts();
    const Expr *RHSE = BO->getRHS()->IgnoreParenImpCasts();
    const auto *LHSDeclRef = dyn_cast<DeclRefExpr>(LHSE);
    const auto *SFRDecl =
        LHSDeclRef ? dyn_cast<VarDecl>(LHSDeclRef->getDecl()) : nullptr;
    if (!SFRDecl || !isBitAddressBaseDecl(SFRDecl) || !SFRDecl->hasInit()) {
      Diag(Init->getExprLoc(), diag::err_mcs51_sbit_invalid_initializer)
          << Init->getSourceRange();
      VDecl->setInvalidDecl();
      return false;
    }

    llvm::APSInt SFRAddr(32);
    if (!EvalConstInt(SFRDecl->getInit()->IgnoreParenImpCasts(), SFRAddr) ||
        !CheckBitAddressRange(SFRAddr, SFRDecl->getLocation())) {
      VDecl->setInvalidDecl();
      return false;
    }

    llvm::APSInt BitIndex(32);
    if (!EvalConstInt(RHSE, BitIndex)) {
      Diag(Init->getExprLoc(), diag::err_mcs51_sbit_invalid_initializer)
          << Init->getSourceRange();
      VDecl->setInvalidDecl();
      return false;
    }

    if (BitIndex.isSigned() && BitIndex.isNegative()) {
      Diag(RHSE->getExprLoc(), diag::err_mcs51_sbit_bit_index_range);
      VDecl->setInvalidDecl();
      return false;
    }
    llvm::APSInt BitMax(BitIndex.getBitWidth(), BitIndex.isUnsigned());
    BitMax = 7;
    if (BitIndex > BitMax) {
      Diag(RHSE->getExprLoc(), diag::err_mcs51_sbit_bit_index_range);
      VDecl->setInvalidDecl();
      return false;
    }

    llvm::APSInt Normalized = SFRAddr;
    Normalized += BitIndex;
    if (!CheckBitAddressRange(Normalized, Init->getExprLoc())) {
      VDecl->setInvalidDecl();
      return false;
    }

    Init = IntegerLiteral::Create(getASTContext(), Normalized,
                                  getASTContext().IntTy, Init->getExprLoc());
    return true;
  }

  llvm::APSInt Addr(32);
  if (!EvalConstInt(CoreInit, Addr)) {
    Diag(Init->getExprLoc(), diag::err_mcs51_sbit_invalid_initializer)
        << Init->getSourceRange();
    VDecl->setInvalidDecl();
    return false;
  }
  if (!CheckBitAddressRange(Addr, Init->getExprLoc())) {
    VDecl->setInvalidDecl();
    return false;
  }

  return true;
}

bool SemaMCS51::checkBitAddressDeclHasInitializer(VarDecl *VDecl) {
  if (!isBitAddressDecl(VDecl))
    return true;

  if (VDecl->hasInit())
    return true;

  Diag(VDecl->getLocation(), diag::err_mcs51_sbit_requires_initializer);
  VDecl->setInvalidDecl();
  return false;
}
