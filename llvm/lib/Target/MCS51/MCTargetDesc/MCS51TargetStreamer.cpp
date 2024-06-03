//===-- MCS51TargetStreamer.cpp - MCS51 Target Streamer Methods ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides MCS51 specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "MCS51TargetStreamer.h"

#include "llvm/MC/MCContext.h"

namespace llvm {

MCS51TargetStreamer::MCS51TargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}

MCS51TargetAsmStreamer::MCS51TargetAsmStreamer(MCStreamer &S)
    : MCS51TargetStreamer(S) {}

} // end namespace llvm
