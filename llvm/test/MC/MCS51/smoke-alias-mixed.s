; RUN: llvm-mc -triple avr -arch=mcs51 -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Phase-0 alias/case matrix: mixed-case mnemonics and mixed a/acc, c/cy forms
; should stay accepted by parser bridge rewrites.

        mOv Acc, #0x11
        aDd AcC, #0x01
        SeTb cY
        cLr C
        jB cY, target
        jNb C, target
        pUsH Acc
        pOp a

target:
        rEt

; CHECK: ldi{{[[:space:]]+}}r16, 17
; CHECK: subi{{[[:space:]]+}}r16, -1
; CHECK: sec
; CHECK: clc
; CHECK: brbs{{[[:space:]]+}}0,
; CHECK: brbc{{[[:space:]]+}}0,
; CHECK: push{{[[:space:]]+}}r16
; CHECK: pop{{[[:space:]]+}}r16
; CHECK: ret
