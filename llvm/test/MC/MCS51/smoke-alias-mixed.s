; RUN: llvm-mc -triple avr -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Mixed-case mnemonics and operand spellings.

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

; CHECK: mov{{[[:space:]]+}}a, #17
; CHECK: encoding: [0x74,0x11]
; CHECK: add{{[[:space:]]+}}a, #1
; CHECK: encoding: [0x24,0x01]
; CHECK: setb{{[[:space:]]+}}c
; CHECK: encoding: [0xd3]
; CHECK: clr{{[[:space:]]+}}c
; CHECK: encoding: [0xc3]
; CHECK: jb{{[[:space:]]+}}PSW.7, target
; CHECK: encoding: [0x20,0xd7,0x00]
; CHECK: jnb{{[[:space:]]+}}PSW.7, target
; CHECK: encoding: [0x30,0xd7,0x00]
; CHECK: push{{[[:space:]]+}}ACC
; CHECK: encoding: [0xc0,0xe0]
; CHECK: pop{{[[:space:]]+}}ACC
; CHECK: encoding: [0xd0,0xe0]
; CHECK: ret
