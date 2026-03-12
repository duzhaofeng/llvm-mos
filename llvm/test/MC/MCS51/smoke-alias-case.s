; RUN: llvm-mc -triple avr -arch=mcs51 -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Phase-0 alias/case smoke: verify case-insensitive alias spellings for
; accumulator and carry flag across bridge rewrites.

        MOV ACC, #0x2A
        CPL ACC
        SETB CY
        CLR CY
        JB CY, target
        JNB CY, target
        PUSH ACC
        POP ACC

target:
        RET

; CHECK: ldi{{[[:space:]]+}}r16, 42
; CHECK: cpl{{[[:space:]]+}}r16
; CHECK: sec
; CHECK: clc
; CHECK: jb{{[[:space:]]+}}0, target
; CHECK: jnb{{[[:space:]]+}}0, target
; CHECK: push{{[[:space:]]+}}r16
; CHECK: pop{{[[:space:]]+}}r16
; CHECK: ret
