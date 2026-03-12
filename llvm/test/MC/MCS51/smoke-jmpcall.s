; RUN: llvm-mc -triple avr -arch=mcs51 -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Phase-0 jmp/call smoke: default mcs51 now enables jmp/call support, and
; ljmp/lcall bridge to those absolute forms.

        jmp target
        call target
        ljmp target
        ajmp target
        acall target
        lcall target
target:
        ret

; CHECK: jmp{{[[:space:]]+}}target
; CHECK: call{{[[:space:]]+}}target
; CHECK: jmp{{[[:space:]]+}}target
; CHECK: call{{[[:space:]]+}}target
; CHECK: ret