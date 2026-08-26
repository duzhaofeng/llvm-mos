; RUN: llvm-mc -triple avr -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Native 8051 jump/call forms.

        sjmp target
        ajmp target
        ljmp target
        acall target
        lcall target
target:
        ret

; CHECK: sjmp{{[[:space:]]+}}target
; CHECK: ajmp{{[[:space:]]+}}target
; CHECK: ljmp{{[[:space:]]+}}target
; CHECK: acall{{[[:space:]]+}}target
; CHECK: lcall{{[[:space:]]+}}target
; CHECK: ret