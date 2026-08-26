; RUN: llvm-mc -triple avr -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; push/pop of the accumulator (SFR 0xE0).

        push a
        pop a
        ret

; CHECK: push{{[[:space:]]+}}ACC
; CHECK: encoding: [0xc0,0xe0]
; CHECK: pop{{[[:space:]]+}}ACC
; CHECK: encoding: [0xd0,0xe0]
; CHECK: ret
