; RUN: llvm-mc -triple avr -arch=mcs51 -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Phase-0 stack smoke: push/pop accumulator spelling is bridged through the
; temporary R16 mapping and now available with default mcs51 SRAM feature.

        push a
        pop a
        ret

; CHECK: push{{[[:space:]]+}}r16
; CHECK: pop{{[[:space:]]+}}r16
; CHECK: ret
