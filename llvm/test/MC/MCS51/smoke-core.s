; RUN: llvm-mc -triple avr -arch=mcs51 -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Minimal end-to-end sanity test kept as a stable fast check.

        mov a, #0x12
        sjmp .
func:
        ret

; CHECK: ldi{{[[:space:]]+}}r16, 18
; CHECK: sjmp
; CHECK: ret
