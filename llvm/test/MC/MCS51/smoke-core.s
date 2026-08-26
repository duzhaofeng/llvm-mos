; RUN: llvm-mc -triple avr -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Minimal end-to-end sanity test kept as a stable fast check.

        mov a, #0x12
        sjmp .
func:
        ret

; CHECK: mov{{[[:space:]]+}}a, #18
; CHECK: encoding: [0x74,0x12]
; CHECK: sjmp
; CHECK: ret
