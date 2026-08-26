; RUN: not llvm-mc -triple avr -mcpu=mcs51 < %s 2>&1 | FileCheck %s --check-prefix=ERR
;
; Unknown register names are rejected.

        mov r8, #1
        mov a, r8
        add r8, #1

target:
        ret

; ERR: error: invalid operand for instruction
; ERR: mov r8, #1
; ERR: error: invalid operand for instruction
; ERR: mov a, r8
; ERR: error: invalid operand for instruction
; ERR: add r8, #1
