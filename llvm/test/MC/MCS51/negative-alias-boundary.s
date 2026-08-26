; RUN: not llvm-mc -triple avr -mcpu=mcs51 < %s 2>&1 | FileCheck %s --check-prefix=ERR
;
; Accumulator and carry tokens cannot be used as bit operands.

        setb acc
        mov cy, #1
        jb acc, target
        jnb acc, target

target:
        ret

; ERR: error: invalid operand for instruction
; ERR: setb acc
; ERR: error: invalid operand for instruction
; ERR: mov cy, #1
; ERR: error: invalid operand for instruction
; ERR: jb acc, target
; ERR: error: invalid operand for instruction
; ERR: jnb acc, target
