; RUN: not llvm-mc -triple avr -arch=mcs51 -mcpu=mcs51 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=ERR
;
; Phase-0 boundary test for parser aliases:
; - `acc` is an accumulator register alias, not a carry flag operand.
; - `cy` is only accepted in carry-specific bridge forms.

        setb acc
        setb p1.0
        setb psw.foo
        clr ie.ex2
        mov cy, #1
        jb acc, target
        jnb acc, target

target:
        ret

; ERR: error: invalid instruction
; ERR: setb acc
; ERR: error: invalid instruction
; ERR: setb p1.0
; ERR: error: invalid instruction
; ERR: setb psw.foo
; ERR: error: invalid operand for instruction
; ERR: clr ie.ex2
; ERR: error: invalid operand for instruction
; ERR: mov cy, #1
; ERR: error: invalid instruction
; ERR: jb acc, target
; ERR: error: invalid instruction
; ERR: jnb acc, target
