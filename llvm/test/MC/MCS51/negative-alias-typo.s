; RUN: not llvm-mc -triple avr -arch=mcs51 -mcpu=mcs51 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=ERR
;
; Phase-0 alias typo boundary:
; misspelled accumulator/carry/flag aliases should be rejected.

        mov accc, #1
        setb cyy
        clr zerro
        jb cyy, target
        jnb irqq, target
        push accc

target:
        ret

; ERR: error: invalid operand for instruction
; ERR: mov accc, #1
; ERR: error: invalid operand for instruction
; ERR: setb cyy
; ERR: error: invalid operand for instruction
; ERR: clr zerro
; ERR: error: invalid operand for instruction
; ERR: jb cyy, target
; ERR: error: invalid operand for instruction
; ERR: jnb irqq, target
; ERR: error: invalid operand for instruction
; ERR: push accc
