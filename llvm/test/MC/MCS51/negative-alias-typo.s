; RUN: not llvm-mc -triple avr -arch=mcs51 -mcpu=mcs51 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=ERR
;
; Phase-0 alias typo boundary:
; spellings outside {a, acc} and {c, cy} should be rejected.

        mov accc, #1
        setb cyy
        jb cyy, target
        push accc

target:
        ret

; ERR: error: invalid operand for instruction
; ERR: mov accc, #1
; ERR: error: invalid instruction
; ERR: setb cyy
; ERR: error: invalid instruction
; ERR: jb cyy, target
; ERR: error: invalid operand for instruction
; ERR: push accc
