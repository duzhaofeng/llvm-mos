; RUN: not llvm-mc -triple avr -mcpu=mcs51 < %s 2>&1 | FileCheck %s --check-prefix=ERR
;
; Dotted bit forms must use a known bit-addressable SFR base and a bit
; index in range 0..7.

        jb p1.8, target
        jb foo.3, target
        jnb psw.foo, target
        jb ie.ex2, target

target:
        ret

; ERR: error: invalid bit address
; ERR: jb p1.8, target
; ERR: error: invalid bit address
; ERR: jb foo.3, target
; ERR: error: invalid bit address
; ERR: jnb psw.foo, target
; ERR: error: invalid bit address
; ERR: jb ie.ex2, target
