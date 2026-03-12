; RUN: not llvm-mc -triple avr -arch=mcs51 -mcpu=mcs51 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=ERR
;
; Phase-0 boundary test:
; - direct bit-address constants are still unsupported
; - dotted bit suffix forms only allow bit index range 0..7
; - dotted forms must use known 8051 bit-address base names

        jb 0x20, target
        jnb p1.8, target
        jb foo.3, target
        jnb psw.foo, target
        jb ie.ex2, target

target:
        ret

; ERR: error: invalid instruction
; ERR: jb 0x20, target
; ERR: error: invalid instruction
; ERR: jnb p1.8, target
; ERR: error: invalid instruction
; ERR: jb foo.3, target
; ERR: error: invalid instruction
; ERR: jnb psw.foo, target
; ERR: error: invalid instruction
; ERR: jb ie.ex2, target
