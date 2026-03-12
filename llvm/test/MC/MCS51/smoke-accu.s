; RUN: llvm-mc -triple avr -arch=mcs51 -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Phase-0 bridge test (accumulator/carry forms): parser accepts a minimal
; 8051-like syntax subset while internals are still AVR-derived.

        mov a, #0x12
        mov acc, #0x34
        add a, #0x01
        anl a, #0xF0
        orl a, #0x0F
        inc a
        dec a
        clr a
        setb c
        setb cy
        clr c
        clr cy
        cpl a
        ret

; CHECK: ldi{{[[:space:]]+}}r16, 18
; CHECK: ldi{{[[:space:]]+}}r16, 52
; CHECK: subi{{[[:space:]]+}}r16, -1
; CHECK: andi{{[[:space:]]+}}r16, 240
; CHECK: ori{{[[:space:]]+}}r16, 15
; CHECK: inc{{[[:space:]]+}}r16
; CHECK: dec{{[[:space:]]+}}r16
; CHECK: clr{{[[:space:]]+}}r16
; CHECK: sec
; CHECK: sec
; CHECK: clc
; CHECK: clc
; CHECK: cpl{{[[:space:]]+}}r16
; CHECK: ret
