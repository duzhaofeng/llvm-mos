; RUN: llvm-mc -triple avr -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Accumulator and carry forms with native 8051 encodings.

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

; CHECK: mov{{[[:space:]]+}}a, #18
; CHECK: encoding: [0x74,0x12]
; CHECK: mov{{[[:space:]]+}}a, #52
; CHECK: encoding: [0x74,0x34]
; CHECK: add{{[[:space:]]+}}a, #1
; CHECK: encoding: [0x24,0x01]
; CHECK: anl{{[[:space:]]+}}a, #240
; CHECK: encoding: [0x54,0xf0]
; CHECK: orl{{[[:space:]]+}}a, #15
; CHECK: encoding: [0x44,0x0f]
; CHECK: inc{{[[:space:]]+}}a
; CHECK: encoding: [0x04]
; CHECK: dec{{[[:space:]]+}}a
; CHECK: encoding: [0x14]
; CHECK: clr{{[[:space:]]+}}a
; CHECK: encoding: [0xe4]
; CHECK: setb{{[[:space:]]+}}c
; CHECK: encoding: [0xd3]
; CHECK: clr{{[[:space:]]+}}c
; CHECK: encoding: [0xc3]
; CHECK: cpl{{[[:space:]]+}}a
; CHECK: encoding: [0xf4]
; CHECK: ret
