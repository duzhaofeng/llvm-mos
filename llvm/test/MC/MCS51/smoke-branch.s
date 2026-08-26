; RUN: llvm-mc -triple avr -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Native 8051 conditional branches and bit-address resolution.

        sjmp .
        ajmp func
        acall func
        jz func
        jnz func
        jc func
        jnc func
        jb c, func
        jnb c, func
        jb 3, func
        jnb 5, func
        jb p1.6, func
        jnb p1.5, func
        jb psw.0, func
        jnb ie.7, func
        jb psw.c, func
        jnb psw.ac, func
func:
        ret

; CHECK: sjmp
; CHECK: ajmp{{[[:space:]]+}}func
; CHECK: acall{{[[:space:]]+}}func
; CHECK: jz{{[[:space:]]+}}func
; CHECK: encoding: [0x60,0x00]
; CHECK: jnz{{[[:space:]]+}}func
; CHECK: encoding: [0x70,0x00]
; CHECK: jc{{[[:space:]]+}}func
; CHECK: encoding: [0x40,0x00]
; CHECK: jnc{{[[:space:]]+}}func
; CHECK: encoding: [0x50,0x00]
; CHECK: jb{{[[:space:]]+}}PSW.7, func
; CHECK: encoding: [0x20,0xd7,0x00]
; CHECK: jnb{{[[:space:]]+}}PSW.7, func
; CHECK: encoding: [0x30,0xd7,0x00]
; CHECK: jb{{[[:space:]]+}}0x20.3, func
; CHECK: encoding: [0x20,0x03,0x00]
; CHECK: jnb{{[[:space:]]+}}0x20.5, func
; CHECK: encoding: [0x30,0x05,0x00]
; CHECK: jb{{[[:space:]]+}}P1.6, func
; CHECK: encoding: [0x20,0x96,0x00]
; CHECK: jnb{{[[:space:]]+}}P1.5, func
; CHECK: encoding: [0x30,0x95,0x00]
; CHECK: jb{{[[:space:]]+}}PSW.0, func
; CHECK: encoding: [0x20,0xd0,0x00]
; CHECK: jnb{{[[:space:]]+}}IE.7, func
; CHECK: encoding: [0x30,0xaf,0x00]
; CHECK: jb{{[[:space:]]+}}PSW.7, func
; CHECK: jnb{{[[:space:]]+}}PSW.6, func
; CHECK: ret
