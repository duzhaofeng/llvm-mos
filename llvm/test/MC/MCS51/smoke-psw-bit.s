; RUN: llvm-mc -triple avr -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; PSW bit forms resolve to absolute 8051 bit addresses.

        setb psw.0
        clr psw.0
        setb psw.3
        clr psw.7
        setb psw.c
        clr psw.ac
        ret

; CHECK: setb{{[[:space:]]+}}PSW.0
; CHECK: encoding: [0xd2,0xd0]
; CHECK: clr{{[[:space:]]+}}PSW.0
; CHECK: encoding: [0xc2,0xd0]
; CHECK: setb{{[[:space:]]+}}PSW.3
; CHECK: encoding: [0xd2,0xd3]
; CHECK: clr{{[[:space:]]+}}PSW.7
; CHECK: encoding: [0xc2,0xd7]
; CHECK: setb{{[[:space:]]+}}PSW.7
; CHECK: encoding: [0xd2,0xd7]
; CHECK: clr{{[[:space:]]+}}PSW.6
; CHECK: encoding: [0xc2,0xd6]
; CHECK: ret