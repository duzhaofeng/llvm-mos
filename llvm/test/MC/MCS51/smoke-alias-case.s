; RUN: llvm-mc -triple avr -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Case-insensitive accumulator and carry spellings.

        MOV ACC, #0x2A
        CPL ACC
        SETB CY
        CLR CY
        JB CY, target
        JNB CY, target
        PUSH ACC
        POP ACC

target:
        RET

; CHECK: mov{{[[:space:]]+}}a, #42
; CHECK: encoding: [0x74,0x2a]
; CHECK: cpl{{[[:space:]]+}}a
; CHECK: encoding: [0xf4]
; CHECK: setb{{[[:space:]]+}}c
; CHECK: encoding: [0xd3]
; CHECK: clr{{[[:space:]]+}}c
; CHECK: encoding: [0xc3]
; CHECK: jb{{[[:space:]]+}}PSW.7, target
; CHECK: encoding: [0x20,0xd7,0x00]
; CHECK: jnb{{[[:space:]]+}}PSW.7, target
; CHECK: encoding: [0x30,0xd7,0x00]
; CHECK: push{{[[:space:]]+}}ACC
; CHECK: encoding: [0xc0,0xe0]
; CHECK: pop{{[[:space:]]+}}ACC
; CHECK: encoding: [0xd0,0xe0]
; CHECK: ret
