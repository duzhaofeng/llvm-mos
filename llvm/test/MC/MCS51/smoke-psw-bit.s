; RUN: llvm-mc -triple avr -arch=mcs51 -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Phase-0 bridge test (PSW bit forms): setb/clr on psw.N should lower to
; AVR status-bit set/clear operations.

        setb psw.0
        clr psw.0
        setb psw.3
        clr psw.7
        setb psw.c
        clr psw.z
        ret

; CHECK: sec
; CHECK: clc
; CHECK: sev
; CHECK: cli
; CHECK: sec
; CHECK: {{(clz|bclr[[:space:]]+1)}}
; CHECK: ret