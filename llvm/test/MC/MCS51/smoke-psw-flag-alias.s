; RUN: llvm-mc -triple avr -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; psw.<flag> long-name suffix aliases.

        setb psw.carry
        clr psw.ac
        jb psw.overflow, done
        jnb psw.parity, done
        setb psw.ovf
        clr psw.rs0
        jb psw.auxcarry, done
        jnb psw.f0, done
done:
        ret

; CHECK: setb{{[[:space:]]+}}PSW.7
; CHECK: encoding: [0xd2,0xd7]
; CHECK: clr{{[[:space:]]+}}PSW.6
; CHECK: encoding: [0xc2,0xd6]
; CHECK: jb{{[[:space:]]+}}PSW.2, done
; CHECK: encoding: [0x20,0xd2,0x00]
; CHECK: jnb{{[[:space:]]+}}PSW.0, done
; CHECK: encoding: [0x30,0xd0,0x00]
; CHECK: setb{{[[:space:]]+}}PSW.2
; CHECK: encoding: [0xd2,0xd2]
; CHECK: clr{{[[:space:]]+}}PSW.3
; CHECK: encoding: [0xc2,0xd3]
; CHECK: jb{{[[:space:]]+}}PSW.6, done
; CHECK: encoding: [0x20,0xd6,0x00]
; CHECK: jnb{{[[:space:]]+}}PSW.5, done
; CHECK: encoding: [0x30,0xd5,0x00]
; CHECK: ret