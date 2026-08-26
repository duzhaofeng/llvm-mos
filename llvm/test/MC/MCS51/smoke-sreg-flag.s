; RUN: llvm-mc -triple avr -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Bare flag names resolve to PSW bit addresses.

        setb carry
        clr ac
        jb overflow, done
        jnb parity, done
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
; CHECK: ret