; RUN: llvm-mc -triple avr -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Named bit suffixes on bit-addressable SFR bases.

        setb ie.ea
        clr ip.px0
        jb tcon.tf1, done
        jnb scon.ri, done
        jb ie.ex0, done
        jnb ip.pt2, done
done:
        ret

; CHECK: setb{{[[:space:]]+}}IE.7
; CHECK: encoding: [0xd2,0xaf]
; CHECK: clr{{[[:space:]]+}}IP.0
; CHECK: encoding: [0xc2,0xb8]
; CHECK: jb{{[[:space:]]+}}TCON.7, done
; CHECK: encoding: [0x20,0x8f,0x00]
; CHECK: jnb{{[[:space:]]+}}SCON.0, done
; CHECK: encoding: [0x30,0x98,0x00]
; CHECK: jb{{[[:space:]]+}}IE.0, done
; CHECK: encoding: [0x20,0xa8,0x00]
; CHECK: jnb{{[[:space:]]+}}IP.5, done
; CHECK: encoding: [0x30,0xbd,0x00]
; CHECK: ret