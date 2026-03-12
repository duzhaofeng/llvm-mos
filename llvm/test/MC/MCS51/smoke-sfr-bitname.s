; RUN: llvm-mc -triple avr -arch=mcs51 -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Phase-0 bridge test: named bit suffixes on selected SFR bases are accepted.

        setb ie.ea
        clr ip.px0
        jb tcon.tf1, done
        jnb scon.ri, done
        jb ie.ex0, done
        jnb ip.pt2, done
done:
        ret

; CHECK: sei
; CHECK: clc
; CHECK: {{(brie[[:space:]]+done|brbs[[:space:]]+7,[[:space:]]+done)}}
; CHECK: {{(brbc[[:space:]]+0,[[:space:]]+done|brcc[[:space:]]+done)}}
; CHECK: {{(brbs[[:space:]]+0,[[:space:]]+done|brcs[[:space:]]+done)}}
; CHECK: {{(brhc[[:space:]]+done|brbc[[:space:]]+5,[[:space:]]+done)}}
; CHECK: ret