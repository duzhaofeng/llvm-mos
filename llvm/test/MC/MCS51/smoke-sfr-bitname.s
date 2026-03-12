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
; CHECK: jb{{[[:space:]]+}}7, done
; CHECK: jnb{{[[:space:]]+}}0, done
; CHECK: jb{{[[:space:]]+}}0, done
; CHECK: jnb{{[[:space:]]+}}5, done
; CHECK: ret