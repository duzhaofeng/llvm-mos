; RUN: llvm-mc -triple avr -arch=mcs51 -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Phase-0 bridge test: psw.<flag> supports readable long-name suffix aliases.

        setb psw.carry
        clr psw.zero
        jb psw.overflow, done
        jnb psw.interrupt, done
        setb psw.ovf
        clr psw.irq
        jb psw.auxcarry, done
        jnb psw.sign, done
done:
        ret

; CHECK: sec
; CHECK: {{(clz|bclr[[:space:]]+1)}}
; CHECK: {{(brvs[[:space:]]+done|brbs[[:space:]]+3,[[:space:]]+done)}}
; CHECK: {{(brid[[:space:]]+done|brbc[[:space:]]+7,[[:space:]]+done)}}
; CHECK: sev
; CHECK: cli
; CHECK: brhs{{[[:space:]]+}}done
; CHECK: brbc{{[[:space:]]+}}4, done
; CHECK: ret