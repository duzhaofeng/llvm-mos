; RUN: llvm-mc -triple avr -arch=mcs51 -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Phase-0 bridge test: bare SREG-style flag names are accepted in
; setb/clr and jb/jnb forms.

        setb carry
        clr zero
        jb overflow, done
        jnb irq, done
done:
        ret

; CHECK: sec
; CHECK: clz
; CHECK: jb{{[[:space:]]+}}3, done
; CHECK: jnb{{[[:space:]]+}}7, done
; CHECK: ret