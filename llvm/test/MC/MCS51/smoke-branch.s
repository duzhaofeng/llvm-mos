; RUN: llvm-mc -triple avr -arch=mcs51 -mcpu=mcs51 -show-encoding < %s | FileCheck %s
;
; Phase-0 bridge test (branch forms): verify canonicalization of common
; 8051 jump mnemonics and carry-based branch aliases.

        sjmp .
        ajmp func
        acall func
        jz func
        jnz func
        jc func
        jnc func
        jb c, func
        jnb c, func
        jb cy, func
        jnb cy, func
        jb 3, func
        jnb 5, func
        jb p1.6, func
        jnb p1.5, func
        jb psw.0, func
        jnb ie.7, func
        jb psw.c, func
        jnb psw.z, func
func:
        ret

; CHECK: jmp
; CHECK: call
; CHECK: jz
; CHECK: jnz
; CHECK: jb{{[[:space:]]+}}0, func
; CHECK: jnb{{[[:space:]]+}}0, func
; CHECK: jb{{[[:space:]]+}}0, func
; CHECK: jnb{{[[:space:]]+}}0, func
; CHECK: jb{{[[:space:]]+}}0, func
; CHECK: jnb{{[[:space:]]+}}0, func
; CHECK: jb{{[[:space:]]+}}3, func
; CHECK: jnb{{[[:space:]]+}}5, func
; CHECK: jb{{[[:space:]]+}}6, func
; CHECK: jnb{{[[:space:]]+}}5, func
; CHECK: jb{{[[:space:]]+}}0, func
; CHECK: jnb{{[[:space:]]+}}7, func
; CHECK: jb{{[[:space:]]+}}0, func
; CHECK: {{(jnb[[:space:]]+1,[[:space:]]+func|jnz[[:space:]]+func)}}
; CHECK: ret
