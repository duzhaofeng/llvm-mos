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
; CHECK: brbs{{[[:space:]]+}}0,
; CHECK: brbc{{[[:space:]]+}}0,
; CHECK: brbs{{[[:space:]]+}}0,
; CHECK: brbc{{[[:space:]]+}}0,
; CHECK: brbs{{[[:space:]]+}}0,
; CHECK: brbc{{[[:space:]]+}}0,
; CHECK: brvs{{[[:space:]]+}}func
; CHECK: brhc{{[[:space:]]+}}func
; CHECK: brts{{[[:space:]]+}}func
; CHECK: brhc{{[[:space:]]+}}func
; CHECK: brbs{{[[:space:]]+}}0,
; CHECK: {{(brbc[[:space:]]+7,|brid[[:space:]]+)}}func
; CHECK: brbs{{[[:space:]]+}}0,
; CHECK: {{(brbc[[:space:]]+1,|jnz[[:space:]]+)}}func
; CHECK: ret
