	.zeropage	__rc0
	.zeropage	__rc1
	.zeropage	__rc2
	.zeropage	__rc3
	.zeropage	__rc4
	.zeropage	__rc5
	.zeropage	__rc6
	.zeropage	__rc7
	.zeropage	__rc8
	.zeropage	__rc9
	.zeropage	__rc10
	.zeropage	__rc11
	.zeropage	__rc12
	.zeropage	__rc13
	.zeropage	__rc14
	.zeropage	__rc15
	.zeropage	__rc16
	.zeropage	__rc17
	.zeropage	__rc18
	.zeropage	__rc19
	.zeropage	__rc20
	.zeropage	__rc21
	.zeropage	__rc22
	.zeropage	__rc23
	.zeropage	__rc24
	.zeropage	__rc25
	.zeropage	__rc26
	.zeropage	__rc27
	.zeropage	__rc28
	.zeropage	__rc29
	.zeropage	__rc30
	.zeropage	__rc31
	.file	"test.c"
	.text
	.globl	f                               ; -- Begin function f
	.type	f,@function
f:                                      ; @f
; %bb.0:                                ; %entry
	cpx	#0
	bne	.LBB0_3
; %bb.1:                                ; %entry
	tax
	bne	.LBB0_3
; %bb.2:                                ; %entry
	lda	#0
	tax
	rts
.LBB0_3:                                ; %entry
	lda	#1
	ldx	#0
	rts
.Lfunc_end0:
	.size	f, .Lfunc_end0-f
                                        ; -- End function
	.ident	"clang version 23.0.0git (https://github.com/duzhaofeng/llvm-mos.git 751cafccae5b6d6f1300760fa5dfb4f1699bf65d)"
	.section	".note.GNU-stack","",@progbits
	;Declaring this symbol tells the CRT that the stack pointer needs to be initialized.
	.globl	__do_init_stack
