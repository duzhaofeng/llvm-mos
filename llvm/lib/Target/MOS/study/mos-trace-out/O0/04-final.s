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
	pha
	clc
	lda	__rc0
	adc	#252
	sta	__rc0
	lda	__rc1
	adc	#255
	sta	__rc1
	pla
	clc
	ldy	__rc0
	sty	__rc2
	ldy	__rc1
	sty	__rc3
	ldy	#0
	sta	(__rc2),y
	iny
	txa
	sta	(__rc2),y
	dey
	lda	(__rc2),y
	sta	__rc4
	iny
	lda	(__rc2),y
	bne	.LBB0_2
	jmp	.LBB0_1
.LBB0_1:                                ; %entry
	lda	__rc4
	beq	.LBB0_3
	jmp	.LBB0_2
.LBB0_2:                                ; %if.then
	ldy	#0
	clc
	lda	__rc0
	adc	#2
	sta	__rc2
	lda	__rc1
	adc	#0
	sta	__rc3
	lda	#1
	sta	(__rc2),y
	tay
	lda	#0
	sta	(__rc2),y
	jmp	.LBB0_4
.LBB0_3:                                ; %if.end
	ldy	#0
	clc
	lda	__rc0
	adc	#2
	sta	__rc2
	lda	__rc1
	adc	#0
	sta	__rc3
	tya
	sta	(__rc2),y
	iny
	sta	(__rc2),y
.LBB0_4:                                ; %return
	ldy	#0
	clc
	lda	__rc0
	adc	#2
	sta	__rc2
	lda	__rc1
	adc	#0
	sta	__rc3
	lda	(__rc2),y
	sta	__rc4
	iny
	lda	(__rc2),y
	tax
	lda	__rc4
	pha
	clc
	lda	__rc0
	adc	#4
	sta	__rc0
	lda	__rc1
	adc	#0
	sta	__rc1
	pla
	rts
.Lfunc_end0:
	.size	f, .Lfunc_end0-f
                                        ; -- End function
	.ident	"clang version 23.0.0git (https://github.com/duzhaofeng/llvm-mos.git 751cafccae5b6d6f1300760fa5dfb4f1699bf65d)"
	.section	".note.GNU-stack","",@progbits
	;Declaring this symbol tells the CRT that the stack pointer needs to be initialized.
	.globl	__do_init_stack
