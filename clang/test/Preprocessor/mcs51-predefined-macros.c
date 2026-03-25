// RUN: %clang_cc1 -triple avr -target-cpu mcs51 -E -dM %s | FileCheck %s --check-prefix=MCS51
// RUN: %clang_cc1 -triple avr -target-cpu mcs251 -E -dM %s | FileCheck %s --check-prefix=MCS251

// MCS51-DAG: #define __MCS51_ARCH__ 51
// MCS51-DAG: #define __MCS51__ 1
// MCS51-DAG: #define __bit __bdata _Bool
// MCS51-DAG: #define __data __attribute__((__address_space__(1)))
// MCS51-DAG: #define __idata __attribute__((__address_space__(2)))
// MCS51-DAG: #define __xdata __attribute__((__address_space__(3)))
// MCS51-DAG: #define __pdata __attribute__((__address_space__(5)))
// MCS51-DAG: #define __bdata __attribute__((__address_space__(6)))
// MCS51-DAG: #define __code __attribute__((__address_space__(4)))
// MCS51-DAG: #define __sfr __attribute__((annotate("mcs51_sfr"))) __attribute__((__address_space__(7))) volatile unsigned char
// MCS51-DAG: #define __sfr16 __attribute__((__address_space__(7))) volatile unsigned short
// MCS51-DAG: #define __sbit __attribute__((annotate("mcs51_sbit"))) __attribute__((__address_space__(7))) volatile unsigned char
// MCS51-NOT: #define __MCS251__ 1
// MCS51-NOT: #define __AVR__

// MCS251-DAG: #define __MCS51_ARCH__ 251
// MCS251-DAG: #define __MCS51__ 1
// MCS251-DAG: #define __MCS251__ 1
// MCS251-DAG: #define __bit __bdata _Bool
// MCS251-DAG: #define __data __attribute__((__address_space__(1)))
// MCS251-DAG: #define __idata __attribute__((__address_space__(2)))
// MCS251-DAG: #define __xdata __attribute__((__address_space__(3)))
// MCS251-DAG: #define __pdata __attribute__((__address_space__(5)))
// MCS251-DAG: #define __bdata __attribute__((__address_space__(6)))
// MCS251-DAG: #define __code __attribute__((__address_space__(4)))
// MCS251-DAG: #define __sfr __attribute__((annotate("mcs51_sfr"))) __attribute__((__address_space__(7))) volatile unsigned char
// MCS251-DAG: #define __sfr16 __attribute__((__address_space__(7))) volatile unsigned short
// MCS251-DAG: #define __sbit __attribute__((annotate("mcs51_sbit"))) __attribute__((__address_space__(7))) volatile unsigned char
// MCS251-NOT: #define __AVR__
