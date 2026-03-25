// RUN: %clang_cc1 %s -triple avr -target-cpu mcs51 -fsyntax-only -verify

__sfr P0 = 0x80;
__sfr IE = 0xA8;

// Supported forms.
__sbit P0_0 = P0 ^ 0;
__sbit IE_EA = IE ^ 7;
__sbit ABS_0 = 0x80;
__sbit ABS_1 = 0x80 + 1;

// Rejected forms.
__sbit BAD_NOINIT; // expected-error {{'__sbit' declaration requires an initializer}}
__sbit BAD_BIT_RANGE = P0 ^ 8; // expected-error {{'__sbit' bit index must be in range [0,7]}}
__sbit BAD_NEG_BIT = P0 ^ -1; // expected-error {{'__sbit' bit index must be in range [0,7]}}
__sbit BAD_ADDR_RANGE = 0x1FF; // expected-error {{'__sbit' bit address must be in range [0,255]}}

enum { NONSFR = 0x90 };
__sbit BAD_XOR_NONSFR = NONSFR ^ 1; // expected-error {{invalid '__sbit' initializer; expected '__sfr ^ bit(0..7)' or a bit-address constant in [0,255]}}

void test_local(void) {
  __sbit local = 0x80; // expected-error {{'__sbit' declaration must be at file scope}}
  static __sbit local_static = 0x81; // expected-error {{'__sbit' declaration must be at file scope}}
}
