// RUN: %clang_cc1 %s -triple avr -target-cpu mcs51 -fsyntax-only -verify

__data int D;
__xdata int X;
__pdata int P;
__bdata int B;
__code const int C = 7;
__sfr SFR0 = 0x80;
__sfr16 SFRW;
__sbit SB0 = SFR0 ^ 0;
__bit Flag;
__bdata _Bool FlagBData;

void takes_data(__data int *P); // expected-note {{passing argument to parameter 'P' here}}
void takes_idata(__idata int *P); // expected-note {{passing argument to parameter 'P' here}}
void takes_xdata(__xdata int *P); // expected-note {{passing argument to parameter 'P' here}}
void takes_pdata(__pdata int *P); // expected-note {{passing argument to parameter 'P' here}}
void takes_bdata(__bdata int *P); // expected-note {{passing argument to parameter 'P' here}}
void takes_code(__code const int *P);

void test(void) {
  takes_data(&D);
  __idata int I;
  takes_idata(&I);
  takes_xdata(&X);
  takes_pdata(&P);
  takes_bdata(&B);
  takes_code(&C);

  takes_data(&X); // expected-error {{changes address space of pointer}}
  takes_xdata(&D); // expected-error {{changes address space of pointer}}
  takes_idata(&X); // expected-error {{changes address space of pointer}}
  takes_pdata(&D); // expected-error {{changes address space of pointer}}
  takes_bdata(&X); // expected-error {{changes address space of pointer}}
  takes_data((__data int *)&X);
}

void local_temps(void) {
  __data int d = 1;
  __idata int i = 2;
  __xdata int x = 3;
  __pdata int p = 4;
  __bdata int b = 5;
  (void)d;
  (void)i;
  (void)x;
  (void)p;
  (void)b;

  (void)(__data int){1};
  (void)(__idata int){2};
  (void)(__xdata int){3};
  (void)(__pdata int){4};
  (void)(__bdata int){5};
  (__code int){4}; // expected-error {{compound literal in function scope may not be qualified with an address space}}

  __sfr local_sfr; // expected-error {{automatic variable qualified with an address space}}
  __sbit local_sbit; // expected-error {{'__sbit' declaration must be at file scope}}
  static __sbit local_static_sbit; // expected-error {{'__sbit' declaration must be at file scope}}
}

void bit_bdata_relation(void) {
  __bit lb = Flag;
  __bdata _Bool bb = FlagBData;
  bb = lb;
  lb = bb;
}
