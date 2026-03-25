// RUN: %clang_cc1 -triple avr -target-cpu mcs51 -emit-llvm -o - %s | FileCheck %s

__data const int D = 1;
__idata const int I = 5;
__xdata const int X = 2;
__pdata const int P = 6;
__bdata const int B = 7;
__code const int C[2] = {3, 4};
__sfr SFR0 = 0;
__sfr16 SFR16 = 0;
__sbit SB0 = 0;
__bit const FLAG = 1;

// CHECK: @D{{.*}} addrspace(1) constant i16 1
// CHECK: @I{{.*}} addrspace(2) constant i16 5
// CHECK: @X{{.*}} addrspace(3) constant i16 2
// CHECK: @P{{.*}} addrspace(5) constant i16 6
// CHECK: @B{{.*}} addrspace(6) constant i16 7
// CHECK: @C{{.*}} addrspace(4) constant [2 x i16] [i16 3, i16 4]
// CHECK: @SFR0{{.*}} addrspace(7) global i8 0
// CHECK: @SFR16{{.*}} addrspace(7) global i16 0
// CHECK: @SB0{{.*}} addrspace(7) global i8 0
// CHECK: @FLAG{{.*}} addrspace(6) constant i8 1

__data const int *get_data(void) {
  return &D;
}

__xdata const int *get_xdata(void) {
  return &X;
}

__pdata const int *get_pdata(void) {
  return &P;
}

__bdata const int *get_bdata(void) {
  return &B;
}

__idata const int *get_idata(void) {
  return &I;
}

__code const int *get_code(void) {
  return C;
}

// CHECK-DAG: define{{.*}} ptr addrspace(1) @get_data()
// CHECK-DAG: define{{.*}} ptr addrspace(2) @get_idata()
// CHECK-DAG: define{{.*}} ptr addrspace(3) @get_xdata()
// CHECK-DAG: define{{.*}} ptr addrspace(5) @get_pdata()
// CHECK-DAG: define{{.*}} ptr addrspace(6) @get_bdata()
// CHECK-DAG: define{{.*}} ptr addrspace(4) @get_code()

int sum_locals(void) {
  __data int d = 1;
  __idata int i = 2;
  __xdata int x = 3;
  __pdata int p = 4;
  __bdata int b = 5;
  return d + i + x + p + b;
}

__bit test_bit_use(void) {
  __bit local = FLAG;
  return local && FLAG;
}

// CHECK-LABEL: define{{.*}} i16 @sum_locals()
// CHECK: addrspacecast ptr %{{.*}} to ptr addrspace(2)
// CHECK: addrspacecast ptr %{{.*}} to ptr addrspace(3)
// CHECK: addrspacecast ptr %{{.*}} to ptr addrspace(5)
// CHECK: addrspacecast ptr %{{.*}} to ptr addrspace(6)

// CHECK-LABEL: define{{.*}} i1 @test_bit_use()
// CHECK: [[LOCAL_ADDR:%.*]] = alloca i8, align 1
// CHECK: [[LOCAL_AS6:%.*]] = addrspacecast ptr [[LOCAL_ADDR]] to ptr addrspace(6)
// CHECK: store i8 1, ptr addrspace(6) [[LOCAL_AS6]], align 1
// CHECK: [[LOCAL_VAL_I8:%.*]] = load i8, ptr addrspace(6) [[LOCAL_AS6]], align 1
// CHECK: [[LOCAL_BOOL:%.*]] = trunc i8 [[LOCAL_VAL_I8]] to i1
// CHECK: br i1 [[LOCAL_BOOL]], label %land.rhs, label %land.end
// CHECK: land.rhs:
// CHECK: br label %land.end
// CHECK: land.end:
// CHECK: [[BOOL_PHI:%.*]] = phi i1 [ false, %entry ], [ true, %land.rhs ]
// CHECK: ret i1 [[BOOL_PHI]]
