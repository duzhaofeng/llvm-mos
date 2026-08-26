; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-p:16:8-p1:8:8-i16:8-i32:8-i64:8-f32:8-f64:8-a:8-Fi8-n8"
target triple = "mos"

; Function Attrs: noinline nounwind optnone
define dso_local i16 @f(i16 noundef %x) #0 {
entry:
  %retval = alloca i16, align 1
  %x.addr = alloca i16, align 1
  store i16 %x, ptr %x.addr, align 1
  %0 = load i16, ptr %x.addr, align 1
  %tobool = icmp ne i16 %0, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i16 1, ptr %retval, align 1
  br label %return

if.end:                                           ; preds = %entry
  store i16 0, ptr %retval, align 1
  br label %return

return:                                           ; preds = %if.end, %if.then
  %1 = load i16, ptr %retval, align 1
  ret i16 %1
}

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 23.0.0git (https://github.com/duzhaofeng/llvm-mos.git 751cafccae5b6d6f1300760fa5dfb4f1699bf65d)"}
