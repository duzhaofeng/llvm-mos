; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-p:16:8-p1:8:8-i16:8-i32:8-i64:8-f32:8-f64:8-a:8-Fi8-n8"
target triple = "mos"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define dso_local range(i16 0, 2) i16 @f(i16 noundef %x) local_unnamed_addr #0 {
entry:
  %tobool.not = icmp ne i16 %x, 0
  %. = zext i1 %tobool.not to i16
  ret i16 %.
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!llvm.errno.tbaa = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 23.0.0git (https://github.com/duzhaofeng/llvm-mos.git 751cafccae5b6d6f1300760fa5dfb4f1699bf65d)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
