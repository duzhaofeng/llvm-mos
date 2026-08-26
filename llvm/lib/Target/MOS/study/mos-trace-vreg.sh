#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   sh mos-trace-vreg.sh [source.c] [function_name] [triple]
# 例子:
#   sh mos-trace-vreg.sh test.c f mos
SRC="${1:-test.c}"
FUNC="${2:-f}"
TRIPLE="${3:-mos}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)"

if [[ -x "${REPO_ROOT}/build/bin/clang" ]]; then
  CLANG="${REPO_ROOT}/build/bin/clang"
elif command -v clang >/dev/null 2>&1; then
  CLANG=clang
else
  echo "error: clang not found" >&2
  exit 1
fi

if [[ -x "${REPO_ROOT}/build/bin/llc" ]]; then
  LLC="${REPO_ROOT}/build/bin/llc"
elif command -v llc >/dev/null 2>&1; then
  LLC=llc
else
  echo "error: llc not found" >&2
  exit 1
fi

search_lines() {
  local pattern="$1"
  local file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -n "$pattern" "$file" || true
  else
    grep -En "$pattern" "$file" || true
  fi
}

OUTDIR="mos-trace-vreg-out"
mkdir -p "${OUTDIR}"

if [[ ! -f "${SRC}" ]]; then
  cat > "${SRC}" <<'EOF'
int f(int a, int b, int c) {
  int x = a + b;
  int y = x - c;
  return (y > 0) ? y : (y + 1);
}
EOF
fi

echo "[1/11] 生成 LLVM IR"
"${CLANG}" -target "${TRIPLE}" -O0 -S -emit-llvm "${SRC}" -o "${OUTDIR}/00-input.ll"

echo "[2/11] IRTranslator 后 MIR"
"${LLC}" -mtriple="${TRIPLE}" -global-isel -stop-after=irtranslator \
  "${OUTDIR}/00-input.ll" -o "${OUTDIR}/01-irtranslator.mir"

echo "[3/11] Legalizer 后 MIR"
"${LLC}" -mtriple="${TRIPLE}" -global-isel -stop-after=legalizer \
  "${OUTDIR}/00-input.ll" -o "${OUTDIR}/02-legalizer.mir"

echo "[4/11] RegBankSelect 后 MIR"
"${LLC}" -mtriple="${TRIPLE}" -global-isel -stop-after=regbankselect \
  "${OUTDIR}/00-input.ll" -o "${OUTDIR}/03-regbankselect.mir"

echo "[5/11] InstructionSelect 后 MIR"
"${LLC}" -mtriple="${TRIPLE}" -global-isel -stop-after=instruction-select \
  "${OUTDIR}/00-input.ll" -o "${OUTDIR}/04-isel.mir"

echo "[6/11] 最终汇编"
"${LLC}" -mtriple="${TRIPLE}" -global-isel "${OUTDIR}/00-input.ll" \
  -o "${OUTDIR}/05-final.s"

echo "[7/11] MOSLowerSelect 前后日志"
"${LLC}" -mtriple="${TRIPLE}" -global-isel "${OUTDIR}/00-input.ll" -o /dev/null \
  -print-before=mos-lower-select -print-after=mos-lower-select \
  -filter-print-funcs="${FUNC}" \
  2> "${OUTDIR}/06-mos-lower-select.log"

echo "[8/11] 全 pass changed 日志"
"${LLC}" -mtriple="${TRIPLE}" -global-isel "${OUTDIR}/00-input.ll" -o /dev/null \
  -filter-print-funcs="${FUNC}" -print-changed \
  2> "${OUTDIR}/07-all-pass-changed.log"

echo "[9/11] 抽取种子寄存器（从 early MIR 的 ALU/SELECT/ICMP 定义）"
SEED_OPS='G_ADD|G_SUB|G_SELECT|G_ICMP|G_PTR_ADD|G_SEXT|G_ZEXT'
SEED_FILE="${OUTDIR}/08-seed-vregs.txt"
: > "${SEED_FILE}"

search_lines "${SEED_OPS}" "${OUTDIR}/01-irtranslator.mir" \
  | sed -E 's/^[0-9]+:[[:space:]]*//' \
  | sed -nE 's/^([%][0-9]+)(:.*)?[[:space:]]*=.*/\1/p' \
  | awk '!seen[$0]++' \
  | head -n 10 > "${SEED_FILE}" || true

if [[ ! -s "${SEED_FILE}" ]]; then
  search_lines "${SEED_OPS}" "${OUTDIR}/02-legalizer.mir" \
    | sed -E 's/^[0-9]+:[[:space:]]*//' \
    | sed -nE 's/^([%][0-9]+)(:.*)?[[:space:]]*=.*/\1/p' \
    | awk '!seen[$0]++' \
    | head -n 10 > "${SEED_FILE}" || true
fi

echo "[10/11] 生成逐值追踪报告"
TRACE_REPORT="${OUTDIR}/09-vreg-trace.log"
: > "${TRACE_REPORT}"

{
  echo "==== Seed VRegs ===="
  if [[ -s "${SEED_FILE}" ]]; then
    cat "${SEED_FILE}"
  else
    echo "(none)"
  fi
  echo

  for STAGE in \
    "01-irtranslator.mir" \
    "02-legalizer.mir" \
    "03-regbankselect.mir" \
    "04-isel.mir"
  do
    echo "==== Stage: ${STAGE} ===="
    if [[ -s "${SEED_FILE}" ]]; then
      while IFS= read -r VREG; do
        [[ -z "${VREG}" ]] && continue
        echo "-- ${VREG}"
        search_lines "${VREG}" "${OUTDIR}/${STAGE}"
      done < "${SEED_FILE}"
    else
      echo "(no seed vregs found)"
    fi
    echo
  done

  echo "==== Return-related lines (for backward inspection hint) ===="
  search_lines "G_RETURN|RET|return|JMP|Br|CmpBr|G_BRCOND|G_BRCOND_IMM" \
    "${OUTDIR}/04-isel.mir"
  echo
  echo "==== Final asm branch/alu summary ===="
  search_lines "adc|sbc|inc|dec|cmp|beq|bne|bcc|bcs|bmi|bpl|bvc|bvs|jmp|bra|lda|sta" \
    "${OUTDIR}/05-final.s"
} >> "${TRACE_REPORT}"

echo "[11/11] 生成总摘要"
SUMMARY="${OUTDIR}/99-summary.log"
{
  echo "Output dir: ${OUTDIR}"
  echo
  echo "Key files:"
  echo "  ${OUTDIR}/00-input.ll"
  echo "  ${OUTDIR}/01-irtranslator.mir"
  echo "  ${OUTDIR}/02-legalizer.mir"
  echo "  ${OUTDIR}/03-regbankselect.mir"
  echo "  ${OUTDIR}/04-isel.mir"
  echo "  ${OUTDIR}/05-final.s"
  echo "  ${OUTDIR}/06-mos-lower-select.log"
  echo "  ${OUTDIR}/07-all-pass-changed.log"
  echo "  ${OUTDIR}/08-seed-vregs.txt"
  echo "  ${OUTDIR}/09-vreg-trace.log"
  echo
  echo "Quick peek:"
  echo "  sed -n '1,200p' ${OUTDIR}/09-vreg-trace.log"
} | tee "${SUMMARY}"

echo
echo "完成。先看 ${OUTDIR}/09-vreg-trace.log。"
echo "提示: 跨阶段 vreg 可能改号，这是正常现象；此报告用于快速定位候选 def/use 片段。"
