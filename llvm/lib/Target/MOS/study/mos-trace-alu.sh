#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   sh mos-trace-alu.sh [source.c] [function_name] [triple]
# 例子:
#   sh mos-trace-alu.sh test.c f mos
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

OUTDIR="mos-trace-alu-out"
mkdir -p "${OUTDIR}"

# 默认样例: 触发 add/sub/select
if [[ ! -f "${SRC}" ]]; then
  cat > "${SRC}" <<'EOF'
int f(int a, int b, int c) {
  int x = a + b;
  int y = x - c;
  return (y > 0) ? y : (y + 1);
}
EOF
fi

echo "[1/9] 生成 LLVM IR"
"${CLANG}" -target "${TRIPLE}" -O0 -S -emit-llvm "${SRC}" -o "${OUTDIR}/00-input.ll"

echo "[2/9] IRTranslator 后 MIR"
"${LLC}" -mtriple="${TRIPLE}" -global-isel -stop-after=irtranslator \
  "${OUTDIR}/00-input.ll" -o "${OUTDIR}/01-irtranslator.mir"

echo "[3/9] Legalizer 后 MIR"
"${LLC}" -mtriple="${TRIPLE}" -global-isel -stop-after=legalizer \
  "${OUTDIR}/00-input.ll" -o "${OUTDIR}/02-legalizer.mir"

echo "[4/9] RegBankSelect 后 MIR"
"${LLC}" -mtriple="${TRIPLE}" -global-isel -stop-after=regbankselect \
  "${OUTDIR}/00-input.ll" -o "${OUTDIR}/03-regbankselect.mir"

echo "[5/9] InstructionSelect 后 MIR"
"${LLC}" -mtriple="${TRIPLE}" -global-isel -stop-after=instruction-select \
  "${OUTDIR}/00-input.ll" -o "${OUTDIR}/04-isel.mir"

echo "[6/9] MOSLowerSelect 前后日志"
"${LLC}" -mtriple="${TRIPLE}" -global-isel "${OUTDIR}/00-input.ll" -o /dev/null \
  -print-before=mos-lower-select -print-after=mos-lower-select \
  -filter-print-funcs="${FUNC}" \
  2> "${OUTDIR}/05-mos-lower-select.log"

echo "[7/9] 全 pass changed 日志"
"${LLC}" -mtriple="${TRIPLE}" -global-isel "${OUTDIR}/00-input.ll" -o /dev/null \
  -filter-print-funcs="${FUNC}" -print-changed \
  2> "${OUTDIR}/06-all-pass-changed.log"

echo "[8/9] 最终汇编"
"${LLC}" -mtriple="${TRIPLE}" -global-isel "${OUTDIR}/00-input.ll" \
  -o "${OUTDIR}/07-final.s"

echo "[9/9] 关键摘要"
{
  echo "==== IR: add/sub/select/cmp/br ===="
  search_lines " add | sub |select|icmp| br " "${OUTDIR}/00-input.ll"
  echo
  echo "==== 01-irtranslator.mir: generic ALU/select ===="
  search_lines "G_ADD|G_SUB|G_PTR_ADD|G_SELECT|G_ICMP|G_BRCOND|G_BRCOND_IMM" \
    "${OUTDIR}/01-irtranslator.mir"
  echo
  echo "==== 02-legalizer.mir: legalized generic ALU/select ===="
  search_lines "G_ADD|G_SUB|G_PTR_ADD|G_SELECT|G_ICMP|G_BRCOND|G_BRCOND_IMM" \
    "${OUTDIR}/02-legalizer.mir"
  echo
  echo "==== 04-isel.mir: selected target-ish ops ===="
  search_lines "G_ADD|G_SUB|Add|Sub|CmpBr|Br|ADC|SBC|INC|DEC|JMP|B[A-Z]" \
    "${OUTDIR}/04-isel.mir"
  echo
  echo "==== 05-mos-lower-select.log: select lowering evidence ===="
  search_lines "G_SELECT|G_BRCOND|G_BRCOND_IMM|MachineFunction" \
    "${OUTDIR}/05-mos-lower-select.log"
  echo
  echo "==== 07-final.s: asm ALU/branch mnemonics ===="
  search_lines "adc|sbc|clc|sec|inc|dec|cmp|bne|beq|jmp|bra|bpl|bmi|bcs|bcc|bvs|bvc|lda|sta" \
    "${OUTDIR}/07-final.s"
} | tee "${OUTDIR}/99-summary.log"

echo

echo "完成。输出目录: ${OUTDIR}"
echo "建议先看:"
echo "  ${OUTDIR}/99-summary.log"
echo "再按阶段看:"
echo "  ${OUTDIR}/01-irtranslator.mir"
echo "  ${OUTDIR}/02-legalizer.mir"
echo "  ${OUTDIR}/04-isel.mir"
echo "  ${OUTDIR}/07-final.s"
