#!/usr/bin/env bash
set -euo pipefail

# 可按需改
SRC="${1:-test.c}"
FUNC="${2:-f}"
TRIPLE="${3:-mos}"
OPT_LEVELS="${4:-O0 O1 O2}"

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

OUTDIR="mos-trace-out"
mkdir -p "${OUTDIR}"

if [[ ! -f "${SRC}" ]]; then
  cat > "${SRC}" <<'EOF'
int f(int x) {
  if (x) return 1;
  return 0;
}
EOF
fi

rm -f "${OUTDIR}/99-summary-all.log"

idx=0
for OPT in ${OPT_LEVELS}; do
  idx=$((idx + 1))
  LEVEL_DIR="${OUTDIR}/${OPT}"
  mkdir -p "${LEVEL_DIR}"

  echo "[${idx}] ${OPT}: 生成 LLVM IR"
  "${CLANG}" -target "${TRIPLE}" -${OPT} -S -emit-llvm "${SRC}" -o "${LEVEL_DIR}/00-input.ll"

  echo "[${idx}] ${OPT}: Legalizer 后 MIR"
  "${LLC}" -mtriple="${TRIPLE}" -global-isel -stop-after=legalizer \
    "${LEVEL_DIR}/00-input.ll" -o "${LEVEL_DIR}/01-legalizer.mir"

  echo "[${idx}] ${OPT}: MOSLowerSelect 前后日志"
  "${LLC}" -mtriple="${TRIPLE}" -global-isel "${LEVEL_DIR}/00-input.ll" -o /dev/null \
    -print-before=mos-lower-select -print-after=mos-lower-select \
    -filter-print-funcs="${FUNC}" \
    2> "${LEVEL_DIR}/02-mos-lower-select.log"

  echo "[${idx}] ${OPT}: InstructionSelect 后 MIR"
  "${LLC}" -mtriple="${TRIPLE}" -global-isel -stop-after=instruction-select \
    "${LEVEL_DIR}/00-input.ll" -o "${LEVEL_DIR}/03-isel.mir"

  echo "[${idx}] ${OPT}: 最终汇编"
  "${LLC}" -mtriple="${TRIPLE}" -global-isel "${LEVEL_DIR}/00-input.ll" \
    -o "${LEVEL_DIR}/04-final.s"

  echo "[${idx}] ${OPT}: 全 pass changed 日志"
  "${LLC}" -mtriple="${TRIPLE}" -global-isel "${LEVEL_DIR}/00-input.ll" -o /dev/null \
    -filter-print-funcs="${FUNC}" -print-changed \
    2> "${LEVEL_DIR}/05-all-pass-changed.log"

  {
    echo "==== ${OPT} / 01-legalizer.mir: G_BRCOND / G_BRCOND_IMM ===="
    search_lines "G_BRCOND|G_BRCOND_IMM" "${LEVEL_DIR}/01-legalizer.mir"
    echo
    echo "==== ${OPT} / 02-mos-lower-select.log: G_SELECT / G_BRCOND / G_BRCOND_IMM ===="
    search_lines "G_SELECT|G_BRCOND|G_BRCOND_IMM" "${LEVEL_DIR}/02-mos-lower-select.log"
    echo
    echo "==== ${OPT} / 03-isel.mir: branch-like opcodes ===="
    search_lines "G_BRCOND|CmpBr|Br|JMP|B[A-Z]" "${LEVEL_DIR}/03-isel.mir"
    echo
    echo "==== ${OPT} / 04-final.s: asm branch mnemonics ===="
    search_lines "bne|beq|jmp|bra|bpl|bmi|bcs|bcc|bvs|bvc" "${LEVEL_DIR}/04-final.s"
  } | tee "${LEVEL_DIR}/99-summary.log"

  {
    echo "===== ${OPT} ====="
    sed -n '1,200p' "${LEVEL_DIR}/99-summary.log"
    echo
  } >> "${OUTDIR}/99-summary-all.log"
done

echo "完成。输出目录: ${OUTDIR}"
echo "优化级别: ${OPT_LEVELS}"
echo "每个级别文件:"
echo "  <OUTDIR>/<OPT>/00-input.ll"
echo "  <OUTDIR>/<OPT>/01-legalizer.mir"
echo "  <OUTDIR>/<OPT>/02-mos-lower-select.log"
echo "  <OUTDIR>/<OPT>/03-isel.mir"
echo "  <OUTDIR>/<OPT>/04-final.s"
echo "  <OUTDIR>/<OPT>/05-all-pass-changed.log"
echo "  <OUTDIR>/<OPT>/99-summary.log"
echo "总摘要: ${OUTDIR}/99-summary-all.log"
