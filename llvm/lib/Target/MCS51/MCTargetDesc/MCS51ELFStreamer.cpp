#include "MCS51ELFStreamer.h"
#include "MCS51MCTargetDesc.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/TargetParser/SubtargetFeature.h"

namespace llvm {

static unsigned getEFlagsForFeatureSet(const FeatureBitset &Features) {
  unsigned EFlags = 0;

  // Set architecture
  if (Features[MCS51::ELFArchMCS511])
    EFlags |= ELF::EF_AVR_ARCH_AVR1;
  else if (Features[MCS51::ELFArchMCS512])
    EFlags |= ELF::EF_AVR_ARCH_AVR2;
  else if (Features[MCS51::ELFArchMCS5125])
    EFlags |= ELF::EF_AVR_ARCH_AVR25;
  else if (Features[MCS51::ELFArchMCS513])
    EFlags |= ELF::EF_AVR_ARCH_AVR3;
  else if (Features[MCS51::ELFArchMCS5131])
    EFlags |= ELF::EF_AVR_ARCH_AVR31;
  else if (Features[MCS51::ELFArchMCS5135])
    EFlags |= ELF::EF_AVR_ARCH_AVR35;
  else if (Features[MCS51::ELFArchMCS514])
    EFlags |= ELF::EF_AVR_ARCH_AVR4;
  else if (Features[MCS51::ELFArchMCS515])
    EFlags |= ELF::EF_AVR_ARCH_AVR5;
  else if (Features[MCS51::ELFArchMCS5151])
    EFlags |= ELF::EF_AVR_ARCH_AVR51;
  else if (Features[MCS51::ELFArchMCS516])
    EFlags |= ELF::EF_AVR_ARCH_AVR6;
  else if (Features[MCS51::ELFArchTiny])
    EFlags |= ELF::EF_AVR_ARCH_AVRTINY;
  else if (Features[MCS51::ELFArchXMEGA1])
    EFlags |= ELF::EF_AVR_ARCH_XMEGA1;
  else if (Features[MCS51::ELFArchXMEGA2])
    EFlags |= ELF::EF_AVR_ARCH_XMEGA2;
  else if (Features[MCS51::ELFArchXMEGA3])
    EFlags |= ELF::EF_AVR_ARCH_XMEGA3;
  else if (Features[MCS51::ELFArchXMEGA4])
    EFlags |= ELF::EF_AVR_ARCH_XMEGA4;
  else if (Features[MCS51::ELFArchXMEGA5])
    EFlags |= ELF::EF_AVR_ARCH_XMEGA5;
  else if (Features[MCS51::ELFArchXMEGA6])
    EFlags |= ELF::EF_AVR_ARCH_XMEGA6;
  else if (Features[MCS51::ELFArchXMEGA7])
    EFlags |= ELF::EF_AVR_ARCH_XMEGA7;

  return EFlags;
}

MCS51ELFStreamer::MCS51ELFStreamer(MCStreamer &S, const MCSubtargetInfo &STI)
    : MCS51TargetStreamer(S) {

  ELFObjectWriter &W = getStreamer().getWriter();
  unsigned EFlags = W.getELFHeaderEFlags();

  EFlags |= getEFlagsForFeatureSet(STI.getFeatureBits());
  EFlags |= ELF::EF_AVR_LINKRELAX_PREPARED;

  W.setELFHeaderEFlags(EFlags);
}

} // end namespace llvm
