add_llvm_component_library(LLVMMCS51AsmParser
  MCS51AsmParser.cpp

  LINK_COMPONENTS
  MCS51Desc
  MCS51Info
  CodeGenTypes
  MC
  MCParser
  Support

  ADD_TO_COMPONENT
  MCS51
  )
