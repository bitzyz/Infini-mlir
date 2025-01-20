file(GLOB _sources *.cpp)

add_llvm_library(INFINIMLIRSupport
  ${_sources}

  DEPENDS
  ModuleAttrGen
  MLIRInfiniOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
