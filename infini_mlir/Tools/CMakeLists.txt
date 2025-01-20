
set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

set(LIBS
  MLIRFuncDialect
  MLIROptLib
  INFINIMLIRInitAll
  )

add_llvm_executable(infini-opt
  infini-opt.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(infini-opt PRIVATE ${LIBS})
llvm_update_compile_flags(infini-opt)

mlir_check_all_link_libraries(infini-opt)

install(TARGETS infini-opt DESTINATION bin)
