# 添加包含路径
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Dialect/IR/include ${CMAKE_CURRENT_BINARY_DIR}/../Interfaces/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Dialect ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Transforms/include)
add_mlir_library(INFINIMLIRInitAll
  InitAll.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  INFINIMLIRDialect
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
)
