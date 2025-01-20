include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/IR/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Dialect/IR/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Interfaces/include)
add_mlir_public_c_api_library(INFINIMLIRCAPI
  src/RegisterEverything.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  INFINIMLIRDialect
  MLIRFuncDialect
)
