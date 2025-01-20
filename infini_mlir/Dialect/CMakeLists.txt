set(Dialect_Dir ${CMAKE_CURRENT_BINARY_DIR}/IR/include)
include_directories(${Dialect_Dir} ${CMAKE_CURRENT_BINARY_DIR}/../Interfaces/include ${CMAKE_CURRENT_SOURCE_DIR}/../Interfaces/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/Transforms/include ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/../Support/include)
add_subdirectory(IR/include)
add_subdirectory(IR/src)
add_subdirectory(Transforms/include)
add_library(InfiniDialectDirectories INTERFACE)
target_include_directories(InfiniDialectDirectories INTERFACE ${Dialect_Dir})
file(GLOB _sources
          Interfaces/*.cpp
          IR/src/*.cc
          Transforms/src/*.cpp
    )

add_mlir_library(INFINIMLIRDialect
  ${_sources}

  DEPENDS
  MLIRInfiniOpsIncGen
  MLIRInfiniPassIncGen
  MLIRShapeInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTransforms
  MLIRSupport
  MLIRFuncDialect
  MLIRShapeInterface
  INFINIMLIRSupport
)