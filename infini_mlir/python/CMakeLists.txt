include(AddMLIRPython)

include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

set(INFINI_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/infini_mlir")
set(INFINI_MLIR_PYTHON_PACKAGES_DIR "${CMAKE_BINARY_DIR}/python")

include_directories("${CMAKE_SOURCE_DIR}/infini_mlir/Dialect/IR/include")
include_directories("${CMAKE_SOURCE_DIR}/infini_mlir/CAPI/include")

declare_mlir_python_sources(INFINIMLIRPythonSources)
declare_mlir_python_sources(INFINIMLIRPythonSources.Dialects
  ROOT_DIR "${INFINI_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT INFINIMLIRPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT INFINIMLIRPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/infini_mlir"
  TD_FILE dialects/InfiniOps.td
  SOURCES dialects/infini.py
  DIALECT_NAME infini)

declare_mlir_python_extension(INFINIMLIRPythonExtension.RegisterEverything
  MODULE_NAME _mlirRegisterEverything
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/infini_mlir"
  SOURCES
    RegisterEverything.cpp
  EMBED_CAPI_LINK_LIBS
    INFINIMLIRCAPI
)

add_mlir_python_common_capi_library(INFINIMLIRPythonCAPI
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/mlir/_mlir_libs"
  DECLARED_SOURCES
    INFINIMLIRPythonSources
    INFINIMLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

add_mlir_python_modules(INFINIMLIRPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python/mlir"
  DECLARED_SOURCES
    INFINIMLIRPythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.func
    INFINIMLIRPythonExtension.RegisterEverything
  COMMON_CAPI_LINK_LIBS
    INFINIMLIRPythonCAPI
)
