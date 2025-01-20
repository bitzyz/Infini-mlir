
include_directories(Dialect/IR/include)
include_directories(Traits/include)
include_directories(Interfaces/include)
include_directories(Support/include)
include_directories(include)

add_subdirectory(Interfaces)
add_subdirectory(Support)
add_subdirectory(Dialect)
add_subdirectory(Translation)
add_subdirectory(CAPI)
add_subdirectory(python)
add_subdirectory(Tools)
add_subdirectory(src)

add_library(infini_mlir INTERFACE)
target_include_directories(infini_mlir INTERFACE Dialect/IR/include Traits/include Interfaces/include Support/include include ${CMAKE_CURRENT_BINARY_DIR}/Interfaces/include)
target_link_libraries(infini_mlir INTERFACE translation)
