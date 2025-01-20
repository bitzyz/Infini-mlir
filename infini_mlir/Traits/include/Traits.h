#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace infini {
namespace infinimlir {
namespace trait {

template <typename ConcreteType>
class SupportFuseRelu
    : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportFuseRelu> {};


} // namespace trait
} // namespace infinimlir
} // namespace infini