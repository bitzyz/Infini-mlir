#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace infini {
namespace infinimlir {
namespace trait {

template <typename ConcreteType>
class ScalarProducer
    : public ::mlir::OpTrait::TraitBase<ConcreteType, ScalarProducer> {};

template <typename ConcreteType>
class ScalarConsumer
    : public ::mlir::OpTrait::TraitBase<ConcreteType, ScalarConsumer> {};

template <typename ConcreteType>
class SupportFuseRelu
    : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportFuseRelu> {};

template <typename ConcreteType>
class SupportConstant
    : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportConstant> {};

} // namespace trait
} // namespace infinimlir
} // namespace infini