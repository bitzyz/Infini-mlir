#pragma once
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Attributes.h"
#include "ModuleEnum.h.inc"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringRef.h"

namespace infini {
namespace infinimlir {
    
typedef std::shared_ptr<std::vector<int64_t>> i64_array_t;

namespace module {

using mlir::Value;
using mlir::Type;
using mlir::func::FuncOp;
using mlir::ModuleOp;
using llvm::StringRef;
using mlir::func::CallOp;
using mlir::Operation;
using mlir::ArrayAttr;

Type getElementType(Value v);
llvm::ArrayRef<int64_t> getShape(Value v);
bool isUnranked(Value v);
void setShapeOrVerify(Value v, llvm::ArrayRef<int64_t> shape);

FuncOp getFuncOp(ModuleOp mod, StringRef func_name);

FuncOp getMainFuncOp(ModuleOp module);
CallOp getCallOp(FuncOp func);
std::shared_ptr<std::vector<ModuleOp>> getAllModules();

void updateModuleTypes(ModuleOp s);

void assert_with_dump(bool cond, Operation *op, const char *info,
                      const char *file = nullptr, unsigned line = 0);
void unreachable(const char *info, Operation *op = nullptr,
                 const char *file = nullptr, unsigned line = 0);
i64_array_t getI64Array(ArrayAttr arrayAttr);
i64_array_t getI64Array(std::optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        int64_t default_value);

int64_t getNumElements(Value v);
} // namespace module

#define ASSERT_OP(COND, OP)                                                    \
  module::assert_with_dump(COND, OP, #COND, __FILE__, __LINE__)
#define ASSERT_THIS(COND) ASSERT_OP(COND, this->getOperation())

} // namespace infinimlir
} // namespace infini
