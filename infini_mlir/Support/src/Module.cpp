#include <optional>
#include <cstdint>
#include <iostream>
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ModuleEnum.h.inc"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ModuleEnum.cpp.inc"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace infini {
namespace infinimlir {
typedef std::shared_ptr<std::vector<int64_t>> i64_array_t;
namespace module {

using mlir::Value;
using mlir::RankedTensorType;
using mlir::ShapedType;
using mlir::NoneType;
using mlir::UnrankedTensorType;
using mlir::Type;
using mlir::func::FuncOp;
using mlir::ModuleOp;
using llvm::StringRef;
using mlir::Builder;
using mlir::Block;
using mlir::MLIRContext;
using mlir::func::ReturnOp;
using llvm::dyn_cast; 
using llvm::cast;
using mlir::func::CallOp;
using mlir::WalkResult;
using mlir::WalkOrder;
using mlir::Operation;
using mlir::ArrayAttr;
using mlir::IntegerAttr;

// static MLIRContext *ctx = nullptr;
// static ModuleOp m = nullptr;

struct Attr {
  static constexpr llvm::StringRef WEIGHT_FILE = "module.weight_file";
  static constexpr llvm::StringRef PLATFORM = "module.platform";
};

bool isUnranked(Value v) { return v.getType().isa<mlir::UnrankedTensorType>(); }

bool isDynamicShape(Value v) {
  int ret = false;
  auto tensorTy = v.getType().dyn_cast<RankedTensorType>();
  if (tensorTy) {
    for (int64_t dim : tensorTy.getShape()) {
      if (ShapedType::isDynamic(dim) || dim == 0)
        ret = true;
    }
  }
  return ret;
}

llvm::ArrayRef<int64_t> getShape(Value v) {
  if (v.getType().isa<NoneType>()) {
    v.dump();
    llvm_unreachable("v is none type");
  }
  if (!isUnranked(v)) {
    auto type = v.getType().cast<RankedTensorType>();
    return type.getShape();
  } else {
    return v.getType().cast<UnrankedTensorType>().getShape();
  }
}

Type getElementType(Value v) {
  auto type = v.getType();
  if (type.isa<RankedTensorType>()) {
    auto rtype = type.cast<RankedTensorType>();
    return rtype.getElementType();
  } else if (type.isa<UnrankedTensorType>()) {
    auto rtype = type.cast<UnrankedTensorType>();
    return rtype.getElementType();
  }
  return type;
}

void setShapeOrVerify(Value v, llvm::ArrayRef<int64_t> shape) {
  if (isUnranked(v) || isDynamicShape(v)) {
    auto newType = RankedTensorType::get(shape, getElementType(v));
    v.setType(newType);
  } else {
    auto s = getShape(v);
    /* unranked tensor is okay, for example:
       tensor<*xf32>->tensor<1xf32> */
    if ((std::max(s.size(), shape.size()) > 1) && s != shape) {
      v.dump();
      llvm_unreachable("Shape Verify failed");
    }
  }
}

FuncOp getFuncOp(ModuleOp mod, StringRef func_name) {
  for (auto func : mod.getOps<FuncOp>()) {
    if (func.getName() == func_name) {
      return func;
    }
  }
  llvm::errs() << "Can't find FuncOp:" << func_name << "\n";
  llvm_unreachable("Error getFuncOp !!\n");
  return nullptr;
}

FuncOp getMainFuncOp(ModuleOp module) { return getFuncOp(module, "main_graph"); }

CallOp getCallOp(FuncOp func) {
  auto parent = func->getParentOp();
  auto s = cast<ModuleOp>(parent);
  CallOp call = nullptr;
  for (auto each_func : s.getOps<FuncOp>()) {
    WalkResult result =
        each_func.walk<WalkOrder::PreOrder>([&](CallOp op) {
          if (!call && op.getCallee() == func.getName()) {
            call = op;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted())
      break;
  }
  return call;
}

void updateModuleTypes(ModuleOp s) {
  Builder builder(s->getContext());
  // update callee func's return types
  for (auto func : s.getOps<FuncOp>()) {
    if (func.getName() == "main_graph") {
      continue;
    }
    std::vector<Type> returns;
    Block &entryBlock = func.front();
    auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
    for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
      returns.push_back(returnOp->getOperand(i).getType());
    }
    auto fnType = builder.getFunctionType(func.getArgumentTypes(),
                                          llvm::ArrayRef<Type>{returns});
    func.setType(fnType);
    auto callee = getCallOp(func);
    if (callee) {
      for (auto it : llvm::zip(callee.getResults(), returns)) {
        std::get<0>(it).setType(std::get<1>(it));
      }
    }
  }
  // update callee arg types
  for (auto func : s.getOps<FuncOp>()) {
    if (func.getName() == "main_graph") {
      continue;
    }
    auto callee = getCallOp(func);
    if (!callee) {
      continue;
    }
    std::vector<Type> arguments;
    for (auto it :
         llvm::zip(callee.getOperandTypes(), func.front().getArguments())) {
      arguments.push_back(std::get<0>(it));
      std::get<1>(it).setType(std::get<0>(it));
    }
    auto fnType = builder.getFunctionType(llvm::ArrayRef<Type>(arguments),
                                          func.getResultTypes());
    func.setType(fnType);
  }
  // update main op return types
  auto mainFunc = getMainFuncOp(s);
  Block &entryBlock = mainFunc.front();
  auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
  std::vector<Type> returns;
  for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
    returns.push_back(returnOp->getOperand(i).getType());
  }
  std::vector<Type> inputs;
  auto args = mainFunc.getArguments();
  for (auto arg : args) {
    inputs.push_back(arg.getType());
  }
  auto fnType = builder.getFunctionType(llvm::ArrayRef<Type>{inputs},
                                        llvm::ArrayRef<Type>{returns});
  mainFunc.setType(fnType);
}

std::shared_ptr<std::vector<ModuleOp>> getAllModules(ModuleOp module) {
  auto modules = std::make_shared<std::vector<ModuleOp>>();
  auto sub = module.getOps<ModuleOp>();
  if (sub.empty()) {
    modules->push_back(module);
  } else {
    modules->assign(sub.begin(), sub.end());
  }
  return modules;
}

// void updateModuleTypes(ModuleOp module) {
//   auto modules = getAllModules(module);
//   for (auto s : *modules) {
//     updateModuleTypes(s);
//   }
// }

void unreachable(const char *info, Operation *op, const char *file,
                 unsigned line) {
  std::cerr << "ASSERT executed at" << file << ":" << line << std::endl;
  std::cerr << "ASSERT INFO:" << info << std::endl << "Operation:" << std::endl;
  if (op != nullptr) {
    auto inputs = op->getOperands();
    if (!inputs.empty()) {
        for (auto input : inputs)
          input.dump();
    }
    std::cerr << "-> ";
    op->dump();
    for (auto out : op->getResults()) {
      for (auto user : out.getUsers())
        user->dump();
    }
  }
  exit(-1);
}

void assert_with_dump(bool cond, Operation *op, const char *info,
                      const char *file, unsigned line) {
  if (cond) {
    return;
  }
  unreachable(info, op, file, line);
}

i64_array_t getI64Array(ArrayAttr arrayAttr) {
  auto data = std::make_shared<std::vector<int64_t>>();
  for (auto en : llvm::enumerate(arrayAttr)) {
    auto attr = en.value().dyn_cast<IntegerAttr>();
    if (attr) {
      data->push_back(attr.getInt());
    } else {
      arrayAttr.dump();
      llvm_unreachable("not int64_t type");
    }
  }
  return data;
}

i64_array_t getI64Array(std::optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        int64_t default_value) {
  if (arrayAttr.has_value()) {
    auto arr = getI64Array(arrayAttr.value());
    assert(static_cast<int64_t>(arr->size()) == num_elem);
    return arr;
  }
  return std::make_shared<std::vector<int64_t>>(num_elem, default_value);
}

int64_t getNumElements(Value v) {
  if (v.getType().isa<RankedTensorType>() == false) {
    return 0;
  }
  auto type = v.getType().cast<RankedTensorType>();
  return type.getNumElements();
}

} // namespace module
} // namespace infinimlir
} // namespace infini