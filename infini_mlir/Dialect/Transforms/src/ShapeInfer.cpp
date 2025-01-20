#include "Transforms/include/Passes.h"
#include "Module.h"
#include <llvm/Support/Debug.h>
#define DEBUG_TYPE "shape_infer"

using namespace llvm;

namespace infini {
namespace infinimlir {
using mlir::Operation;

class ShapeInferPass : public ShapeInferBase<ShapeInferPass> {
public:
  ShapeInferPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    // Do shape infer
    for (auto func : mOp.getOps<FuncOp>()) {
      // func.walk([&](ShapeInterface op) {
      //   LLVM_DEBUG(llvm::dbgs() << "shape infer: " << op << "\n";);
      //   op.shape_inference();
      //   op.dump();
      // });
      func.walk([&](Operation *op) {
        if (auto shapeOp = dyn_cast<ShapeInterface>(op)) {
          LLVM_DEBUG(llvm::dbgs() << "shape infer: " << *op << "\n";);
          shapeOp.shape_inference();
          op->dump();
        }
      });
    }
    module::updateModuleTypes(mOp);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createShapeInferPass() {
  return std::make_unique<ShapeInferPass>();
}
} // namespace infinimlir
} // namespace infini