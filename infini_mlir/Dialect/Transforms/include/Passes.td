#ifndef INFINI_MLIR_DIALECT_PASSES
#define INFINI_MLIR_DIALECT_PASSES

include "mlir/Pass/PassBase.td"

def ShapeInfer : Pass<"shape-infer", "ModuleOp"> {
  let summary = "do shape inference for each op";
  let constructor = "createShapeInferPass()";
  let dependentDialects = ["InfiniDialect"];
}

#endif
