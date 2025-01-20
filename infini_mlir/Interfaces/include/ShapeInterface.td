#ifndef INFINI_MLIR_SHAPEINTERFACE
#define INFINI_MLIR_SHAPEINTERFACE

include "mlir/IR/OpBase.td"

def ShapeInterface : OpInterface<"ShapeInterface"> {
  let description = [{
    For shape inference. 
  }];
  let cppNamespace = "infini::infinimlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          shape_inference.
        }],
        /*retType=*/"void",
        /*methodName=*/"shape_inference",
        /*args=*/(ins)
      >,
  ];
}
#endif // INFINI_MLIR_SHAPEINTERFACE
