#ifndef INFINI_OPS
#define INFINI_OPS

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Traits.td"
include "ShapeInterface.td"

def Infini_Dialect : Dialect {
    let name = "infini";
    let summary = "Infini dialect for Deep Learning Operations";
    let cppNamespace = "infini::infinimlir";
}

// InfiniTypes
def AnyTensorOrNone: AnyTypeOf<[AnyTensor, NoneType]>;

// BaseOp
class Infini_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<Infini_Dialect, mnemonic, traits> ;

def Infini_NoneOp : Infini_BaseOp<"None"> {
  let summary = "none operation";

  let description = [{
    A none Op to return a NoneType.
  }];
  let results = (outs NoneType);
}

def Infini_WeightOp : Infini_BaseOp<"Weight"> {
  let summary = "Weight operation";

  let description = [{
      Load weight from model.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$weight
  );

  let results = (outs AnyRankedTensor:$output);
  // let hasCanonicalizer = 1;
}

def Infini_InputOp: Infini_BaseOp<"Input"> {
  let summary = "Input operation";

  let description = [{
    Input operation for shapeinterface and preprocess.
  }];

  let arguments = (
    ins AnyRankedTensor:$input,
    DefaultValuedAttr<BoolAttr, "false">:$preprocess
  );

  let results = (outs AnyTensor:$output);
}

// InfiniOp
class Infini_Op<string mnemonic, list<Trait> traits = []> :
    Infini_BaseOp<mnemonic, [DeclareOpInterfaceMethods<ShapeInterface>]>;

def Infini_AddOp: Infini_Op<"Add", [SupportFuseRelu, SupportConstant, ScalarProducer, ScalarConsumer]> {
  let summary = "Add operation";

  let description = [{
    Add two tensor.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$input2
  );

  let results = (outs AnyTensor:$output);
  // let hasCanonicalizer = 1;
}

def Infini_BatchNormOp: Infini_Op<"BatchNorm", [SupportFuseRelu]> {
  let summary = "BatchNormalization operation";
  let description = [{
    TODO: add the description of batchnormalization.
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$mean,
    AnyTensor:$variance,
    AnyTensorOrNone:$gamma,
    AnyTensorOrNone:$beta,
    DefaultValuedAttr<F32Attr, "1e-05">:$epsilon
  );
  let results = (outs AnyTensor:$output);
  // let hasCanonicalizer = 1;
}

def Infini_ConstantOp : Infini_Op<"Constant", [Pure]> {
  let summary = "Constant operation";
  let description = [{
    Create a constant tensor.
  }];
  let arguments = (ins
    DefaultValuedAttr<I64Attr, "0">:$size,
    I64Attr:$data_ptr
  );
  let results = (outs AnyType:$output);

  let extraClassDeclaration = [{
    int64_t getBitWidth();
  }];
}

def Infini_ConvOp: Infini_Op<"Conv", [SupportFuseRelu]> {
  let summary = "Convolution operation";

  let description = [{
    TDDO: add the description of convolution.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight,
    AnyTensorOrNone:$bias,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    DefaultValuedAttr<I64Attr, "1">:$group,
    OptionalAttr<I64ArrayAttr>:$dilations
  );

  let results = (outs AnyTensor:$output);
  // let hasCanonicalizer = 1;
}

def Infini_MatMulOp: Infini_Op<"MatMul", [SupportFuseRelu]> {
  let summary = "Matmul operation";

  let description = [{
    Support gemm.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$right,
    AnyTensorOrNone:$bias,
    DefaultValuedAttr<BoolAttr, "false">:$right_transpose,
    DefaultValuedAttr<BoolAttr, "false">:$left_transpose,
    DefaultValuedAttr<BoolAttr, "false">:$output_transpose,
    DefaultValuedAttr<BoolAttr, "false">:$hdim_is_batch,
    DefaultValuedAttr<BoolAttr, "true">:$keep_dims
  );

  let results = (outs AnyTensor:$output);
  // let hasCanonicalizer = 1;
}

class Infini_PoolOp <string mnemonic> : Infini_Op<mnemonic,[SupportFuseRelu]> {
  let summary = "Pool operation";

  let description = [{
    TODO: add the description of pooling.
  }];

  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    OptionalAttr<BoolAttr>:$ceil_mode,
    DefaultValuedAttr<BoolAttr, "true">:$keepdims,
    DefaultValuedAttr<BoolAttr, "false">:$count_include_pad
  );

  let results = (outs AnyTensor:$output);
  
}

def Infini_AvgPoolOp:Infini_PoolOp<"AvgPool">;
def Infini_MaxPoolOp:Infini_PoolOp<"MaxPool">;

def Infini_ReluOp: Infini_Op<"Relu"> {
  let summary = "Relu operation";

  let description = [{
     TODO: add the description of relu.
  }];

  let arguments = (
    ins AnyTensor:$input
  );

  let results = (outs AnyTensor:$output);

  // let hasCanonicalizer = 1;
}

def Infini_ReshapeOp:Infini_Op<"Reshape"> {
  let summary = "Reshape operation";
  let description = [{
    TODO: add the description of reshape.
  }];
  let arguments = (ins
    AnyTensor:$input,
    OptionalAttr<I64ArrayAttr>:$shape
  );
  let results = (outs AnyTensor:$output);
  // let hasCanonicalizer = 1;
  // let hasFolder = 1;
}

#endif // INFINI_OPS
