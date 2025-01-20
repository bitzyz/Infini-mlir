from MLIRBuilder import MLIRBuilder, Platform
from BaseImporter import BaseImporter
from onnx import numpy_helper, mapping
from numbers import Number
import onnx
import numpy as np
import mlir.dialects.infini as infini
from mlir.ir import *
import copy
import logging

onnx_attr_translator = {
    "axis": lambda x: int(x),
    "axes": lambda x: [int(a) for a in x],
    "dtype": lambda x: onnx_dtype(x),
    "keepdims": lambda x: bool(x),
    "to": lambda x: onnx_dtype(x),
}

def translate_onnx(key, val):
    return onnx_attr_translator.get(key, lambda x: x)(val)

def onnx_dtype(dtype):
    if isinstance(dtype, Number):
        onnx_dtype = dtype
    elif isinstance(dtype, str):
        onnx_dtype = onnx.TensorProto.DataType.Value(dtype)
    else:
        raise RuntimeError("dtype should be number or str.")
    return mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype]

def convert_onnx_attribute_proto(attr_proto):
    if attr_proto.HasField('f'):
        return attr_proto.f
    elif attr_proto.HasField('i'):
        return attr_proto.i
    elif attr_proto.HasField('s'):
        return attr_proto.s
    elif attr_proto.HasField('t'):
        return attr_proto.t  # this is a proto!
    elif attr_proto.floats:
        return list(attr_proto.floats)
    elif attr_proto.ints:
        return list(attr_proto.ints)
    elif attr_proto.strings:
        str_list = list(attr_proto.strings)
        return str_list
    elif attr_proto.name:
        name_list = list(attr_proto.name)
        return name_list
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))

class BaseNode():
    def __init__(self, info):
        self.name = str(info["name"])
        self.op_type = str(info["op_type"])
        self.attrs = dict(info["attrs"])
        self.inputs = list(info["inputs"])
        self.outputs = list(info["outputs"])
        self.shape_info = dict()

class OnnxNode(BaseNode):
    def __init__(self, node):
        info = dict()
        info["name"] = node.output[0]
        info["op_type"] = node.op_type
        info["attrs"] = [(attr.name, translate_onnx(attr.name, convert_onnx_attribute_proto(attr)))
                         for attr in node.attribute]
        info["inputs"] = node.input
        info["outputs"] = node.output
        super().__init__(info)
        logging.info("node:{}".format(info))
        self.node_proto = node

class OnnxImporter(BaseImporter):
    def __init__(self,
                 model_name: str,
                 onnx_file,
                 input_shapes: list,
                 output_names: list,
                 dynamic_shape_input_names=[],
                 dynamic=False):
        super().__init__()

        self.dynamic_shape_input_names = dynamic_shape_input_names
        if self.dynamic_shape_input_names:
            self.dynamic = "manual"
            dynamic = True
        elif dynamic:
            self.dynamic = "auto"
        else:
            self.dynamic = "off"
        self.model_name = model_name
        self.weight_file = "{}_infini_origin_weight.npz".format(model_name)
        self.model = None
        self.mlir = None
        self.origin_output_names = output_names.copy()
        self.load_onnx_model(onnx_file, input_shapes, output_names)
        self.init_MLIRBuilder()
        self.unranked_type = self.mlir.get_tensor_type([])
        self.converted_nodes = list()

        self.onnxop_factory = {
            "Add": lambda node: self.convert_add_op(node),
            "AveragePool": lambda node: self.convert_avgpool_op(node),
            "BatchNormalization": lambda node: self.convert_batchnorm_op(node),
            "Conv": lambda node: self.convert_conv_op(node),
            "Gemm": lambda node: self.convert_gemm_op(node),
            "GlobalAveragePool": lambda node: self.convert_global_avgpool_op(node),
            "GlobalMaxPool": lambda node: self.convert_global_maxpool_op(node),
            "MatMul": lambda node: self.convert_gemm_op(node),
            "MaxPool": lambda node: self.convert_maxpool_op(node),
            "Relu": lambda node: self.convert_relu_op(node),
            "Reshape": lambda node: self.convert_reshape_op(node),
            "Sigmoid": lambda node: self.convert_sigmoid_op(node),
            "Transpose": lambda node: self.convert_transpose_op(node),
        }

    def __del__(self):
        if self.mlir != None:
            del self.mlir
            self.mlir = None

    def get_outputs(self, model: onnx.ModelProto):
        initializer_names = [x.name for x in model.graph.initializer]
        return [opt for opt in model.graph.output if opt.name not in initializer_names]

    def get_inputs(self, model: onnx.ModelProto):
        initializer_names = [x.name for x in model.graph.initializer]
        return [ipt for ipt in model.graph.input if ipt.name not in initializer_names]

    def get_input_names(self, model: onnx.ModelProto):
        input_names = [ipt.name for ipt in self.get_inputs(model)]
        return input_names

    def get_input_types(self, model: onnx.ModelProto):
        input_types = []
        for input in self.get_inputs(model):
            if input.type.tensor_type.elem_type in [onnx.TensorProto.INT64, onnx.TensorProto.INT32]:
                input_types.append('INT32')
            else:
                input_types.append('F32')
        return input_types

    def get_output_types(self, model: onnx.ModelProto):
        output_types = []
        for output in self.get_outputs(model):
            if output.type.tensor_type.elem_type in [
                    onnx.TensorProto.INT64, onnx.TensorProto.INT32
            ]:
                output_types.append('INT32')
            else:
                output_types.append('F32')
        return output_types

    def get_shape_from_value_info_proto(self, v: onnx.ValueInfoProto):
        return [dim.dim_value for dim in v.type.tensor_type.shape.dim]

    def get_input_shapes(self, model: onnx.ModelProto):
        inputs = self.get_inputs(model)
        return [self.get_shape_from_value_info_proto(i) for i in inputs]

    def get_loc(self, names):
        if isinstance(names, str):
            return Location.fused([Location.name(names)], context=self.mlir.ctx)
        elif isinstance(names, list):
            return Location.fused([Location.name(n) for n in names], context=self.mlir.ctx)
        else:
            raise RuntimeError("Unknown names:{}".format(names))

    def load_onnx_model(self, onnx_file, input_shapes: list, output_names: list):
        if isinstance(onnx_file, str):
            self.model = onnx.load(onnx_file)
        else:
            self.model = onnx_file
        self.input_names = self.get_input_names(self.model)
        self.num_input = len(self.input_names)


        self.dynamic_input_names_auto_assign()

        self.input_shape_assign(input_shapes)

        self.input_shapes = self.get_input_shapes(self.model)
        self.input_types = self.get_input_types(self.model)
        self.output_types = self.get_output_types(self.model)
        # add all weight
        for tensor in self.model.graph.initializer:
            name = tensor.name
            data = numpy_helper.to_array(tensor).astype(np.float32)
            self.addWeight(name, data)
        self.get_output_name(self.model.graph)
        logging.basicConfig(filename='output.log', level=logging.INFO)
        logging.info("input_names:{}".format(self.input_names))
        logging.info("output_names:{}".format(self.output_names))
        logging.info("shapes:{}".format(self.shapes))
        logging.info("operands:{}".format(self.operands))
        logging.info("tensors:{}".format(self.tensors))
        logging.info("input_shapes:{}".format(self.input_shapes))
        logging.info("input_types:{}".format(self.input_types))
        logging.info("output_types:{}".format(self.output_types))


    def get_output_name(self, graph):
        for output in graph.output:
            if not self.isWeight(output.name):
                self.output_names.append(output.name)

    def dynamic_input_names_auto_assign(self):
        if self.dynamic != "auto":
            return
        inputs = self.get_inputs(self.model)
        for input in inputs:
            _dims = input.type.tensor_type.shape.dim
            if _dims:
                for _dim in _dims:
                    if _dim.dim_value == 0 and _dim.dim_param:
                        self.dynamic_shape_input_names.append(input.name)
                        break

    def input_shape_assign(self, input_shapes):
        inputs = self.get_inputs(self.model)
        outputs = self.get_outputs(self.model)
        shape_changed = False
        no_shape = True

        def check_shape(l, r):
            if no_shape == False and l != r:
                raise KeyError("input shapes error:{}, {} vs {}".format(input_shapes, l, r))

        if len(input_shapes) > 0:
            no_shape = False
            check_shape(self.num_input, len(input_shapes))
        for idx, input in enumerate(inputs):
            _dims = input.type.tensor_type.shape.dim
            # for 1-element scalars that has no shape, assign [1] as shape to convert to tensor
            if not _dims:
                _dims.append(onnx.TensorShapeProto.Dimension(dim_value=1))
            num_dims = len(_dims)
            if no_shape == False:
                check_shape(num_dims, len(input_shapes[idx]))
            _shape = []
            for _i, _dim in enumerate(_dims):
                if _dim.dim_value <= 0:
                    if no_shape:
                        assert 0, "Please check --input_shapes formula or check if there is any dynamic dim"
                    else:
                        _dim.dim_value = input_shapes[idx][_i]
                # elif not no_shape:
                #     check_shape(_dim_value, input_shapes)
                elif not no_shape and input_shapes[idx][_i] != _dim.dim_value:
                    _dim.dim_value = input_shapes[idx][_i]
                    shape_changed = True
                _shape.append(_dim.dim_value)
            self.addShape(input.name, _shape)
        idx = 0  # avoid confilict for multi dynamic axes
        for o in outputs:
            # for set arbitrary axes
            _odims = o.type.tensor_type.shape.dim
            for _odim in _odims:
                if _odim.dim_value <= 0 or shape_changed:
                    _odim.dim_param = '?_' + str(idx)
                    idx += 1
    
    def init_MLIRBuilder(self):
        input_shapes = list()
        for _name in self.input_names:
            input_shapes.append(self.getShape(_name))
        output_shapes = list()
        output_shapes = len(self.output_names) * [[]]
        # init builder
        self.mlir = MLIRBuilder(input_shapes, output_shapes, self.model_name, Platform.ONNX, self.input_types)
        self.weight_file = self.mlir.weight_file

    def get_shape_for_node(self, input, output, value_info, name):
        for i in value_info:
            if i.name == name:
                return i.type.tensor_type.shape.dim
        for i in input:
            if i.name == name:
                return i.type.tensor_type.shape.dim
        for i in output:
            if i.name == name:
                return i.type.tensor_type.shape.dim

    def generate_mlir(self, mlir_file: str):
        # add input op
        for idx, _name in enumerate(self.input_names):
            input_ = self.mlir.create_input_op(self.get_loc(_name), idx)
            self.addOperand(_name, input_)

        self.converted_nodes.clear()
        for n in self.model.graph.node:
            node = OnnxNode(n)
            self.converted_nodes.append(node)

        for n in self.converted_nodes:
            self.onnxop_factory.get(n.op_type, lambda x: NoneAndRaise(x))(n)
        # add return op
        return_op = list()
        # Set output
        final_output_names = []
        if self.origin_output_names:
            final_output_names = self.origin_output_names
        else:
            final_output_names = self.output_names
        for idx, _name in enumerate(final_output_names):
            op = self.getOperand(_name)
            return_op.append(op)

        self.mlir.create_return_op(return_op)
        logging.info("operands:{}".format(self.operands))
        mlir_txt = self.mlir.print_module()
        with open(mlir_file, "w") as f:
            f.write(mlir_txt)
        self.WeightToNpz(self.weight_file)

    def convert_add_op(self, onnx_node):
        assert (onnx_node.op_type == "Add")
        assert (len(onnx_node.inputs) == 2)
        lhs = onnx_node.inputs[0]
        rhs = onnx_node.inputs[1]
        if self.isWeight(lhs) and not self.isWeight(rhs):
            onnx_node.inputs[0], onnx_node.inputs[1] = onnx_node.inputs[1], onnx_node.inputs[0]
            self.convert_add_op(onnx_node)
            return
        name = "{}_{}".format(onnx_node.name, onnx_node.op_type)
        lhs_op = self.getOp(lhs)
        rhs_op = self.getOp(rhs)
        new_op = infini.AddOp(self.unranked_type, lhs_op, rhs_op,
                            loc=self.get_loc(name),
                            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_batchnorm_op(self, onnx_node):
        assert (onnx_node.op_type == "BatchNormalization")
        op = self.getOperand(onnx_node.inputs[0])
        gamma = self.getWeightOp(onnx_node.inputs[1])
        beta = self.getWeightOp(onnx_node.inputs[2])
        mean = self.getWeightOp(onnx_node.inputs[3])
        variance = self.getWeightOp(onnx_node.inputs[4])
        epsilon = onnx_node.attrs.get("epsilon")
        new_op = infini.BatchNormOp(self.unranked_type,
                                 op,
                                 mean,
                                 variance,
                                 gamma,
                                 beta,
                                 epsilon=epsilon,
                                 loc=self.get_loc("{}_{}".format(onnx_node.name,
                                                                 onnx_node.op_type)),
                                 ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_conv_op(self, onnx_node):
        assert (onnx_node.op_type == "Conv")
        op = self.getOp(onnx_node.inputs[0])
        kernel_shape = onnx_node.attrs['kernel_shape']
        dim = len(kernel_shape)
        dilations = onnx_node.attrs.get("dilations", dim * [1])
        group = onnx_node.attrs.get("group", 1)
        strides = onnx_node.attrs.get("strides", dim * [1])
        pads = onnx_node.attrs.get("pads", dim * 2 * [0])
        operands = list()
        operands.append(op)
        filter_op = self.getOp(onnx_node.inputs[1])
        operands.append(filter_op)
        if len(onnx_node.inputs) > 2:
            bias_op = self.getWeightOp(onnx_node.inputs[2])
        else:
            bias_op = self.mlir.none_op
        operands.append(bias_op)
        new_op = infini.ConvOp(self.unranked_type,
                            *operands,
                            kernel_shape=kernel_shape,
                            strides=strides,
                            dilations=dilations,
                            pads=pads,
                            group=group,
                            loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_gemm_op(self, onnx_node):
        assert (onnx_node.op_type == "Gemm" or onnx_node.op_type == 'MatMul')
        # (M, K) * (K, N) => (M, N)
        alpha = onnx_node.attrs.get('alpha', 1)
        beta = onnx_node.attrs.get('beta', 1)
        trans_a = onnx_node.attrs.get('transA', 0)
        trans_b = onnx_node.attrs.get('transB', 0)
        assert (trans_a == 0)
        operands = list()
        A = onnx_node.inputs[0]
        B = onnx_node.inputs[1]
        if self.isWeight(A):
            if trans_a == 1 or alpha != 1:
                _tensor = self.getWeight(A)
                _tensor = copy.deepcopy(_tensor)
                if trans_a == 1:
                    _tensor = np.ascontiguousarray(np.transpose(_tensor, (1, 0)))
                if alpha != 1:
                    _tensor *= alpha
                A += '_fix'
                self.addWeight(A, _tensor)
            operands.append(self.getWeightOp(A))
        else:
            operands.append(self.getOperand(A))

        if self.isWeight(B):
            if trans_b == 1 or alpha != 1:
                _tensor = self.getWeight(B)
                _tensor = copy.deepcopy(_tensor)  #if change weight,should do deepcopy
                if trans_b == 1:
                    _tensor = np.ascontiguousarray(np.transpose(_tensor, (1, 0)))
                if alpha != 1:
                    _tensor *= alpha
                B += '_fix'
                self.addWeight(B, _tensor)
            operands.append(self.getWeightOp(B))
        else:
            operands.append(self.getOperand(B))
        if len(onnx_node.inputs) > 2 and beta != 0:
            C = onnx_node.inputs[2]
            if self.isWeight(C):
                if beta != 1:
                    _tensor = self.getWeight(C)
                    _tensor = copy.deepcopy(_tensor)
                    _tensor *= beta
                    C += '_fix'
                    self.addWeight(C, _tensor)
                operands.append(self.getWeightOp(C))
            else:
                operands.append(self.getOperand(C))

        new_op = infini.MatMulOp(self.unranked_type,
                              *operands,
                              loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                              ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_global_avgpool_op(self, onnx_node):
        assert (onnx_node.op_type == "GlobalAveragePool")
        op = self.getOperand(onnx_node.inputs[0])
        new_op = infini.AvgPoolOp(self.unranked_type,
                               op,
                               kernel_shape=[],
                               strides=[],
                               pads=[],
                               count_include_pad=True,
                               keepdims=True,
                               loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                               ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_maxpool_op(self, onnx_node):
        assert (onnx_node.op_type == "MaxPool")
        op = self.getOperand(onnx_node.inputs[0])
        ceil_mode = onnx_node.attrs.get("ceil_mode", False)
        kernel_shape = onnx_node.attrs['kernel_shape']
        count_include_pad = onnx_node.attrs.get('count_include_pad', False)
        dim = len(kernel_shape)
        strides = onnx_node.attrs.get("strides", kernel_shape)
        pads = onnx_node.attrs.get("pads", dim * 2 * [0])
        new_op = infini.MaxPoolOp(self.unranked_type,
                               op,
                               kernel_shape=kernel_shape,
                               strides=strides,
                               pads=pads,
                               ceil_mode=ceil_mode,
                               count_include_pad=count_include_pad,
                               loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                               ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_relu_op(self, onnx_node):
        assert (onnx_node.op_type == "Relu")
        op = self.getOperand(onnx_node.inputs[0])
        new_op = infini.ReluOp(self.unranked_type,
                            op,
                            loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                            ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)

    def convert_reshape_op(self, onnx_node):
        assert (onnx_node.op_type == "Reshape")
        op = self.getOperand(onnx_node.inputs[0])
        if self.isWeight(onnx_node.inputs[1]):
            shape = self.getWeight(onnx_node.inputs[1])
            new_op = infini.ReshapeOp(self.unranked_type,
                                op,
                                shape=shape,
                                loc=self.get_loc("{}_{}".format(onnx_node.name, onnx_node.op_type)),
                                ip=self.mlir.insert_point).output
        self.addOperand(onnx_node.name, new_op)
