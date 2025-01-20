import numpy as np

class BaseImporter(object):
    def __init__(self):
        self.operands = dict()
        self.tensors = dict()
        self.shapes = dict()
        self.input_names = list()
        self.output_names = list()

    def generate_mlir(self, mlir_file: str):
        raise NotImplementedError('generate_mlir')

    def addShape(self, name, shape):
        if len(shape) == 0:
            shape = [1]
        self.shapes[name] = shape

    def getShape(self, name):
        return self.shapes[name]

    def addOperand(self, name, op):
        self.operands[name] = op

    def getOperand(self, name):
        return self.operands[name]

    def getOp(self, name):
        if self.isWeight(name):
            return self.getWeightOp(name)
        return self.getOperand(name)

    def addWeight(self, name, data):
        self.tensors[name] = data
        self.addShape(name, data.shape)

    def isWeight(self, name):
        if name in self.tensors:
            return True
        return False

    def getWeight(self, name):
        if name not in self.tensors:
            raise KeyError("No {} tensor in model".format(name))
        return self.tensors[name]

    def getWeightOp(self, name, shape: list = []):
        if name not in self.tensors:
            raise KeyError("Should addWeight first:{}!!!".format(name))
        old_shape = self.getShape(name)
        if shape and old_shape != shape:
            assert (np.prod(old_shape) == np.prod(shape))
            old_shape = shape
        ori_type = str(self.tensors[name].dtype)
        type_dict = {
            'int8': "INT8",
            'uint8': "UINT8",
            'float32': "F32",
            'int32': "INT32",
            'int16': "INT16",
            'uint16': "UINT16",
        }
        if ori_type not in type_dict:
            raise KeyError("type {} not implemented".format(ori_type))
        op = self.mlir.create_weight_op(name, old_shape, type_dict[ori_type])
        self.addOperand(name, op)
        return op
        
    def WeightToNpz(self, weight_file):
        tensor_npz = {}
        for name in self.tensors:
            if name in self.operands:
                tensor_npz[name] = self.tensors[name]
        np.savez(weight_file, **tensor_npz)
