import argparse
from BaseImporter import BaseImporter
from utils import str2shape, str2list, mlir_opt
class ModelTranslator(object):
    def __init__(self, model_name, model_file):
        self.model_name = model_name
        self.model_file = model_file
        self.importer = BaseImporter()

    def model_translate(self, mlir_file: str):
        self.mlir_file = mlir_file
        self.importer.generate_mlir(mlir_file)
        self.mlir_file = mlir_file
        mlir_origin = mlir_file.replace('.mlir', '_origin.mlir', 1)
        if self.importer:
            self.importer.generate_mlir(mlir_origin)
        else:
            mlir_origin = self.model_file
        mlir_opt(mlir_origin, self.mlir_file)

class OnnxTranslator(ModelTranslator):
    def __init__(self,
                 model_name,
                 model_file,
                 input_shapes: list = [],
                 output_names: list = [],
                 dynamic_shape_input_names: list = [],
                 dynamic=False):
        super().__init__(model_name, model_file)
        from OnnxImporter import OnnxImporter
        self.importer = OnnxImporter(self.model_name, self.model_file, input_shapes, output_names, 
                                     dynamic_shape_input_names=dynamic_shape_input_names, 
                                     dynamic=dynamic)


def get_model_translate(args):
    tool = None
    if args.model_file.endswith('.onnx'):
        tool = OnnxTranslator(args.model_name, args.model_file, args.input_shapes,
                              dynamic_shape_input_names=args.dynamic_shape_input_names,
                              dynamic=args.dynamic)
    # TODO: support more model types.
    return tool

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="model name")
    parser.add_argument("--model_file", required=True, help="model file.")
    parser.add_argument("--input_shapes", type=str2shape, default=list(), help="list of input shapes")
    parser.add_argument("--mlir", type=str, required=True, help="output mlir file")
    parser.add_argument("--dynamic_shape_input_names", type=str2list, default=list(), help="name list of inputs with dynamic shape")
    parser.add_argument("--dynamic", action='store_true', help='dynamic shape')
    args = parser.parse_args()
    tool = get_model_translate(args)
    tool.model_translate(args.mlir)
