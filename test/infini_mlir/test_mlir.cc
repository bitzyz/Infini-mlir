#include "core/graph.h"
#include "core/runtime.h"
#include "operators/element_wise.h"
#include "gtest/gtest.h"
#include "test.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "ConvertToInfini.h"
#include "mlir/IR/BuiltinOps.h"

namespace infini {
namespace infinimlir {

TEST(Graph, coverttomlir) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i1 = g->addTensor({2, 3, 4, 5}, DataType::UInt32);
    Tensor i2 = g->addTensor({2, 3, 4, 5}, DataType::UInt32);
    Tensor o = g->addTensor({2, 3, 4, 5}, DataType::UInt32);
    i1->setInput();
    i2->setWeight();
    o->setOutput();
    g->addOpWithOutputs<AddObj>(i1, i2, o);

    g->dataMalloc();
    i2->setData(OneGenerator());

    g->print();
    g->optimize();
    g->print();
}

TEST(Graph, mlir2graph) {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<infini::infinimlir::InfiniDialect>();
    std::string filename = "../../infini_mlir/python/translation/output.mlir";
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    auto moduleRef = mlir::parseSourceString(buffer.str(), &context);
    if (!moduleRef) {
        std::cerr << "Failed to parse MLIR module" << std::endl;
        return;
    }

    auto module = moduleRef->clone();
    mlir::ModuleOp moduleOp = llvm::cast<mlir::ModuleOp>(module);
    moduleOp.dump();

    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Graph new_graph = convertMLIRToInfini(moduleOp, g->getRuntime());
    new_graph->print();
}

} // namespace infinimlir
} // namespace infini
