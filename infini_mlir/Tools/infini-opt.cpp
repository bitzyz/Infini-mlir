#include<fstream>

#include "InitAll.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
using namespace mlir;

int main(int argc, char **argv) {
  infini::infinimlir::registerAllPasses();
  DialectRegistry registry;
  infini::infinimlir::registerAllDialects(registry);

  // 处理 debug_cmd (如果需要)
  if (argc > 1) {
    std::string debug_cmd = argv[argc - 1];
    std::string substring = "--debug_cmd=";
    if (debug_cmd.find(substring) != std::string::npos) {
      std::ofstream ofs;
      ofs.open("/tmp/debug_cmd", std::ios::out | std::ios::trunc);
      ofs << debug_cmd.substr(substring.size()) << std::endl;
      argc -= 1;
    }
  }

  return asMainReturnCode(MlirOptMain(
      argc, argv, "INFINI MLIR module optimizer driver\n", registry));
}