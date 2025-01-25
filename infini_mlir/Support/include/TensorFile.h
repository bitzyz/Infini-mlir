#ifndef MLIR_SUPPORT_TENSORFILE_H_
#define MLIR_SUPPORT_TENSORFILE_H_

#include "cnpy.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h" 
#include <fstream>
#include <string>


namespace mlir {

class TensorFile {
public:
  TensorFile(llvm::StringRef filename, bool readOnly, bool newCreate = false);

  ~TensorFile();

  template <typename T>
  LogicalResult readTensor(llvm::StringRef name, T *data, size_t count);
  template <typename T>
  std::unique_ptr<std::vector<T>>
  readTensor(llvm::StringRef name, RankedTensorType &type);

private:
  /// load the file
  LogicalResult load(void);

  std::string filename;
  bool readOnly;
  cnpy::npz_t map;
};

} // namespace mlir

#endif // MLIR_SUPPORT_TENSORFILE_H_
