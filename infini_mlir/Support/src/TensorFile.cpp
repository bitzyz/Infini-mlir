#include "TensorFile.h"

namespace mlir {

template <typename T> static bool check_type(Type eltType) {
  bool same;
  if (eltType.isF32()) {
    same = std::is_same<T, float>::value;
  } else {
    same = false;
  }
  if (same != true) {
    eltType.dump();
    llvm::errs() << "\nnot equal to Type "
                 << "\n";
  }
  return same;
}

TensorFile::TensorFile(llvm::StringRef filename, bool readOnly, bool newCreate)
    : filename(filename), readOnly(readOnly) {
  if (!newCreate) {
    std::ifstream f(filename.str());
    if (!f.good()) {
      llvm::errs() << "WARNING, " << filename
                   << " doesn't exist, please check\n";
    }
    auto ret = load();
    if (!succeeded(ret)) {
      if (readOnly) {
        llvm::errs() << filename << " not exist, failed to read for read\n";
        llvm_unreachable("TensorFile error!");
      }
      map.clear();
    }
  } else {
    map.clear();
  }
}

TensorFile::~TensorFile() {}


template std::unique_ptr<std::vector<float>>
TensorFile::readTensor<float>(llvm::StringRef name, RankedTensorType &type);

template <typename T>
LogicalResult TensorFile::readTensor(llvm::StringRef name, T *data, size_t count) {
  auto it = map.find(name.str());
  if (it == map.end()) {
    llvm::errs() << "failed to find tensor " << name.str() << " to read\n";
    llvm_unreachable("readTensor failed");
    return failure();
  }
  auto arr = it->second;
  if (arr.num_bytes() != count * sizeof(T)) {
    llvm::errs() << "size does not match for tensor " << name.str() << "\n";
    llvm_unreachable("readTensor failed");
    return failure();
  }
  
  memcpy(data, arr.data_holder->data(), arr.num_bytes());
  return success();
}

template <typename T>
std::unique_ptr<std::vector<T>> TensorFile::readTensor(llvm::StringRef name, RankedTensorType &type) {
  size_t count = 1;
  auto s = type.getShape();
  if (s.size() > 0 ) {
    count = type.getNumElements();
    assert(check_type<T>(type.getElementType()) == true);
  }

  auto data = std::make_unique<std::vector<T>>(count);
  auto ret = readTensor(name, (T *)data.get()->data(), count);
  assert(succeeded(ret));
  return data;
}

LogicalResult TensorFile::load(void) {
  map = cnpy::npz_load(filename);
  if (map.size() > 0) {
    return success();
  } else {
    return failure();
  }
}

std::string filename;
bool readOnly;
cnpy::npz_t map;
}; // namespace mlir
