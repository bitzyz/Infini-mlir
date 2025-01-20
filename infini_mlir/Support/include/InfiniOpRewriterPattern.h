#pragma once
#include "Module.h"
#include <iostream>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/PatternMatch.h>
#include <string>
#include <unordered_map>

template <typename SourceOp>
class InfiniOpRewriterPattern : public mlir::OpRewritePattern<SourceOp> {
public:
  InfiniOpRewriterPattern(mlir::MLIRContext *context,
                      llvm::StringRef patternName = "",
                      mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<SourceOp>(context, benefit),
        patternName(patternName) {}

  mlir::LogicalResult matchAndRewrite(SourceOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::LogicalResult result = matchAndRewriteImpl(op, rewriter);
    if (mlir::succeeded(result)) {
      if (!patternName.empty()) {
        ++infini::infinimlir::module::patternMatchCounts[patternName];
      }
    }
    return result;
  }

protected:
  virtual mlir::LogicalResult matchAndRewriteImpl(SourceOp op, mlir::PatternRewriter &rewriter) const = 0;

private:
  std::string patternName;
  static void printPatternMatchCounts() {
    for (const auto &entry : infini::infinimlir::module::patternMatchCounts) {
      std::cout << "Pattern [" << entry.first << "] matched " << entry.second
                << " times.\n";
    }
  }
};
