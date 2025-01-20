#include "InfiniOpRewriterPattern.h"
#include "Traits.h"
#include "InfiniOps.h"
#include <iostream>

using namespace infini::infinimlir;
using namespace infini::infinimlir::trait;
using mlir::RewritePatternSet;
using mlir::MLIRContext;
using mlir::LogicalResult;
using mlir::PatternRewriter;

struct FuseRelu : public InfiniOpRewriterPattern<ReluOp> {
  using InfiniOpRewriterPattern::InfiniOpRewriterPattern;
  FuseRelu(mlir::MLIRContext *context)
      : InfiniOpRewriterPattern<ReluOp>(context, "FuseRelu") {}

  LogicalResult matchAndRewriteImpl(ReluOp op,
                                    PatternRewriter &rewriter) const override {
    std::cout << "FuseRelu\n";
    auto formerOp = op.getInput().getDefiningOp();
    formerOp->dump();
    if (!formerOp->getResult(0).hasOneUse()) {
      std::cout<<"oneuse\n";
      return mlir::failure();
    }

    if (formerOp->hasTrait<SupportFuseRelu>() == false) {
      formerOp->dump();
      std::cout<<"trait\n";
      return mlir::failure();
    }
    formerOp->setAttr("do_relu", rewriter.getBoolAttr(true));
    formerOp->setLoc(op.getLoc());
    // remove the relu Op
    rewriter.replaceOp(op, {op.getInput()});
    std::cout<<"Finish\n";
    return mlir::success();

  }
};

void ReluOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<FuseRelu>(context);
}
