//===- PartTensorConversion.cpp - Part tensor primitives conversion ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass that converts part tensor primitives into calls into a runtime
// support library. Part tensor types are converted into opaque pointers
// to the underlying part storage schemes. The use of opaque pointers
// together with runtime support library keeps the conversion relatively
// simple, but at the expense of IR opacity, which obscures opportunities
// for subsequent optimization of the IR. An alternative is provided by
// the PartTensorCodegen pass.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"

#include "lapis/Dialect/PartTensor/IR/PartTensor.h"
#include "lapis/Dialect/PartTensor/IR/PartTensorType.h"
#include "lapis/Dialect/PartTensor/Transforms/Passes.h"

#include "CodegenUtils.h"

using namespace mlir;
using namespace mlir::part_tensor;

namespace {

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

/// Maps each part tensor type to an opaque pointer.
static std::optional<Type> convertPartTensorTypes(Type type) {
  if (mlir::part_tensor::getPartTensorEncoding(type) != nullptr)
    return LLVM::LLVMPointerType::get(type.getContext());
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Conversion rules.
//===----------------------------------------------------------------------===//

constexpr auto getBackendPrefix = [](mlir::PartTensorDistBackend backend) {
  switch (backend) {
  case PartTensorDistBackend::kNone:
    return std::string("");
  case PartTensorDistBackend::kKRS:
    return std::string("krs_");
  case PartTensorDistBackend::kMPI:
    return std::string("mpi_");
  default:
    llvm_unreachable("Unknown PartTensorDistBackend");
  }
};

/// Part conversion rule for position accesses.
template <PartTensorDistBackend backend>
class PartTensorGetPartitionsConverter
    : public OpConversionPattern<GetPartitionsOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GetPartitionsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    const Type crdTp = cast<ShapedType>(resType).getElementType();
    Location loc = op->getLoc();
    MemRefType callRetType = MemRefType::get({ShapedType::kDynamic}, crdTp);
    SmallVector<Value> operands{adaptor.getOperands()[0]};
    auto fn = mlir::sparse_tensor::getFunc(
        op->getParentOfType<ModuleOp>(),
        getBackendPrefix(backend) + "getPartitions", callRetType, operands,
        mlir::sparse_tensor::EmitCInterface::On);
    Value callRet =
        rewriter.create<func::CallOp>(loc, callRetType, fn, operands)
            .getResult(0);
    if (resType != callRetType)
      callRet = rewriter.create<memref::CastOp>(loc, resType, callRet);
    rewriter.replaceOp(op, callRet);
    return success();
  }
};

template <PartTensorDistBackend backend>
class PartTensorGetActiveMaskConverter
    : public OpConversionPattern<GetActiveMaskOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GetActiveMaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    Location loc = op->getLoc();
    Type callRetType = resType;
    SmallVector<Value> operands{adaptor.getOperands()[0],
                                adaptor.getOperands()[1],
                                adaptor.getOperands()[2]};
    auto fn = mlir::sparse_tensor::getFunc(
        op->getParentOfType<ModuleOp>(),
        getBackendPrefix(backend) + "getActiveMask", callRetType, operands,
        mlir::sparse_tensor::EmitCInterface::On);
    Value callRet =
        rewriter.create<func::CallOp>(loc, callRetType, fn, operands)
            .getResult(0);
    if (resType != callRetType)
      callRet = rewriter.create<memref::CastOp>(loc, resType, callRet);
    rewriter.replaceOp(op, callRet);
    return success();
  }
};

template <PartTensorDistBackend backend>
class PartTensorGetSliceForActiveMask
    : public OpConversionPattern<GetSliceForActiveMaskOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GetSliceForActiveMaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Note using the namespace sparse_tensor here is not significant.
    // It just happens to be convenient to reuse the existing utility.
    Type resType = sparse_tensor::getOpaquePointerType(rewriter);
    Type origResType = op.getType();
    Location loc = op->getLoc();
    // replace %a = part_tensor.get_slice : part_tensor, ... -> sparse_tensor
    // with    %a1 = call @getSlice(%a) : (partTensor) -> i8*
    //         %a2 = unrealized_conversion_cast %a1 : i8* to %sparseTensor
    Value callRet =
        createFuncCall(
            rewriter, loc, getBackendPrefix(backend) + "getSliceForActiveMask",
            resType, adaptor.getOperands(), sparse_tensor::EmitCInterface::On)
            .getResult(0);
    callRet =
        rewriter.create<UnrealizedConversionCastOp>(loc, origResType, callRet)
            .getResult(0);
    rewriter.replaceOp(op, callRet);
    return success();
  }
};

template <PartTensorDistBackend backend>
class PartTensorGetSliceConverter : public OpConversionPattern<GetSliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GetSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Note using the namespace sparse_tensor here is not significant.
    // It just happens to be convenient to reuse the existing utility.
    Type resType = sparse_tensor::getOpaquePointerType(rewriter);
    Type origResType = op.getType();
    Location loc = op->getLoc();
    SmallVector<Value> operands{adaptor.getOperands()[0]};
    // replace %a = part_tensor.get_slice : part_tensor, ... -> sparse_tensor
    // with    %a1 = call @getSlice(%a) : (partTensor) -> i8*
    //         %a2 = unrealized_conversion_cast %a1 : i8* to %sparseTensor
    Value callRet =
        createFuncCall(rewriter, loc, getBackendPrefix(backend) + "getSlice",
                       resType, adaptor.getOperands(),
                       sparse_tensor::EmitCInterface::On)
            .getResult(0);
    callRet =
        rewriter.create<UnrealizedConversionCastOp>(loc, origResType, callRet)
            .getResult(0);
    rewriter.replaceOp(op, callRet);
    return success();
  }
};

template <PartTensorDistBackend backend>
class PartTensorGetNumPartitionsConverter
    : public OpConversionPattern<GetNumPartitionsOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GetNumPartitionsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    Location loc = op->getLoc();
    auto fn = mlir::sparse_tensor::getFunc(
        op->getParentOfType<ModuleOp>(),
        getBackendPrefix(backend) + "getNumPartitions", resType,
        adaptor.getOperands(), mlir::sparse_tensor::EmitCInterface::On);
    Value callRet =
        rewriter.create<func::CallOp>(loc, resType, fn, adaptor.getOperands())
            .getResult(0);
    rewriter.replaceOp(op, callRet);
    return success();
  }
};

template <PartTensorDistBackend backend>
class PartTensorSetSliceConverter : public OpConversionPattern<SetSliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SetSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    createFuncCall(rewriter, loc, getBackendPrefix(backend) + "setSlice", {},
                   {adaptor.getInPartTensor(), adaptor.getPartSpec(),
                    adaptor.getSparseTensor()},
                   mlir::sparse_tensor::EmitCInterface::On);
    rewriter.replaceOp(op, adaptor.getInPartTensor());
    return success();
  }
};

template <PartTensorDistBackend backend>
class PartTensorUpdateSliceConverter
    : public OpConversionPattern<UpdateSliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UpdateSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto sparseTensor = adaptor.getSparseTensor();
    (void)sparseTensor;
    createFuncCall(rewriter, loc, getBackendPrefix(backend) + "updateSlice", {},
                   {adaptor.getInPartTensor(), adaptor.getPartSpec(),
                    adaptor.getSparseTensor()},
                   mlir::sparse_tensor::EmitCInterface::On);
    rewriter.replaceOp(op, adaptor.getInPartTensor());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Part tensor type conversion into opaque pointer.
//===----------------------------------------------------------------------===//

mlir::PartTensorTypeToPtrConverter::PartTensorTypeToPtrConverter() {
  addConversion([](Type type) { return type; });
  addConversion(convertPartTensorTypes);
}

//===----------------------------------------------------------------------===//
// Public method for populating conversion rules.
//===----------------------------------------------------------------------===//

/// Populates the given patterns list with conversion rules required for
/// the sparsification of linear algebra operations.
void mlir::populatePartTensorConversionPatterns(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns,
                                                PartTensorDistBackend backend) {
  switch (backend) {
  case PartTensorDistBackend::kNone: {
    patterns
        .add<PartTensorGetPartitionsConverter<PartTensorDistBackend::kNone>>(
            typeConverter, patterns.getContext());
    patterns.add<PartTensorGetSliceConverter<PartTensorDistBackend::kNone>>(
        typeConverter, patterns.getContext());

    patterns
        .add<PartTensorGetNumPartitionsConverter<PartTensorDistBackend::kNone>>(
            typeConverter, patterns.getContext());

    patterns.add<PartTensorSetSliceConverter<PartTensorDistBackend::kNone>>(
        typeConverter, patterns.getContext());

    patterns.add<PartTensorUpdateSliceConverter<PartTensorDistBackend::kNone>>(
        typeConverter, patterns.getContext());
  } break;
  case PartTensorDistBackend::kKRS: {
    patterns.add<PartTensorGetPartitionsConverter<PartTensorDistBackend::kKRS>>(
        typeConverter, patterns.getContext());
    patterns.add<PartTensorGetSliceConverter<PartTensorDistBackend::kKRS>>(
        typeConverter, patterns.getContext());

    patterns
        .add<PartTensorGetNumPartitionsConverter<PartTensorDistBackend::kKRS>>(
            typeConverter, patterns.getContext());

    patterns.add<PartTensorSetSliceConverter<PartTensorDistBackend::kKRS>>(
        typeConverter, patterns.getContext());

    patterns.add<PartTensorUpdateSliceConverter<PartTensorDistBackend::kKRS>>(
        typeConverter, patterns.getContext());
  } break;
  case PartTensorDistBackend::kMPI: {
    patterns.add<PartTensorGetPartitionsConverter<PartTensorDistBackend::kMPI>>(
        typeConverter, patterns.getContext());
    patterns.add<PartTensorGetActiveMaskConverter<PartTensorDistBackend::kMPI>>(
        typeConverter, patterns.getContext());
    patterns.add<PartTensorGetSliceForActiveMask<PartTensorDistBackend::kMPI>>(
        typeConverter, patterns.getContext());
    patterns.add<PartTensorGetSliceConverter<PartTensorDistBackend::kMPI>>(
        typeConverter, patterns.getContext());

    patterns
        .add<PartTensorGetNumPartitionsConverter<PartTensorDistBackend::kMPI>>(
            typeConverter, patterns.getContext());

    patterns.add<PartTensorSetSliceConverter<PartTensorDistBackend::kMPI>>(
        typeConverter, patterns.getContext());

    patterns.add<PartTensorUpdateSliceConverter<PartTensorDistBackend::kMPI>>(
        typeConverter, patterns.getContext());
  } break;
  default:
    llvm_unreachable("Unknown PartTensorDistBackend");
  }
}
