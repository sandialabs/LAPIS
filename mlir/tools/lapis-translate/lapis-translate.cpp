//===- lapis-translate.cpp - MLIR Translate Driver -------------------------===//
//
// **** This file has been modified from its original in llvm-project ****
// Original file was mlir/lib/Tools/mlir-translate/MlirTranslateMain.cpp
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates an MLIR module to Kokkos C++.
//
//===----------------------------------------------------------------------===//

#include "lapis/InitAllKokkosTranslations.h"
#include "lapis/Target/KokkosCpp/KokkosCppEmitter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Tools/ParseUtilities.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

// All dialects that the translation depends on
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;

int main(int argc, char **argv) {
  registerAllKokkosTranslations();

  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> cxxFilename(
      "o", llvm::cl::desc("C++ output filename"), llvm::cl::value_desc("C++ (.cpp) filename"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> cxxHeaderFilename(
      "hpp", llvm::cl::desc("C++ header output filename (optional)"), llvm::cl::value_desc("C++ header filename (optional)"),
      llvm::cl::init(""));

  static llvm::cl::opt<std::string> pythonFilename(
      "py", llvm::cl::desc("Python output filename (optional)"), llvm::cl::value_desc("Python filename (optional)"),
      llvm::cl::init(""));

  static llvm::cl::opt<bool> isLastKernel(
      "finalize", llvm::cl::desc("Whether this module's Python destructor should finalize Kokkos"), llvm::cl::value_desc("finalize"),
      llvm::cl::init(false));

  static llvm::cl::opt<bool> emitTeamLevel(
      "team-level", llvm::cl::desc("Whether this module should be emitted as team-level code"), llvm::cl::value_desc("team-level"),
      llvm::cl::init(false));

  // For backward compatiblity, allow the --mlir-to-kokkos flag to be passed (but the value is ignored;
  // lapis-translate always does this translation)
  static llvm::cl::opt<bool> unused(
      "mlir-to-kokkos", llvm::cl::desc("Perform MLIR to Kokkos translation (always on)"), llvm::cl::value_desc(""),
      llvm::cl::init(true));

  llvm::InitLLVM y(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv, "lapis-translate");

  if(pythonFilename.size() && emitTeamLevel) {
    llvm::errs() << "Cannot emit python wrapper when emitting team-level kernels\n";
    return 1;
  }

  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  std::unique_ptr<llvm::ToolOutputFile> cxxOutput = openOutputFile(cxxFilename, &errorMessage);
  if (!cxxOutput) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  std::unique_ptr<llvm::ToolOutputFile> pythonOutput = nullptr;
  if(pythonFilename.length()) {
    pythonOutput = openOutputFile(pythonFilename, &errorMessage);
    if (!pythonOutput) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }
  }

  // Header output path (optional)
  std::unique_ptr<llvm::ToolOutputFile> cxxHeaderOutput = nullptr;
  if(cxxHeaderFilename.length()) {
    cxxHeaderOutput = openOutputFile(cxxHeaderFilename, &errorMessage);
    if (!cxxHeaderOutput) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }
  }

  MLIRContext context;
  context.allowUnregisteredDialects(false);
  context.printOpOnDiagnostic(true);
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(input), SMLoc());
  SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, &context);

  DialectRegistry registry;
  registry.insert<arith::ArithDialect,
                  cf::ControlFlowDialect,
                  emitc::EmitCDialect,
                  func::FuncDialect,
                  kokkos::KokkosDialect,
                  LLVM::LLVMDialect,
                  math::MathDialect,
                  memref::MemRefDialect,
                  scf::SCFDialect,
                  vector::VectorDialect>();
  context.appendDialectRegistry(registry);

  FallbackAsmResourceMap fallbackResourceMap;
  ParserConfig parseConfig(&context, /*verifyAfterParse=*/true, &fallbackResourceMap);

  OwningOpRef<Operation *> op = parseSourceFileForTool(sourceMgr, parseConfig, true);
  if (!op || failed(verify(*op))) {
    llvm::errs() << "Failed to parse input module\n";
    return 1;
  }
  raw_ostream* cxxHeaderStream = nullptr;
  if(cxxHeaderOutput)
    cxxHeaderStream = &cxxHeaderOutput->os();
  if(emitTeamLevel) {
    if(failed(kokkos::translateToKokkosCppTeamLevel(*op, &cxxOutput->os(), cxxHeaderStream, cxxHeaderFilename)))
      return 1;
  }
  else {
    if(pythonOutput) {
      if(failed(kokkos::translateToKokkosCpp(*op, &cxxOutput->os(), cxxHeaderStream, cxxHeaderFilename, &pythonOutput->os(), isLastKernel)))
        return 1;
    }
    else {
      if(failed(kokkos::translateToKokkosCpp(*op, &cxxOutput->os(), cxxHeaderStream, cxxHeaderFilename)))
        return 1;
    }
  }

  // Everything succeeded; don't delete the output files 
  cxxOutput->keep();
  if(cxxHeaderOutput)
    cxxHeaderOutput->keep();
  if(pythonOutput)
    pythonOutput->keep();

  return 0;
}

