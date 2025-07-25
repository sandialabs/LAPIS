//===- KokkosPasses.cpp - Passes for lowering to Kokkos dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <map>
#include <utility> // pair
#include <random>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h" //for SparseParallelizationStrategy

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_KOKKOSMDRANGEITERATION

#include "lapis/Dialect/Kokkos/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::kokkos;

namespace {


/* The basic idea:

we have something like this

scf.parallel (%i, %j) {                             (1)
  memref.store[%j, %i] : memref<10, 20, f32>        (2)
  scf.parallel (%k, %l) {                           (3)
    _ = memref.load[%i, %k] : memref<10, ?, f32>    (4)
  }
}

Given
  * scf.parallel will be convert to a Kokkos MDRange with Iterate::Right for inner and outer iteration orders
  * memrefs point to LayoutRight allocations
What is the best order for the scf.parallel induction variables? e.g. for (1), do we want (%i, %j) or (%j, %i)?
To answer, we build a cost model of each memref load/store, and choose the induction variable ordering for all scf.parallels that minimizes that cost.
The cost model of a memref has three parts:
  * The number of times the memref is executed: a memref that is executed more is more costly
  * How the memref's access offset changes w.r.t an induction variable: the smaller the offset, the better locality
  * Whether the memref is a load or a store: stores are expected to be more costly

=====================================================================
==== Part 0: Exploring scf.parallel Induction Variable Orderings ====
=====================================================================

All possible configurations of induction variable orderings are explored.
This number is pretty tractable.

Each scf.parallel has N! configurations for N induction variables.
If they are nested, those values are mulitplied together.<unnamed>
In practice we're limited to ~trillions of elements, so even a 5- or 6-D loop is quite large (each extent would have to be small)
Even an 8-D loop is only 40k configurations.

scf.parallel (%1 %2 %3 %4) { // 4!
  scf.parallel (%5 %6 %7)    // 4! * 3!
     
  }
}

The cost of consecutive scf.parallel is just the sum of the costs of each of them independently.<unnamed>

==========================================================
==== Part 1: The Number of Times a memref is Executed ====
==========================================================
For each memref, we look at all the scf.parallel regions that it's nested inside and multiply their iteration spaces together.
Limitations:
  * Ignore conditional execution

==== Part 2: The Stride of a memref w.r.t an scf.parallel induction variable ===

In general, it's impossible to compute the stride of a memref w.r.t an induction variable.
Consider the following example pseudocode

scf.parallel (%i) ... {
   %j = f(%i)
   memref.load(%j)
}

f could do literally anything

However, in practice, it is common for memref indices to be "simple" functions of induction varibles - scalings, offsets, etc.
So, model this stride as a derivative:

  d(memref) / d(induction variable), the partial derivative of the accessed offset w.r.t the induction variable. We can ignore the base address, because it's derivative w.r.t all induction variables is 0

Via the chain rule, the sum of various partial derivatives (index variable meaning an argument to the memref)
d(memref) / d(induction variable) = SUM (
                                         pd(memref)         / pd(index variable)     *
                                         pd(index variable) / pd(induction variable)
                                        )

pd(memref) / pd(index variable) is in principle simple, it's just the product of all strides of right-ward (due to LayoutRight) dimension of the index variable.
See Part 4 for discussion of runtime extents.

pd(index variable) / pd(induction variable) is computed by recursively following the inputs to the operation and applying differentiation rules.
For simplicity, We only try to differentiate simple arithmetic functions, e.g. if f(x) = g(x) + h(x), df/dx = dg/dx + dh/dx. Similarly for multiply, divide, etc. Any other f we just give up and say who knows.


==== Part 3: Memref Load or Store ====

Easily determined. A load has its otherwise-computed cost scaled by 1.0. A store is scaled by 3.0.

=========================================
==== Part 4: Handling Runtime Values ====
=========================================

Any loop trip counts and memref extents that are not statically known (whether due to insufficient analysis or runtime-derived values) are considered "unknowns".
The expressions of interest in Parts 1, 2, and 3 are modeled as an expression tree, with symbolic values inserted for these unknowns.

The Monte-Carlo method is used to estimate the cost in the face of these unknowns.
For each unknown, a value between 1 and 100,000 is selected in a log-uniform way (the range 1-10 is roughly as likely as 10-100, etc.)
The cost expressions are evaluted in this "context" to arrive at a concrete numerical cost for the memref.
The median of 500 simulations is chosen as the single final number.


=========================================
==== Part 5: Putting it all Together ====
=========================================

* For each possible parallel loop induction variable ordering
  * Model the cost of the program memrefs under that configuration using Monte-Carlo method
  * Track the best identified ordering so far
* Apply the selected ordering to each scf.parallel


=============================
==== Part 6: Limitations ====
=============================

It is easy for the anlysis to lose track of the relationship between an induction variable and an index variable. In that case, the contributed cost is 0.

Being conditionally-executed (inside an `if`) does not effect the cost model. If a memref is present in the code, we consider it unconditionally executed.

The analysis does not attempt to actually model the memory hierarchy:
  * The cost is proportional to the stride, which is probably not true on any actual computer
  * Imagine if v(0,9) is next to v(10, 0) in memory. We still count the increment of the left index variable as stride 10, rather than stride 1.
    * This is semi-defensible because the iteration order in Kokkos is not specified, but we actually know the Kokkos implementation so we could probably do something better here.
  * We don't model memrefs in the context of previously-executed memrefs. Conescutive accesses to the same address are probably cached, but we ignore that.

*/

// Put these out here so we can overload operator<< easily
namespace {

  // a context for expression evaluation
  struct Ctx {
    std::unordered_map<std::string, int> values; // FIXME: llvm data structures
  };

  struct Expr {
    enum class Kind {
      Add, Sub, Mul, Div, Constant, Unknown
    };

    Expr(Kind kind) : kind_(kind) {}
    Kind kind_;

    virtual size_t eval(const Ctx &ctx) = 0;
    virtual void dump(llvm::raw_fd_ostream &os) const = 0;
    virtual void dump(llvm::raw_ostream &os) const = 0;
    virtual std::vector<std::string> unknowns() const = 0;
    virtual std::shared_ptr<Expr> clone() const = 0;
    virtual ~Expr() {}
  };

template <typename OS>
OS & operator<<(OS &os, const std::shared_ptr<Expr> &e) {
  e->dump(os);
  return os;
}


} // namespace



struct KokkosMdrangeIterationPass
    : public impl::KokkosMdrangeIterationBase<KokkosMdrangeIterationPass> {

#if 0
#define MDRANGE_DEBUG(x) \
  llvm::outs() << x;
#else
#define MDRANGE_DEBUG(x)
#endif

  KokkosMdrangeIterationPass() = default;
  KokkosMdrangeIterationPass(const KokkosMdrangeIterationPass& pass) = default;

  using ParallelOpVec = llvm::SmallVector<scf::ParallelOp, 4>;
  using ValueVec = llvm::SmallVector<Value, 8>;
  using Permutation = llvm::SmallVector<size_t, 16>;

  // generate a log-random integer within a specified range
  static size_t log_random_int(std::mt19937 &gen, size_t min, size_t max) {
      // Create a uniform real distribution between log(min) and log(max)
      std::uniform_real_distribution<> dis(std::log(min), std::log(max));

      // Generate a random number in the log space and exponentiate it
      double logRandom = dis(gen);
      size_t res = std::exp(logRandom);

      // Ensure the result is within the desired range
      if (res < min) res = min;
      if (res > max) res = max;

      return res;
  }



  struct Binary : public Expr {
    Binary(Kind kind, const std::string sym, std::shared_ptr<Expr> lhs, std::shared_ptr<Expr> rhs) : Expr(kind), sym_(sym), lhs_(lhs), rhs_(rhs) {}

    template <typename OS>
    void dump_impl(OS &os) const {
      os << "(";
      lhs_->dump(os);
      os << sym_;
      rhs_->dump(os);
      os << ")";
    }

    virtual void dump(llvm::raw_fd_ostream &os) const override {
      dump_impl(os);
    }

    virtual void dump(llvm::raw_ostream &os) const override {
      dump_impl(os);
    }

    virtual std::vector<std::string> unknowns() const override {
      std::vector<std::string> ret;

      if (!lhs_) {
        llvm::report_fatal_error("lhs_ is null");
      }
      if (!rhs_) {
        llvm::report_fatal_error("rhs_ is null");
      }

      for (auto &op : {lhs_, rhs_}) {
        for (auto &name : op->unknowns()) {
          ret.push_back(name);
        }
      }
      return ret;
    }

    protected:
      std::string sym_;
      std::shared_ptr<Expr> lhs_;
      std::shared_ptr<Expr> rhs_;
  };

  struct Add : public Binary {
    Add(std::shared_ptr<Expr> lhs, std::shared_ptr<Expr> rhs) : Binary(Kind::Add, "+", lhs, rhs) {}

    virtual size_t eval(const Ctx &ctx) override {
      return lhs_->eval(ctx) + rhs_->eval(ctx);
    }

    virtual std::shared_ptr<Expr> clone() const override {
      return make(lhs_->clone(), rhs_->clone());
    }

    static std::shared_ptr<Expr> make(std::shared_ptr<Expr> lhs, std::shared_ptr<Expr> rhs) {
      auto lhs_const = llvm::dyn_cast<Constant>(lhs.get());
      auto rhs_const = llvm::dyn_cast<Constant>(rhs.get());

      if (lhs_const && lhs_const->value_ == 0) {
          return rhs;
      } else if (rhs_const && rhs_const->value_ == 0) {
          return lhs;
      } else if (rhs_const && lhs_const) {
        return Constant::make(lhs_const->value_ + rhs_const->value_);
      }

      return std::make_shared<Add>(lhs, rhs);
    }

    static bool classof(const Expr *e) {
      return e->kind_ == Expr::Kind::Add;
    }
  };

  struct Sub : public Binary {
    Sub(std::shared_ptr<Expr> lhs, std::shared_ptr<Expr> rhs) : Binary(Kind::Add, "-", lhs, rhs) {}

    virtual size_t eval(const Ctx &ctx) override {
      return lhs_->eval(ctx) + rhs_->eval(ctx);
    }

    virtual std::shared_ptr<Expr> clone() const override {
      return make(lhs_->clone(), rhs_->clone());
    }

    static std::shared_ptr<Expr> make(std::shared_ptr<Expr> lhs, std::shared_ptr<Expr> rhs) {
      auto lhs_const = llvm::dyn_cast<Constant>(lhs.get());
      auto rhs_const = llvm::dyn_cast<Constant>(rhs.get());

      if (lhs_const && lhs_const->value_ == 0) { // 0 - x --> -x
          return Mul::make(rhs, Constant::make(-1));
      } else if (rhs_const && rhs_const->value_ == 0) { // x - 0 --> x
          return lhs;
      } else if (rhs_const && lhs_const) { // c1 - c2 --> c3
        return Constant::make(lhs_const->value_ - rhs_const->value_);
      }

      return std::make_shared<Sub>(lhs, rhs);
    }

    static bool classof(const Expr *e) {
      return e->kind_ == Expr::Kind::Sub;
    }
  };

  struct Mul : public Binary {
    Mul(std::shared_ptr<Expr> lhs, std::shared_ptr<Expr> rhs) : Binary(Kind::Mul, "*", lhs, rhs) {}

    virtual size_t eval(const Ctx &ctx) override {
      return lhs_->eval(ctx) * rhs_->eval(ctx);
    }

    virtual std::shared_ptr<Expr> clone() const override {
      return make(lhs_->clone(), rhs_->clone());
    }

    static std::shared_ptr<Expr> make(std::shared_ptr<Expr> lhs, std::shared_ptr<Expr> rhs) {
      auto lhs_const = llvm::dyn_cast<Constant>(lhs.get());
      auto rhs_const = llvm::dyn_cast<Constant>(rhs.get());

      if (rhs_const && lhs_const) {
        return Constant::make(lhs_const->value_ * rhs_const->value_);
      } else if (lhs_const && lhs_const->value_ == 1) { // 1 * x
          return rhs;
      } else if (lhs_const && lhs_const->value_ == 0) { // 0 * x
          return Constant::make(0);
      } else if (rhs_const && rhs_const->value_ == 1) { // x * 1
          return lhs;
      } else if (rhs_const && rhs_const->value_ == 0) { // x * 0
          return Constant::make(0);
      } else if (rhs_const && rhs_const->value_ == -1) { // x * -1
          return Constant::make(-lhs_const->value_);
      } else if (lhs_const && lhs_const->value_ == -1) { // -1 * x
          return Constant::make(-rhs_const->value_);
      }

      return std::make_shared<Mul>(lhs, rhs);
    }

    static bool classof(const Expr *e) {
      return e->kind_ == Expr::Kind::Mul;
    }
  };

  struct Div : public Binary {
    Div(std::shared_ptr<Expr> lhs, std::shared_ptr<Expr> rhs) : Binary(Kind::Div, "/", lhs, rhs) {}

    virtual size_t eval(const Ctx &ctx) override {
      return lhs_->eval(ctx) / rhs_->eval(ctx);
    }

    virtual std::shared_ptr<Expr> clone() const override {
      return make(lhs_->clone(), rhs_->clone());
    }

    static std::shared_ptr<Expr> make(std::shared_ptr<Expr> lhs, std::shared_ptr<Expr> rhs) {
      auto lhs_const = llvm::dyn_cast<Constant>(lhs.get());
      auto rhs_const = llvm::dyn_cast<Constant>(rhs.get());

      if (rhs_const && lhs_const) {
        return Constant::make(lhs_const->value_ * rhs_const->value_);
      } else if (lhs_const && lhs_const->value_ == 0) { // 0 / x
          return Constant::make(0);
      } else if (rhs_const && rhs_const->value_ == 1) { // x / 1
          return lhs;
      } else if (rhs_const && rhs_const->value_ == -1) { // x / -1
          return Constant::make(-lhs_const->value_);
      }

      return std::make_shared<Div>(lhs, rhs);
    }

    static bool classof(const Expr *e) {
      return e->kind_ == Expr::Kind::Div;
    }
  };


  struct Constant : public Expr {
    Constant(int value) : Expr(Kind::Constant), value_(value) {}
    int value_;

    virtual size_t eval(const Ctx &ctx) override {
      return value_;
    }

    virtual std::shared_ptr<Expr> clone() const override {
      return make(value_);
    }

    virtual void dump(llvm::raw_fd_ostream &os) const override {
      dump_impl(os);
    }

    virtual void dump(llvm::raw_ostream &os) const override {
      dump_impl(os);
    }

    template <typename OS>
    void dump_impl(OS &os) const {
      os << value_;
    }

    virtual std::vector<std::string> unknowns() const override {
      return {};
    }

    static std::shared_ptr<Constant> make(int c) {
      return std::make_shared<Constant>(c);
    }

    static bool classof(const Expr *e) {
      return e->kind_ == Expr::Kind::Constant;
    }
  };

  struct Unknown : public Expr {
    Unknown(const std::string &name) : Expr(Kind::Unknown), name_(name) {}
    std::string name_;

    virtual size_t eval(const Ctx &ctx) override {
      return ctx.values.at(name_);
    }

    virtual std::shared_ptr<Expr> clone() const override {
      return make(name_);
    }

    virtual void dump(llvm::raw_fd_ostream &os) const override {
      dump_impl(os);
    }

    virtual void dump(llvm::raw_ostream &os) const override {
      dump_impl(os);
    }

    template <typename OS>
    void dump_impl(OS &os) const {
      os << "(" << name_ << ")";
    }

    virtual std::vector<std::string> unknowns() const override {
      return {name_};
    }

    static std::shared_ptr<Unknown> make(const std::string &name) {
      return std::make_shared<Unknown>(name);
    }

    static bool classof(const Expr *e) {
      return e->kind_ == Expr::Kind::Unknown;
    }

  };



  // cost model for memref / induction variable pair
  struct Cost {

    Cost(std::shared_ptr<Expr> stride, std::shared_ptr<Expr> count, int sf) : stride_(stride), count_(count), sf_(sf) {}
    Cost() = default;

    std::shared_ptr<Expr> stride_; // stride of the memref w.r.t an induction variable
    std::shared_ptr<Expr> count_;  // number of times the memref is executed
    size_t sf_; // scaling factor, 1 for load, 3 for store

    std::vector<std::string> unknowns() const {
      std::vector<std::string> ret;
      for (const std::string &u : stride_->unknowns()) ret.push_back(u);
      for (const std::string &u : count_->unknowns()) ret.push_back(u);
      return ret;
    }

    template <typename Memref>
    static constexpr int scale_factor() {
      static_assert(std::is_same_v<Memref, memref::LoadOp> || std::is_same_v<Memref, memref::StoreOp>);
      if constexpr (std::is_same_v<Memref, memref::LoadOp>) return 1;
      else if constexpr (std::is_same_v<Memref, memref::StoreOp>) return 3;
    }
  };

// partial derivative df/dx
// FIXME: does this handle block arguments appropriately?
static std::shared_ptr<Expr> df_dx(Value &f, Value &x) {
  if (f == x) {
    MDRANGE_DEBUG("Info: df_dx of equal values\n");
    return Constant::make(1);
  } else if (mlir::isa<BlockArgument>(f)) {
    MDRANGE_DEBUG("Info: f is a block argument\n");
    return Constant::make(0);
  } else if (mlir::isa<BlockArgument>(x)) {
    MDRANGE_DEBUG("Info: x is a block argument\n");
    return Constant::make(0);
  } else {
    // FIXME: what other scenarios if there is no defining op.
    if (auto fOp = f.getDefiningOp()) {
      if (auto xOp = x.getDefiningOp()) {
        return df_dx(fOp, xOp);
      }
    }
    {
      std::stringstream ss;
      ss << "Unexpected lack of defining op for value\n";
      MDRANGE_DEBUG(f << " <- " << f.getDefiningOp() << "\n");
      MDRANGE_DEBUG(x << " <- " << x.getDefiningOp() << "\n");
      llvm::report_fatal_error(ss.str().c_str());
    }
    
    return nullptr;
  }
}

  // FIXME: better written as df_dx(f, x) I guess
  static std::shared_ptr<Expr> df_dx(Operation *df, Operation *dx) {
    if (!df) {
      MDRANGE_DEBUG("Warn: df_dx requested on null df\n");
      return nullptr;
    } else if (!dx) {
      MDRANGE_DEBUG("Warn: df_dx requested on null dx\n");
      return nullptr;
    } else if (df == dx) {
      // df/dx (dx) = 1
      return Constant::make(1);
    } else if (auto constOp = dyn_cast<mlir::arith::ConstantIntOp>(df)) { // f is +
      return Constant::make(0);
    } else if (auto addOp = dyn_cast<mlir::arith::AddFOp>(df)) { // f is +
      // d(lhs + rhs)/dx = dlhs/dx + drhs/dx
      Value lhs = addOp.getOperand(0);
      Value rhs = addOp.getOperand(1);
      std::shared_ptr<Expr> dLhs = df_dx(lhs.getDefiningOp(), dx);
      std::shared_ptr<Expr> dRhs = df_dx(rhs.getDefiningOp(), dx);
      if (dLhs && dRhs) {
        return Add::make(dLhs, dRhs);
      } 
    } else if (auto mulOp = dyn_cast<mlir::arith::MulFOp>(df)) { // f is *
      // d(lhs * rhs)/dx = lhs * drhs/dx + rhs * dlhs/dx
      // we'll only bother to compute this one if lhs or rhs is a constant
      Value lhs = mulOp.getOperand(0);
      Value rhs = mulOp.getOperand(1);
      
      if (auto lhsConst = lhs.getDefiningOp<mlir::arith::ConstantIntOp>()) { // FIXME: is this all integral values?
        // lhs is a constant, so the derivative is lhs * drhs/dx
        std::shared_ptr<Expr> dRhs = df_dx(rhs.getDefiningOp(), dx);
        if (dRhs) {
          return Mul::make(Constant::make(cast<IntegerAttr>(lhsConst.getValue()).getInt()), dRhs); // FIXME: can this cast fail?
        }
      }
      
      if (auto rhsConst = rhs.getDefiningOp<mlir::arith::ConstantIntOp>()) { // FIXME: is this all integral values?
        // rhs is a constant, so the derivative is rhs * dlhs/dx
        std::shared_ptr<Expr> dLhs = df_dx(lhs.getDefiningOp(), dx);
        if (dLhs) {
          return Mul::make(Constant::make(cast<IntegerAttr>(rhsConst.getValue()).getInt()), dLhs); // FIXME: can this cast fail?
        }
      }
    } // TODO: sub, div

    MDRANGE_DEBUG("WARN: unhandled case in df_dx of " << df << " w.r.t " << dx << "\n");
    return nullptr;
  }



  // computes d(offset) / d(index variable)
  template <typename Memref>
  static std::shared_ptr<Expr> do_di(Memref &memrefOp, Value indexVar) {

    static_assert(std::is_same_v<Memref, memref::LoadOp> || std::is_same_v<Memref, memref::StoreOp>, "Memref must be either LoadOp or StoreOp");

    // find the index var
    int indexVarDim = 0;
    for (mlir::Value var : memrefOp.getIndices()) {
      if (var == indexVar) {

        auto memrefType = dyn_cast<MemRefType>(memrefOp.getMemRef().getType()); // FIXME: can this fail?

        // Get the size in bits of the element type
        mlir::Type elementType = memrefType.getElementType();
        const unsigned sizeInBytes = elementType.getIntOrFloatBitWidth() / CHAR_BIT;

        std::shared_ptr<Expr> res = std::make_shared<Constant>(sizeInBytes);
        
        // LayoutRight: work in from the right, multiplying in dimensions
        auto memrefShape = memrefType.getShape();
        const int nDim = memrefShape.size();
        for (int dim = nDim - 1; dim > indexVarDim; --dim) {
          if (memrefShape[dim] == ShapedType::kDynamic) {
            std::string name = std::string("memref") + std::to_string(uintptr_t(memrefOp.getOperation())) + "_extent" + std::to_string(dim);
            res = std::make_shared<Mul>(res, Unknown::make(name)); 
          } else {
            res = std::make_shared<Mul>(res, std::make_shared<Constant>(memrefShape[dim]));
          }
        }

        return res;
      }
      ++indexVarDim;
    }

    // memref address is not a function of this variable
    MDRANGE_DEBUG("Info: " << memrefOp << " is not a function of " << indexVar << "\n");
    return std::make_shared<Constant>(0);
  }

  static void dump_ops(ModuleOp &mod) {
    mod.walk([&](Operation *op) {
      if (auto parallelOp = dyn_cast<scf::ParallelOp>(op)) {
        MDRANGE_DEBUG("Found scf.parallel operation:\n");
        MDRANGE_DEBUG("Induction variables and strides:\n");
        for (auto iv : llvm::zip(parallelOp.getInductionVars(), parallelOp.getStep())) {
          (void) iv;
          MDRANGE_DEBUG(std::get<0>(iv) << " with stride " << std::get<1>(iv) << "\n");
        }
        MDRANGE_DEBUG("\n\n");
      }

      if (auto memrefOp = dyn_cast<memref::LoadOp>(op)) {
        MDRANGE_DEBUG("Found memref.load operation:\n");
        MDRANGE_DEBUG("MemRef: " << memrefOp.getMemRef() << "\nIndex variables:\n");
        for (Value index : memrefOp.getIndices()) {
          (void) index;
          MDRANGE_DEBUG(index << "\n");
        }
        if (auto memrefType = dyn_cast<MemRefType>(memrefOp.getMemRef().getType())) {
          MDRANGE_DEBUG("MemRef extents:\n");
          for (int64_t dim : memrefType.getShape()) {
            (void) dim;
            MDRANGE_DEBUG(dim << "\n");
          }
        }
        MDRANGE_DEBUG("\n\n");
      }

      if (auto memrefOp = dyn_cast<memref::StoreOp>(op)) {
        MDRANGE_DEBUG("Found memref.store operation:\n");
        MDRANGE_DEBUG("MemRef: " << memrefOp.getMemRef() << "\nIndex variables:\n");
        for (Value index : memrefOp.getIndices()) {
          (void) index;
          MDRANGE_DEBUG(index << "\n");
        }
        if (auto memrefType = dyn_cast<MemRefType>(memrefOp.getMemRef().getType())) {
          MDRANGE_DEBUG("MemRef extents:\n");
          for (int64_t dim : memrefType.getShape()) {
            (void) dim;
            MDRANGE_DEBUG(dim << "\n");
          }
        }
        MDRANGE_DEBUG("\n\n");
      }
    });
  }




// Get a unique name for the provided value
static std::string get_value_name(mlir::Value &value) {
  if (mlir::isa<BlockArgument>(value)) {
    auto ba = mlir::cast<BlockArgument>(value);
    return std::string("block") +std::to_string(uintptr_t(ba.getOwner())) + "_arg" + std::to_string(ba.getArgNumber());
  } else {
    // mlir::Operation *op = value.getDefiningOp();

    std::string name;
    llvm::raw_string_ostream os(name);
    value.print(os);
    name = name.substr(0, name.find(' ')); // "%blah = ..." -> "%blah"

    // std::string name = op->getName().getStringRef().str() + std::to_string(uintptr_t(op));
    return name;
  }
}

// Get an expression representing the size of the iteration space of `op` in the
// `dim` dimension.
static std::shared_ptr<Expr> iteration_space_expr(scf::ParallelOp &op, int dim) {

    auto lb = op.getLowerBound()[dim];
    auto ub = op.getUpperBound()[dim];
    auto st = op.getStep()[dim];

    std::shared_ptr<Expr> lbExpr;
    std::shared_ptr<Expr> ubExpr;
    std::shared_ptr<Expr> stExpr;

    if (auto lbConst = lb.getDefiningOp<arith::ConstantIndexOp>()) {
      lbExpr = Constant::make(lbConst.value());
    } else {
      lbExpr = Unknown::make(get_value_name(lb));
    }

   if (auto ubConst = ub.getDefiningOp<arith::ConstantIndexOp>()) {
      ubExpr = Constant::make(ubConst.value());
    } else {
      ubExpr = Unknown::make(get_value_name(ub));
    }
  
  if (auto stepConst = st.getDefiningOp<arith::ConstantIndexOp>()) {
      stExpr = Constant::make(stepConst.value());
    } else {
      stExpr = Unknown::make(get_value_name(st));
    }

    // (ub - lb + step - 1) / step
    // TODO: this could be a special DivCeil operation or something
    auto num = Add::make(Sub::make(ubExpr, lbExpr), Sub::make(stExpr, Constant::make(1)));
    auto ret = Div::make(num, stExpr);

    MDRANGE_DEBUG("Trip count (dim " << dim << ") of:\n" << op << "\n: " << ret << "\n");
    return ret;
}

// Get an expression representing the size of the iteration space of `op`
static std::shared_ptr<Expr> iteration_space_expr(scf::ParallelOp &op) {
    auto lowerBounds = op.getLowerBound();
    std::shared_ptr<Expr> total = iteration_space_expr(op, 0);
    for (unsigned i = 1; i < lowerBounds.size(); ++i) {
      total = Mul::make(total, iteration_space_expr(op, i));
    }
    return total;
}

using IterationSpaceExprs = llvm::DenseMap<scf::ParallelOp, std::shared_ptr<Expr>>;

// Get expressions represeting the iteration space for all parallel loops in the module
static IterationSpaceExprs build_parallel_trip_counts(ModuleOp &mod) {
  // create expr for each op, ignoring nesting
  IterationSpaceExprs ISE1;
  mod.walk([&](scf::ParallelOp op){
    std::shared_ptr<Expr> expr = iteration_space_expr(op);
    ISE1[op] = expr;
  });

  // fixup, incorporate parent trip counts into expression
  IterationSpaceExprs ISE2;
  mod.walk([&](scf::ParallelOp op){
    std::shared_ptr<Expr> expr = ISE1[op];
    for (auto parent : enclosing_parallel_ops(op)) {
      expr = Mul::make(expr, ISE1[parent]);
    }
    ISE2[op] = expr;
  });

  return ISE2;
}

static IterationSpaceExprs build_parallel_trip_counts(scf::ParallelOp &parentOp, std::shared_ptr<Expr> parentCount) {
  IterationSpaceExprs ISE;

  parentOp.getBody()->walk([&](Operation *op) {
      if (auto parallelOp = dyn_cast<scf::ParallelOp>(op)) {
        // create an expression representing the trip count for this loop
        std::shared_ptr<Expr> expr = iteration_space_expr(parallelOp);
        ISE[parallelOp] = Mul::make(expr, parentCount);

        // descend into the body of the loop
        IterationSpaceExprs exprs = build_parallel_trip_counts(parallelOp, expr);
        ISE.insert(exprs.begin(), exprs.end());
      }
  });

  return ISE;
}

// map of (Operation*, Value) -> Cost
// map of the cost model for a given memref / induction variable pair
using MemrefInductionCosts = llvm::DenseMap<std::pair<Operation*, mlir::Value>, Cost>;


// return all induction variables for all parallel ops
static ValueVec all_induction_variables(ParallelOpVec &ops) {
  ValueVec vars;
  for (auto &op : ops) {
    for (auto &var : op.getInductionVars()) {
      vars.push_back(var);
    }
  }
  return vars;
}

  // compute the partial derivative of each memref with respect to all enclosing induction variables via the chain rule:
  // d(offset)/d(indvar) = sum( 
  //    d(offset)/d(index) * d(index)/d(indvar), 
  //    for each index in indices)
template <typename Memref>
static MemrefInductionCosts get_costs(Memref &memrefOp, IterationSpaceExprs &tripCounts) {
  static_assert(std::is_same_v<Memref, memref::LoadOp> || std::is_same_v<Memref, memref::StoreOp>);

  if constexpr (std::is_same_v<Memref, memref::LoadOp>) {
    MDRANGE_DEBUG("get_cost: memref::LoadOp\n");
  } else if constexpr (std::is_same_v<Memref, memref::StoreOp>) {
    MDRANGE_DEBUG("get_cost: memref::StoreOp\n");
  }

  MemrefInductionCosts MIC;

  // get all the parallel ops that enclose this memref
  auto ancestors = enclosing_parallel_ops(memrefOp);
  if (ancestors.empty()) {
    MDRANGE_DEBUG("get_costs: memref is not enclosed in an scf::ParallelOp\n");
    return MIC;
  }
  scf::ParallelOp &parentOp = *ancestors.begin();

  ValueVec indVars = all_induction_variables(ancestors);
  

  for (Value indVar : indVars) {
    std::shared_ptr<Expr> dodi = Constant::make(0);
    for (Value indexVar : memrefOp.getIndices()) {
      auto e1 = do_di(memrefOp, indexVar);

      MDRANGE_DEBUG("∂(" << memrefOp << ")/∂(" << indexVar << ") = ");
      if (e1) {
        MDRANGE_DEBUG(e1);
      } else {
        MDRANGE_DEBUG("undefined");
      }
      MDRANGE_DEBUG("\n");

      auto e2 = df_dx(indexVar, indVar);

      MDRANGE_DEBUG("∂(" << indexVar << ")/∂(" << indVar << ") = ");
      if (e2) {
        MDRANGE_DEBUG(e2);
      } else {
        MDRANGE_DEBUG("undefined");
      }
      MDRANGE_DEBUG("\n");

      if (e1 && e2) {
        dodi = Add::make(dodi, Mul::make(e1, e2));
      } else {
        dodi = nullptr;
        break;
      }
    }

    MDRANGE_DEBUG("∂(" << memrefOp << ")/∂(" << indVar << ") = ");
    if (dodi) {
      MDRANGE_DEBUG(dodi);
    } else {
      MDRANGE_DEBUG("undefined");
    }
    MDRANGE_DEBUG("\n");

    std::shared_ptr<Expr> tripCount = tripCounts[parentOp];

    MIC[std::make_pair(memrefOp, indVar)] = Cost(dodi, tripCount, Cost::scale_factor<Memref>());
  }
  return MIC;
}

  static MemrefInductionCosts build_cost_table(ModuleOp &mod, IterationSpaceExprs &tripCounts) {
    MemrefInductionCosts MIC;

    // compute for loads
    mod.walk([&](Operation *op){
      if (auto memrefOp = dyn_cast<memref::LoadOp>(op)) {
        MemrefInductionCosts costs = get_costs(memrefOp, tripCounts);
        MIC.insert(costs.begin(), costs.end());
      } else if (auto memrefOp = dyn_cast<memref::StoreOp>(op)) {
        MemrefInductionCosts costs = get_costs(memrefOp, tripCounts);
        MIC.insert(costs.begin(), costs.end());
      }
    });

    return MIC;
  }


  

  struct ParallelConfig {
    // permutation of induction variables for each parallel op
    llvm::DenseMap<scf::ParallelOp, Permutation> perms_;
  };


  static size_t get_num_induction_vars(scf::ParallelOp &parallelOp) {
    return parallelOp.getInductionVars().size();
  }

  // call f on every operation immediately nested under op
  template <typename Lambda>
  void for_each_nested(mlir::Operation *op, Lambda &&f) {

    // assert f can be called on mlir::Operation*
    static_assert(std::is_invocable_v<Lambda, mlir::Operation *>);

    for (mlir::Region &region : op->getRegions()) {
      for (mlir::Block &block : region.getBlocks()) {
        for (mlir::Operation &nestedOp : block.getOperations()) {
          f(&nestedOp);
        }
      }
    }
  }

  static ParallelOpVec enclosing_parallel_ops(mlir::Operation *op) {
    ParallelOpVec ops;
    scf::ParallelOp parent = op->getParentOfType<scf::ParallelOp>();
    while (parent) {
      ops.push_back(parent);
      parent = parent->getParentOfType<scf::ParallelOp>();
    }
    return ops;
  }


  // get top-level (not nested) scf::ParallelOp in a block
  static ParallelOpVec get_parallel_ops(Block &block) {
    ParallelOpVec ret;
    for (Operation &op : block.getOperations()) {
      if (auto parallelOp = dyn_cast<scf::ParallelOp>(op)) {
        ret.push_back(parallelOp);
      } else {
        ParallelOpVec nestedOps = get_parallel_ops(op);
        for (auto &nestedOp : nestedOps) {
          ret.push_back(nestedOp);
        }
      }
    }
    return ret;
  }

  // get top-level (not nested) scf::ParallelOp in a region
  static ParallelOpVec get_parallel_ops(Region &region) {
    ParallelOpVec ret;
    for (Block &block : region.getBlocks()) {
      ParallelOpVec blockOps = get_parallel_ops(block);
      for (scf::ParallelOp &op : blockOps) {
        ret.push_back(op);
      }
    }
    return ret;
  }

  // get top-level (not nested) scf::ParallelOp in a module
  static ParallelOpVec get_parallel_ops(mlir::Operation &op) {
    ParallelOpVec ret;
    for (Region &region : op.getRegions()) {
      ParallelOpVec regionOps = get_parallel_ops(region);
      for (scf::ParallelOp &op : regionOps) {
        ret.push_back(op);
      }
    }
    return ret;
  }

  // get top-level (not nested) scf::ParallelOp in a module
  static ParallelOpVec get_parallel_ops(mlir::ModuleOp &mod) {
    ParallelOpVec ret;
    ParallelOpVec regionOps = get_parallel_ops(mod.getRegion());
    for (scf::ParallelOp &op : regionOps) {
      ret.push_back(op);
    }
    return ret;
  }

  // get top-level (not nested) scf::ParallelOp in a ParallelOp
  static ParallelOpVec get_parallel_ops(scf::ParallelOp &op) {
    ParallelOpVec ret;
    ParallelOpVec regionOps = get_parallel_ops(op.getRegion());
    for (scf::ParallelOp &op : regionOps) {
      ret.push_back(op);
    }
    return ret;
  }

  // TODO: unknowns is all unknowns combined from all models
  // this means unkowns is the same for all calls here
  // this is a bit confusing since model also has model.unkowns() which is a subset
  static size_t monte_carlo(const std::vector<std::string> &unknowns, 
                            const MemrefInductionCosts &mic,
                            const ParallelConfig &cfg,
                            std::vector<mlir::Operation*> &memrefOps,
                            int n = 500, int seed = 31337) {
    std::mt19937 gen(seed);

    std::vector<size_t> costs;

    for (int i = 0; i < n; i++) {

      // MDRANGE_DEBUG("MC iteration " << i << ":\n");

      // generate random values for all unknowns in cost model
      Ctx ctx;
      for (auto &unk : unknowns) {
        auto val = log_random_int(gen, 1, 100000);
        // MDRANGE_DEBUG(unk << ": " << val << "\n");
        ctx.values[unk] = val;
      }
      
      size_t cost = 0;

      for (mlir::Operation *op : memrefOps) {

        auto ancestors = enclosing_parallel_ops(op);
        for (scf::ParallelOp &ancestor : ancestors) {
          if (auto it = cfg.perms_.find(ancestor); it != cfg.perms_.end()) {
            const Permutation &perm = it->second;

            // assume MDRange iteration is inner = right, outer = right
            mlir::Value rightIndVar = ancestor.getInductionVars()[perm[perm.size() - 1]];
            auto key = std::make_pair(op, rightIndVar);
            if (auto it = mic.find(key); it != mic.end()) {
              const Cost &model = it->second;
              cost += model.stride_->eval(ctx) * model.count_->eval(ctx) * model.sf_;
            } else {
              llvm::report_fatal_error("couldn't find model for memref / induction variable combo");
            }
          } else {
            llvm::report_fatal_error("couldn't find config for nesting scf.parallel of memref");
          }
        }
      }

      // MDRANGE_DEBUG("MC iteration " << i << " cost=" << cost << "\n");
      costs.push_back(cost);
    }

    // FIXME: here we do median, is there a principled aggregation strategy?
    // kth pctile cost?
    // average of worst k?
    // worst / average ("competitive ratio")?
    // geometric mean?
    // trimmed mean?
    //
    // robustness metrics?
    // coefficient of variation
    std::sort(costs.begin(), costs.end());
    return costs[costs.size() / 2];
  }


  // modify `parallelOp` so that its induction variables are permuted according to `permutation`
  static void permute_parallel_op(scf::ParallelOp parallelOp, const Permutation &permutation) {
    OpBuilder builder(parallelOp);
    SmallVector<Value, 4> newLowerBounds, newUpperBounds, newSteps;

    for (int index : permutation) {
      newLowerBounds.push_back(parallelOp.getLowerBound()[index]);
      newUpperBounds.push_back(parallelOp.getUpperBound()[index]);
      newSteps.push_back(parallelOp.getStep()[index]);
    }

    auto newParallelOp = builder.create<scf::ParallelOp>(
        parallelOp.getLoc(), newLowerBounds, newUpperBounds, newSteps);

    // Move the body of the original parallelOp to the new parallelOp.
    newParallelOp.getBody()->getTerminator()->erase(); // splicing in the new body has a terminator already
    newParallelOp.getBody()->getOperations().splice(
        newParallelOp.getBody()->begin(), parallelOp.getBody()->getOperations());

    // replace uses of original induction variable perm[i] with new induction variable [i]
    for (size_t i = 0; i < permutation.size(); ++i) {
      parallelOp.getInductionVars()[permutation[i]].replaceAllUsesWith(newParallelOp.getInductionVars()[i]);
    }

    parallelOp.erase();
  }


  // returns best config and minimal cost for all scf::ParallelOp in mod
  std::pair<ParallelConfig, size_t> best_configuration(const std::vector<std::string> &unknowns, const MemrefInductionCosts &costs, ModuleOp &mod) {

    // best configuration for a module is the combination of the best configurations
    // of the parallel ops in that module
    // minimal cost is the sum

    ParallelConfig best;
    size_t cost = 0;

    ParallelOpVec modOps = get_parallel_ops(mod);
    for (scf::ParallelOp &op : modOps) {

      auto [opCfg, opCost] = best_configuration(unknowns, costs, best, op);

      for (auto &kv : opCfg.perms_) {
        best.perms_[kv.first] = kv.second;
      }
      cost += opCost;
    }

    return std::make_pair(best, cost);
  }

  // returns the best configuration and minimal cost for all configurations of `op` given configuration `cfg` for parent ParallelOps
  std::pair<ParallelConfig, size_t> best_configuration(const std::vector<std::string> &unknowns, const MemrefInductionCosts &costs, ParallelConfig &cfg, scf::ParallelOp &op) {

    // find parallel nested operations and cache
    ParallelOpVec nestedOps = get_parallel_ops(op);


    // best configuration among all permutations of this parallel op
    ParallelConfig bestCfg;
    size_t bestCost = std::numeric_limits<size_t>::max();


    // generate all possible parallel configurations of this operation
    Permutation perm(get_num_induction_vars(op));
    std::iota(perm.begin(), perm.end(), 0);

    do {

      MDRANGE_DEBUG("Modeling permutation {");
      for (int i : perm) {
        (void) i;
        MDRANGE_DEBUG(i << ", ");
      }
      MDRANGE_DEBUG("} ...\n");

      ParallelConfig newCfg = cfg;
      newCfg.perms_[op] = perm;

      // get all top-level memrefs in this parallel region
      std::vector<Operation *> memrefOps;
      for (Block &block : op.getRegion().getBlocks()) {
        for (Operation &child : block.getOperations())
          if (auto memrefOp = dyn_cast<memref::LoadOp>(child)) {
            memrefOps.push_back(memrefOp);
          } else if (auto memrefOp = dyn_cast<memref::StoreOp>(child)) {
            memrefOps.push_back(memrefOp);
          }
        }

      


      size_t memrefCost = 0;
      if (memrefOps.empty()) {
        MDRANGE_DEBUG("scf.parallel has no memrefs\n");
      } else {

        // model the cost of the memrefs under this loop configuration
        // here we use Monte Carlo method
        memrefCost = monte_carlo(unknowns, costs, newCfg, memrefOps);
        MDRANGE_DEBUG("... under permutation, memref cost is " << memrefCost << "\n");
      }

      // total cost is the cost of the top-level memrefs + the cost of any nested parallel ops under this configuration
      size_t nestedCost = 0;

      for (scf::ParallelOp &nestedOp : nestedOps) {
        auto [parOpCfg, parOpCost] = best_configuration(unknowns, costs, newCfg, nestedOp);

        // augment our configuration with the best discovered nested ones
        // FIXME: this overwrites nesting k-v pairs but I think it's okay
        for (auto &kv : parOpCfg.perms_) {
          newCfg.perms_[kv.first] = kv.second;
        }
        nestedCost += parOpCost;
      }

      const size_t permCost = nestedCost + memrefCost;

      // check if this permutation is the best cost so far
      if (permCost < bestCost) {
        bestCfg = newCfg;
        bestCost = permCost;
      }

    } while (std::next_permutation(perm.begin(), perm.end()));


    return std::make_pair(bestCfg, bestCost);
  }


  void runOnOperation() override {
    ModuleOp mod = getOperation();

    MDRANGE_DEBUG("====\nprint module\n====\n");
    MDRANGE_DEBUG(mod << "\n");


    MDRANGE_DEBUG("====\ndump_ops\n====\n");
    dump_ops(mod);

    MDRANGE_DEBUG("====\nbuild_parallel_trip_counts\n====\n");
    IterationSpaceExprs tripCounts = build_parallel_trip_counts(mod);

    for (auto &kv : tripCounts) {
      const std::shared_ptr<Expr> &trip = kv.second;
      (void) trip;
      MDRANGE_DEBUG("parallel op: " << kv.first << " trip: " << trip << "\n");
    }

    MDRANGE_DEBUG("====\nbuild_cost_table\n====\n");
    MemrefInductionCosts costTable = build_cost_table(mod, tripCounts);

    MDRANGE_DEBUG("====\nunknowns\n====\n");
    std::vector<std::string> unknowns;
    for (const auto &kv : costTable) {
      for (const std::string &unk : kv.second.unknowns()) {
        if (unknowns.end() == std::find(unknowns.begin(), unknowns.end(), unk)) {
          unknowns.push_back(unk);
        }
      }
    }
    MDRANGE_DEBUG(unknowns.size() << " unknowns:\n");
    std::sort(unknowns.begin(), unknowns.end());
    for (const std::string &unk : unknowns) {
      (void) unk;
      MDRANGE_DEBUG(unk << "\n");
    }

    MDRANGE_DEBUG("====\nModel Reordered Induction variables\n====\n");  
    // ParallelConfig, size_t
    auto [bestCfg, bestCost] = best_configuration(unknowns, costTable, mod);
    

    MDRANGE_DEBUG("min cost: " << bestCost << "\n");

    MDRANGE_DEBUG("====\nbuild new module\n====\n");
    // modify the parallel ops in the module
    mod.walk([&](scf::ParallelOp parallelOp) {
      MDRANGE_DEBUG("modifying " << parallelOp << "\n");
      if (auto it = bestCfg.perms_.find(parallelOp); it != bestCfg.perms_.end()) {
        const Permutation &permutation = it->second;
        MDRANGE_DEBUG("applying permutation ");
        for (auto i : permutation) {
          (void) i;
          MDRANGE_DEBUG(i << " ");
        }
        MDRANGE_DEBUG("\n");
        permute_parallel_op(parallelOp, permutation);
      } else {
        llvm::report_fatal_error("no configuration for scf.parallel in configuration");
      }
    });
    MDRANGE_DEBUG("====\ndone\n====\n");
  }
};

} // anonymous namespace

std::unique_ptr<Pass> mlir::createKokkosMdrangeIterationPass() {
  return std::make_unique<KokkosMdrangeIterationPass>();
}

