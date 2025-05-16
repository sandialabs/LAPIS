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

Presuming that the scf.parallel will actually be implemented in a "layout-right" iteration order, and given that memrefs are layout right, how to do we want to order the scf.parallel induction variables? e.g. for (1), do we want (%i, %j) or (%j, %i)?
To answer, we build a cost model of each memref load/store, and choose the induction variable ordering for all scf.parallels that minimizes that cost.
The foundation of the cost model is the reuse distance of the memref, under the theory that accesses with better locality will be faster due to coalescing/caching.
The stride of the memref depends on whichever induction variable is the "right-most" one in the scf.parallel region, due to our "layout-right" iteration order assumption.

Some examples:

For (2), the reuse distance w.r.t. %i is 4 (sizeof f32), and the reuse distance with respect to %j is 20 * 4 (size of 1st dimension * sizeof f32)

For (4), the reuse distance with respect to (%i) is 4 * whatever the 1st memref dimension is.
The reuse distance w.r.t %j is undefined (address does not change when %j changes).
The reuse distance w.r.t %k is 4.
The reuse distance w.r.t %l is undefined.

The way to understand this is that if the index variable of the memref is some kind of simple function of the induction variable, we can compute the reuse distance. If it is not a function of the induction variable, or is a function of the induction variable but we don't know the function, we can't compute the reuse distance.

------

So, what kind of simple functions can we compute? This takes the following approach: it tries to compute 

d(memref) / d(induction variable), the partial derivative of the accessed offset w.r.t the induction variable. We can ignore the base address, because it's derivative w.r.t all induction variables is 0

Via the chain rule;
d(memref) / d(induction variable) = d(memref)         / d(index variable)     *
                                    d(index variable) / d(induction variable)

Let's take d(index variable) / d(induction variable) first.

------

d(index variable) / d(induction variable) is computed by recursively following the inputs to the operation and applying differentiation rules.

To make this problem tractable, we make two simplifying assumptions:

We only care about about results of the form df / dx = a * x
We only try to differentiate simple arithmetic functions, e.g. if f(x) = g(x) + h(x), df/dx = dg/dx + dh/dx. Similarly for multiply, divide, etc. Any other f we just give up and say who knows.

------

d(memref) / d(index variable) is in principle simple, it's just the product of all strides of lower dimension than the index variable. In practice, however, most strides are unknown at compile time, so we won't be able to get an actual number, we'll just get expressions like

d(memref) / d(index variable) = stride_0 * stride_2 * sizeof(datatype)

if we're lucky, and all dimensions are known or there are no lower dimensions, then we can get an actual number.

------

So in the end, d(memref) / d(induction variable) ends up being something of this form:

stride_0 * stride_2 * sizeof(datatype) * a * x
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
memref / index var component
                                         ^^^^^
                                         index var / induction var component


the second component might just be ???, and/or the first component might be a known integer number.

------

In principle, each memref has a different cost for each induction variable ordering.
In practice, we just consider the induction variable that is incrementing the fastest for each memref - that is, the right-most induction variable for the closest enclosing loop.

The reuse distance is just looked up in the previously computed table of d(memref) / d(induction variable)

------

We generate all possible combinations of
  * choose an induction variable from each parallel region to be the right-most one.
We compute the cost under each combination.
  * Since the cost expression will contain many unknowns, we do monte-carlo simulation of the cost model for each induction variable ordering
We chosoe the induction variable ordering with the lowest cost

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

    virtual int eval(const Ctx &ctx) = 0;
    virtual void dump(llvm::raw_fd_ostream &os) const = 0;
    virtual void dump(llvm::raw_ostream &os) const = 0;
    virtual std::vector<std::string> unknowns() const = 0;
    virtual std::shared_ptr<Expr> clone() const = 0;
    virtual ~Expr() {}
  };

} // namespace

static llvm::raw_fd_ostream & operator<<(llvm::raw_fd_ostream &os, const std::shared_ptr<Expr> &e) {
  e->dump(os);
  return os;
}

static llvm::raw_ostream & operator<<(llvm::raw_ostream &os, const std::shared_ptr<Expr> &e) {
  e->dump(os);
  return os;
}

struct KokkosMdrangeIterationPass
    : public impl::KokkosMdrangeIterationBase<KokkosMdrangeIterationPass> {

#if 1
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

    virtual int eval(const Ctx &ctx) override {
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

    virtual int eval(const Ctx &ctx) override {
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

    virtual int eval(const Ctx &ctx) override {
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

    virtual int eval(const Ctx &ctx) override {
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

    virtual int eval(const Ctx &ctx) override {
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

    virtual int eval(const Ctx &ctx) override {
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
    int sf_; // scaling factor, 1 for load, 3 for store

    std::vector<std::string> unknowns() const {
      std::vector<std::string> ret;
      for (auto &u : stride_->unknowns()) ret.push_back(u);
      for (auto &u : count_->unknowns()) ret.push_back(u);
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
static std::shared_ptr<Expr> df_dx(Value &f, Value &x) {
  if (f == x) {
    MDRANGE_DEBUG("Info: df_dx of equal values\n");
    return Constant::make(1);
  } else if (mlir::isa<BlockArgument>(f) && mlir::isa<BlockArgument>(x)) {
    MDRANGE_DEBUG("Info: df_dx of different block arguments:\n");
    return Constant::make(0);
  } else {
    // FIXME: what other scenarios if there is no defining op.
    if (auto fOp = f.getDefiningOp()) {
      if (auto xOp = x.getDefiningOp()) {
        return df_dx(fOp, xOp);
      }
    }
    MDRANGE_DEBUG("ERROR: One of the values has no defining operation\n");
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

  static ParallelOpVec get_parallel_ops(ModuleOp &mod) {
    ParallelOpVec ret;
    mod.walk([&](Operation *op) {
      if (auto parallelOp = dyn_cast<scf::ParallelOp>(op)) {
        ret.push_back(parallelOp);
      }
    });
    return ret;
  }

  // cost of a module with a given parallel configuration
  static size_t model_cost(ModuleOp &mod, const ParallelConfig &cfg, const MemrefInductionCosts &costTable) {
    size_t cost = 0;
    mod.walk([&](Operation *op) {
      if (auto memrefOp = dyn_cast<memref::LoadOp>(op)) {
        cost += model_cost(memrefOp, cfg, costTable);
      } else if (auto memrefOp = dyn_cast<memref::StoreOp>(op)) {
        cost += model_cost(memrefOp, cfg, costTable);
      }
    }); // walk
    return cost;
  }

  static size_t monte_carlo(const Cost &model, int n = 100, int seed = 31337) {
    std::mt19937 gen(seed);

    std::vector<size_t> costs;

    std::vector<std::string> unknowns = model.unknowns();

    for (int i = 0; i < n; i++) {

      // generate random values for all unknowns in cost model
      Ctx ctx;
      for (auto &name : unknowns) {
        auto val = log_random_int(gen, 1, 1000000);
        // llvm::outs() << name << ": " << val << "\n";
        ctx.values[name] = val;
      }
      
      costs.push_back(model.stride_->eval(ctx) * model.count_->eval(ctx) * model.sf_);
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

  template <typename MemrefOp>
  static size_t model_cost(MemrefOp &memrefOp, const ParallelConfig &cfg, const MemrefInductionCosts &costTable) {
    static_assert(std::is_same_v<MemrefOp, memref::LoadOp> || std::is_same_v<MemrefOp, memref::StoreOp>);
    MDRANGE_DEBUG("model cost of " << memrefOp << "...\n");

    auto parents = enclosing_parallel_ops(memrefOp);

    MDRANGE_DEBUG("... memref op above has " << parents.size() << " enclosing parallels\n");

    size_t cost = 0;
    for (scf::ParallelOp &parallelOp : parents) {
      if (auto it = cfg.perms_.find(parallelOp); it != cfg.perms_.end()) {

        const Permutation &perm = it->second;
        Value leftMostVar = parallelOp.getInductionVars()[perm[0]];

        MDRANGE_DEBUG("... under permutation, left-most induction var of enclosing parallel is " << leftMostVar << "\n");
        
        // FIXME: why does this work? the table should expect key to be pair<Operation*, Value> not pair<Operation, Value>
        auto costKey = std::make_pair(memrefOp, leftMostVar);
        if (auto jt = costTable.find(costKey); jt != costTable.end()) {
          Cost model = jt->second;
          size_t parallelContrib = monte_carlo(model);
          cost += parallelContrib;
          MDRANGE_DEBUG("..." << memrefOp << " contributes " << parallelContrib << " (now " << cost << ")\n");
        } else {
          MDRANGE_DEBUG("WARN: couldn't find model for memref / induction variable combo\n");
          return 0;
        }
      } else {
        MDRANGE_DEBUG("WARN: couldn't find permutation for parent parallel op\n");
        return 0;
      }
    }
    return cost;
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

  template <typename Lambda>
  void walk_configurations(ParallelOpVec &ops, Lambda &&f) {
    ParallelConfig cfg;
    walk_configurations(ops, std::forward<Lambda>(f), cfg);
  }

// return true if op, or any of its nested children, were scf parallel
  template <typename Lambda>
  void walk_configurations(ParallelOpVec &ops, Lambda &&f, const ParallelConfig &cfg) {
    if (ops.empty()) {
      f(cfg);
    } else {
      scf::ParallelOp &first = ops[0];
      ParallelOpVec rest;
      for (size_t oi = 1; oi < ops.size(); ++oi) {
        rest.push_back(ops[oi]);
      }
      Permutation perm(get_num_induction_vars(first));
      std::iota(perm.begin(), perm.end(), 0);

      do {
        ParallelConfig newCfg = cfg;
        newCfg.perms_[first] = perm;
        walk_configurations(rest, std::forward<Lambda>(f), newCfg);
      } while (std::next_permutation(perm.begin(), perm.end()));
    }
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    MDRANGE_DEBUG(module << "\n");

    MDRANGE_DEBUG("====\ndump_ops\n====\n");
    dump_ops(module);

    MDRANGE_DEBUG("====\nbuild_parallel_trip_counts\n====\n");
    IterationSpaceExprs tripCounts = build_parallel_trip_counts(module);

    for (auto &kv : tripCounts) {
      const std::shared_ptr<Expr> &trip = kv.second;
      MDRANGE_DEBUG("parallel op: " << kv.first << " trip: " << trip << "\n");
    }

    MDRANGE_DEBUG("====\nbuild_cost_table\n====\n");
    MemrefInductionCosts costTable = build_cost_table(module, tripCounts);

    MDRANGE_DEBUG("====\nExtract parallel ops\n====\n");
    auto parallelOps = get_parallel_ops(module);

    MDRANGE_DEBUG("====\nModel Reordered Induction variables\n====\n");
    size_t minCost = std::numeric_limits<size_t>::max();
    ParallelConfig minCfg;
    walk_configurations(parallelOps, [&](const ParallelConfig &cfg){
      MDRANGE_DEBUG("modeling ParallelConfig:\n");
      for (const auto &kv : cfg.perms_) {
        MDRANGE_DEBUG(kv.first << " -> {");
        for(const auto &e : kv.second) {
          (void)e;
          MDRANGE_DEBUG(e << ", ");
        }
        MDRANGE_DEBUG("}\n");
      }

      size_t cost = model_cost(module, cfg, costTable);
      MDRANGE_DEBUG("cost was " << cost << "\n");
      if (cost < minCost) {
        MDRANGE_DEBUG("Info: new optimal! cost=" << cost << "\n");

        for (const auto &kv : cfg.perms_) {
          MDRANGE_DEBUG(kv.first << " with permutation: ");
          for (const size_t e : kv.second) {
            MDRANGE_DEBUG(e << " ");
            
          } 
          MDRANGE_DEBUG("\n");
        }

        minCost = cost;
        minCfg = cfg;
      }

    });
    MDRANGE_DEBUG("min cost: " << minCost << "\n");

    MDRANGE_DEBUG("====\nbuild new module\n====\n");
#if 0
    // clone the existing module
    ModuleOp newModule = module.clone();

    // TODO: modify the parallel ops in the new module
    newModule.walk([&](scf::ParallelOp parallelOp) {

      llvm::outs() << "modifying " << parallelOp << "\n";


      // TODO: replace this placeholder permutation with the computed one
      // fake permutation that just reverses stuff
      Permutation permutation(parallelOp.getInductionVars().size());
      std::iota(permutation.begin(), permutation.end(), 0);
      std::reverse(permutation.begin(), permutation.end());

      llvm::outs() << "applying permutation ";
      for (auto i : permutation) {
        llvm::outs() << i << " ";
      }
      llvm::outs() << "\n";

      permute_parallel_op(parallelOp, permutation);
    });


    // FIXME: this seems like it might introduce an extra scf.reduce at the end
    // of the parallel region, probably because it clones one and then one gets inserted
    // --mlir-print-ir-after-failure
    // overwrite the module with the new module
    // Replace the original module with the new module.
    module.getBody()->getOperations().clear();
    module.getBody()->getOperations().splice(module.getBody()->begin(),
                                             newModule.getBody()->getOperations());
#else
    // modify the parallel ops in the module
    module.walk([&](scf::ParallelOp parallelOp) {

      MDRANGE_DEBUG("modifying " << parallelOp << "\n");

#if 1
      const Permutation &permutation = minCfg.perms_[parallelOp];
#else
      // TODO: replace this placeholder permutation with the computed one
      // fake permutation that just reverses stuff
      Permutation permutation(parallelOp.getInductionVars().size());
      std::iota(permutation.begin(), permutation.end(), 0);
      std::reverse(permutation.begin(), permutation.end());
#endif
      MDRANGE_DEBUG("applying permutation ");
      for (auto i : permutation) {
        (void) i;
        MDRANGE_DEBUG(i << " ");
      }
      MDRANGE_DEBUG("\n");

      permute_parallel_op(parallelOp, permutation);
    });
#endif
    MDRANGE_DEBUG("====\ndone\n====\n");
  }
};

} // anonymous namespace

std::unique_ptr<Pass> mlir::createKokkosMdrangeIterationPass() {
  return std::make_unique<KokkosMdrangeIterationPass>();
}

