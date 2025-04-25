//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file has been modified from its original version.
//
//@HEADER

/// \file KokkosSparse_CrsMatrix.hpp
/// \brief Local sparse matrix interface
///
/// This file provides KokkosSparse::CrsMatrix.  This implements a
/// local (no MPI) sparse matrix stored in compressed row sparse
/// ("Crs") format.

#ifndef KOKKOS_SPARSE_CRSMATRIX_HPP_
#define KOKKOS_SPARSE_CRSMATRIX_HPP_

#include "Kokkos_Core.hpp"
#include "Kokkos_StaticCrsGraph.hpp"
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace KokkosSparse {

/// \class CrsMatrix
/// \brief Compressed sparse row implementation of a sparse matrix.
/// \tparam ScalarType The type of entries in the sparse matrix.
/// \tparam OrdinalType The type of column indices in the sparse matrix.
/// \tparam Device The Kokkos Device type.
/// \tparam MemoryTraits Traits describing how Kokkos manages and
///   accesses data.  The default parameter suffices for most users.
///
/// "Crs" stands for "compressed row sparse."  This is the phrase
/// Trilinos traditionally uses to describe compressed sparse row
/// storage for sparse matrices, as described, for example, in Saad
/// (2nd ed.).
template <class ScalarType, class OrdinalType, class Device,
          class MemoryTraits = void,
          class SizeType     = typename Kokkos::ViewTraits<OrdinalType*, Device,
                                                       void, void>::size_type>
class CrsMatrix {
  static_assert(
      std::is_signed<OrdinalType>::value,
      "CrsMatrix requires that OrdinalType is a signed integer type.");

 private:
  typedef typename Kokkos::ViewTraits<ScalarType*, Device, void,
                                      MemoryTraits>::host_mirror_space
      host_mirror_space;

 public:
  //! Type of the matrix's execution space.
  typedef typename Device::execution_space execution_space;
  //! Type of the matrix's memory space.
  typedef typename Device::memory_space memory_space;
  //! Canonical device type
  typedef Kokkos::Device<execution_space, memory_space> device_type;

  //! Type of each value in the matrix.
  typedef ScalarType value_type;
  //! Type of each (column) index in the matrix.
  typedef OrdinalType ordinal_type;
  typedef MemoryTraits memory_traits;
  /// \brief Type of each entry of the "row map."
  ///
  /// The "row map" corresponds to the \c ptr array of row offsets in
  /// compressed sparse row (CSR) storage.
  typedef SizeType size_type;

  //! Type of a host-memory mirror of the sparse matrix.
  typedef CrsMatrix<ScalarType, OrdinalType, host_mirror_space, MemoryTraits,
                    SizeType>
      HostMirror;
  //! Type of the graph structure of the sparse matrix.
  typedef Kokkos::StaticCrsGraph<ordinal_type, Kokkos::LayoutRight, device_type,
                                 memory_traits, size_type>
      StaticCrsGraphType;
  //! Type of the graph structure of the sparse matrix - consistent with Kokkos.
  typedef Kokkos::StaticCrsGraph<ordinal_type, Kokkos::LayoutRight, device_type,
                                 memory_traits, size_type>
      staticcrsgraph_type;
  //! Type of column indices in the sparse matrix.
  typedef typename staticcrsgraph_type::entries_type index_type;
  //! Const version of the type of column indices in the sparse matrix.
  typedef typename index_type::const_value_type const_ordinal_type;
  //! Nonconst version of the type of column indices in the sparse matrix.
  typedef typename index_type::non_const_value_type non_const_ordinal_type;
  //! Type of the "row map" (which contains the offset for each row's data).
  typedef typename staticcrsgraph_type::row_map_type row_map_type;
  //! Const version of the type of row offsets in the sparse matrix.
  typedef typename row_map_type::const_value_type const_size_type;
  //! Nonconst version of the type of row offsets in the sparse matrix.
  typedef typename row_map_type::non_const_value_type non_const_size_type;
  //! Kokkos Array type of the entries (values) in the sparse matrix.
  typedef Kokkos::View<value_type*, Kokkos::LayoutRight, device_type,
                       MemoryTraits>
      values_type;
  //! Const version of the type of the entries in the sparse matrix.
  typedef typename values_type::const_value_type const_value_type;
  //! Nonconst version of the type of the entries in the sparse matrix.
  typedef typename values_type::non_const_value_type non_const_value_type;

  typedef CrsMatrix<const_value_type, ordinal_type, device_type, memory_traits,
                    size_type>
      const_type;

  /// \name Storage of the actual sparsity structure and values.
  ///
  /// CrsMatrix uses the compressed sparse row (CSR) storage format to
  /// store the sparse matrix.  CSR is also called "compressed row
  /// storage"; hence the name, which it inherits from Tpetra and from
  /// Epetra before it.
  //@{
  //! The graph (sparsity structure) of the sparse matrix.
  staticcrsgraph_type graph;
  //! The 1-D array of values of the sparse matrix.
  values_type values;
  //@}

 private:
  /// \brief The number of distinct column indices used by the matrix
  ///
  /// This value might not be exact but rather an upper bound of the
  /// number of distinct column indices used by the matrix.
  /// It provides multiple sparse algorithms to allocate appropriate
  /// amount of temporary work space or to allocate memory for the
  /// output of the kernel.
  ordinal_type numCols_;

 public:
  /// \brief Default constructor; constructs an empty sparse matrix.
  ///
  /// FIXME (mfh 09 Aug 2013) numCols and nnz should be properties of
  /// the graph, not the matrix.  Then CrsMatrix needs methods to get
  /// these from the graph.
  KOKKOS_INLINE_FUNCTION
  CrsMatrix() : numCols_(0) {}

  //! Copy constructor (shallow copy).
  template <typename InScalar, typename InOrdinal, class InDevice,
            class InMemTraits, typename InSizeType>
  KOKKOS_INLINE_FUNCTION CrsMatrix(
      const CrsMatrix<InScalar, InOrdinal, InDevice, InMemTraits, InSizeType>&
          B)
      : graph(B.graph.entries, B.graph.row_map),
        values(B.values),
        numCols_(B.numCols()) {
    graph.row_block_offsets = B.graph.row_block_offsets;
    // TODO: MD 07/2017: Changed the copy constructor of graph
    // as the constructor of StaticCrsGraph does not allow copy from non const
    // version.
  }

  //! Deep copy constructor (can cross spaces)
  template <typename InScalar, typename InOrdinal, typename InDevice,
            typename InMemTraits, typename InSizeType>
  CrsMatrix(const std::string&,
            const CrsMatrix<InScalar, InOrdinal, InDevice, InMemTraits,
                            InSizeType>& mat_) {
    typename row_map_type::non_const_type rowmap(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "rowmap"),
        mat_.graph.row_map.extent(0));
    index_type cols(Kokkos::view_alloc(Kokkos::WithoutInitializing, "cols"),
                    mat_.nnz());
    values = values_type(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "values"), mat_.nnz());
    Kokkos::deep_copy(rowmap, mat_.graph.row_map);
    Kokkos::deep_copy(cols, mat_.graph.entries);
    Kokkos::deep_copy(values, mat_.values);

    numCols_ = mat_.numCols();
    graph    = StaticCrsGraphType(cols, rowmap);
  }

  /// \brief Construct with a graph that will be shared.
  ///
  /// Allocate the values array for subsquent fill.
  template <typename InOrdinal, typename InLayout, typename InDevice,
            typename InMemTraits, typename InSizeType>
  [[deprecated(
      "Use the constructor that accepts ncols as input "
      "instead.")]] CrsMatrix(const std::string& label,
                              const Kokkos::StaticCrsGraph<
                                  InOrdinal, InLayout, InDevice, InMemTraits,
                                  InSizeType>& graph_)
      : graph(graph_.entries, graph_.row_map),
        values(label, graph_.entries.extent(0)),
        numCols_(maximum_entry(graph_) + 1) {}

  /// \brief Constructor that accepts a a static graph, and numCols.
  ///
  /// The matrix will store and use the row map, indices
  /// (by view, not by deep copy) and allocate the values view.
  ///
  /// \param label  [in] The sparse matrix's label.
  /// \param graph_ [in] The graph for storing the rowmap and col ids.
  /// \param ncols  [in] The number of columns.
  template <typename InOrdinal, typename InLayout, typename InDevice,
            typename InMemTraits, typename InSizeType>
  CrsMatrix(const std::string& label,
            const Kokkos::StaticCrsGraph<InOrdinal, InLayout, InDevice,
                                         InMemTraits, InSizeType>& graph_,
            const OrdinalType& ncols)
      : graph(graph_.entries, graph_.row_map),
        values(label, graph_.entries.extent(0)),
        numCols_(ncols) {}

  /// \brief Constructor that accepts a a static graph, and values.
  ///
  /// The matrix will store and use the row map, indices, and values
  /// directly (by view, not by deep copy).
  ///
  /// \param ncols [in] The number of columns.
  /// \param vals [in/out] The entries.
  /// \param graph_ The graph for storing the rowmap and col ids.
  template <typename InOrdinal, typename InLayout, typename InDevice,
            typename InMemTraits, typename InSizeType>
  CrsMatrix(const std::string&, const OrdinalType& ncols,
            const values_type& vals,
            const Kokkos::StaticCrsGraph<InOrdinal, InLayout, InDevice,
                                         InMemTraits, InSizeType>& graph_)
      : graph(graph_.entries, graph_.row_map), values(vals), numCols_(ncols) {}

  /// \brief Constructor that copies raw arrays of host data in
  ///   3-array CRS (compresed row storage) format.
  ///
  /// On input, the entries must be sorted by row. \c rowmap determines where
  /// each row begins and ends. For each entry k (0 <= k < annz), \c cols[k]
  /// gives the adjacent column, and \c val[k] gives the corresponding matrix
  /// value.
  ///
  /// This constructor is mainly useful for benchmarking or for
  /// reading the sparse matrix's data from a file.
  ///
  /// \param nrows [in] The number of rows.
  /// \param ncols [in] The number of columns.
  /// \param annz [in] The number of entries.
  /// \param val [in] The values.
  /// \param rowmap [in] The row offsets. The values/columns in row k begin at
  /// index
  ///   \c rowmap[k] and end at \c rowmap[k+1]-1 (inclusive). This means the
  ///   array must have length \c nrows+1.
  /// \param cols [in] The column indices. \c cols[k] is the column
  ///   index of entry k, with a corresponding value of \c val[k] .
  CrsMatrix(const std::string& /*label*/, OrdinalType nrows, OrdinalType ncols,
            size_type annz, ScalarType* val, OrdinalType* rowmap,
            OrdinalType* cols) {
    using Kokkos::Unmanaged;
    using HostRowmap       = Kokkos::View<SizeType*, Kokkos::HostSpace>;
    using UnmanagedRowmap  = Kokkos::View<const SizeType*, Kokkos::HostSpace,
                                         Kokkos::MemoryTraits<Unmanaged>>;
    using UnmanagedEntries = Kokkos::View<const OrdinalType*, Kokkos::HostSpace,
                                          Kokkos::MemoryTraits<Unmanaged>>;
    using UnmanagedValues  = Kokkos::View<const ScalarType*, Kokkos::HostSpace,
                                         Kokkos::MemoryTraits<Unmanaged>>;
    // Allocate device rowmap, entries, values views
    typename row_map_type::non_const_type rowmapDevice(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "rowmap"), nrows + 1);
    index_type entriesDevice(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "entries"), annz);
    // given rowmap in ordinal_type, so may need to convert to size_type
    // explicitly
    HostRowmap rowmapConverted;
    UnmanagedRowmap rowmapRaw;
    if (!std::is_same<OrdinalType, SizeType>::value) {
      rowmapConverted = HostRowmap(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "rowmap raw"),
          nrows + 1);
      for (OrdinalType i = 0; i <= nrows; i++) rowmapConverted(i) = rowmap[i];
      rowmapRaw = rowmapConverted;
    } else {
      rowmapRaw = UnmanagedRowmap((const SizeType*)rowmap, nrows + 1);
    }
    Kokkos::deep_copy(rowmapDevice, rowmapRaw);
    UnmanagedEntries entriesRaw(cols, annz);
    Kokkos::deep_copy(entriesDevice, entriesRaw);
    // Construct graph and populate all members
    this->numCols_ = ncols;
    this->graph    = StaticCrsGraphType(entriesDevice, rowmapDevice);
    this->values   = values_type(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "values"), annz);
    UnmanagedValues valuesRaw(val, annz);
    Kokkos::deep_copy(this->values, valuesRaw);
  }

  /// \brief Constructor that accepts a row map, column indices, and
  ///   values.
  ///
  /// The matrix will store and use the row map, indices, and values
  /// directly (by view, not by deep copy).
  ///
  /// \param nrows [in] The number of rows.
  /// \param ncols [in] The number of columns.
  /// \param annz [in] The number of entries.
  /// \param vals [in] The entries.
  /// \param rowmap [in] The row map (containing the offsets to the
  ///   data in each row).
  /// \param cols [in] The column indices.
  CrsMatrix(const std::string& /* label */, const OrdinalType nrows,
            const OrdinalType ncols, const size_type annz,
            const values_type& vals, const row_map_type& rowmap,
            const index_type& cols)
      : graph(cols, rowmap), values(vals), numCols_(ncols) {
    const ordinal_type actualNumRows =
        (rowmap.extent(0) != 0)
            ? static_cast<ordinal_type>(rowmap.extent(0) -
                                        static_cast<size_type>(1))
            : static_cast<ordinal_type>(0);
    if (nrows != actualNumRows) {
      std::ostringstream os;
      os << "Input argument nrows = " << nrows
         << " != the actual number of "
            "rows "
         << actualNumRows << " according to the 'rows' input argument.";
      throw std::invalid_argument(os.str());
    }
    if (annz != nnz()) {
      std::ostringstream os;
      os << "Input argument annz = " << annz << " != this->nnz () = " << nnz()
         << ".";
      throw std::invalid_argument(os.str());
    }
  }

  //! Attempt to assign the input matrix to \c *this.
  template <typename aScalarType, typename aOrdinalType, class aDevice,
            class aMemoryTraits, typename aSizeType>
  CrsMatrix& operator=(const CrsMatrix<aScalarType, aOrdinalType, aDevice,
                                       aMemoryTraits, aSizeType>& mtx) {
    numCols_ = mtx.numCols();
    graph    = mtx.graph;
    values   = mtx.values;
    return *this;
  }

  //! The number of rows in the sparse matrix.
  KOKKOS_INLINE_FUNCTION ordinal_type numRows() const {
    return graph.numRows();
  }

  //! The number of columns in the sparse matrix.
  KOKKOS_INLINE_FUNCTION ordinal_type numCols() const { return numCols_; }

  //! The number of "point" (non-block) rows in the matrix. Since Crs is not
  //! blocked, this is just the number of regular rows.
  KOKKOS_INLINE_FUNCTION ordinal_type numPointRows() const { return numRows(); }

  //! The number of "point" (non-block) columns in the matrix. Since Crs is not
  //! blocked, this is just the number of regular columns.
  KOKKOS_INLINE_FUNCTION ordinal_type numPointCols() const { return numCols(); }

  //! The number of stored entries in the sparse matrix.
  KOKKOS_INLINE_FUNCTION size_type nnz() const {
    return graph.entries.extent(0);
  }
};

}  // namespace KokkosSparse
#endif
