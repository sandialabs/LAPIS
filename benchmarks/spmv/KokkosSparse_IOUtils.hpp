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
#ifndef _KOKKOSSPARSE_IOUTILS_HPP
#define _KOKKOSSPARSE_IOUTILS_HPP

#include "KokkosSparse_CrsMatrix.hpp"
#include <iomanip>
#include <iostream>
#include <fstream>

namespace KokkosSparse {

template <typename idx, typename wt>
struct Edge {
  idx src;
  idx dst;
  wt ew;
  bool operator<(const Edge<idx, wt> &a) const {
    // return !((this->src < a.src) || (this->src == a.src && this->dst <
    // a.dst));
    return (this->src < a.src) || (this->src == a.src && this->dst < a.dst);
  }
};

inline bool endswith(std::string const &fullString, std::string const &ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(),
                                    ending.length(), ending));
  } else {
    return false;
  }
}

// MM: types and utility functions for parsing the MatrixMarket format
namespace MM {
enum MtxObject { UNDEFINED_OBJECT, MATRIX, VECTOR };
enum MtxFormat { UNDEFINED_FORMAT, COORDINATE, ARRAY };
enum MtxField {
  UNDEFINED_FIELD,
  REAL,     // includes both float and double
  COMPLEX,  // includes complex<float> and complex<double>
  INTEGER,  // includes all integer types
  PATTERN   // not a type, but means the value for every entry is 1
};
enum MtxSym {
  UNDEFINED_SYMMETRY,
  GENERAL,
  SYMMETRIC,       // A(i, j) = A(j, i)
  SKEW_SYMMETRIC,  // A(i, j) = -A(j, i)
  HERMITIAN        // A(i, j) = a + bi; A(j, i) = a - bi
};

// readScalar/writeScalar: read and write a scalar in the form that it appears
// in an .mtx file. The >> and << operators won't work, because complex appears
// as "real imag", not "(real, imag)"
template <typename scalar_t>
scalar_t readScalar(std::istream &is) {
  scalar_t val;
  is >> val;
  return val;
}

template <>
inline Kokkos::complex<float> readScalar(std::istream &is) {
  float r, i;
  is >> r;
  is >> i;
  return Kokkos::complex<float>(r, i);
}

template <>
inline Kokkos::complex<double> readScalar(std::istream &is) {
  double r, i;
  is >> r;
  is >> i;
  return Kokkos::complex<double>(r, i);
}

template <typename scalar_t>
void writeScalar(std::ostream &os, scalar_t val) {
  os << val;
}

template <>
inline void writeScalar(std::ostream &os, Kokkos::complex<float> val) {
  os << val.real() << ' ' << val.imag();
}

template <>
inline void writeScalar(std::ostream &os, Kokkos::complex<double> val) {
  os << val.real() << ' ' << val.imag();
}

// symmetryFlip: given a value for A(i, j), return the value that
// should be inserted at A(j, i) (if any)
template <typename scalar_t>
scalar_t symmetryFlip(scalar_t val, MtxSym symFlag) {
  if (symFlag == SKEW_SYMMETRIC) return -val;
  return val;
}

template <>
inline Kokkos::complex<float> symmetryFlip(Kokkos::complex<float> val,
                                           MtxSym symFlag) {
  if (symFlag == HERMITIAN)
    return Kokkos::conj(val);
  else if (symFlag == SKEW_SYMMETRIC)
    return -val;
  return val;
}

template <>
inline Kokkos::complex<double> symmetryFlip(Kokkos::complex<double> val,
                                            MtxSym symFlag) {
  if (symFlag == HERMITIAN)
    return Kokkos::conj(val);
  else if (symFlag == SKEW_SYMMETRIC)
    return -val;
  return val;
}
}  // namespace MM

template <typename lno_t, typename size_type, typename scalar_t>
void write_matrix_mtx(lno_t nrows, lno_t ncols, size_type nentries,
                      const size_type *xadj, const lno_t *adj,
                      const scalar_t *vals, const char *filename) {
  std::ofstream myFile(filename);
  myFile << "%%MatrixMarket matrix coordinate ";
  if (std::is_same<scalar_t, Kokkos::complex<float>>::value ||
      std::is_same<scalar_t, Kokkos::complex<double>>::value)
    myFile << "complex";
  else
    myFile << "real";
  myFile << " general\n";
  myFile << nrows << " " << ncols << " " << nentries << '\n';
  myFile << std::setprecision(17) << std::scientific;
  for (lno_t i = 0; i < nrows; ++i) {
    size_type b = xadj[i];
    size_type e = xadj[i + 1];
    for (size_type j = b; j < e; ++j) {
      myFile << i + 1 << " " << adj[j] + 1 << " ";
      MM::writeScalar<scalar_t>(myFile, vals[j]);
      myFile << '\n';
    }
  }
  myFile.close();
}

template <typename lno_t, typename size_type, typename scalar_t>
void write_graph_mtx(lno_t nv, size_type ne, const size_type *xadj,
                     const lno_t *adj, const scalar_t *ew,
                     const char *filename) {
  std::ofstream myFile(filename);
  myFile << "%%MatrixMarket matrix coordinate ";
  if (std::is_same<scalar_t, Kokkos::complex<float>>::value ||
      std::is_same<scalar_t, Kokkos::complex<double>>::value)
    myFile << "complex";
  else
    myFile << "real";
  myFile << " general\n";
  myFile << nv << " " << nv << " " << ne << '\n';
  myFile << std::setprecision(8) << std::scientific;
  for (lno_t i = 0; i < nv; ++i) {
    size_type b = xadj[i];
    size_type e = xadj[i + 1];
    for (size_type j = b; j < e; ++j) {
      myFile << i + 1 << " " << (adj)[j] + 1 << " ";
      MM::writeScalar<scalar_t>(myFile, ew[j]);
      myFile << '\n';
    }
  }

  myFile.close();
}

template <typename lno_t, typename size_type, typename scalar_t>
void read_graph_bin(lno_t *nv, size_type *ne, size_type **xadj, lno_t **adj,
                    scalar_t **ew, const char *filename) {
  std::ifstream myFile(filename, std::ios::in | std::ios::binary);

  myFile.read((char *)nv, sizeof(lno_t));
  myFile.read((char *)ne, sizeof(size_type));
  *xadj = new size_type[*nv + 1];
  *adj  = new lno_t[*ne];
  *ew   = new scalar_t[*ne];
  myFile.read((char *)*xadj, sizeof(size_type) * (*nv + 1));
  myFile.read((char *)*adj, sizeof(lno_t) * (*ne));
  myFile.read((char *)*ew, sizeof(scalar_t) * (*ne));
  myFile.close();
}

// When Kokkos issue #2313 is resolved, can delete
// parseScalar and just use operator>>
template <typename scalar_t>
scalar_t parseScalar(std::istream &is) {
  scalar_t val;
  is >> val;
  return val;
}

template <>
inline Kokkos::complex<float> parseScalar(std::istream &is) {
  std::complex<float> val;
  is >> val;
  return Kokkos::complex<float>(val);
}

template <>
inline Kokkos::complex<double> parseScalar(std::istream &is) {
  std::complex<double> val;
  is >> val;
  return Kokkos::complex<double>(val);
}

template <typename lno_t, typename size_type, typename scalar_t>
void read_graph_crs(lno_t *nv, size_type *ne, size_type **xadj, lno_t **adj,
                    scalar_t **ew, const char *filename) {
  std::ifstream myFile(filename, std::ios::in);
  myFile >> *nv >> *ne;

  *xadj = new size_type[*nv + 1];
  *adj  = new lno_t[*ne];
  *ew   = new scalar_t[*ne];

  for (lno_t i = 0; i <= *nv; ++i) {
    myFile >> (*xadj)[i];
  }

  for (size_type i = 0; i < *ne; ++i) {
    myFile >> (*adj)[i];
  }
  for (size_type i = 0; i < *ne; ++i) {
    (*ew)[i] = parseScalar<scalar_t>(myFile);
  }
  myFile.close();
}

template <typename lno_t, typename size_type, typename scalar_t>
int read_mtx(const char *fileName, lno_t *nrows, lno_t *ncols, size_type *ne,
             size_type **xadj, lno_t **adj, scalar_t **ew,
             bool symmetrize = false, bool remove_diagonal = true,
             bool transpose = false) {
  using namespace MM;
  std::ifstream mmf(fileName, std::ifstream::in);
  if (!mmf.is_open()) {
    throw std::runtime_error("File cannot be opened\n");
  }

  std::string fline = "";
  getline(mmf, fline);

  if (fline.size() < 2 || fline[0] != '%' || fline[1] != '%') {
    throw std::runtime_error("Invalid MM file. Line-1\n");
  }

  // make sure every required field is in the file, by initializing them to
  // UNDEFINED_*
  MtxObject mtx_object = UNDEFINED_OBJECT;
  MtxFormat mtx_format = UNDEFINED_FORMAT;
  MtxField mtx_field   = UNDEFINED_FIELD;
  MtxSym mtx_sym       = UNDEFINED_SYMMETRY;

  if (fline.find("matrix") != std::string::npos) {
    mtx_object = MATRIX;
  } else if (fline.find("vector") != std::string::npos) {
    mtx_object = VECTOR;
    throw std::runtime_error(
        "MatrixMarket \"vector\" is not supported by KokkosKernels read_mtx()");
  }

  if (fline.find("coordinate") != std::string::npos) {
    // sparse
    mtx_format = COORDINATE;
  } else if (fline.find("array") != std::string::npos) {
    // dense
    mtx_format = ARRAY;
  }

  if (fline.find("real") != std::string::npos ||
      fline.find("double") != std::string::npos) {
    if (std::is_same<scalar_t, Kokkos::Experimental::half_t>::value ||
        std::is_same<scalar_t, Kokkos::Experimental::bhalf_t>::value)
      mtx_field = REAL;
    else {
      if (!std::is_floating_point<scalar_t>::value)
        throw std::runtime_error(
            "scalar_t in read_mtx() incompatible with float or double typed "
            "MatrixMarket file.");
      else
        mtx_field = REAL;
    }
  } else if (fline.find("complex") != std::string::npos) {
    if (!(std::is_same<scalar_t, Kokkos::complex<float>>::value ||
          std::is_same<scalar_t, Kokkos::complex<double>>::value))
      throw std::runtime_error(
          "scalar_t in read_mtx() incompatible with complex-typed MatrixMarket "
          "file.");
    else
      mtx_field = COMPLEX;
  } else if (fline.find("integer") != std::string::npos) {
    if (std::is_integral<scalar_t>::value ||
        std::is_floating_point<scalar_t>::value ||
        std::is_same<scalar_t, Kokkos::Experimental::half_t>::value ||
        std::is_same<scalar_t, Kokkos::Experimental::bhalf_t>::value)
      mtx_field = INTEGER;
    else
      throw std::runtime_error(
          "scalar_t in read_mtx() incompatible with integer-typed MatrixMarket "
          "file.");
  } else if (fline.find("pattern") != std::string::npos) {
    mtx_field = PATTERN;
    // any reasonable choice for scalar_t can represent "1" or "1.0 + 0i", so
    // nothing to check here
  }

  if (fline.find("general") != std::string::npos) {
    mtx_sym = GENERAL;
  } else if (fline.find("skew-symmetric") != std::string::npos) {
    mtx_sym = SKEW_SYMMETRIC;
  } else if (fline.find("symmetric") != std::string::npos) {
    // checking for "symmetric" after "skew-symmetric" because it's a substring
    mtx_sym = SYMMETRIC;
  } else if (fline.find("hermitian") != std::string::npos ||
             fline.find("Hermitian") != std::string::npos) {
    mtx_sym = HERMITIAN;
  }
  // Validate the matrix attributes
  if (mtx_format == ARRAY) {
    if (mtx_sym == UNDEFINED_SYMMETRY) mtx_sym = GENERAL;
    if (mtx_sym != GENERAL)
      throw std::runtime_error(
          "array format MatrixMarket file must have general symmetry (optional "
          "to include \"general\")");
  }
  if (mtx_object == UNDEFINED_OBJECT)
    throw std::runtime_error(
        "MatrixMarket file header is missing the object type.");
  if (mtx_format == UNDEFINED_FORMAT)
    throw std::runtime_error("MatrixMarket file header is missing the format.");
  if (mtx_field == UNDEFINED_FIELD)
    throw std::runtime_error(
        "MatrixMarket file header is missing the field type.");
  if (mtx_sym == UNDEFINED_SYMMETRY)
    throw std::runtime_error(
        "MatrixMarket file header is missing the symmetry type.");

  while (1) {
    getline(mmf, fline);
    if (fline[0] != '%') break;
  }
  std::stringstream ss(fline);
  lno_t nr = 0, nc = 0;
  size_type nnz = 0;
  ss >> nr >> nc;
  if (mtx_format == COORDINATE)
    ss >> nnz;
  else
    nnz = nr * nc;
  size_type numEdges = nnz;
  symmetrize         = symmetrize || mtx_sym != GENERAL;
  if (symmetrize && nr != nc) {
    throw std::runtime_error("A non-square matrix cannot be symmetrized.");
  }
  if (mtx_format == ARRAY) {
    // Array format only supports general symmetry and non-pattern
    if (symmetrize)
      throw std::runtime_error(
          "array format MatrixMarket file cannot be symmetrized.");
    if (mtx_field == PATTERN)
      throw std::runtime_error(
          "array format MatrixMarket file can't have \"pattern\" field type.");
  }
  if (symmetrize) {
    numEdges = 2 * nnz;
  }
  // numEdges is only an upper bound (diagonal entries may be removed)
  std::vector<Edge<lno_t, scalar_t>> edges(numEdges);
  size_type nE      = 0;
  lno_t numDiagonal = 0;
  for (size_type i = 0; i < nnz; ++i) {
    getline(mmf, fline);
    std::stringstream ss2(fline);
    Edge<lno_t, scalar_t> tmp;
    // read source, dest (edge) and weight (value)
    lno_t s, d;
    scalar_t w;
    if (mtx_format == ARRAY) {
      // In array format, entries are listed in column major order,
      // so the row and column can be determined just from the index i
      //(but make them 1-based indices, to match the way coordinate works)
      s = i % nr + 1;  // row
      d = i / nr + 1;  // col
    } else {
      // In coordinate format, row and col of each entry is read from file
      ss2 >> s >> d;
    }
    if (mtx_field == PATTERN)
      w = 1;
    else
      w = readScalar<scalar_t>(ss2);
    if (!transpose) {
      tmp.src = s - 1;
      tmp.dst = d - 1;
      tmp.ew  = w;
    } else {
      tmp.src = d - 1;
      tmp.dst = s - 1;
      tmp.ew  = w;
    }
    if (tmp.src == tmp.dst) {
      numDiagonal++;
      if (!remove_diagonal) {
        edges[nE++] = tmp;
      }
      continue;
    }
    edges[nE++] = tmp;
    if (symmetrize) {
      Edge<lno_t, scalar_t> tmp2;
      tmp2.src = tmp.dst;
      tmp2.dst = tmp.src;
      // the symmetrized value is w, -w or conj(w) if mtx_sym is
      // SYMMETRIC, SKEW_SYMMETRIC or HERMITIAN, respectively.
      tmp2.ew     = symmetryFlip<scalar_t>(tmp.ew, mtx_sym);
      edges[nE++] = tmp2;
    }
  }
  mmf.close();
  std::sort(edges.begin(), edges.begin() + nE);
  if (transpose) {
    lno_t tmp = nr;
    nr        = nc;
    nc        = tmp;
  }
  // idx *nv, idx *ne, idx **xadj, idx **adj, wt **wt
  *nrows = nr;
  *ncols = nc;
  *ne    = nE;

  *xadj = new size_type[nr + 1];
  *adj  = new lno_t[nE];
  *ew   = new scalar_t[nE];

  size_type eind   = 0;
  size_type actual = 0;
  for (lno_t i = 0; i < nr; ++i) {
    (*xadj)[i]    = actual;
    bool is_first = true;
    while (eind < nE && edges[eind].src == i) {
      if (is_first || !symmetrize || eind == 0 ||
          (eind > 0 && edges[eind - 1].dst != edges[eind].dst)) {
        (*adj)[actual] = edges[eind].dst;
        (*ew)[actual]  = edges[eind].ew;
        ++actual;
      }
      is_first = false;
      ++eind;
    }
  }
  (*xadj)[nr] = actual;
  *ne         = actual;
  return 0;
}

// Version of read_mtx which does not capture the number of columns.
// This is the old interface; it's kept for backwards compatibility.
template <typename lno_t, typename size_type, typename scalar_t>
int read_mtx(const char *fileName, lno_t *nv, size_type *ne, size_type **xadj,
             lno_t **adj, scalar_t **ew, bool symmetrize = false,
             bool remove_diagonal = true, bool transpose = false) {
  lno_t ncol;  // will discard
  return read_mtx<lno_t, size_type, scalar_t>(fileName, nv, &ncol, ne, xadj,
                                              adj, ew, symmetrize,
                                              remove_diagonal, transpose);
}

template <typename lno_t, typename size_type, typename scalar_t>
void read_matrix(lno_t *nv, size_type *ne, size_type **xadj, lno_t **adj,
                 scalar_t **ew, const char *filename) {
  std::string strfilename(filename);
  if (endswith(strfilename, ".mtx") || endswith(strfilename, ".mm")) {
    read_mtx(filename, nv, ne, xadj, adj, ew, false, false, false);
  }

  else if (endswith(strfilename, ".bin")) {
    read_graph_bin(nv, ne, xadj, adj, ew, filename);
  }

  else if (endswith(strfilename, ".crs")) {
    read_graph_crs(nv, ne, xadj, adj, ew, filename);
  }

  else {
    throw std::runtime_error("Reader is not available\n");
  }
}

template <typename crsMat_t>
crsMat_t read_kokkos_crst_matrix(const char *filename_) {
  std::string strfilename(filename_);
  bool isMatrixMarket =
      endswith(strfilename, ".mtx") || endswith(strfilename, ".mm");

  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename graph_t::entries_type::non_const_type cols_view_t;
  typedef typename crsMat_t::values_type::non_const_type values_view_t;

  typedef typename row_map_view_t::value_type size_type;
  typedef typename cols_view_t::value_type lno_t;
  typedef typename values_view_t::value_type scalar_t;

  lno_t nr, nc, *adj;
  size_type *xadj, nnzA;
  scalar_t *values;

  if (isMatrixMarket) {
    // MatrixMarket file contains the exact number of columns
    read_mtx<lno_t, size_type, scalar_t>(filename_, &nr, &nc, &nnzA, &xadj,
                                         &adj, &values, false, false, false);
  } else {
    //.crs and .bin files don't contain #cols, so will compute it later based on
    // the entries
    read_matrix<lno_t, size_type, scalar_t>(&nr, &nnzA, &xadj, &adj, &values,
                                            filename_);
  }

  row_map_view_t rowmap_view("rowmap_view", nr + 1);
  cols_view_t columns_view("colsmap_view", nnzA);
  values_view_t values_view("values_view", nnzA);

  {
    Kokkos::View<size_type *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        hr(xadj, nr + 1);
    Kokkos::View<lno_t *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        hc(adj, nnzA);
    Kokkos::View<scalar_t *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        hv(values, nnzA);
    Kokkos::deep_copy(rowmap_view, hr);
    Kokkos::deep_copy(columns_view, hc);
    Kokkos::deep_copy(values_view, hv);
  }

  graph_t static_graph(columns_view, rowmap_view);
  crsMat_t crsmat("CrsMatrix", nc, values_view, static_graph);
  delete[] xadj;
  delete[] adj;
  delete[] values;
  return crsmat;
}

template <typename crsGraph_t>
crsGraph_t read_kokkos_crst_graph(const char *filename_) {
  typedef typename crsGraph_t::row_map_type::non_const_type row_map_view_t;
  typedef typename crsGraph_t::entries_type::non_const_type cols_view_t;

  typedef typename row_map_view_t::value_type size_type;
  typedef typename cols_view_t::value_type lno_t;
  typedef double scalar_t;

  lno_t nv, *adj;
  size_type *xadj, nnzA;
  scalar_t *values;
  read_matrix<lno_t, size_type, scalar_t>(&nv, &nnzA, &xadj, &adj, &values,
                                          filename_);

  row_map_view_t rowmap_view("rowmap_view", nv + 1);
  cols_view_t columns_view("colsmap_view", nnzA);

  typename row_map_view_t::HostMirror hr(xadj, nv + 1);
  typename cols_view_t::HostMirror hc(adj, nnzA);
  Kokkos::deep_copy(rowmap_view, hr);
  Kokkos::deep_copy(columns_view, hc);

  delete[] xadj;
  delete[] adj;
  delete[] values;

  crsGraph_t static_graph(columns_view, rowmap_view);
  return static_graph;
}

}  // namespace KokkosSparse
#endif  // _KOKKOSSPARSE_IOUTILS_HPP
