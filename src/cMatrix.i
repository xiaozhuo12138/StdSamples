////////////////////////////////////////////////////////////////////////
//
// Copyright (C) 1996-2022 The Octave Project Developers
//
// See the file COPYRIGHT.md in the top-level directory of this
// distribution or <https://octave.org/copyright/>.
//
// This file is part of Octave.
//
// Octave is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Octave is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Octave; see the file COPYING.  If not, see
// <https://www.gnu.org/licenses/>.
//
////////////////////////////////////////////////////////////////////////

%{
#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>
#include <octave/interpreter.h>
%}

class ComplexMatrix : public ComplexNDArray
{
public:

  typedef ComplexColumnVector column_vector_type;
  typedef ComplexRowVector row_vector_type;

  typedef ColumnVector real_column_vector_type;
  typedef RowVector real_row_vector_type;

  typedef Matrix real_matrix_type;
  typedef ComplexMatrix complex_matrix_type;

  typedef DiagMatrix real_diag_matrix_type;
  typedef ComplexDiagMatrix complex_diag_matrix_type;

  typedef double real_elt_type;
  typedef Complex complex_elt_type;

  typedef void (*solve_singularity_handler) (double rcon);

  ComplexMatrix (void) = default;

  ComplexMatrix (const ComplexMatrix& a) = default;

  ComplexMatrix& operator = (const ComplexMatrix& a) = default;

  ~ComplexMatrix (void) = default;

  ComplexMatrix (octave_idx_type r, octave_idx_type c);
  ComplexMatrix (octave_idx_type r, octave_idx_type c, const Complex& val);
  ComplexMatrix (const dim_vector& dv) : ComplexNDArray (dv.redim (2));
  ComplexMatrix (const dim_vector& dv, const Complex& val);
  template <typename U>
  ComplexMatrix (const MArray<U>& a);
  template <typename U>
  ComplexMatrix (const Array<U>& a);
  ComplexMatrix (const Matrix& re, const Matrix& im);

  explicit ComplexMatrix (const Matrix& a);

  explicit ComplexMatrix (const RowVector& rv);

  explicit ComplexMatrix (const ColumnVector& cv);

  explicit ComplexMatrix (const DiagMatrix& a);

  explicit ComplexMatrix (const MDiagArray2<double>& a);

  explicit ComplexMatrix (const DiagArray2<double>& a);

  explicit ComplexMatrix (const ComplexRowVector& rv);

  explicit ComplexMatrix (const ComplexColumnVector& cv);

  explicit ComplexMatrix (const ComplexDiagMatrix& a);

  explicit ComplexMatrix (const MDiagArray2<Complex>& a);

  explicit ComplexMatrix (const DiagArray2<Complex>& a);

  explicit ComplexMatrix (const boolMatrix& a);

  explicit ComplexMatrix (const charMatrix& a);

  bool operator == (const ComplexMatrix& a) const;
  bool operator != (const ComplexMatrix& a) const;

  bool ishermitian (void) const;

  // destructive insert/delete/reorder operations

  ComplexMatrix&
  insert (const Matrix& a, octave_idx_type r, octave_idx_type c);
  ComplexMatrix&
  insert (const RowVector& a, octave_idx_type r, octave_idx_type c);
  ComplexMatrix&
  insert (const ColumnVector& a, octave_idx_type r, octave_idx_type c);
  ComplexMatrix&
  insert (const DiagMatrix& a, octave_idx_type r, octave_idx_type c);

  ComplexMatrix&
  insert (const ComplexMatrix& a, octave_idx_type r, octave_idx_type c);
  ComplexMatrix&
  insert (const ComplexRowVector& a, octave_idx_type r, octave_idx_type c);
  ComplexMatrix&
  insert (const ComplexColumnVector& a, octave_idx_type r, octave_idx_type c);
  ComplexMatrix&
  insert (const ComplexDiagMatrix& a, octave_idx_type r, octave_idx_type c);

  ComplexMatrix& fill (double val);
  ComplexMatrix& fill (const Complex& val);
  ComplexMatrix&
  fill (double val, octave_idx_type r1, octave_idx_type c1,
        octave_idx_type r2, octave_idx_type c2);
  ComplexMatrix&
  fill (const Complex& val, octave_idx_type r1, octave_idx_type c1,
        octave_idx_type r2, octave_idx_type c2);

  ComplexMatrix append (const Matrix& a) const;
  ComplexMatrix append (const RowVector& a) const;
  ComplexMatrix append (const ColumnVector& a) const;
  ComplexMatrix append (const DiagMatrix& a) const;

  ComplexMatrix append (const ComplexMatrix& a) const;
  ComplexMatrix append (const ComplexRowVector& a) const;
  ComplexMatrix append (const ComplexColumnVector& a) const;
  ComplexMatrix append (const ComplexDiagMatrix& a) const;

  ComplexMatrix stack (const Matrix& a) const;
  ComplexMatrix stack (const RowVector& a) const;
  ComplexMatrix stack (const ColumnVector& a) const;
  ComplexMatrix stack (const DiagMatrix& a) const;

  ComplexMatrix stack (const ComplexMatrix& a) const;
  ComplexMatrix stack (const ComplexRowVector& a) const;
  ComplexMatrix stack (const ComplexColumnVector& a) const;
  ComplexMatrix stack (const ComplexDiagMatrix& a) const;

  ComplexMatrix hermitian (void) const;
  ComplexMatrix transpose (void) const;
  friend ComplexMatrix conj (const ComplexMatrix& a);

  // resize is the destructive equivalent for this one

  ComplexMatrix
  extract (octave_idx_type r1, octave_idx_type c1,
           octave_idx_type r2, octave_idx_type c2) const;

  ComplexMatrix
  extract_n (octave_idx_type r1, octave_idx_type c1,
             octave_idx_type nr, octave_idx_type nc) const;

  // extract row or column i.

  ComplexRowVector row (octave_idx_type i) const;

  ComplexColumnVector column (octave_idx_type i) const;

  void resize (octave_idx_type nr, octave_idx_type nc,
               const Complex& rfv = Complex (0));

  ComplexMatrix inverse (void) const;
  ComplexMatrix inverse (octave_idx_type& info) const;
  ComplexMatrix
  inverse (octave_idx_type& info, double& rcon,
           bool force = false, bool calc_cond = true) const;

  ComplexMatrix inverse (MatrixType& mattype) const;
  ComplexMatrix
  inverse (MatrixType& mattype, octave_idx_type& info) const;
  ComplexMatrix
  inverse (MatrixType& mattype, octave_idx_type& info, double& rcon,
           bool force = false, bool calc_cond = true) const;

  ComplexMatrix pseudo_inverse (double tol = 0.0) const;

  ComplexMatrix fourier (void) const;
  ComplexMatrix ifourier (void) const;

  ComplexMatrix fourier2d (void) const;
  ComplexMatrix ifourier2d (void) const;

  ComplexDET determinant (void) const;
  ComplexDET determinant (octave_idx_type& info) const;
  ComplexDET
  determinant (octave_idx_type& info, double& rcon,
               bool calc_cond = true) const;
  ComplexDET
  determinant (MatrixType& mattype, octave_idx_type& info, double& rcon,
               bool calc_cond = true) const;

  double rcond (void) const;
  double rcond (MatrixType& mattype) const;

  // Generic interface to solver with no probing of type
  ComplexMatrix solve (MatrixType& mattype, const Matrix& b) const;
  ComplexMatrix
  solve (MatrixType& mattype, const Matrix& b, octave_idx_type& info) const;
  ComplexMatrix
  solve (MatrixType& mattype, const Matrix& b, octave_idx_type& info,
         double& rcon) const;
  ComplexMatrix
  solve (MatrixType& mattype, const Matrix& b, octave_idx_type& info,
         double& rcon, solve_singularity_handler sing_handler,
         bool singular_fallback = true,
         blas_trans_type transt = blas_no_trans) const;

  ComplexMatrix
  solve (MatrixType& mattype, const ComplexMatrix& b) const;
  ComplexMatrix
  solve (MatrixType& mattype, const ComplexMatrix& b,
         octave_idx_type& info) const;
  ComplexMatrix
  solve (MatrixType& mattype, const ComplexMatrix& b, octave_idx_type& info,
         double& rcon) const;
  ComplexMatrix
  solve (MatrixType& mattype, const ComplexMatrix& b, octave_idx_type& info,
         double& rcon, solve_singularity_handler sing_handler,
         bool singular_fallback = true,
         blas_trans_type transt = blas_no_trans) const;

  ComplexColumnVector
  solve (MatrixType& mattype, const ColumnVector& b) const;
  ComplexColumnVector
  solve (MatrixType& mattype, const ColumnVector& b,
         octave_idx_type& info) const;
  ComplexColumnVector
  solve (MatrixType& mattype, const ColumnVector& b,
         octave_idx_type& info, double& rcon) const;
  ComplexColumnVector
  solve (MatrixType& mattype, const ColumnVector& b, octave_idx_type& info,
         double& rcon, solve_singularity_handler sing_handler,
         blas_trans_type transt = blas_no_trans) const;

  ComplexColumnVector
  solve (MatrixType& mattype, const ComplexColumnVector& b) const;
  ComplexColumnVector
  solve (MatrixType& mattype, const ComplexColumnVector& b,
         octave_idx_type& info) const;
  ComplexColumnVector
  solve (MatrixType& mattype, const ComplexColumnVector& b,
         octave_idx_type& info, double& rcon) const;
  ComplexColumnVector
  solve (MatrixType& mattype, const ComplexColumnVector& b,
         octave_idx_type& info, double& rcon,
         solve_singularity_handler sing_handler,
         blas_trans_type transt = blas_no_trans) const;

  // Generic interface to solver with probing of type
  ComplexMatrix solve (const Matrix& b) const;
  ComplexMatrix
  solve (const Matrix& b, octave_idx_type& info) const;
  ComplexMatrix
  solve (const Matrix& b, octave_idx_type& info, double& rcon) const;
  ComplexMatrix
  solve (const Matrix& b, octave_idx_type& info, double& rcon,
         solve_singularity_handler sing_handler,
         blas_trans_type transt = blas_no_trans) const;

  ComplexMatrix solve (const ComplexMatrix& b) const;
  ComplexMatrix
  solve (const ComplexMatrix& b, octave_idx_type& info) const;
  ComplexMatrix
  solve (const ComplexMatrix& b, octave_idx_type& info, double& rcon) const;
  ComplexMatrix
  solve (const ComplexMatrix& b, octave_idx_type& info, double& rcon,
         solve_singularity_handler sing_handler,
         blas_trans_type transt = blas_no_trans) const;

  ComplexColumnVector solve (const ColumnVector& b) const;
  ComplexColumnVector
  solve (const ColumnVector& b, octave_idx_type& info) const;
  ComplexColumnVector
  solve (const ColumnVector& b, octave_idx_type& info, double& rcon) const;
  ComplexColumnVector
  solve (const ColumnVector& b, octave_idx_type& info, double& rcon,
         solve_singularity_handler sing_handler,
         blas_trans_type transt = blas_no_trans) const;

  ComplexColumnVector solve (const ComplexColumnVector& b) const;
  ComplexColumnVector
  solve (const ComplexColumnVector& b, octave_idx_type& info) const;
  ComplexColumnVector
  solve (const ComplexColumnVector& b, octave_idx_type& info,
         double& rcon) const;
  ComplexColumnVector
  solve (const ComplexColumnVector& b, octave_idx_type& info, double& rcon,
         solve_singularity_handler sing_handler,
         blas_trans_type transt = blas_no_trans) const;

  ComplexMatrix lssolve (const Matrix& b) const;
  ComplexMatrix
  lssolve (const Matrix& b, octave_idx_type& info) const;
  ComplexMatrix
  lssolve (const Matrix& b, octave_idx_type& info,
           octave_idx_type& rank) const;
  ComplexMatrix
  lssolve (const Matrix& b, octave_idx_type& info,
           octave_idx_type& rank, double& rcon) const;

  ComplexMatrix lssolve (const ComplexMatrix& b) const;
  ComplexMatrix
  lssolve (const ComplexMatrix& b, octave_idx_type& info) const;
  ComplexMatrix
  lssolve (const ComplexMatrix& b, octave_idx_type& info,
           octave_idx_type& rank) const;
  ComplexMatrix
  lssolve (const ComplexMatrix& b, octave_idx_type& info,
           octave_idx_type& rank, double& rcon) const;

  ComplexColumnVector lssolve (const ColumnVector& b) const;
  ComplexColumnVector
  lssolve (const ColumnVector& b, octave_idx_type& info) const;
  ComplexColumnVector
  lssolve (const ColumnVector& b, octave_idx_type& info,
           octave_idx_type& rank) const;
  ComplexColumnVector
  lssolve (const ColumnVector& b, octave_idx_type& info,
           octave_idx_type& rank, double& rcon) const;

  ComplexColumnVector lssolve (const ComplexColumnVector& b) const;
  ComplexColumnVector
  lssolve (const ComplexColumnVector& b, octave_idx_type& info) const;
  ComplexColumnVector
  lssolve (const ComplexColumnVector& b, octave_idx_type& info,
           octave_idx_type& rank) const;
  ComplexColumnVector
  lssolve (const ComplexColumnVector& b, octave_idx_type& info,
           octave_idx_type& rank, double& rcon) const;

  // matrix by diagonal matrix -> matrix operations

  ComplexMatrix& operator += (const DiagMatrix& a);
  ComplexMatrix& operator -= (const DiagMatrix& a);

  ComplexMatrix& operator += (const ComplexDiagMatrix& a);
  ComplexMatrix& operator -= (const ComplexDiagMatrix& a);

  // matrix by matrix -> matrix operations

  ComplexMatrix& operator += (const Matrix& a);
  ComplexMatrix& operator -= (const Matrix& a);

  // other operations

  boolMatrix all (int dim = -1) const;
  boolMatrix any (int dim = -1) const;

  ComplexMatrix cumprod (int dim = -1) const;
  ComplexMatrix cumsum (int dim = -1) const;
  ComplexMatrix prod (int dim = -1) const;
  ComplexMatrix sum (int dim = -1) const;
  ComplexMatrix sumsq (int dim = -1) const;
  Matrix abs (void) const;

  ComplexMatrix diag (octave_idx_type k = 0) const;

  ComplexDiagMatrix
  diag (octave_idx_type m, octave_idx_type n) const;

  bool row_is_real_only (octave_idx_type) const;
  bool column_is_real_only (octave_idx_type) const;

  ComplexColumnVector row_min (void) const;
  ComplexColumnVector row_max (void) const;

  ComplexColumnVector row_min (Array<octave_idx_type>& index) const;
  ComplexColumnVector row_max (Array<octave_idx_type>& index) const;

  ComplexRowVector column_min (void) const;
  ComplexRowVector column_max (void) const;

  ComplexRowVector column_min (Array<octave_idx_type>& index) const;
  ComplexRowVector column_max (Array<octave_idx_type>& index) const;

  // i/o
    /*
  friend std::ostream&
  operator << (std::ostream& os, const ComplexMatrix& a);
  friend std::istream&
  operator >> (std::istream& is, ComplexMatrix& a);
  */
};
