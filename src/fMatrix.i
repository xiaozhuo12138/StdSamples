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

class FloatMatrix : public FloatNDArray
{
public:

  typedef FloatColumnVector column_vector_type;
  typedef FloatRowVector row_vector_type;

  typedef FloatColumnVector real_column_vector_type;
  typedef FloatRowVector real_row_vector_type;

  typedef FloatMatrix real_matrix_type;
  typedef FloatComplexMatrix complex_matrix_type;

  typedef FloatDiagMatrix real_diag_matrix_type;
  typedef FloatComplexDiagMatrix complex_diag_matrix_type;

  typedef float real_elt_type;
  typedef FloatComplex complex_elt_type;

  typedef void (*solve_singularity_handler) (float rcon);

  FloatMatrix (void);
  FloatMatrix (const FloatMatrix& a);
  FloatMatrix& operator = (const FloatMatrix& a);
  ~FloatMatrix (void);

  FloatMatrix (octave_idx_type r, octave_idx_type c);
  FloatMatrix (octave_idx_type r, octave_idx_type c, float val);
  FloatMatrix (const dim_vector& dv);
  FloatMatrix (const dim_vector& dv, float val);

  template <typename U>
  FloatMatrix (const MArray<U>& a);

  template <typename U>
  FloatMatrix (const Array<U>& a);

  explicit FloatMatrix (const FloatRowVector& rv);

  explicit FloatMatrix (const FloatColumnVector& cv);

  explicit FloatMatrix (const FloatDiagMatrix& a);

  explicit FloatMatrix (const MDiagArray2<float>& a);

  explicit FloatMatrix (const DiagArray2<float>& a);

  explicit FloatMatrix (const PermMatrix& a);

  explicit FloatMatrix (const boolMatrix& a);

  explicit FloatMatrix (const charMatrix& a);

  bool operator == (const FloatMatrix& a) const;
  bool operator != (const FloatMatrix& a) const;

  bool issymmetric (void) const;

  // destructive insert/delete/reorder operations

  FloatMatrix&
  insert (const FloatMatrix& a, octave_idx_type r, octave_idx_type c);
  FloatMatrix&
  insert (const FloatRowVector& a, octave_idx_type r, octave_idx_type c);
  FloatMatrix&
  insert (const FloatColumnVector& a, octave_idx_type r, octave_idx_type c);
  FloatMatrix&
  insert (const FloatDiagMatrix& a, octave_idx_type r, octave_idx_type c);

  FloatMatrix& fill (float val);
  FloatMatrix&
  fill (float val, octave_idx_type r1, octave_idx_type c1,
        octave_idx_type r2, octave_idx_type c2);

  FloatMatrix append (const FloatMatrix& a) const;
  FloatMatrix append (const FloatRowVector& a) const;
  FloatMatrix append (const FloatColumnVector& a) const;
  FloatMatrix append (const FloatDiagMatrix& a) const;

  FloatMatrix stack (const FloatMatrix& a) const;
  FloatMatrix stack (const FloatRowVector& a) const;
  FloatMatrix stack (const FloatColumnVector& a) const;
  FloatMatrix stack (const FloatDiagMatrix& a) const;

  //friend FloatMatrix real (const FloatComplexMatrix& a);
  //friend FloatMatrix imag (const FloatComplexMatrix& a);

  //friend class FloatComplexMatrix;

  FloatMatrix hermitian (void) const;
  FloatMatrix transpose (void) const;

  // resize is the destructive equivalent for this one

  FloatMatrix
  extract (octave_idx_type r1, octave_idx_type c1,
           octave_idx_type r2, octave_idx_type c2) const;

  FloatMatrix
  extract_n (octave_idx_type r1, octave_idx_type c1,
             octave_idx_type nr, octave_idx_type nc) const;

  // extract row or column i.

  FloatRowVector row (octave_idx_type i) const;

  FloatColumnVector column (octave_idx_type i) const;

  void resize (octave_idx_type nr, octave_idx_type nc, float rfv = 0);


  FloatMatrix inverse (void) const;
  FloatMatrix inverse (octave_idx_type& info) const;
  FloatMatrix
  inverse (octave_idx_type& info, float& rcon, bool force = false,
           bool calc_cond = true) const;

  FloatMatrix inverse (MatrixType& mattype) const;
  FloatMatrix
  inverse (MatrixType& mattype, octave_idx_type& info) const;
  FloatMatrix
  inverse (MatrixType& mattype, octave_idx_type& info, float& rcon,
           bool force = false, bool calc_cond = true) const;

  FloatMatrix pseudo_inverse (float tol = 0.0) const;

  FloatComplexMatrix fourier (void) const;
  FloatComplexMatrix ifourier (void) const;

  FloatComplexMatrix fourier2d (void) const;
  FloatComplexMatrix ifourier2d (void) const;

  FloatDET determinant (void) const;
  FloatDET determinant (octave_idx_type& info) const;
  FloatDET
  determinant (octave_idx_type& info, float& rcon,
               bool calc_cond = true) const;
  FloatDET
  determinant (MatrixType& mattype, octave_idx_type& info, float& rcon,
               bool calc_cond = true) const;

  float rcond (void) const;
  float rcond (MatrixType& mattype) const;


public:
  // Generic interface to solver with no probing of type
  FloatMatrix
  solve (MatrixType& mattype, const FloatMatrix& b) const;
  FloatMatrix
  solve (MatrixType& mattype, const FloatMatrix& b,
         octave_idx_type& info) const;
  FloatMatrix
  solve (MatrixType& mattype, const FloatMatrix& b, octave_idx_type& info,
         float& rcon) const;
  FloatMatrix
  solve (MatrixType& mattype, const FloatMatrix& b, octave_idx_type& info,
         float& rcon, solve_singularity_handler sing_handler,
         bool singular_fallback = true,
         blas_trans_type transt = blas_no_trans) const;

  FloatComplexMatrix
  solve (MatrixType& mattype, const FloatComplexMatrix& b) const;
  FloatComplexMatrix
  solve (MatrixType& mattype, const FloatComplexMatrix& b,
         octave_idx_type& info) const;
  FloatComplexMatrix
  solve (MatrixType& mattype, const FloatComplexMatrix& b,
         octave_idx_type& info, float& rcon) const;
  FloatComplexMatrix
  solve (MatrixType& mattype, const FloatComplexMatrix& b,
         octave_idx_type& info, float& rcon,
         solve_singularity_handler sing_handler, bool singular_fallback = true,
         blas_trans_type transt = blas_no_trans) const;

  FloatColumnVector
  solve (MatrixType& mattype, const FloatColumnVector& b) const;
  FloatColumnVector
  solve (MatrixType& mattype, const FloatColumnVector& b,
         octave_idx_type& info) const;
  FloatColumnVector
  solve (MatrixType& mattype, const FloatColumnVector& b,
         octave_idx_type& info, float& rcon) const;
  FloatColumnVector
  solve (MatrixType& mattype, const FloatColumnVector& b,
         octave_idx_type& info, float& rcon,
         solve_singularity_handler sing_handler,
         blas_trans_type transt = blas_no_trans) const;

  FloatComplexColumnVector
  solve (MatrixType& mattype, const FloatComplexColumnVector& b) const;
  FloatComplexColumnVector
  solve (MatrixType& mattype, const FloatComplexColumnVector& b,
         octave_idx_type& info) const;
  FloatComplexColumnVector
  solve (MatrixType& mattype, const FloatComplexColumnVector& b,
         octave_idx_type& info, float& rcon) const;
  FloatComplexColumnVector
  solve (MatrixType& mattype, const FloatComplexColumnVector& b,
         octave_idx_type& info, float& rcon,
         solve_singularity_handler sing_handler,
         blas_trans_type transt = blas_no_trans) const;

  // Generic interface to solver with probing of type
  FloatMatrix solve (const FloatMatrix& b) const;
  FloatMatrix
  solve (const FloatMatrix& b, octave_idx_type& info) const;
  FloatMatrix
  solve (const FloatMatrix& b, octave_idx_type& info, float& rcon) const;
  FloatMatrix
  solve (const FloatMatrix& b, octave_idx_type& info, float& rcon,
         solve_singularity_handler sing_handler,
         blas_trans_type transt = blas_no_trans) const;

  FloatComplexMatrix solve (const FloatComplexMatrix& b) const;
  FloatComplexMatrix
  solve (const FloatComplexMatrix& b, octave_idx_type& info) const;
  FloatComplexMatrix
  solve (const FloatComplexMatrix& b, octave_idx_type& info,
         float& rcon) const;
  FloatComplexMatrix
  solve (const FloatComplexMatrix& b, octave_idx_type& info,
         float& rcon, solve_singularity_handler sing_handler,
         blas_trans_type transt = blas_no_trans) const;

  FloatColumnVector solve (const FloatColumnVector& b) const;
  FloatColumnVector
  solve (const FloatColumnVector& b, octave_idx_type& info) const;
  FloatColumnVector
  solve (const FloatColumnVector& b, octave_idx_type& info, float& rcon) const;
  FloatColumnVector
  solve (const FloatColumnVector& b, octave_idx_type& info, float& rcon,
         solve_singularity_handler sing_handler,
         blas_trans_type transt = blas_no_trans) const;

  FloatComplexColumnVector
  solve (const FloatComplexColumnVector& b) const;
  FloatComplexColumnVector
  solve (const FloatComplexColumnVector& b, octave_idx_type& info) const;
  FloatComplexColumnVector
  solve (const FloatComplexColumnVector& b, octave_idx_type& info,
         float& rcon) const;
  FloatComplexColumnVector
  solve (const FloatComplexColumnVector& b, octave_idx_type& info, float& rcon,
         solve_singularity_handler sing_handler,
         blas_trans_type transt = blas_no_trans) const;

  // Singular solvers
  FloatMatrix lssolve (const FloatMatrix& b) const;
  FloatMatrix
  lssolve (const FloatMatrix& b, octave_idx_type& info) const;
  FloatMatrix
  lssolve (const FloatMatrix& b, octave_idx_type& info,
           octave_idx_type& rank) const;
  FloatMatrix
  lssolve (const FloatMatrix& b, octave_idx_type& info,
           octave_idx_type& rank, float& rcon) const;

  FloatComplexMatrix lssolve (const FloatComplexMatrix& b) const;
  FloatComplexMatrix
  lssolve (const FloatComplexMatrix& b, octave_idx_type& info) const;
  FloatComplexMatrix
  lssolve (const FloatComplexMatrix& b, octave_idx_type& info,
           octave_idx_type& rank) const;
  FloatComplexMatrix
  lssolve (const FloatComplexMatrix& b, octave_idx_type& info,
           octave_idx_type& rank, float& rcon) const;

  FloatColumnVector lssolve (const FloatColumnVector& b) const;
  FloatColumnVector
  lssolve (const FloatColumnVector& b, octave_idx_type& info) const;
  FloatColumnVector
  lssolve (const FloatColumnVector& b, octave_idx_type& info,
           octave_idx_type& rank) const;
  FloatColumnVector
  lssolve (const FloatColumnVector& b, octave_idx_type& info,
           octave_idx_type& rank, float& rcon) const;

  FloatComplexColumnVector
  lssolve (const FloatComplexColumnVector& b) const;
  FloatComplexColumnVector
  lssolve (const FloatComplexColumnVector& b, octave_idx_type& info) const;
  FloatComplexColumnVector
  lssolve (const FloatComplexColumnVector& b, octave_idx_type& info,
           octave_idx_type& rank) const;
  FloatComplexColumnVector
  lssolve (const FloatComplexColumnVector& b, octave_idx_type& info,
           octave_idx_type& rank, float& rcon) const;

  FloatMatrix& operator += (const FloatDiagMatrix& a);
  FloatMatrix& operator -= (const FloatDiagMatrix& a);

  FloatMatrix cumprod (int dim = -1) const;
  FloatMatrix cumsum (int dim = -1) const;
  FloatMatrix prod (int dim = -1) const;
  FloatMatrix sum (int dim = -1) const;
  FloatMatrix sumsq (int dim = -1) const;
  FloatMatrix abs (void) const;

  FloatMatrix diag (octave_idx_type k = 0) const;

  FloatDiagMatrix diag (octave_idx_type m, octave_idx_type n) const;

  FloatColumnVector row_min (void) const;
  FloatColumnVector row_max (void) const;

  FloatColumnVector row_min (Array<octave_idx_type>& index) const;
  FloatColumnVector row_max (Array<octave_idx_type>& index) const;

  FloatRowVector column_min (void) const;
  FloatRowVector column_max (void) const;

  FloatRowVector column_min (Array<octave_idx_type>& index) const;
  FloatRowVector column_max (Array<octave_idx_type>& index) const;

  // i/o
  /*
  friend std::ostream&
  operator << (std::ostream& os, const FloatMatrix& a);
  friend std::istream&
  operator >> (std::istream& is, FloatMatrix& a);
  */
};
