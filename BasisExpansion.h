#pragma once

#include "ShardedMatrix.h"
#include <armadillo>
#include "LAARDD_Utils.h"
#include "LAARDD.h"
#include "Predictor.h"

template<class Pred>
class Transformed;


// Simple class to make a datamatrix useable for linear model.
// This could be more efficient if it didn't have to materialize the whole intermidiate matrix,
// but this way is easier to prototype with.
// It could be trivial to code-gen ShardedMatrix instances that represent specific
// basis expansions that only one column at a time, and each column exactly once.  
//
class BasisExpansionShardedMatrix : public ShardedMatrix
{
public:
  BasisExpansionShardedMatrix(ShardedMatrix * mat) : ShardedMatrix(0,mat->SegmentsNumRows()), m_data_matrix(mat) { }

  bool WriteMatrixSegment(uint seg_num, arma::mat & out)
  {
    arma::mat internal_matrix;
    CHECK(m_data_matrix->WriteMatrixSegment(seg_num, internal_matrix), false);
    out.set_size(internal_matrix.n_rows, NumColumns());
    for (uint i = 0; i < NumColumns(); ++i)
    {
      m_basis_functions[i](internal_matrix, out, out.colptr(i));
    }
    return true;
  }

  typedef std::function<void(const arma::mat & data_segment, const arma::mat & current_matrix, double * out)>  ColumnFun;

  static ColumnFun ConstantColumn(double c)
  {
    return [c] (const arma::mat & data_segment,  const arma::mat & current_matrix, double * out)
    {
      for (uint i = 0; i < data_segment.n_rows; ++i)
      {
	out[i] = c;
      }
    };
  }

  static ColumnFun CopyColumn(uint col_num)
  {
    return [col_num] (const arma::mat & data_segment,  const arma::mat & current_matrix, double * out)
    {
      std::memcpy(out, data_segment.colptr(col_num), sizeof(double) * data_segment.n_rows);
    };
  }

  static ColumnFun MonomialColumn(std::vector<uint> & col_data)
  {
    return [col_data] (const arma::mat & data_segment,  const arma::mat & current_matrix, double * out)
    {
      for (uint i = 0; i < data_segment.n_rows; ++i)
      {
	out[i] = std::pow(data_segment(i,col_data[0]), col_data[1]);
      }
      for (uint j = 2; j < col_data.size(); j += 2)
      {
	for (uint i = 0; i < data_segment.n_rows; ++i)
	{
	  out[i] *= std::pow(data_segment(i,col_data[j]), col_data[j+1]);
	}
      }
    };
  }
  
  static ColumnFun NaturalCubicSplineColumn(uint col_num, double xi_k, double xi_K_1, double xi_K)
  {
    return [col_num, xi_k, xi_K_1, xi_K] (const arma::mat & data_segment,  const arma::mat & current_matrix, double * out)
    {
      for (uint i = 0; i < data_segment.n_rows; ++i)
      {
	double d_k = data_segment(i,col_num) - xi_k;
	double d_K = data_segment(i,col_num) - xi_K;
	double d_K_1 = data_segment(i,col_num) - xi_K_1;
	double l_K = (data_segment(i,col_num) < xi_K) * d_K * d_K * d_K;
	double l_K_1 = (data_segment(i,col_num) < xi_K_1) * d_K_1 * d_K_1 * d_K_1;
	double l_k = (data_segment(i,col_num) < xi_k) * d_k * d_k * d_k;
	out[i] = (l_k-l_K)/(xi_K - xi_k) - (l_K_1-l_K)/(xi_K - xi_K_1);
      }
    };
  }

  // Tensor two existing basis functions.
  // Note the indices passed in are supposed to be from the basis expansion being written, not the original data matrix!
  //
  static ColumnFun TensorColumns(uint basis_fun_one_col, uint basis_fun_two_col)
  {
    return [ basis_fun_one_col, basis_fun_two_col] (const arma::mat & data_segment,  const arma::mat & current_matrix, double * out)
    {
      for (uint i = 0; i < data_segment.n_rows; ++i)
      {
	out[i] = current_matrix(i,basis_fun_one_col) * current_matrix(i,basis_fun_two_col);
      }
    };
  }

  void AddColumn(ColumnFun c)
  {
    m_basis_functions.push_back(c);
    ShardedMatrix::AddColumn();
  }

  void AddIntercept()
  {
    AddColumn(ConstantColumn(1));
  }
  
  void AddLinearBases()
  {
    for (uint i = 0; i < m_data_matrix->NumColumns(); ++i)
    {
      AddColumn(CopyColumn(i));
    }
  }
  
  void AddQuadradicBases()
  {
    for (uint i = 0; i < m_data_matrix->NumColumns(); ++i)
    {
      std::vector<uint> base;
      base.push_back(i);
      base.push_back(2);
      AddColumn(MonomialColumn(base));
      base[1] = 1;
      base.push_back(i);
      base.push_back(1);
      for (uint j = i+1; j < m_data_matrix->NumColumns(); ++j)
      {
	base[2] = j;
	AddColumn(MonomialColumn(base));
      }
    }
  }

  // Adds a natural cubic spline without intercept, knots uniformly placed.
  // TODO: knots should be placed according to empirical distribution, not uniformly, but I'm knot sure if it matters.  
  //
  void AddNaturalCubicSpline(uint col_num, double knot_0, double knot_final, uint num_knots)
  {
    AddColumn(CopyColumn(col_num));
    double knot_final_1 = knot_0 + (num_knots - 2) * (knot_final - knot_0) / (num_knots - 1);
    for (uint i = 0; i < num_knots - 2; ++i)
    {
      AddColumn(NaturalCubicSplineColumn(col_num, knot_0 + i * (knot_final - knot_0) / (num_knots - 1), knot_final_1, knot_final));
    }
  }
  
  template<class Pred>
  friend class Transformed;

private:

  ShardedMatrix * m_data_matrix;
  std::vector<ColumnFun> m_basis_functions;
};


template<class Pred>
class Transformed : public Predictor
{
public:
  Transformed(BasisExpansionShardedMatrix * data, ShardedMatrix * Y) : m_data(data)
  {
    reg.Train(m_data, Y);
  }
  
  arma::mat Predict(arma::mat & x0)
  {
    arma::mat basis;
    basis.set_size(x0.n_rows, m_data->NumColumns());
    for (uint i = 0; i < m_data->NumColumns(); ++i)
    {
      m_data->m_basis_functions[i](x0, basis, basis.colptr(i));
    }
    return reg.Predict(basis);
  }
  
  void Train(ShardedMatrix * X, ShardedMatrix * Y) 
  {
    // reg.Train(m_data.Transform(X),Y) TODO: implement BasisExpansionShardedMatrix::Transform
  }
  
private:
  BasisExpansionShardedMatrix * m_data;
  Pred reg;
};

