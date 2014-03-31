#pragma once

#include "ShardedMatrix.h"
#include <armadillo>
#include "LAARDD_Utils.h"
#include "LAARDD.h"

class Predictor
{
public:
  virtual arma::mat Predict(arma::mat & x0) = 0;
  virtual void Train(ShardedMatrix * X, ShardedMatrix * Y) = 0;
};

// Does linear regression with multiple left hand sides.  
// 
class LinearRegression : public Predictor
{
public:
  LinearRegression() { }
  ~LinearRegression() { }

  arma::mat Predict(arma::mat & x0)
  {
    return x0 * m_beta_hat;
  }
  
  void Train(ShardedMatrix * X, ShardedMatrix * Y)
  {
    arma::mat QtY;
    LAARDD::QRPair QR = LAARDD::SequentialQR(X, Y, &QtY);
    assert(QR.R);
    arma::solve(m_beta_hat, arma::trimatu(*QR.R), QtY);
    std::cout << "beta_hat = " << m_beta_hat << std::endl;
    QR.Free();
  }

  void TrainFromQR(LAARDD::QRPair QR, ShardedMatrix * Y)
  {
    arma::solve(m_beta_hat, arma::trimatu(*QR.R), ShardedMatrix::InnerProduct(QR.Q,Y));
  }
  void TrainFromSVD(LAARDD::SVDTriplet SVD, ShardedMatrix * Y)
  {
    arma::mat UtY = ShardedMatrix::InnerProduct(SVD.U,Y);
    arma::vec s_inv = arma::vec(s_inv.size(), arma::fill::zeros);
    for (uint i = 0; i < s_inv.size(); ++i)
    {
      if (std::abs((*SVD.s)(i)) <= arma::datum::eps)
      {
	break;
      }
      s_inv(i) = 1/((*SVD.s)(i));
    }
    m_beta_hat = (*SVD.V) * arma::diagmat(*SVD.s) * UtY;
  }
  
private:
  arma::mat m_beta_hat;
};

// Simple class to make a datamatrix useable for linear model.
// This could be more efficient if it didn't have to materialize the whole intermidiate matrix,
// but this way is easier to prototype with.
// It could be trivial to code-gen ShardedMatrix instances that represent specific
// basis expansions that only one column at a time, and each column exactly once.  
//
class BasisExpansionShardedMatrix : public ShardedMatrix
{
  BasisExpansionShardedMatrix() : ShardedMatrix(0) { }

  bool WriteMatrixSegment(uint seg_num, arma::mat & out)
  {
    arma::mat internal_matrix;
    CHECK(m_data_matrix->WriteMatrixSegment(seg_num, internal_matrix), false);
    out.set_size(internal_matrix.n_rows, NumColumns());
    for (uint i = 0; i < NumColumns(); ++i)
    {
      m_basis_functions[i](internal_matrix, out.colptr(i));
    }
    return true;
  }

  typedef std::function<void(const arma::mat & data_segment, double * out)>  ColumnFun;

  static ColumnFun ConstantColumn(double c)
  {
    return [c] (const arma::mat & data_segment, double * out)
    {
      for (uint i = 0; i < data_segment.n_rows; ++i)
      {
	out[i] = c;
      }
    };
  }

  static ColumnFun CopyColumn(uint col_num)
  {
    return [col_num] (const arma::mat & data_segment, double * out)
    {
      std::memcpy(out, data_segment.colptr(col_num), sizeof(uint) * data_segment.n_rows);
    };
  }

  static ColumnFun MonomialColumn(std::vector<uint> & col_data)
  {
    return [col_data] (const arma::mat & data_segment, double * out)
    {
      for (uint i = 0; i < data_segment.n_rows; ++i)
      {
	out[i] = std::pow(data_segment[col_data[0]], data_segment[col_data[1]]);
      }
      for (uint j = 2; j < col_data.size(); j += 2)
      {
	for (uint i = 0; i < data_segment.n_rows; ++i)
	{
	  out[i] *= std::pow(data_segment[col_data[j]], data_segment[col_data[j+1]]);
	}
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
	m_data->m_basis_functions[i](x0, basis.colptr(i));
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

  template<class Pred>
  static Transformed<Pred> * TransformModel(BasisExpansionShardedMatrix * X, ShardedMatrix * Y)
  {
    return new Transformed<Pred>(X, Y);
  }

private:

  ShardedMatrix * m_data_matrix;
  std::vector<ColumnFun> m_basis_functions;
};
