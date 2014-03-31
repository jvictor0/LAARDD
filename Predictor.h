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
    LAARDD::QRPair QR = LAARDD::SequentialQR(X);
    assert(QR.Q);
    TrainFromQR(QR, Y);
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
    for (int i = 0; i < s_inv.size(); ++i)
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
// Probably could be more efficient for simple linear models, 
// but should be efficient for complex models using each column multiple times.
//
class BasisExpansionShardedMatrix : public ShardedMatrix
{
  bool WriteMatrixSegment(uint seg_num, arma::mat & out)
  {
    arma::mat internal_matrix;
    CHECK(m_data_matrix->WriteMatrixSegment(seg_num, internal_matrix), false);
    out.set_size(internal_matrix.n_rows, NumColumns());
    for (int i = 0; i < NumColumns(); ++i)
    {
      m_basis_functions[i](internal_matrix, out.colptr(i));
    }
  }

  typedef std::function<void(const arma::mat & data_segment, double * out)>  ColumnFun;

  static ColumnFun ConstantColumn(double c)
  {
    return [c] (const arma::mat & data_segment, double * out)
    {
      for (int i = 0; i < data_segment.n_rows; ++i)
      {
	out[i] = c;
      }
    };
  }

  static ColumnFun CopyColumn(int col_num)
  {
    return [col_num] (const arma::mat & data_segment, double * out)
    {
      std::memcpy(out, data_segment.colptr(col_num), sizeof(uint) * data_segment.n_rows);
    };
  }

  static BasisExpansionShardedMatrix* WithIntercept(ShardedMatrix * data)
  {
    BasisExpansionShardedMatrix* result =  new BasisExpansionShardedMatrix(data->NumColumns() + 1);
    result->m_basis_functions.push_back(ConstantColumn(1));
    for (int i = 0; i < data->NumColumns(); ++i)
    {
      result->m_basis_functions.push_back(CopyColumn(i));
    }
    result->m_data_matrix = data;
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
      for (int i = 0; i < m_data->NumColumns(); ++i)
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

  BasisExpansionShardedMatrix(uint num_cols) : ShardedMatrix(num_cols) { }

  ShardedMatrix * m_data_matrix;
  std::vector<ColumnFun> m_basis_functions;
};
