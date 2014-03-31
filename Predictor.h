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
// Might have numerical stability issues, not quite sure how stable Armadillo's ut-solve is.
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

private:
  arma::mat m_beta_hat;
};

// Does RIDGED linear regression with multiple left hand sides.  
// Should be super numerically stable.  
// 
class RidgeRegression : public Predictor
{
public:
  RidgeRegression(double lambda) : m_lambda(lambda) { }
  ~RidgeRegression() { }

  arma::mat Predict(arma::mat & x0)
  {
    return x0 * m_beta_hat;
  }
  
  void Train(ShardedMatrix * X, ShardedMatrix * Y)
  {
    arma::mat QtY;
    LAARDD::QRPair QR = LAARDD::SequentialQR(X, Y, &QtY);
    assert(QR.R);
    arma::mat U,V;
    arma::vec s;
    arma::svd(U,s,V,*QR.R);
    QtY = U.t() * QtY;
    Train_(V, s, QtY);
    QR.Free();
  }

  static void Path(ShardedMatrix * X, ShardedMatrix * Y, 
		   double lambda_min, double lambda_max, uint num_lambdas, 
		   std::vector<RidgeRegression> & result)
  {
    arma::mat QtY;
    LAARDD::QRPair QR = LAARDD::SequentialQR(X, Y, &QtY);
    assert(QR.R);
    arma::mat U,V;
    arma::vec s;
    arma::svd(U,s,V,*QR.R);
    result.reserve(num_lambdas);
    QtY = U.t() * QtY;
    for (uint i = 0; i < num_lambdas; ++i)
    {
      result.push_back(RidgeRegression(lambda_min + i * (lambda_max - lambda_min) / (num_lambdas - 1)));
      result.back().Train_(V,s,QtY);
    }
    QR.Free();
  }

private:

  // \hat\beta = V(S+\lambda I)^{-1}U^TQ^Ty, where R=USV^T
  //
  void Train_(arma::mat & V, arma::vec & s, arma::mat & UtY)
  {
    arma::vec s_plus_lambdaI_inv(s.size());
    for (uint i = 0; i < s.size(); ++i)
    {
      s_plus_lambdaI_inv(i) = 1/(s(i) + m_lambda);
    }
    m_beta_hat = V * arma::diagmat(s_plus_lambdaI_inv) * UtY;
    std::cout << "||beta_hat|| = " << arma::norm(m_beta_hat,2) << std::endl;
  }

  arma::mat m_beta_hat;
  double m_lambda;
};

