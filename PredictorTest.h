#pragma once

#include "ShardedMatrix.h"
#include <armadillo>
#include "LAARDD_Utils.h"
#include "LAARDD.h"
#include "Predictor.h"
#include "BasisExpansion.h"

struct XYPair
{
  ShardedMatrix * XTrain;
  ShardedMatrix * YTrain;
  arma::mat XTest;
  arma::mat YTest;
  void Free()
  {
    delete XTrain;
    delete YTrain;
  }
};

XYPair SimulateLM(uint s, uint p, uint b, double epsilon)
{
  DiskShardedMatrix *X = new DiskShardedMatrix(p);
  DiskShardedMatrix *Y = new DiskShardedMatrix(1);
  arma::mat beta = arma::randn<arma::mat>(p,1);
  arma::mat XSeg;
  for (uint i = 0; i < b; ++i)
  {
    std::cout << "Simulating segment " << i << std::endl;
    XSeg = arma::randn<arma::mat>(s,p);
    X->AddSegment(XSeg);
    Y->AddSegment(XSeg * beta + epsilon * arma::randn<arma::mat>(s,1));
  }
  XYPair result;
  result.XTrain = X;
  result.YTrain = Y;
  result.XTest = arma::randn<arma::mat>(s,p);
  result.YTest = result.XTest * beta + epsilon * arma::randn<arma::mat>(s,1);
  std::cout << "beta = " << beta << std::endl;
  return result;
}

#define PLUS_CUBED(X,knot) arma::pow(arma::max(arma::join_horiz(X-knot * arma::ones(X.n_rows,1), arma::zeros(X.n_rows,1)),1),3)

XYPair SimulateNaturalCubicSpline(uint s, uint b, double epsilon, uint num_knots)
{
  DiskShardedMatrix *X = new DiskShardedMatrix(1);
  DiskShardedMatrix *Y = new DiskShardedMatrix(1);
  arma::mat XSeg,YSeg;
  XYPair XY;
  XY.XTrain = X;
  XY.YTrain = Y;
  arma::mat beta = arma::randn<arma::mat>(num_knots,1);
  arma::vec knots = 2 * arma::randu<arma::mat>(num_knots);
  for (uint i = 1; i < num_knots; ++i)
  {
    knots(i) = knots(i) + knots(i-1);
  }
  for (uint i = 0; i < b; ++i)
  {
    std::cout << "Simulating segment " << i << std::endl;
    XSeg = num_knots * arma::ones(s,1) + arma::randn<arma::mat>(s,1) * num_knots;
    YSeg = beta(0) * arma::ones(s,1) + beta(1) * XSeg;
    for (uint j = 2; j < num_knots; ++j)
    {
      YSeg = YSeg + beta(j) *
	((PLUS_CUBED(XSeg,knots(j-2)) - PLUS_CUBED(XSeg,knots(num_knots - 1)))/(knots(j-2) - knots(num_knots - 1)) 
	 - (PLUS_CUBED(XSeg,knots(num_knots - 2)) - PLUS_CUBED(XSeg,knots(num_knots - 1)))/(knots(num_knots - 2) - knots(num_knots - 1)));
    }
    YSeg = YSeg + epsilon * arma::randn<arma::mat>(s,1);
    if (i < b - 1)
    {
      X->AddSegment(XSeg);
      Y->AddSegment(YSeg);
    }
    else
    {
      XY.XTest = XSeg;
      XY.YTest = YSeg;
    }
  }
  return XY;
}

bool TestLinearRegression(uint s, uint p, uint b, double epsilon)
{
  XYPair XY = SimulateLM(s, p, b, epsilon);
  LinearRegression lm;
  lm.Train(XY.XTrain,XY.YTrain);
  arma::mat YHat = lm.Predict(XY.XTest);
  double rms = arma::sum(arma::sum(arma::square(XY.YTest-YHat)))/s;
  std::cout << "rms = " << rms << std::endl;
  XY.Free();
  return true;
}


bool TestRidgeRegression(uint s, uint p, uint b, double epsilon, double lambda_min, double lambda_max, uint num_lambdas)
{
  XYPair XY = SimulateLM(s, p, b, epsilon);
  std::vector<RidgeRegression> path;
  RidgeRegression::Path(XY.XTrain ,XY.YTrain, lambda_min, lambda_max, num_lambdas, path);
  for (uint i = 0; i < num_lambdas; ++i)
  {
    arma::mat YHat = path[i].Predict(XY.XTest);
    double rms = arma::sum(arma::sum(arma::square(XY.YTest-YHat)))/s;
    std::cout << "lambda = " << lambda_min + i * (lambda_max - lambda_min) / (num_lambdas - 1) << ", rms = " << rms << std::endl;
  }
  XY.Free();
  return true;
}

bool TestNaturalSplineRegression(uint s, uint b, double epsilon, uint num_knots_gen, uint num_knots_train)
{
  XYPair XY = SimulateNaturalCubicSpline(s, b, epsilon, num_knots_gen);
  BasisExpansionShardedMatrix spline_model = BasisExpansionShardedMatrix(XY.XTrain);
  spline_model.AddIntercept();
  spline_model.AddNaturalCubicSpline(0, 0, 2.0* num_knots_gen, num_knots_train);
  Transformed<LinearRegression> lm = Transformed<LinearRegression>(&spline_model, XY.YTrain);
  arma::mat YHat = lm.Predict(XY.XTest);
  double rms = arma::sum(arma::sum(arma::square(XY.YTest-YHat)))/s;
  std::cout << "rms = " << rms << std::endl;
  XY.Free();
  //  delete lm;
  return true;  
}
