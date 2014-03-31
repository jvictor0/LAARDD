#pragma once

#include "ShardedMatrix.h"
#include <armadillo>
#include "LAARDD_Utils.h"
#include "LAARDD.h"
#include "Predictor.h"


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
  arma::vec lambdas(num_lambdas);
  for (uint i = 0; i < num_lambdas; ++i)
  {
    lambdas(i) = lambda_min + i * (lambda_max - lambda_min) / (num_lambdas-1);
  }
  std::vector<RidgeRegression> path;
  RidgeRegression::Path(XY.XTrain ,XY.YTrain, lambdas, path);
  for (uint i = 0; i < num_lambdas; ++i)
  {
    arma::mat YHat = path[i].Predict(XY.XTest);
    double rms = arma::sum(arma::sum(arma::square(XY.YTest-YHat)))/s;
    std::cout << "lambda = " << lambdas(i) << ", rms = " << rms << std::endl;
  }
  XY.Free();
  return true;
}
