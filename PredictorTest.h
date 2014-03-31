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
  for (int i = 0; i < b; ++i)
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
