#pragma once

#include "ShardedMatrix.h"
#include <armadillo>
#include "assert.h"
#include <stdio.h>

#include "LAARDD_Utils.h"

static bool SequentialQRAndSVDTest(ShardedMatrix * matrix)
{
  LAARDD::QRPair QandR = LAARDD::SequentialQR(matrix);
  LAARDD::SVDTriplet svd = LAARDD::SVDFromQR(QandR);
  ShardedMatrixProduct QR(QandR.Q, QandR.R);
  arma::mat UVt = arma::diagmat(*svd.s) * svd.V->t();
  ShardedMatrixProduct USVt(svd.U, &UVt);
  double max_diff = 0;
  arma::mat ATemp, BTemp, CTemp;
  for (int i = 0; i < matrix->NumSegments(); ++i)
  {
    matrix->WriteMatrixSegment(i, ATemp);
    QR.WriteMatrixSegment(i, BTemp);
    USVt.WriteMatrixSegment(i, CTemp);
    max_diff = std::max(max_diff, arma::max(arma::max(arma::abs(ATemp - BTemp))));
    max_diff = std::max(max_diff, arma::max(arma::max(arma::abs(ATemp - CTemp))));
  }
  QandR.Free();
  svd.Free();
  return max_diff < 0.00000000001;
}

static bool SmallRandomTests(int lowestN, int highestN)
{
  int count = 0;
  for (int n = lowestN; n < highestN; ++n)
  {
    for (int p = 1; p < n; ++p)
    {
      for (int b = 1; b <= n/p; ++b)
      {
	arma::mat A = arma::randu<arma::mat>(n,p);
	InMemoryShardedMatrix matrix(A,b);
	if (!SequentialQRAndSVDTest(& matrix))
	{
	  std::cout << "FAILED n = " << n << " p = " << p << " b = " << b << std::endl;
	  return false;
	}
      }
    }
  }
  return true;
}


static bool SmallRandomLowRankTests(int lowestN, int highestN)
{
  for (int n = lowestN; n < highestN; ++n)
  {
    for (int p = 1; p < n; ++p)
    {
      for (int b = 1; b <= n/p; ++b)
      {
	for (int r = 0; r < p; ++r)
	{
	  arma::mat A = arma::randu<arma::mat>(n,p);
	  arma::mat U,V;
	  arma::vec s;
	  arma::svd_econ(U,s,V,A);
	  for (int i = 0; i < r; ++i)
	  {
	    s(s.size() - 1 - i) = 0;
	  }
	  A = U * arma::diagmat(s) * V.t();
	  InMemoryShardedMatrix matrix(A,b);
	  if (!SequentialQRAndSVDTest(& matrix))
	  {
	    std::cout << "FAILED n = " << n << " p = " << p << " b = " << b << std::endl;
	    return false;
	  }
	}
      }
    }
  }
  return true;
}

void LAARDDTest()
{
  if(SmallRandomTests(5,30))
  {
    std::cout << "SmallRandomTests Passed" << std::endl;
  }
  if(SmallRandomLowRankTests(5,30))
  {
    std::cout << "SmallRandomLowRankTests Passed" << std::endl;
  }
}
