<<<<<<< HEAD
static bool SequentialQRTest(ShardedMatrix * matrix)
{
  QRPair QandR = SequentialQR(matrix);
  ShardedMatrixProduct QR(QandR.Q, *QandR.R);
  //    std::cout << "my R = " << *QandR.R << std::endl;
  double max_diff = 0;
  arma::mat ATemp, BTemp;
  for (int i = 0; i < matrix->NumSegments(); ++i)
  {
    matrix->WriteMatrixSegment(i, ATemp);
    QR.WriteMatrixSegment(i, BTemp);
    //     std::cout << "ATemp = " << ATemp << std::endl;
    // std::cout << "BTemp = " << BTemp << std::endl;
    max_diff = std::max(max_diff, arma::max(arma::max(arma::abs(ATemp - BTemp))));
  }
  return max_diff < 0.00000000001;
}

static bool SmallRandomTests(int lowestN, int highestN)
{
  for (int n = lowestN; n < highestN; ++n)
  {
    for (int p = 1; p < n; ++p)
    {
      for (int b = 1; b <= n/p; ++b)
      {
	arma::mat A = arma::randu<arma::mat>(n,p);
	InMemoryShardedMatrix matrix(A,b);
	if (!LAARDD::SequentialQRTest(& matrix))
	{
	  std::cout << "FAILED n = " << n << " p = " << p << " b = " << b << std::endl;
	  return false;
	}
      }
    }
  }
  return true;
}
=======
  static void SequentialQRTest(ShardedMatrix * matrix)
  {
    QRPair QandR = SequentialQR(matrix);
    ShardedMatrixProduct QR(QandR.Q, *QandR.R);
    std::cout << "my R = " << *QandR.R << std::endl;
    double max_diff = 0;
    arma::mat ATemp, BTemp;
    for (int i = 0; i < matrix->NumSegments(); ++i)
    {
      matrix->WriteMatrixSegment(i, ATemp);
      QR.WriteMatrixSegment(i, BTemp);
      std::cout << "ATemp = " << ATemp << std::endl;
      std::cout << "BTemp = " << BTemp << std::endl;
      max_diff = std::max(max_diff, arma::max(arma::max(arma::abs(ATemp - BTemp))));
    }
    std::cout << "biggest difference between QR and A = " << max_diff << std::endl;
  }
>>>>>>> parent of 5806df2... Little fixes with ShardedMatrix interface, and better test for random matrices.
