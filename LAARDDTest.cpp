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
