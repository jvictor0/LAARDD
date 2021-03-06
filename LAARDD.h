#pragma once

#include "ShardedMatrix.h"
#include <armadillo>
#include "assert.h"
#include <stdio.h>

#include "LAARDD_Utils.h"

class LAARDD
{

public: 

  typedef int StoredMatrixID;

  static StoredMatrixID Store(arma::Mat<double> & matrix)
  {
    StoredMatrixID id = GetNextMatrixID();
    CHECK(matrix.eval().save("tmp/__LAARDD_TEMPORARY_" + std::to_string(id) + ".mat"), -1); // why do I eval?
    return id;
  }

  static bool Retrieve(StoredMatrixID id, arma::Mat<double> & matrix)
  {
    return matrix.load("tmp/__LAARDD_TEMPORARY_" + std::to_string(id) + ".mat");
  }

  static bool Delete(StoredMatrixID id)
  {
    return std::remove(("tmp/__LAARDD_TEMPORARY_" + std::to_string(id) + ".mat").c_str()) == 0;
  }


  struct QRPair
  {
    QRPair()
    {
      Q = NULL;
      R = NULL;
    }
    ShardedMatrix * Q;
    arma::Mat<double> * R;
    void Free()
    {
      delete Q;
      delete R;
    }
  };

  struct SVDTriplet
  {
    ShardedMatrix * U;
    arma::vec * s;
    arma::mat * V;
    void Free()
    {
      delete U;
      delete s;
      delete V;
    }
  };
  
  // Computes a QR factorization.
  // If the second argument not provided, both Q and R are computed explicitly and linear disk space is used.
  // If Y is provided, the Q part of the result is left NULL and Q^TY is computed into the third argument, 
  // and NO additional disk space is used.
  // 
  static QRPair SequentialQR(ShardedMatrix * matrix, ShardedMatrix * Y = NULL, arma::mat * QtY = NULL)
  {
    StoredMatrixID * Q_ids = Y ? NULL : new StoredMatrixID[matrix->NumSegments()];
    arma::mat Q_seg, A_seg, Y_seg;
    arma::mat * R_seg = new arma::mat(0, matrix->NumColumns());
    
    QRPair result;
    
    if (Y)
    {
      assert(QtY);
      *QtY = arma::zeros(matrix->NumColumns(), Y->NumColumns());
    }
    
    // factor [R_{i-1}; A_i], store the Q_i factors for later
    //
    for (uint i = 0; i < matrix->NumSegments(); ++i)
    {
      std::cout << "Calculating Q,R_" << i << std::endl;
      matrix->WriteMatrixSegment(i, A_seg);
      if (!qr_econ(Q_seg, *R_seg, arma::join_vert(*R_seg, A_seg)))
      {
	delete Q_ids;
	delete R_seg;
	return result;
      }
      assert(Q_seg.n_cols == matrix->NumColumns());
      
      if (Y)
      {
	if (i > 0)
	{
	  *QtY = Q_seg.rows(0,Q_seg.n_cols - 1).t() * (*QtY);
	}
	Y->WriteMatrixSegment(i, Y_seg);
	*QtY = (*QtY) + Q_seg.rows(Q_seg.n_rows - Y_seg.n_rows,Q_seg.n_rows - 1).t() * Y_seg;
      }
      else if (i != matrix->NumSegments() - 1) // no need to store last Q_seg, will use immidiately
      {
	Q_ids[i] = Store(Q_seg);
	assert(Q_ids[i] != -1);
      }
    }
    
    // R_{s-1} is now the final R result of the QR factorization.
    //
    result.R = R_seg;
    if (Y)
    {
      return result;
    }
    DiskShardedMatrix * Q = new DiskShardedMatrix(*matrix, false);

    // Reconstruct Q from the Q_i
    //
    arma::mat next_Q_seg;
    for (uint i = matrix->NumSegments() - 1; i >= 1; --i)
    {
      std::cout << "Calculating Q segment " << i << std::endl;
      Q->SetSegment(i, Q_seg.rows(matrix->NumColumns(), Q_seg.n_rows - 1));
      bool ret_worked = Retrieve(Q_ids[i-1],next_Q_seg);
      assert(ret_worked);
      assert(Delete(Q_ids[i-1]));
      Q_seg = next_Q_seg * Q_seg.rows(0, matrix->NumColumns() - 1);
    }
    
    Q->SetSegment(0, Q_seg); // last segment uses entire Q_seg matrix
    
    delete [] Q_ids;
    result.Q = Q;
    return result;
  }
  
  static SVDTriplet SVDFromQR(QRPair QR)
  {
    SVDTriplet result;
    assert(QR.R->n_cols == QR.R->n_rows);
    result.s = new arma::vec(QR.R->n_cols);
    result.V = new arma::mat(QR.R->n_cols, QR.R->n_cols);
    arma::mat * U_tilde = new arma::mat();
    svd(*U_tilde, *result.s, *result.V, *QR.R);
    ShardedMatrixProduct * U = new ShardedMatrixProduct(QR.Q,U_tilde);
    U->SetOwnsB(true);
    result.U = U;
    return result;
  }
  
private:

  static StoredMatrixID GetNextMatrixID() { return s_counter++; }
  static uint s_counter;

};


uint LAARDD::s_counter = 0;

