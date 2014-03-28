#include "ShardedMatrix.h"
#include <armadillo>
#include "assert.h"
#include <stdio.h>

#include "LAARDD_Utils.h"

class LAARDD
{

public: 

  typedef uint StoredMatrixID;

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
    ShardedMatrix * Q;
    arma::Mat<double> * R;
  };
  
  static QRPair SequentialQR(ShardedMatrix * matrix)
  {
    StoredMatrixID * Q_ids = new StoredMatrixID[matrix->NumSegments()];
    arma::mat Q_seg, A_seg;
    arma::mat * R_seg = new arma::mat(0, matrix->NumColumns());
    
    QRPair result;
    result.Q = NULL;
    result.R = NULL;
    
    // factor [R_{i-1}; A_i], store the Q_i factors for later
    //
    for (int i = 0; i < matrix->NumSegments(); ++i)
    {
      matrix->WriteMatrixSegment(i, A_seg);
      CHECK(qr_econ(Q_seg, *R_seg, arma::join_vert(*R_seg, A_seg)), result);
      assert(Q_seg.n_cols == matrix->NumColumns());
      if (i != matrix->NumSegments() - 1) // no need to store last Q_seg, will use immidiately
      {
	Q_ids[i] = Store(Q_seg);
	assert(Q_ids[i] != -1);
      }
    }
    
    // R_{s-1} is now the final R result of the QR factorization.
    //
    result.R = R_seg;
    DiskShardedMatrix * Q = new DiskShardedMatrix(*matrix, false);

    // Reconstruct Q from the Q_i
    //
    arma::mat next_Q_seg;
    for (int i = matrix->NumSegments() - 1; i >= 1; --i)
    {
      Q->SetSegment(i, Q_seg.rows(matrix->NumColumns(), Q_seg.n_rows - 1));
      bool ret_worked = Retrieve(Q_ids[i-1],next_Q_seg);
      assert(ret_worked);
      assert(Delete(Q_ids[i-1]));
      Q_seg = next_Q_seg * Q_seg.rows(0, matrix->NumColumns() - 1);
    }
    
    Q->SetSegment(0, Q_seg); // last segment uses entire Q_seg matrix
    
    result.Q = Q;
    return result;
  }
  
#include "LAARDDTest.cpp"
  
private:

  static StoredMatrixID GetNextMatrixID() { return s_counter++; }

  static uint s_counter;

};


uint LAARDD::s_counter = 0;

int main()
{
  if(LAARDD::SmallRandomTests(5,30))
  {
    std::cout << "SmallRandomTests Passed" << std::endl;
  }
  if(LAARDD::SmallRandomLowRankTests(5,30))
  {
    std::cout << "SmallRandomLowRankTests Passed" << std::endl;
  }
}
