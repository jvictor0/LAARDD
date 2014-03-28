#pragma once

#include <armadillo>
#include <memory.h>
#include <vector> 
#include <string>
#include "assert.h"
#include "LAARDD_Utils.h"


// A Matrix class represents a tall matrix which does not fit in memory.
// Calling WriteMatrixSegment writes one segment to memory.
//
class ShardedMatrix 
{

public:
  ShardedMatrix(uint num_cols) 
    : m_numColumns(num_cols){ }

  ShardedMatrix(uint num_segs, uint num_cols, uint * segSizes) 
    : m_numColumns (num_cols)
  {
    for (int i = 0; i < num_segs; ++i)
    {
      m_segmentRow.push_back(segSizes[i]);
    }
  }
  
  ShardedMatrix(uint num_segs, uint num_cols, uint num_rows) 
    : m_numColumns (num_cols)
  {
    for (int i = 0; i < num_segs - 1; ++i)
    {
      m_segmentRow.push_back(num_rows/num_segs);
    }
    uint last_size = num_rows % num_segs;
    m_segmentRow.push_back(last_size == 0 ? num_rows/num_segs : last_size);
  }


  virtual ~ShardedMatrix() { } 
  
  virtual bool WriteMatrixSegment(uint seg_num, arma::Mat<double> & out) = 0;

  uint NumColumns()
  {
    return m_numColumns;
  }

  uint NumSegments()
  {
    return m_segmentRow.size();
  }

  uint SegmentNumRows(uint seg_num)
  {
    return m_segmentRow[seg_num];
  }

  uint* SegmentsNumRows()
  {
    return &m_segmentRow[0];
  }
  
protected:
  
  void AddSegment(uint seg_size)
  {
    m_segmentRow.push_back(seg_size);
  }
  
  uint m_numColumns;
  std::vector<uint> m_segmentRow;

};

// This class is just for testing convinience, doesn't make sense for actual use.  
//
class InMemoryShardedMatrix : public ShardedMatrix 
{
public:
  // Actual number of segments may be less than num_segments to avoid creating very small segments
  //
  InMemoryShardedMatrix(arma::Mat<double> & matrix, uint num_segments) 
    : ShardedMatrix(num_segments, matrix.n_cols, matrix.n_rows), m_matrix(matrix)
  { 
    if (SegmentNumRows(NumSegments() - 1) < NumColumns() && NumSegments() > 1)
    {
      m_segmentRow[NumSegments() - 2] += m_segmentRow[NumSegments() - 1];
      m_segmentRow.pop_back();
    }
  }


  bool WriteMatrixSegment(uint seg_num, arma::Mat<double> & out)
  {
    out = m_matrix.rows(seg_num * SegmentNumRows(0), seg_num* SegmentNumRows(0) + SegmentNumRows(seg_num) - 1);
  }

  
  arma::Mat<double> m_matrix;
};


// Matrix class where each segment lies on disk
//
class DiskShardedMatrix : public ShardedMatrix
{
public: 
  DiskShardedMatrix(uint num_columns) 
    : ShardedMatrix(num_columns), m_filename("disk_shard")
  { 
    m_unique_id = s_unique_id++;
  }

  // Create a disk sharded matrix with same dimensions as A.
  // Will copy segments A to disk if copy=true, otherwise it is an error to call WriteMatrixSegment without first calling SetSegment.  
  // TODO: copy = true case.  
  //
  DiskShardedMatrix(ShardedMatrix & A, bool copy) 
    : ShardedMatrix(A.NumSegments(), A.NumColumns(), A.SegmentsNumRows()), m_filename("disk_shard")
  { 
    assert(!copy);
    m_unique_id = s_unique_id++;
  }


  DiskShardedMatrix(uint num_columns, const char * filename) 
    : ShardedMatrix(num_columns), m_filename(filename)
  { 
    m_unique_id = s_unique_id++;
  }

  bool WriteMatrixSegment(uint seg_num, arma::Mat<double> & out)
  {
    return out.load("tmp/" + m_filename + "_" +  std::to_string(m_unique_id)  + "_" + std::to_string(seg_num) + ".mat");
  }
  
  bool AddSegment(arma::Mat<double> seg)
  {
    ShardedMatrix::AddSegment(seg.n_rows);
    SetSegment(NumSegments() - 1, seg);
  }
  
  template<class MatType>
  void SetSegment(uint seg_num, MatType seg)
  {
    assert(seg.n_cols == NumColumns());
    assert(seg.n_rows == SegmentNumRows(seg_num)); // or should I set the number or rows?  What's mutable?  Not sure...
    bool worked = seg.eval().save("tmp/" + m_filename + "_" +  std::to_string(m_unique_id) + "_" + std::to_string(seg_num) + ".mat");
    assert(worked);
  }
    
private:
  

  static uint s_unique_id ;
  
  uint m_unique_id;
  std::string m_filename;
};

uint DiskShardedMatrix::s_unique_id =0;

// Matrix class representing the product AB of a ShardedMatrix A with an in memory matrix B.  
//
class ShardedMatrixProduct : public ShardedMatrix 
{
public: 
  ShardedMatrixProduct(ShardedMatrix * A, arma::Mat<double> & B)
    : ShardedMatrix(B.n_rows, A->NumSegments(), A->SegmentsNumRows()), m_A(A), m_B(B) { }

  bool WriteMatrixSegment(uint seg_num, arma::Mat<double> & out)
  {
    CHECK(m_A->WriteMatrixSegment(seg_num,out),false);
    out = out * m_B;
    return true;
  }
  

private:
  ShardedMatrix * m_A;
  arma::Mat<double> & m_B;
};

