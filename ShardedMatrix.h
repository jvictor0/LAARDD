#pragma once

#include <armadillo>
#include <memory.h>
#include <vector> 
#include <string>
#include "assert.h"


// A Matrix class which can create an in-memory matrix which represents one segment of the matrix.  
//
class ShardedMatrix 
{

public:
  ShardedMatrix(uint num_cols) 
    : m_numColumns(num_cols), m_numSegments(0) { }

  ShardedMatrix(uint num_segs, uint num_cols, uint * segSizes) 
    : m_numColumns (num_cols), m_numSegments(num_segs)
  {
    for (int i = 0; i < num_segs; ++i)
    {
      m_segmentRow.push_back(segSizes[i]);
    }
  }
  
  ShardedMatrix(uint num_segs, uint num_cols, uint num_rows) 
    : m_numColumns (num_cols), m_numSegments(num_segs) 
  {
    for (int i = 0; i < num_segs - 1; ++i)
    {
      m_segmentRow.push_back(num_rows/num_segs);
    }
    uint last_size = num_rows % num_segs;
    m_segmentRow.push_back(last_size == 0 ? num_rows/num_segs : last_size);
  }


  virtual ~ShardedMatrix() { } 
  
  virtual void WriteMatrixSegment(uint seg_num, arma::Mat<double> & out) = 0;

  uint NumColumns()
  {
    return m_numColumns;
  }

  uint NumSegments()
  {
    return m_numSegments;
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
    ++m_numSegments;
    m_segmentRow.push_back(seg_size);
  }
  
  uint m_numColumns;
  uint m_numSegments;
  std::vector<uint> m_segmentRow;

};

// This class is just for testing convinience, doesn't make sense for actual use.  
//
class InMemoryShardedMatrix : public ShardedMatrix 
{
public:
  InMemoryShardedMatrix(arma::Mat<double> & matrix, uint num_segments) 
    : ShardedMatrix(num_segments, matrix.n_cols, matrix.n_rows), m_matrix(matrix)
  { }


  void WriteColumnSegment(uint seg_num, uint col_num, double* out)
  {
    for (int i = 0; i < SegmentNumRows(seg_num); ++i)
    {
      out[i] = m_matrix(seg_num * SegmentNumRows(0) + i ,col_num);
    }
  }
  
  void WriteMatrixSegment(uint seg_num, arma::Mat<double> & out)
  {
    out.set_size(SegmentNumRows(seg_num), NumColumns());
    double* matrix_array = out.memptr();
    for (int i = 0; i < NumColumns(); ++i)
    {
      WriteColumnSegment(seg_num, i, matrix_array + i * SegmentNumRows(seg_num));
    }
  }

  
  arma::Mat<double> m_matrix;
};

class DiskShardedMatrix : public ShardedMatrix
{
public: 
  DiskShardedMatrix(uint num_columns) 
    : ShardedMatrix(num_columns), m_filename("disk_shard")
  { 
    m_unique_id = s_unique_id++;
  }

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

  void WriteMatrixSegment(uint seg_num, arma::Mat<double> & out)
  {
    out.load(m_filename + "_" +  std::to_string(m_unique_id)  + "_" + std::to_string(seg_num) + ".mat");
  }
  
  void AddSegment(arma::Mat<double> seg)
  {
    ShardedMatrix::AddSegment(seg.n_rows);
    SetSegment(NumSegments() - 1, seg);
  }
  
  template<class MatType>
  void SetSegment(uint seg_num, MatType seg)
  {
    
    assert(seg.n_cols == NumColumns());
    seg.eval().save(m_filename + "_" +  std::to_string(m_unique_id) + "_" + std::to_string(seg_num) + ".mat");
  }
    
private:
  

  static uint s_unique_id ;
  
  uint m_unique_id;
  std::string m_filename;
};

uint DiskShardedMatrix::s_unique_id =0;

class ShardedMatrixProduct : public ShardedMatrix 
{
public: 
  ShardedMatrixProduct(ShardedMatrix * A, arma::Mat<double> & B)
    : ShardedMatrix(B.n_rows, A->NumSegments(), A->SegmentsNumRows()), m_A(A), m_B(B) { }

  void WriteMatrixSegment(uint seg_num, arma::Mat<double> & out)
  {
    m_A->WriteMatrixSegment(seg_num,out);
    out = out * m_B;
  }
  

private:
  ShardedMatrix * m_A;
  arma::Mat<double> m_B;
};

