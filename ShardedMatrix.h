#pragma once

#include <armadillo>
#include <memory.h>
#include <vector> 
#include <string>
#include "assert.h"
#include "LAARDD_Utils.h"
#include <stdio.h>

// A Matrix class which can create an in-memory matrix which represents one segment of the matrix.  
//
class ShardedMatrix 
{

public:
  ShardedMatrix(uint num_cols) 
    : m_numColumns(num_cols) { }

  ShardedMatrix(uint num_cols, std::vector<uint> & segSizes) 
    : m_numColumns (num_cols), m_segmentRow(segSizes) { }
  
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

  std::vector<uint> & SegmentsNumRows()
  {
    return m_segmentRow;
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
  
  bool WriteMatrixSegment(uint seg_num, arma::Mat<double> & out)
  {
    out.set_size(SegmentNumRows(seg_num), NumColumns());
    double* matrix_array = out.memptr();
    for (int i = 0; i < NumColumns(); ++i)
    {
      WriteColumnSegment(seg_num, i, matrix_array + i * SegmentNumRows(seg_num));
    }
    return true;
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
    : ShardedMatrix(A.NumColumns(), A.SegmentsNumRows()), m_filename("disk_shard")
  { 
    assert(!copy);
    m_unique_id = s_unique_id++;
  }

  DiskShardedMatrix(DiskShardedMatrix & A) : ShardedMatrix(0, A.SegmentsNumRows())
  {
    assert(false); // DO NOT COPY-CONSTRUCT
  }

  ~DiskShardedMatrix()
  {
    for (int i = 0; i < NumSegments(); ++i)
    {
      assert(std::remove(("tmp/" + m_filename + "_" +  std::to_string(m_unique_id)  + "_" + std::to_string(i) + ".mat").c_str())==0);
    }
  }

  DiskShardedMatrix(uint num_columns, const char * filename) 
    : ShardedMatrix(num_columns), m_filename(filename)
  { 
    m_unique_id = s_unique_id++;
  }

  bool WriteMatrixSegment(uint seg_num, arma::Mat<double> & out)
  {
    return out.load("tmp/" + m_filename + "_" +  std::to_string(m_unique_id)  + "_" + std::to_string(seg_num) + ".mat");
    out.eval();
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
    seg.eval().save("tmp/" + m_filename + "_" +  std::to_string(m_unique_id) + "_" + std::to_string(seg_num) + ".mat");
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
  
  ShardedMatrixProduct(ShardedMatrix * A, arma::Mat<double> * B)
    : ShardedMatrix(B->n_rows, A->SegmentsNumRows()), m_A(A), m_B(B), m_ownsA(false), m_ownsB(false) { }

  ~ShardedMatrixProduct()
  {
    if (m_ownsA)
    {
      FreeA();
    }
    if (m_ownsB)
    {
      FreeB();
    }
  }

  bool WriteMatrixSegment(uint seg_num, arma::Mat<double> & out)
  {
    CHECK(m_A->WriteMatrixSegment(seg_num,out),false);
    out = out * (*m_B);
    return true;
  }

  void SetOwnsA(bool b)
  {
    m_ownsA = b;
  }
  void SetOwnsB(bool b)
  {
    m_ownsB = b;
  }

  void FreeA()
  {
    delete m_A;
  }
  void FreeB()
  {
    delete m_B;
  }

private:
  ShardedMatrix * m_A;
  arma::Mat<double> * m_B;
  
  bool m_ownsA;
  bool m_ownsB;
};

