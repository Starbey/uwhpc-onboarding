#pragma once

#include <cstddef>
#include <vector>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::size_t stride_; // number of doubles per row
  std::vector<double> cells_;

public:
  // round the size of a row up to the nearest multiple of the cache line size (64) so that each row starts at the beginning of each cache line
  Grid(std::size_t rows, std::size_t cols) : rows_{rows}, cols_{cols}, stride_{(cols + 7) & ~std::size_t{7}}, cells_(rows * stride_, 0.0) {}

  std::size_t rows() const {
    return rows_;
  }

  std::size_t cols() const {
    return cols_;
  }

  std::size_t stride() const {
    return stride_;
  }

  double& operator()(std::size_t i, std::size_t j) {
    return cells_[i * stride_ + j];
  }

  double  operator()(std::size_t i, std::size_t j) const {
    return cells_[i * stride_ + j];
  }

  const double* data() const { 
    return cells_.data(); 
  }

  double* data() { 
    return cells_.data(); 
  }
};  

/* what restrict does:
- tells the compiler that the pointer is not aliased with any other pointer
- verified two separate Grid objects in the harness code
- allows compiler to hold values in registers and process cells in batches instead of fetching from memory after every write
- without restrict, the compiler can't read all 4 cells at once and process them in parallel
- i noticed that this hardly makes a difference in performance probably because this problem is bottlenecked by memory access and not math
*/

/* views: 
- Grid owns memory
- views hold pointer to that same memory, but it doesn't own it
- view going out of scope does not free the memory
- copying a Grid copies 8 MB
- copying a view copies 8*4 = 32 B
*/
struct ConstGridView {
  const double* __restrict cells;
  std::size_t rows;
  std::size_t cols;
  std::size_t stride;

  double operator()(std::size_t i, std::size_t j) const {
    return cells[i * stride + j];
  }
};

struct GridView {
  double* __restrict cells;
  std::size_t rows;
  std::size_t cols;
  std::size_t stride;

  double& operator()(std::size_t i, std::size_t j) const {
    return cells[i * stride + j];
  }
};

void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  ConstGridView in{old_grid.data(), old_grid.rows(), old_grid.cols(), old_grid.stride()};
  GridView out{new_grid.data(), new_grid.rows(), new_grid.cols(), new_grid.stride()};

  for (std::size_t i = 0; i < in.rows; i++) {
    out(i, 0) = in(i, 0);
    out(i, in.cols - 1) = in(i, in.cols - 1);
  }

  for (std::size_t j = 0; j < in.cols; j++) {
    out(0, j) = in(0, j);
    out(in.rows - 1, j) = in(in.rows - 1, j);
  }

  /* my processor has 6 performance cores and 8 efficiency cores. efficiency cores are slower, so we want less work on those.
  #pragma omp parallel for splits rows into equal chunks by default. 
  this schedule hands out work as threads become free, so fast cores take more chunks

  dynamic: put rows in a pile, threads take a batch of 16 whenever they're free
  
  why 16 rows per chunk? honestly i just tried a few numbers and this worked best for my machine. 
  if the batch size is small, there's too much coordination overhead. if batch size is large, 
  performance cores idle after finishing their batch and now we're stuck waiting for the efficiency cores
  16 isn't optimal for all grid sizes. for example, if grid has 16 interior rows, then each thread only gets 1 row

  TODO: test static vs. dynamic scheduling on evaluator
  
  observation: cores' private caches are too small to hold any meaningful fraction of a whole grid, so they need to fetch from the shared cache
  at the start of each time step. a lot of data is moved on the shared interconnect, so there's a cache bandwidth bottleneck s.t. 
  increasing # of threads only worsens performance
*/ 
  #pragma omp parallel for schedule(static)
  for (std::size_t i = 1; i < in.rows - 1; i++) {  
    const double* top = in.cells + (i - 1) * in.stride;
    const double* center = in.cells + i * in.stride;
    const double* bottom = in.cells + (i + 1) * in.stride;
    double* out_row = out.cells + i * out.stride;

    for (std::size_t j = 1; j < in.cols - 1; j++) {
      // previously each cell access demanded a multiply and an add
      // explicitly defining each stride moves this work out of the inner loop
      out_row[j] = 0.5 * center[j] + 0.125 * (top[j] + bottom[j] + center[j - 1] + center[j + 1]);
    }
  }
}
