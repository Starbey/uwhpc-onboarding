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
  std::size_t per_row_; // number of doubles per row
  std::vector<double> cells_;

public:
  // round the size of a row up to the nearest multiple of the cache line size (64) so that each row starts at the beginning of each cache line
  Grid(std::size_t rows, std::size_t cols) : rows_{rows}, cols_{cols}, per_row_{(cols + 7) & ~std::size_t{7}}, cells_(rows * per_row_, 0.0) {}

  std::size_t rows() const {
    return rows_;
  }

  std::size_t cols() const {
    return cols_;
  }

  std::size_t per_row() const {
    return per_row_;
  }

  double& operator()(std::size_t i, std::size_t j) {
    return cells_[i * per_row_ + j];
  }

  double  operator()(std::size_t i, std::size_t j) const {
    return cells_[i * per_row_ + j];
  }

  const double* data() const { 
    return cells_.data(); 
  }
};  

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  const double* old_cells = old_grid.data();

  const std::size_t rows = old_grid.rows();
  const std::size_t cols = old_grid.cols();

  for (std::size_t i = 0; i < rows; i++) {
    new_grid(i, 0) = old_grid(i, 0);
    new_grid(i, cols - 1) = old_grid(i, cols - 1);
  }

  for (std::size_t j = 0; j < cols; j++) {
    new_grid(0, j) = old_grid(0, j);
    new_grid(rows - 1, j) = old_grid(rows - 1, j);
  }

  for (std::size_t i = 1; i < rows - 1; i++) {  
    const double* top = old_cells + (i - 1) * old_grid.per_row();
    const double* center = old_cells + i * old_grid.per_row();
    const double* bottom = old_cells + (i + 1) * old_grid.per_row();

    for (std::size_t j = 1; j < cols - 1; j++) {
      // previously each cell access demanded a multiply and an add
      // explicitly defining each stride moves this work out of the inner loop
      new_grid(i, j) = 0.5 * center[j] + 0.125 * (top[j] + bottom[j] + center[j - 1] + center[j + 1]);
    }
  }
}
