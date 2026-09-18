#pragma once

#include <cstddef>
#include <vector>
#include <algorithm>
#include <new>

/* padding the row length only guarantees that rows sit at a uniform offset from the
start of the allocation. if that start is not itself divisible by 64, every row is off
by the same amount and no row begins at a cache line boundary. this allocator fixes the
start, and std::vector still owns and frees the memory so no cleanup work moves to Grid */
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
  using value_type = T;

  AlignedAllocator() noexcept = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  // std::vector cannot derive this itself because Alignment is not a type parameter
  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  T* allocate(std::size_t n) {
    return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{Alignment}));
  }

  void deallocate(T* p, std::size_t) noexcept {
    ::operator delete(p, std::align_val_t{Alignment});
  }
};

// two of these allocators are interchangeable whenever their alignment matches
template <typename T, typename U, std::size_t A>
bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept {
  return true;
}

template <typename T, typename U, std::size_t A>
bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept {
  return false;
}

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
  std::vector<double, AlignedAllocator<double, 64>> cells_;

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
- i noticed that this hardly makes a difference in performance probably because this problem is bottlenecked by memory access and not math*/

/* views: 
- Grid owns memory
- views hold pointer to that same memory, but it doesn't own it
- view going out of scope does not free the memory
- copying a Grid copies 8 MB
- copying a view copies 8*4 = 32 B*/
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

  // hand copying of top and bottom boundary rows to copy_n
  // copy_n uses wide instructions: good in this case where memory is contiguous
  std::copy_n(in.cells, in.cols, out.cells);
  std::copy_n(in.cells + (in.rows - 1) * in.stride, in.cols, out.cells + (out.rows - 1) * out.stride);

  /* my processor has 6 performance cores and 8 efficiency cores. efficiency cores are slower, so we want less work on those.
  #pragma omp parallel for splits rows into equal chunks by default. dynamic schedule hands out work as threads become free, 
  so fast cores take more chunks. hinders performance on evaluator though, so i stuck with static
  
  observation: cores' private caches are too small to hold any meaningful fraction of a whole grid, so they need to fetch from the shared cache
  at the start of each time step. a lot of data is moved on the shared interconnect, so there's a cache bandwidth bottleneck s.t. 
  increasing # of threads only worsens performance. on my machine, it flattened after 6 threads.*/ 
  #pragma omp parallel for schedule(static)
  for (std::size_t i = 1; i < in.rows - 1; i++) {  
    const double* top = in.cells + (i - 1) * in.stride;
    const double* center = in.cells + i * in.stride;
    const double* bottom = in.cells + (i + 1) * in.stride;
    double* out_row = out.cells + i * out.stride;

    // rows are consecutive in memory and independent. can fetch and add consecutive top, bottom, and center rows
    #pragma omp simd
    for (std::size_t j = 1; j < in.cols - 1; j++) {
      // previously each cell access demanded a multiply and an add
      // explicitly defining each stride moves this work out of the inner loop
      out_row[j] = 0.5 * center[j] + 0.125 * (top[j] + bottom[j] + center[j - 1] + center[j + 1]);
    }
  }
}
