// Shared definitions between harness.cu and your wildfire.cu (do not modify).
#pragma once
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

// cell states
#define UNBURNT 0
#define BURNING 1
#define BURNT   2
#define WET     3

struct Params {
    int N, T, H, dr, dc, W, S, D, I, R, X;
};

// Runs the whole simulation. You implement this in wildfire.cu.
//
// All pointers are host pointers; copying data to and from the GPU is part of
// your job (and of the timed region).
//
// On entry:
//   state  N*N cells, all UNBURNT (set the initial fires yourself)
//   fuel   N*N cells, the input fuel grid
//   elev   N*N cells, the input elevation grid
//   fires     F pairs (row, col)
//   stations  K pairs (row, col)
//   crews     C triples (row, col, water)
// All grids are stored row by row: cell (r, c) is at index r * N + c.
//
// On exit, state and fuel must hold the final grids, and crews the final
// (row, col, water) of every crew.
//
// The harness has already initialised the GPU before calling this, so CUDA
// start-up cost is not counted against you.
void run_simulation(Params p, uint8_t *state, uint8_t *fuel, const uint8_t *elev,
                    int F, const int *fires, int K, const int *stations,
                    int C, int *crews);
