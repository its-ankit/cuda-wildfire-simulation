// Wildfire simulation on the GPU.
// Compiled together with the provided harness.cu:
//     nvcc -arch=sm_70 harness.cu wildfire.cu -o main
#include "wildfire.h"
#include <algorithm>

#define BLOCK_SIZE 256 // you can change it.

// ----------------------------------------------------------------------------
// Tuning thresholds (see README: find the crossover points experimentally)
// ----------------------------------------------------------------------------

// Ignition stage: use the worklist while at most this fraction of the grid is
// burning; above it, scan every cell instead.
#ifndef FULL_GRID_FRACTION
#define FULL_GRID_FRACTION 0.03
#endif

// Crew targets: one thread per crew while fewer than this many cells burn;
// at or above it, split each crew's search across whole blocks.
#ifndef SPLIT_SEARCH_MIN_CELLS
#define SPLIT_SEARCH_MIN_CELLS 4096
#endif

#define ITEMS_PER_THREAD 8          // work per thread in the split crew search
#define MAX_BLOCKS_PER_CREW 512

// Device counters, stored together in one small array
#define CNT_NEXT 0                  // entries in the next burning list
#define CNT_CAND 1                  // entries in the candidate list
#define NUM_COUNTERS 2

#define NO_TARGET 0xFFFFFFFFFFFFFFFFull

static inline int gridFor(long long n) { return (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE); }

// Everything the kernels need to read about the terrain and the fire rules
struct Grid
{
    int N, H, dr, dc, W, S, D, I;
    uint8_t *state, *fuel;          // change every step
    const uint8_t *elev, *station;  // fixed
};

// ============================================================================
// DEVICE HELPERS
// ============================================================================

// Walking steps between two cells: ceil(max(|dr|, |dc|) / R)
__device__ __forceinline__ int walkSteps(int r1, int c1, int r2, int c2, int R)
{
    int d = max(abs(r1 - r2), abs(c1 - c2));
    return (d + R - 1) / R;
}

// Can this cell catch fire at all? (unburnt, has fuel, not a fire station)
__device__ __forceinline__ bool canIgnite(const Grid &g, int idx)
{
    return g.state[idx] == UNBURNT && g.fuel[idx] > 0 && !g.station[idx];
}

// Algorithm 1: total heat that cell (r, c) receives from the burning cells in
// its 5x5 square. Returns as soon as the ignition threshold is reached, since
// heat only ever adds up.
__device__ bool reachesIgnition(const Grid &g, int r, int c)
{
    const int N = g.N;
    const int myElev = g.elev[r * N + c];
    int heat = 0;

    for (int rs = max(r - 2, 0); rs <= min(r + 2, N - 1); rs++)
    {
        for (int cs = max(c - 2, 0); cs <= min(c + 2, N - 1); cs++)
        {
            int src = rs * N + cs;
            if ((rs == r && cs == c) || g.state[src] != BURNING) continue;

            int a = g.H, b = 1;
            int orr = r - rs, oc = c - cs;

            // wind: straight downwind multiplies, straight upwind divides
            if ((orr == g.dr && oc == g.dc) || (orr == 2 * g.dr && oc == 2 * g.dc))
                a *= g.W;
            else if ((orr == -g.dr && oc == -g.dc) || (orr == -2 * g.dr && oc == -2 * g.dc))
                b *= g.W;

            // slope: uphill multiplies, downhill divides
            int delta = myElev - g.elev[src];
            if (delta > g.D)       a *= g.S;
            else if (delta < -g.D) b *= g.S;

            heat += a / b;
            if (heat >= g.I) return true;
        }
    }
    return false;
}

// ============================================================================
// SETUP KERNELS
// ============================================================================

// Mark the initial fires as burning and write them as the first worklist
__global__ void initFiresKernel(Grid g, const int *fires, int F, int *burning)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= F) return;
    int idx = fires[2 * i] * g.N + fires[2 * i + 1];
    g.state[idx] = BURNING;
    burning[i] = idx;
}

__global__ void initStationsKernel(int N, const int *stations, int K, uint8_t *station)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K) station[stations[2 * i] * N + stations[2 * i + 1]] = 1;
}

// ============================================================================
// FIRE STAGE: IGNITION  (reads only the grid G; no state is written here)
// ============================================================================

// Worklist path, part 1: every burning cell offers its 24 neighbours as
// candidates. A neighbour shared by several fires is added once per step,
// using a per-cell stamp of the step number.
__global__ void markCandidatesKernel(Grid g, const int *burning, int burnCount,
                                     int step, int *stamp, int *cand, int *counters)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)burnCount * 25) return;

    int src = burning[t / 25];
    int k = (int)(t % 25);
    if (k == 12) return;                              // the centre is the fire itself

    int r = src / g.N + k / 5 - 2;
    int c = src % g.N + k % 5 - 2;
    if (r < 0 || r >= g.N || c < 0 || c >= g.N) return;

    int idx = r * g.N + c;
    if (!canIgnite(g, idx)) return;
    if (atomicExch(&stamp[idx], step) != step)        // first visit this step
        cand[atomicAdd(&counters[CNT_CAND], 1)] = idx;
}

// Worklist path, part 2: each candidate gathers its heat; ignitions are only
// recorded in the next list, and applied after the crews have read G.
__global__ void igniteCandidatesKernel(Grid g, const int *cand, int candCount,
                                       int *next, int *counters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= candCount) return;
    int idx = cand[i];
    if (reachesIgnition(g, idx / g.N, idx % g.N))
        next[atomicAdd(&counters[CNT_NEXT], 1)] = idx;
}

// Full-grid path: one thread per cell, used when the fire is large
__global__ void igniteFullGridKernel(Grid g, int *next, int *counters)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)g.N * g.N) return;
    if (!canIgnite(g, (int)idx)) return;
    if (reachesIgnition(g, (int)(idx / g.N), (int)(idx % g.N)))
        next[atomicAdd(&counters[CNT_NEXT], 1)] = (int)idx;
}

// ============================================================================
// CREW STAGE  (reads G; put-out cells are only flagged in `wet`)
// ============================================================================

// Closeness key: walking steps first, then row, then column. Since
// idx = row * N + col, "lowest row, then lowest column" is "lowest idx".
__device__ __forceinline__ unsigned long long targetKey(int steps, int idx)
{
    return ((unsigned long long)steps << 32) | (unsigned int)idx;
}

// Small fire: one thread per crew scans the whole burning list
__global__ void crewSearchSimpleKernel(const int *crews, int C, int N, int R,
                                       const int *burning, int burnCount,
                                       unsigned long long *best)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= C || crews[3 * k + 2] == 0) return;
    int r = crews[3 * k], c = crews[3 * k + 1];

    unsigned long long mine = NO_TARGET;
    for (int i = 0; i < burnCount; i++)
    {
        int idx = burning[i];
        unsigned long long key = targetKey(walkSteps(r, c, idx / N, idx % N, R), idx);
        if (key < mine) mine = key;
    }
    best[k] = mine;
}

// Large fire: blockIdx.y picks the crew, and blockIdx.x splits the burning
// list among several blocks. Each thread finds the closest cell in its share,
// the block combines them with a shared-memory reduction, and each block
// merges its answer into best[crew] with one 64-bit atomicMin.
__global__ void crewSearchSplitKernel(const int *crews, int N, int R,
                                      const int *burning, int burnCount,
                                      unsigned long long *best)
{
    __shared__ unsigned long long partial[BLOCK_SIZE];

    int k = blockIdx.y;
    if (crews[3 * k + 2] == 0) return;                // same for the whole block
    int r = crews[3 * k], c = crews[3 * k + 1];

    unsigned long long mine = NO_TARGET;
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < burnCount; i += stride)
    {
        int idx = burning[i];
        unsigned long long key = targetKey(walkSteps(r, c, idx / N, idx % N, R), idx);
        if (key < mine) mine = key;
    }

    partial[threadIdx.x] = mine;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s && partial[threadIdx.x + s] < partial[threadIdx.x])
            partial[threadIdx.x] = partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMin(&best[k], partial[0]);
}

// Algorithm 2 for every crew: pick the target, walk, put out or refill
__global__ void crewMoveKernel(Grid g, int *crews, int C, int R, int X,
                               const unsigned long long *best, int burnCount,
                               const int *stations, int K, uint8_t *wet)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= C) return;
    const int N = g.N;
    int r = crews[3 * k], c = crews[3 * k + 1], w = crews[3 * k + 2];

    // choose a target
    int tr = -1, tc = -1;
    if (w > 0)
    {
        if (burnCount > 0 && best[k] != NO_TARGET)
        {
            int idx = (int)(best[k] & 0xFFFFFFFFu);
            tr = idx / N;
            tc = idx % N;
        }
    }
    else
    {
        int bestSteps = 0x7FFFFFFF;
        for (int j = 0; j < K; j++)
        {
            int sr = stations[2 * j], sc = stations[2 * j + 1];
            int s = walkSteps(r, c, sr, sc, R);
            if (s < bestSteps || (s == bestSteps && (sr < tr || (sr == tr && sc < tc))))
            {
                bestSteps = s; tr = sr; tc = sc;
            }
        }
    }

    // move up to R rows and R columns towards the target
    if (tr >= 0)
    {
        int mr = min(R, abs(tr - r)), mc = min(R, abs(tc - c));
        r += (tr > r) ? mr : -mr;
        c += (tc > c) ? mc : -mc;
    }

    // put out the cell it stands on (judged from G), or refill at a station
    int idx = r * N + c;
    if (g.state[idx] == BURNING && w > 0)
    {
        wet[idx] = 1;
        w--;
    }
    if (g.station[idx]) w = X;

    crews[3 * k] = r;
    crews[3 * k + 1] = c;
    crews[3 * k + 2] = w;
}

// ============================================================================
// APPLY STAGE: write the end-of-step grid  (G is no longer needed)
// ============================================================================

// Threads [0, burnCount): cells burning in G lose 1 fuel, then become wet
//                         (put out), burnt (no fuel left) or keep burning.
// Threads [burnCount, burnCount + igniteCount): newly ignited cells start burning.
// The next list already holds the ignited cells at [0, igniteCount), and the
// survivors are appended after them.
__global__ void applyStepKernel(Grid g, const int *burning, int burnCount,
                                int *next, int igniteCount, uint8_t *wet, int *counters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < burnCount)
    {
        int idx = burning[i];
        int f = g.fuel[idx] - 1;
        g.fuel[idx] = (uint8_t)f;
        if (wet[idx])
        {
            wet[idx] = 0;
            g.state[idx] = WET;                       // wet wins over burnt
        }
        else if (f == 0)
        {
            g.state[idx] = BURNT;
        }
        else
        {
            next[atomicAdd(&counters[CNT_NEXT], 1)] = idx;
        }
    }
    else if (i < burnCount + igniteCount)
    {
        g.state[next[i - burnCount]] = BURNING;
    }
}

// ============================================================================
// SIMULATION DRIVER
// ============================================================================

static int readCounter(const int *counters, int which)
{
    int v;
    cudaMemcpy(&v, counters + which, sizeof(int), cudaMemcpyDeviceToHost);
    return v;
}

void run_simulation(Params p, uint8_t *state, uint8_t *fuel, const uint8_t *elev,
                    int F, const int *fires, int K, const int *stations,
                    int C, int *crews)
{
    const int N = p.N;
    const size_t cells = (size_t)N * N;

    // ---------------- device buffers (allocated once) ----------------
    uint8_t *d_state, *d_fuel, *d_elev, *d_station, *d_wet;
    int *d_stamp, *d_listA, *d_listB, *d_cand, *d_counters;
    int *d_fires, *d_stations, *d_crews;
    unsigned long long *d_best;

    cudaMalloc(&d_state, cells);
    cudaMalloc(&d_fuel, cells);
    cudaMalloc(&d_elev, cells);
    cudaMalloc(&d_station, cells);
    cudaMalloc(&d_wet, cells);
    cudaMalloc(&d_stamp, cells * sizeof(int));
    cudaMalloc(&d_listA, cells * sizeof(int));
    cudaMalloc(&d_listB, cells * sizeof(int));
    cudaMalloc(&d_cand, cells * sizeof(int));
    cudaMalloc(&d_counters, NUM_COUNTERS * sizeof(int));
    cudaMalloc(&d_fires, (size_t)2 * F * sizeof(int) + sizeof(int));
    cudaMalloc(&d_stations, (size_t)2 * K * sizeof(int) + sizeof(int));
    cudaMalloc(&d_crews, (size_t)3 * C * sizeof(int) + sizeof(int));
    cudaMalloc(&d_best, (size_t)C * sizeof(unsigned long long) + sizeof(unsigned long long));

    cudaMemcpy(d_fuel, fuel, cells, cudaMemcpyHostToDevice);
    cudaMemcpy(d_elev, elev, cells, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fires, fires, (size_t)2 * F * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stations, stations, (size_t)2 * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_crews, crews, (size_t)3 * C * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(d_state, UNBURNT, cells);
    cudaMemset(d_station, 0, cells);
    cudaMemset(d_wet, 0, cells);
    cudaMemset(d_stamp, 0xFF, cells * sizeof(int));   // -1: never visited

    Grid g{N, p.H, p.dr, p.dc, p.W, p.S, p.D, p.I, d_state, d_fuel, d_elev, d_station};

    if (K > 0) initStationsKernel<<<gridFor(K), BLOCK_SIZE>>>(N, d_stations, K, d_station);
    if (F > 0) initFiresKernel<<<gridFor(F), BLOCK_SIZE>>>(g, d_fires, F, d_listA);

    int *burning = d_listA, *next = d_listB;
    int burnCount = F;
    const long long fullGridCutoff = (long long)(FULL_GRID_FRACTION * (double)cells);

    // ---------------- T steps ----------------
    for (int step = 0; step < p.T; step++)
    {
        cudaMemset(d_counters, 0, NUM_COUNTERS * sizeof(int));

        // 1. Fire: find the cells that ignite (recorded in next[], not applied yet)
        if (burnCount > 0)
        {
            if (burnCount > fullGridCutoff)
            {
                igniteFullGridKernel<<<gridFor(cells), BLOCK_SIZE>>>(g, next, d_counters);
            }
            else
            {
                markCandidatesKernel<<<gridFor((long long)burnCount * 25), BLOCK_SIZE>>>(
                    g, burning, burnCount, step, d_stamp, d_cand, d_counters);
                int candCount = readCounter(d_counters, CNT_CAND);
                if (candCount > 0)
                    igniteCandidatesKernel<<<gridFor(candCount), BLOCK_SIZE>>>(
                        g, d_cand, candCount, next, d_counters);
            }
        }

        // 2. Crews: choose targets and move, reading the same grid G
        if (C > 0)
        {
            if (burnCount > 0)
            {
                if (burnCount < SPLIT_SEARCH_MIN_CELLS)
                {
                    crewSearchSimpleKernel<<<gridFor(C), BLOCK_SIZE>>>(
                        d_crews, C, N, p.R, burning, burnCount, d_best);
                }
                else
                {
                    cudaMemset(d_best, 0xFF, (size_t)C * sizeof(unsigned long long));
                    long long want = ((long long)burnCount + BLOCK_SIZE * ITEMS_PER_THREAD - 1)
                                     / (BLOCK_SIZE * ITEMS_PER_THREAD);
                    dim3 grid((unsigned)std::min<long long>(want, MAX_BLOCKS_PER_CREW), C);
                    crewSearchSplitKernel<<<grid, BLOCK_SIZE>>>(
                        d_crews, N, p.R, burning, burnCount, d_best);
                }
            }
            crewMoveKernel<<<gridFor(C), BLOCK_SIZE>>>(
                g, d_crews, C, p.R, p.X, d_best, burnCount, d_stations, K, d_wet);
        }

        // 3. Combine: burn fuel, apply wet / burnt, start the new fires
        if (burnCount > 0)
        {
            int igniteCount = readCounter(d_counters, CNT_NEXT);
            applyStepKernel<<<gridFor((long long)burnCount + igniteCount), BLOCK_SIZE>>>(
                g, burning, burnCount, next, igniteCount, d_wet, d_counters);
            burnCount = readCounter(d_counters, CNT_NEXT);
            std::swap(burning, next);
        }
    }

    // ---------------- results back to the host ----------------
    cudaMemcpy(state, d_state, cells, cudaMemcpyDeviceToHost);
    cudaMemcpy(fuel, d_fuel, cells, cudaMemcpyDeviceToHost);
    cudaMemcpy(crews, d_crews, (size_t)3 * C * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_state);   cudaFree(d_fuel);   cudaFree(d_elev);  cudaFree(d_station);
    cudaFree(d_wet);     cudaFree(d_stamp);  cudaFree(d_listA); cudaFree(d_listB);
    cudaFree(d_cand);    cudaFree(d_counters);
    cudaFree(d_fires);   cudaFree(d_stations); cudaFree(d_crews); cudaFree(d_best);
}
