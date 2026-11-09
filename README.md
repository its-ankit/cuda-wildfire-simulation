# Wildfire Simulation on the GPU (CUDA)

A CUDA simulation of a wildfire spreading over a terrain grid of up to **8192 × 8192 cells**, with firefighter crews chasing the nearest fire and refilling at fire stations.

The focus is on **not wasting work**: instead of updating all 67 million cells every step, the simulation tracks only the cells that can actually change, and switches to a full-grid scan only when the fire is large.

Built for CS6023 (GPU Programming), IIT Madras, Assignment 4.

---

## 1. The big picture

Every step has two actors working on the **same snapshot G** of the grid:

```
            grid G at the start of the step
             /                          \
     FIRE: heat spreads,           CREWS: walk to the closest
     cells ignite, fuel burns      fire, put it out or refill
             \                          /
          COMBINE: wet wins over burnt → grid for next step
```

**Fire.** Each burning cell heats the 24 cells of the 5 × 5 square around it. The heat is boosted or reduced by **wind** (straight downwind ×W, upwind ÷W) and **slope** (uphill ×S, downhill ÷S). A cell ignites when its total heat reaches the threshold I. Burning cells lose 1 fuel per step and become burnt at 0.

**Crews.** Each crew walks up to R rows and R columns per step towards the closest burning cell (closest by walking steps, ties by lowest row then column). Standing on a burning cell puts it out (it becomes **wet**, never burns again) and uses 1 unit of water. An empty crew heads to the closest fire station to refill.

**The key rule:** fire and crews both read G, and nothing changes until the step ends. A cell that ignites this step only starts burning next step; a cell put out this step still gave heat this step.

---

## 2. Why the naive version is too slow

The direct approach launches **one thread per cell, every step**:

- 8192² ≈ 67 million threads per step, × 1000 steps ≈ 6.7 × 10¹⁰ cell updates.
- Almost all of them change nothing: early on the fire is tiny, and later most of its path is burnt and never changes again.
- Each crew scans the **entire grid** for the closest fire.

Only cells **near the fire** can change. That's the idea behind every optimisation below.

---

## 3. Design

### 3.1 Worklist of burning cells

The program keeps a list of the cells burning in G. Every cell that can change in a step is either burning itself or within its 5 × 5 square.

Each step on the worklist path:

1. **Mark candidates** (`markCandidatesKernel`): one thread per (burning cell, neighbour) pair. Neighbours that could ignite (unburnt, fuel > 0, not a station) are added to a candidate list.
2. **Ignite** (`igniteCandidatesKernel`): one thread per candidate gathers its heat from the 5 × 5 square and records the ignition.
3. **Apply** (`applyStepKernel`): burning cells lose fuel and become wet / burnt / stay burning; new fires start. The survivors plus new fires form the next worklist.

Three details make this correct in parallel:

- **No duplicates.** Several fires share neighbours, so one cell can be found many times. A per-cell **stamp** holds the last step in which the cell was added; `atomicExch(&stamp[cell], step)` lets exactly one thread add it. Stamping with the step number means the array never has to be cleared.
- **Reserving list slots.** Many threads append at once; `atomicAdd` on a counter gives each one its own index.
- **Deferred writes.** Ignitions and put-outs are only *recorded* (in the next list and a `wet` flag grid) and applied in the last kernel. Writing the grid earlier would let other threads see a half-updated G.

### 3.2 Gather, not scatter

Each candidate **pulls** heat from its burning neighbours, instead of each fire **pushing** heat with atomic adds. This needs no atomics on the heat itself, and a candidate can stop as soon as its heat reaches I, because heat only adds up.

### 3.3 Full-grid fallback

A worklist is not always faster: per burning cell it pays for scattered reads, atomics and deduplication. When the fire covers a large part of the grid, a plain kernel over all cells (`igniteFullGridKernel`) is cheaper. The program switches when

```
burning cells > FULL_GRID_FRACTION × N²
```

The burning list is still maintained in both modes, since the crews need it.

### 3.4 Crew targets

"Closest" is a three-level comparison: walking steps, then row, then column. Since a cell's index is `row × N + col`, it packs into one 64-bit key:

```
key = (steps << 32) | (row × N + col)        smallest key = correct target
```

So the whole tie-break rule becomes a single `min`.

- **Few burning cells:** one thread per crew loops over the list (`crewSearchSimpleKernel`).
- **Many burning cells:** C threads are far too few to keep the GPU busy. `crewSearchSplitKernel` gives each crew several blocks (`blockIdx.y` = crew). Each thread finds the minimum key in its share, the block combines them with a **shared-memory tree reduction**, and each block merges its result into the crew's answer with a 64-bit `atomicMin`.

The switch happens at `SPLIT_SEARCH_MIN_CELLS` burning cells. A crew heading to a fire station just loops over the stations (at most 256).

### 3.5 Step order on the GPU

```
1. ignition    worklist (mark + gather) or full grid   → new fires recorded
2. crews       search (simple or split) + move          → wet cells flagged
3. apply       fuel −1, wet / burnt / burning, start new fires
```

Steps 1 and 2 only read G; step 3 is the only place the grid is written. All buffers are allocated once per run.

---

## 4. Tuning the thresholds

Both thresholds are defined at the top of `wildfire.cu`:

| Threshold | Starting value | Meaning |
|---|---|---|
| `FULL_GRID_FRACTION` | 0.03 | use the full-grid scan above this fraction of N² burning |
| `SPLIT_SEARCH_MIN_CELLS` | 4096 | use the split crew search at or above this many burning cells |

These are starting points, not measured optima. To tune them: time one step of each version on grids with different numbers of burning cells, find where the two times cross, and check whether the crossover depends on N (the ignition threshold is expressed as a fraction of N² for that reason). Both can be overridden at compile time, e.g. `-DFULL_GRID_FRACTION=0.05`.

---

## 5. Testing

`tools/reference.cpp` is a plain sequential version that follows the problem statement line by line, with no optimisation. It reproduces all 13 public test cases, and was used to generate expected outputs for randomly generated inputs (different sizes, winds, slopes, fuel gaps, crew counts and fire densities).

The GPU logic was checked against it in four configurations, so every code path runs: always worklist + simple search, always full grid + split search, the default thresholds, and a mix. In each, kernel threads ran in a randomly shuffled order, so the result does not depend on thread scheduling.

---

## 6. Repository layout

```
src/wildfire.cu       the GPU simulation (run_simulation and all kernels)
src/wildfire.h        interface: Params struct, cell states (provided by the course)
src/harness.cu        reads input, times run_simulation, writes output (provided by the course)
tools/reference.cpp   sequential CPU version for generating expected outputs
```

## 7. How to run

Requires an NVIDIA GPU and the CUDA Toolkit.

```bash
nvcc -O3 -arch=sm_70 src/harness.cu src/wildfire.cu -o main     # use sm_75 on a Colab T4
./main input.in.bin output.out.bin
```

The program prints the simulation time to stderr.

### Input (binary, int32 little-endian unless noted)

1. Header: `N T H dr dc W S D I R X`
2. Fuel grid: N² bytes (uint8)
3. Elevation grid: N² bytes (uint8)
4. Initial fires: count F, then F pairs `(row, col)`
5. Fire stations: count K, then K pairs `(row, col)`
6. Crews: count C, then C triples `(row, col, water)`

### Output

Final state grid (N² bytes: 0 unburnt, 1 burning, 2 burnt, 3 wet), final fuel grid (N² bytes), then each crew's `(row, col, water)` as int32.

### Constraints

N ≤ 8192, T ≤ 1000, C ≤ 1024 crews, K ≤ 256 stations, wind and slope factors ≤ 8, movement radius R ≤ 64, tank capacity X ≤ 255.
