// Input/output and timing harness (do not modify).
// Reads the input file, warms up the GPU, times one call to run_simulation()
// from wildfire.cu, and writes the output file.
//
// Build: nvcc -arch=sm_70 harness.cu wildfire.cu -o main
#include <chrono>
#include <vector>
#include "wildfire.h"

static void read_or_die(void *dst, size_t bytes, FILE *f, const char *what)
{
    if (bytes > 0 && fread(dst, 1, bytes, f) != bytes) {
        fprintf(stderr, "Error: input file ended while reading %s\n", what);
        exit(1);
    }
}

// reads a count followed by count * width int32 values
static std::vector<int> read_list(FILE *f, int width, int *count, const char *what)
{
    read_or_die(count, sizeof(int), f, what);
    std::vector<int> v((size_t)*count * width + 1);
    read_or_die(v.data(), (size_t)*count * width * sizeof(int), f, what);
    return v;
}

__global__ void warmup_kernel(int *x)
{
    if (x) *x = 0;
}

// Pay CUDA's one-time start-up costs (context creation, module loading,
// first allocation, first launch) before the timer starts.
static void warm_up_gpu()
{
    cudaFree(0);
    int *d_x;
    cudaMalloc(&d_x, sizeof(int));
    warmup_kernel<<<1, 1>>>(d_x);
    int h_x;
    cudaMemcpy(&h_x, d_x, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaDeviceSynchronize();
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }
    // load every kernel when the context is created, so that the first launch
    // of a kernel in wildfire.cu does not pay for loading it inside the timer
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);

    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        fprintf(stderr, "Error: Unable to open input file %s\n", argv[1]);
        return 1;
    }

    Params p;
    int header[11];
    read_or_die(header, sizeof(header), in, "header");
    p.N = header[0]; p.T = header[1]; p.H = header[2]; p.dr = header[3]; p.dc = header[4];
    p.W = header[5]; p.S = header[6]; p.D = header[7]; p.I = header[8]; p.R = header[9]; p.X = header[10];

    size_t cells = (size_t)p.N * p.N;
    std::vector<uint8_t> fuel(cells), elev(cells), state(cells, UNBURNT);
    read_or_die(fuel.data(), cells, in, "fuel grid");
    read_or_die(elev.data(), cells, in, "elevation grid");

    int F, K, C;
    std::vector<int> fires = read_list(in, 2, &F, "fires");
    std::vector<int> stations = read_list(in, 2, &K, "stations");
    std::vector<int> crews = read_list(in, 3, &C, "crews");
    fclose(in);

    warm_up_gpu();

    auto start = std::chrono::high_resolution_clock::now();
    run_simulation(p, state.data(), fuel.data(), elev.data(), F, fires.data(),
                   K, stations.data(), C, crews.data());
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    FILE *out = fopen(argv[2], "wb");
    if (!out) {
        fprintf(stderr, "Error: Unable to open output file %s\n", argv[2]);
        return 1;
    }
    fwrite(state.data(), 1, cells, out);
    fwrite(fuel.data(), 1, cells, out);
    fwrite(crews.data(), sizeof(int), (size_t)3 * C, out);
    fclose(out);

    fprintf(stderr, "Simulation time: %.3f ms\n", ms);
    return 0;
}
