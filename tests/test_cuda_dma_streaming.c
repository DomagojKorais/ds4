/* CUDA DMA streaming: registration lifecycle, decline reasons, and the
 * direct-DMA copy path producing output identical to the resident
 * reference, through a completely different transport (cudaMemcpyAsync
 * straight from a registered mapping instead of pread into a staging
 * buffer). See docs/DMA_STREAMING.md. */
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
enum { EXPERTS = 8, LAYERS = 2, DIM = 256, SELECTED = 4, ROWS = 17 };
/* Q4_K: one 144-byte block per 256-wide row (matches the encoding
 * tests/test_cuda_ssd_cache.c uses for the same type). */
enum { GATE_TYPE = 12, DOWN_TYPE = 12, UNIT = 144 };

static uint32_t random_state = 137;
static unsigned char random_byte(void) {
    random_state = random_state * 1664525u + 1013904223u;
    return random_state >> 24;
}

static ds4_gpu_tensor *tensor(size_t bytes, const void *data) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    CHECK(t && (!data || ds4_gpu_tensor_write(t, 0, data, bytes)));
    return t;
}

/* Q4_K block layout: fp16 scale (d) then fp16 min (dmin) at offset 0, then
 * 12 bytes of 6-bit sub-scales, then 128 bytes of 4-bit quants -- matches
 * the unit==144 case in tests/test_cuda_ssd_cache.c's own scale-setting
 * loop. A valid d/dmin pair keeps the dequantized values finite; random
 * bytes there produce garbage exponents. */
static void fill_q4k(unsigned char *w, size_t rows, size_t block) {
    for (size_t r = 0; r < rows; r++) {
        unsigned char *row = w + r * block;
        for (size_t j = 0; j < block; j++) row[j] = random_byte();
        row[0] = 0; row[1] = 0x14; row[2] = 0; row[3] = 0;
    }
}

int main(void) {
    const size_t gate_bytes = (size_t)DIM * UNIT, down_bytes = (size_t)DIM * UNIT;
    const size_t layer_bytes = EXPERTS * (gate_bytes * 2 + down_bytes);
    const size_t model_bytes = LAYERS * layer_bytes;
    unsigned char *model = malloc(model_bytes);
    CHECK(model);
    for (unsigned l = 0; l < LAYERS; l++) {
        unsigned char *layer = model + l * layer_bytes;
        fill_q4k(layer, (size_t)EXPERTS * DIM, UNIT);
        fill_q4k(layer + EXPERTS * gate_bytes, (size_t)EXPERTS * DIM, UNIT);
        fill_q4k(layer + 2 * EXPERTS * gate_bytes, (size_t)EXPERTS * DIM, UNIT);
    }

    /* cudaHostRegister needs valid host memory, not specifically an mmap --
     * this malloc'd buffer works directly as the "model_map" for the copy
     * path, same as tests/test_cuda_ssd_cache.c does for the pread path.
     * The backing fd only matters to the page-cache/O_DIRECT bookkeeping
     * that this test never exercises. */
    char path[] = "/tmp/ds4-dma-streaming-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0 && unlink(path) == 0 &&
          write(fd, model, model_bytes) == (ssize_t)model_bytes);
    CHECK(ds4_gpu_init());
    /* Set streaming mode before the model map, matching production
     * (ds4.c calls ds4_gpu_set_ssd_streaming() before the model is loaded):
     * ds4_gpu_set_model_map() registers the mapping for the older,
     * unrelated zero-copy resident-access path when streaming is off,
     * which would later collide with this test registering the same
     * pointer for DMA. */
    ds4_gpu_set_ssd_streaming(true);
    CHECK(ds4_gpu_set_model_fd_for_map(fd, model));
    CHECK(ds4_gpu_set_model_map(model, model_bytes));

    float input[ROWS * DIM], weights[ROWS * SELECTED];
    int32_t ids[ROWS * SELECTED];
    for (unsigned i = 0; i < ROWS * DIM; i++) input[i] = ((int)random_byte() - 128) / 512.0f;
    for (unsigned i = 0; i < ROWS * SELECTED; i++) weights[i] = 1.0f / SELECTED;
    for (unsigned i = 0; i < ROWS * SELECTED; i++) ids[i] = (int32_t)((i * 3u) % EXPERTS);
    ds4_gpu_tensor *x = tensor(sizeof(input), input);
    ds4_gpu_tensor *sw = tensor(sizeof(weights), weights);
    ds4_gpu_tensor *si = tensor(sizeof(ids), ids);
    ds4_gpu_tensor *out = tensor((size_t)ROWS * DIM * 4, NULL);
    ds4_gpu_tensor *gate = tensor((size_t)ROWS * SELECTED * DIM * 4, NULL);
    ds4_gpu_tensor *up = tensor((size_t)ROWS * SELECTED * DIM * 4, NULL);
    ds4_gpu_tensor *mid = tensor((size_t)ROWS * SELECTED * DIM * 4, NULL);
    ds4_gpu_tensor *down = tensor((size_t)ROWS * SELECTED * DIM * 4, NULL);

    /* Reference: ordinary --ssd-streaming (pread into a staging buffer),
     * with a cache big enough that nothing is ever evicted mid-run. Compared
     * below against the direct-DMA path driven through the identical cache
     * machinery -- copy source is the only variable being isolated. */
    float reference[LAYERS][ROWS * DIM];
    ds4_gpu_set_dma_streaming(false);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(2 * gate_bytes + down_bytes);
    ds4_gpu_set_streaming_expert_cache_budget(2u * EXPERTS);
    for (unsigned l = 0; l < LAYERS; l++) {
        const uint64_t g = (uint64_t)l * layer_bytes, u = g + EXPERTS * gate_bytes, d = u + EXPERTS * gate_bytes;
        bool half = false;
        CHECK(ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down, model, model_bytes,
            g, u, d, GATE_TYPE, DOWN_TYPE, gate_bytes, UNIT, down_bytes, UNIT,
            DIM, DIM, DIM, si, sw, EXPERTS, SELECTED, 10, x, l, ROWS, &half, false));
        CHECK(!half && ds4_gpu_tensor_read(out, 0, reference[l], (uint64_t)ROWS * DIM * 4));
    }
    ds4_gpu_set_ssd_streaming(false);

    /* Off by default: plain --ssd-streaming semantics never register
     * anything for direct DMA. This is the property the whole flag split
     * exists for -- confirm it directly rather than assume it. */
    ds4_gpu_set_ssd_streaming(true);
    CHECK(!ds4_gpu_dma_streaming_status(NULL));
    ds4_gpu_set_ssd_streaming(false);
    fprintf(stderr, "CUDA DMA streaming: --ssd-streaming alone never registers: PASS\n");

    /* DS4_CUDA_NO_DMA_STREAMING forces a specific decline reason instead of
     * registering, and toggling the mode off and back on (with the
     * override removed) recovers cleanly -- a real bug caught while writing
     * this test: unregister used to leave a prior decline "sticky" and
     * silently skip a later, legitimate registration attempt. */
    CHECK(setenv("DS4_CUDA_NO_DMA_STREAMING", "1", 1) == 0);
    ds4_gpu_set_dma_streaming(true);
    {
        const char *reason = NULL;
        CHECK(!ds4_gpu_dma_streaming_status(&reason));
        CHECK(reason && strstr(reason, "DS4_CUDA_NO_DMA_STREAMING"));
    }
    ds4_gpu_set_dma_streaming(false);
    CHECK(unsetenv("DS4_CUDA_NO_DMA_STREAMING") == 0);
    fprintf(stderr, "CUDA DMA streaming: DS4_CUDA_NO_DMA_STREAMING declines with a specific reason: PASS\n");

    /* The real thing: registration succeeds and every layer's output
     * matches the pread-based reference above, produced through the
     * direct-DMA copy path instead. --dma-streaming always implies
     * ssd_streaming at the engine level (ds4.c); this test calls the
     * ds4_gpu_* setters directly, so it has to set both itself. */
    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_set_dma_streaming(true);
    CHECK(ds4_gpu_dma_streaming_status(NULL));
    ds4_gpu_set_streaming_expert_cache_expert_bytes(2 * gate_bytes + down_bytes);
    ds4_gpu_set_streaming_expert_cache_budget(2u * EXPERTS);
    for (unsigned pass = 0; pass < 3; pass++) for (unsigned k = 0; k < LAYERS; k++) {
        /* Cycle layer order across passes so every slot is evicted and
         * reloaded through the DMA path more than once. */
        const unsigned l = pass & 1 ? LAYERS - 1 - k : k;
        const uint64_t g = (uint64_t)l * layer_bytes, u = g + EXPERTS * gate_bytes, d = u + EXPERTS * gate_bytes;
        bool half = false;
        CHECK(ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down, model, model_bytes,
            g, u, d, GATE_TYPE, DOWN_TYPE, gate_bytes, UNIT, down_bytes, UNIT,
            DIM, DIM, DIM, si, sw, EXPERTS, SELECTED, 10, x, l, ROWS, &half, false));
        float actual[ROWS * DIM];
        CHECK(!half && ds4_gpu_tensor_read(out, 0, actual, (uint64_t)ROWS * DIM * 4));
        for (unsigned i = 0; i < ROWS * DIM; i++)
            CHECK(isfinite(actual[i]) &&
                  fabsf(actual[i] - reference[l][i]) <= 2e-5f * (1 + fabsf(reference[l][i])));
    }
    fprintf(stderr, "CUDA DMA streaming: output through the registered-mapping copy path matches the resident reference: PASS\n");

    ds4_gpu_set_dma_streaming(false);
    CHECK(!ds4_gpu_dma_streaming_status(NULL));
    fprintf(stderr, "CUDA DMA streaming: disabling unregisters cleanly: PASS\n");

    ds4_gpu_cleanup();
    return 0;
}
