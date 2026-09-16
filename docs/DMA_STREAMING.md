# DMA Streaming: Routed Experts By DMA, Not Disk

[README](../README.md)

`--dma-streaming` is a discrete-CUDA-only mode: it registers the whole model
mapping with the CUDA driver once at startup and DMAs routed-expert cache
misses straight out of the host page cache, with no disk read on the miss
path at all. It needs the model to fit in host RAM.

This is a different thing from [SSD streaming](SSD_STREAMING.md), not a
tuning of it, despite both keeping a bounded device-side cache of routed
experts. SSD streaming's job is running a model *larger than RAM*: a miss
reads from disk, and the flag's name says so. DMA streaming's job is running
a model that fits in RAM but not VRAM, as fast as a PCIe-attached card can:
once the mapping is registered, the file is never read again -- misses come
from RAM over PCIe, not from storage. Pick the one that matches your
constraint (model fits in RAM but not VRAM: `--dma-streaming`; model doesn't
fit in RAM either: `--ssd-streaming`) -- or, when the model is bigger than
RAM but not by an amount that leaves nothing worth pinning, pass both
together for [hybrid streaming](#hybrid-ram-tier-plus-disk-tier).

## Use it

```sh
./ds4 --cuda -m ds4flash.gguf --dma-streaming --ctx 32768
```

The one condition: host RAM available to the process must cover the whole
GGUF. If it doesn't, or the card is integrated (Grace, Jetson, Strix Halo --
where the GPU already reaches host memory and this mode has nothing to add),
startup refuses with a specific reason rather than silently falling back to
something slower:

```
ds4: --dma-streaming could not be enabled: model does not fit host RAM (80.76 GiB model, 61.20 GiB available)
```

Like SSD streaming, the expert-cache budget is picked automatically from
free VRAM by default; `--ssd-streaming-cache-experts N|NGB` requests an
explicit one instead, subject to the same fit.

## Hybrid: RAM tier plus disk tier

For a model bigger than host RAM, pass `--dma-streaming` and
`--ssd-streaming` together:

```sh
./ds4 --cuda -m ds4flash.gguf --dma-streaming --ssd-streaming --ctx 32768
```

This is a third mode, not both modes running at once: DMA-register only the
routed-expert layers that fit host RAM, and `pread()` the rest, in the same
bounded device-side expert cache either transport would use alone. Startup
picks whole routed-expert layers, from layer 0 upward, until adding the next
one would exceed the host RAM budget -- available memory minus a 16 GiB
reserve by default (pinned pages are not reclaimable, so this reserve is
real memory held back, not a cushion), or an explicit
`--dma-host-cache NGB`. It reports what it picked:

```
ds4: hybrid streaming: 26 of 40 routed-expert layers pinned (94.00 GiB host RAM budget)
```

A cache miss for an expert inside a pinned layer takes the direct-DMA path
above; a miss for anything else falls back to the ordinary `pread()` +
staging path. Both are byte-identical to a fully resident run at `--temp 0`
-- the split changes throughput, not output.

Picking whole layers by ascending index is a starting point, not a claim
that index order is the ideal one to pin: a future refinement could pick by
measured expert popularity instead (see `ds4_streaming_hotlist.inc`), but
that data doesn't exist for every model yet, and layer order already gets
most of the benefit since every layer is used on every token regardless of
which of its experts are selected.

## Why this and not just faster SSD streaming

A cache miss under the ordinary streaming path is `pread()` into a small
pinned staging buffer, then `cudaMemcpyAsync` to the device -- a bounce
through host userspace on every miss. Registering the mapping with
`cudaHostRegister(map, size, cudaHostRegisterReadOnly)` removes that bounce
entirely: the miss copy comes straight from the registered, page-cache-backed
mapping, and there is no reusable buffer to protect, so there is no per-job
wait for one to be safe to reuse either.

That last point turned out to be the actual win. The first version of this
work tried the more obvious fix -- a small thread pool doing the `pread()`s
in parallel -- and confirmed by direct measurement that the real bottleneck
was not disk bandwidth or the number of copy calls: it was GPU-side
synchronization contention, `cudaEventSynchronize`/`cudaStreamSynchronize`
calls made concurrently with the model's own heavy kernel-launch traffic
during decode and prefill. A reads-only thread with no CUDA call at all held
~27 GB/s next to a busy compute kernel on another stream; the moment that
same thread made *any* synchronization call on its own stream, throughput
collapsed to 5-10 GB/s -- with zero disk I/O in the repro. Removing the wait
entirely, rather than trying to manage the contention around it, is what
direct DMA does.

Registering the whole mapping was tried and rejected once before, for a
concrete reason: on one measurement it cost 7.4 GiB of VRAM to pin an 81 GiB
mapping, which would have starved the expert cache's own sizing budget
outright. That number turned out to be a property of the specific
registration flag used, not of registration as a technique --
`cudaHostRegisterDefault` fails outright against a `PROT_READ` mmap on the
driver this was retested on, and `cudaHostRegisterReadOnly` measured 0.158
GiB for the same 81 GiB mapping. The code does not trust either number: it
re-measures its own cost via `cudaMemGetInfo()` before and after registering,
every time, and declines -- with a specific reason available through
`--dma-streaming`'s startup check -- if a different driver needs more than a
2 GiB budget.

## Measured

RTX 3090 (24 GB, sm_86), 125 GB system RAM, DeepSeek V4 Flash Q2 (IQ2_XXS,
80.8 GiB). Recorded numbers, the exact sweep command, and the raw CSV are in
[PERFORMANCE.md](PERFORMANCE.md#dma-streaming-on-a-single-discrete-card) and
[speed-bench/rtx3090.csv](../speed-bench/rtx3090.csv).

## Automatic cache sizing

CUDA's expert-cache auto-sizing (picking a budget from free VRAM instead of
requiring an explicit `--ssd-streaming-cache-experts`) was validated for
DeepSeek V4.1 Flash and never extended to plain V4 (Flash/PRO) -- an
oversight, not a finding that it is unsafe there. `--dma-streaming` admits
plain V4 too; without that, a `--dma-streaming` run with no explicit budget
would leave the cache sized to whatever the largest single request happens
to be (one full layer's worth of experts on an ordinary prefill chunk), and
a fresh layer's experts would evict the previous layer's on every decode
step -- a 0% hit rate. Plain `--ssd-streaming` on CUDA V4/PRO is unchanged by
this and still needs the explicit flag, since the same fix widened for that
case has not been validated the same way.

## VRAM reserve

A card with little VRAM margin can see the auto-sized expert-cache budget
compete with two other CUDA allocators that do not back off gracefully:
`cuda_tmp_alloc()`'s shared scratch slab (grown to whatever the largest
per-layer temporary has needed so far) and the on-demand non-routed weight
arena, which has no fixed ceiling. Under whole-file `--dma-streaming` the
reserve subtracted before sizing the cache is 10 GiB, raised there through
two real failures on this card: 8 GiB (the reserve plain `--ssd-streaming`
still uses, unchanged) OOM'd through `ds4-server` on a short prompt; 9 GiB
left 0.19 GiB free on an 8355-token tool-calling prefill; 10 GiB left
3.47+ GiB, validated at `--ctx 32768` and a genuine ~100K-token prompt at
`--ctx 131072`. Hybrid streaming uses the plain 8 GiB reserve, not the
10 GiB one: its non-routed resident footprint differs per model, so the
figure tuned specifically for whole-file V4 Flash DMA isn't known to
transfer without its own validation. `DS4_CUDA_SSD_CACHE_RESERVE_MB`
overrides either reserve without a rebuild if a request fails with "cannot
stage N experts with system headroom" or a later out-of-memory during
prefill.

## Diagnostics

`DS4_CUDA_NO_DMA_STREAMING=1` forces `--dma-streaming` to fail its startup
check instead of registering -- useful for isolating whether a regression is
in the DMA path itself or elsewhere in streaming. There is no fallback mode:
this is a diagnostic to make the failure explicit and reproducible, not a way
to keep running without DMA (use `--ssd-streaming` for that; see "Why this
and not just faster SSD streaming" above for what changes between the two
transports). `DS4_CUDA_KEEP_MODEL_PAGES=1` / `DS4_CUDA_DROP_MODEL_PAGES=1`
force the host-page-cache decision either way, mostly useful for measurement
since the automatic decision already tracks whether the model fits.
`DS4_CUDA_WEIGHT_CACHE_VERBOSE=1` logs the page-cache and registration
decisions as they're made.

## Where this applies

CUDA streaming-expert-cache misses only, on a discrete GPU. It does not
touch Metal streaming, ROCm streaming (the `cudaHostRegisterReadOnly`
behavior this relies on is untested there and stays out even under a ROCm
build), the non-streaming resident-weights path, or the background prefetch
thread that predicts and pre-reads the next layer's experts while the
current one computes -- that mechanism is unchanged and still helps hide a
miss regardless of which transport serves it.
