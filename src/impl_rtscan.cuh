// =============================================================================
// File: impl_rtscan.cuh
// Author: Justus Henneberg
// Description: RTScan baseline for the FliX benchmarks. Answers range queries on 32-bit keys by
//              casting rays through a triangle-per-key OptiX BVH and marking hits in
//              per-batch bitmaps. Implemented based on the original work by the original
//              authors: Lv et al., "RTScan: Efficient Scan with Ray Tracing Cores",
//              PVLDB 17(6), 2024 (https://www.vldb.org/pvldb/vol17/p1460-lv.pdf).
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef IMPL_RTSCAN_CUH
#define IMPL_RTSCAN_CUH

// -----------------------------------------------------------------------------
// Overview
//   rtscan<> is a read-only baseline index (no updates). build() turns every key into one
//   triangle and builds an OptiX acceleration structure over them. A range lookup launches
//   rays_for_full_range rays per batch of up to batch_size queries; the any-hit program
//   sets one bit per matching key in a bitmap, and the result is the sum over the set bits.
// -----------------------------------------------------------------------------

#include "cuda_buffer.cuh"
#include "utilities.cuh"
#include "optix_wrapper.cuh"
#include "optix_pipeline.cuh"
#include "build_fine_granular.h"


extern "C" char rtscan_embedded_ptx_code[];

// this will be initialized in main.cu
extern optix_wrapper optix;


// Single-thread kernel: writes the build-time fields (acceleration structure handle, stored
// keys, ray spacing, bitmap buffer and size) into the device-side launch parameters.
GLOBALQUALIFIER
void setup_build_data(
    rtscan_params *launch_params,
    OptixTraversableHandle traversable,
    const key32 *stored_keys,
    double ray_spacing,
    uint32_t *bitmaps,
    smallsize bitmap_entries) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1)
        return;

    launch_params->traversable = traversable;
    launch_params->stored_keys = stored_keys;
    launch_params->ray_spacing = ray_spacing;
    launch_params->bitmaps = bitmaps;
    launch_params->bitmap_entries = bitmap_entries;
}


// Single-thread kernel: writes the per-batch query fields (lower/upper bounds, result pointer
// and batch size) into the device-side launch parameters before each optixLaunch.
GLOBALQUALIFIER
void setup_probe_data(
    rtscan_params *launch_params,
    const key32 *query_lower,
    const key32 *query_upper,
    smallsize *result,
    smallsize batch_size) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1)
        return;

    launch_params->query_lower = query_lower;
    launch_params->query_upper = query_upper;
    launch_params->result = result;
    launch_params->batch_size = batch_size;
}


// RTScan index class.
//   rays_for_full_range: number of rays launched to cover the whole key range
//   batch_size:          queries processed per optixLaunch (one bitmap per query in a batch)
template <typename key_type_, uint32_t rays_for_full_range = 18000, size_t batch_size = 32>
class rtscan {
public:
    using key_type = key32;
    static_assert(std::is_same<key_type_, key32>::value);

private:
    cuda_buffer<rtscan_params> launch_params_buffer;
    cuda_buffer<uint8_t> as_buffer;

    cuda_buffer<uint32_t> bitmap_buffer;
    size_t bitmap_entries;

    double ray_spacing;

    std::optional<optix_pipeline> pipeline;

public:
    static constexpr bool can_lookup = true;
    static constexpr bool can_multi_lookup = true;
    static constexpr bool can_range_lookup = true;
    static constexpr bool can_update = false;

    // Short name used in result files, e.g. "rtscan_18000_32".
    static std::string short_description() {
        return std::string("rtscan_") + std::to_string(rays_for_full_range) + "_" + std::to_string(batch_size);
    }

    // Estimates peak GPU memory (bytes) needed to build over `size` keys:
    // triangle buffer + BVH + auxiliary space + bitmaps.
    static size_t estimate_build_bytes(size_t size) {
        size_t triangle_buffer_bytes = size * 9 * sizeof(float);
        size_t bvh_bytes = triangle_buffer_bytes * 13 / 10;
        size_t aux_bytes = triangle_buffer_bytes * 2 / 10;
        size_t bitmap_bytes = (size / 32 + 1) * batch_size * sizeof(uint32_t);
        return bitmap_bytes + std::max(triangle_buffer_bytes + bvh_bytes, 2 * bvh_bytes) + aux_bytes;
    }

    // Returns the GPU memory (bytes) held after build: acceleration structure,
    // launch parameters and bitmaps.
    size_t gpu_resident_bytes() {
        return as_buffer.size_in_bytes() + launch_params_buffer.size_in_bytes() + bitmap_buffer.size_in_bytes();
    }

    // Builds the index over `size` device-resident keys.
    //   1. creates the OptiX pipeline (first call only)
    //   2. computes the ray spacing and an epsilon from the key range
    //   3. creates one triangle per key and builds the acceleration structure
    //   4. allocates the bitmaps and fills the device launch parameters
    // Outputs: build_time_ms and build_bytes (may be null) accumulate time and memory.
    void build(const key_type *keys, size_t size, double *build_time_ms, size_t *build_bytes) {

        if (!pipeline) {
            pipeline.emplace(
                &optix,
                rtscan_embedded_ptx_code,
                "params",
                "__raygen__rg",
                "",
                "__anyhit__get_prim_id",
                "",
                2);
        }

        launch_params_buffer.alloc(1);
        if (build_bytes)
            *build_bytes += launch_params_buffer.size_in_bytes();

        std::vector<key32> data_on_host(size);
        std::vector<float3> triangles_on_host(3 * size);

        // download data since that is what the original code does
        cudaMemcpy(data_on_host.data(), keys, size * sizeof(key32), D2H);

        key32 data_min = *std::min_element(data_on_host.begin(), data_on_host.end());
        key32 data_max = *std::max_element(data_on_host.begin(), data_on_host.end());

        // "rays_for_full_range" is called "width" in the original codebase
        // this is called "aabb_width" and "ray_interval" in the original codebase
        ray_spacing = (data_max - data_min) / rays_for_full_range + 1.0f;

        // taken from the original codebase ("get_epsilon")
        uint32_t epsilon;
        {
            epsilon = (data_max >> 24) << 3;
            if (epsilon != 0) {
                uint32_t and_val = 0;
                for (uint32_t i = 0; i < 32; i++) {
                    and_val = 1 << (31 - i);
                    if (epsilon & and_val) {
                        break;
                    }
                }
                epsilon = and_val;
            } else {
                // +1 is necessary since the triangle cannot be hit on the edge
                // this was not in the original codebase, not sure why it worked without this change
                epsilon = 1;
            }
        }

        // taken from the original codebase ("vertices_to_triangles")
        // no detour of converting to double3 ("uint32_to_double3")
        for (int i = 0; i < size; i++) {
            triangles_on_host[3 * i] = {
                (float)data_on_host[i],
                (float)0,
                (float)(0 + ray_spacing + epsilon)};
            triangles_on_host[3 * i + 1] = {
                (float)(data_on_host[i] + ray_spacing + epsilon),
                (float)0,
                (float)0};
            triangles_on_host[3 * i + 2] = {
                (float)data_on_host[i],
                (float)(0 + ray_spacing + epsilon),
                (float)(0 - ray_spacing - epsilon)};
        }

        cuda_buffer<float3> primitive_buffer;
        primitive_buffer.alloc_and_upload(triangles_on_host);
        auto traversable = consume_primitives_build_traversable(
            optix, size, primitive_buffer, as_buffer, build_time_ms, build_bytes);
        CUERR

        // allocate bitmap buffer
        bitmap_entries = (size + 31) / 32;
        bitmap_buffer.alloc(bitmap_entries * batch_size);
        if (build_bytes)
            *build_bytes += bitmap_buffer.size_in_bytes();

        setup_build_data<<<1, 1>>>(
            launch_params_buffer.ptr(),
            traversable,
            keys,
            ray_spacing,
            bitmap_buffer.ptr(),
            bitmap_entries);

        cudaDeviceSynchronize();
        CUERR
    }

    // Point lookup: not supported, because misses cannot be detected. Intentionally a no-op.
    void lookup(const key_type *keys, smallsize *result, size_t size, cudaStream_t stream) {
        // we cannot determine misses, so this cannot be implemented for now
    }

    // Multi-lookup: sum of values for each key, run as a range query with lower == upper.
    void multi_lookup_sum(const key_type *keys, smallsize *result, size_t size, cudaStream_t stream) {
        range_lookup_sum(keys, keys, result, size, stream);
    }

    // Range lookup: for each of `size` queries [lower[i], upper[i]] writes the sum of the
    // matching values to result[i]. Queries run in batches of batch_size: reset the bitmaps,
    // set the batch's launch parameters, then launch rays_for_full_range rays.
    void range_lookup_sum(const key_type *lower, const key_type *upper, smallsize *result, size_t size, cudaStream_t stream) {
        // taken from "refineWithOptixRTc1" in "rtc1.cpp"
        // hard-coded all variables that are constant in 1d, and also added batching
        uint32_t launch_width = rays_for_full_range;

        cudaMemsetAsync(result, 0, size * sizeof(smallsize), stream);

        for (size_t batch_offset = 0; batch_offset < size; batch_offset += batch_size) {
            // reset bitmaps
            cudaMemsetAsync(bitmap_buffer.ptr(), 0, batch_size * bitmap_entries * sizeof(uint32_t), stream);

            size_t this_batch_size = std::min(batch_size, size - batch_offset);
            setup_probe_data<<<1, 1, 0, stream>>>(
                launch_params_buffer.ptr(),
                lower + batch_offset,
                upper + batch_offset,
                result + batch_offset,
                this_batch_size);

            OPTIX_CHECK(optixLaunch(
                pipeline->pipeline,
                stream,
                launch_params_buffer.cu_ptr(),
                launch_params_buffer.size_in_bytes(),
                &pipeline.value().sbt,
                launch_width,
                1,
                1))
        }
    }

    // Frees the launch parameters, acceleration structure and bitmap buffers.
    void destroy() {
        launch_params_buffer.free();
        as_buffer.free();
        bitmap_buffer.free();
    }

    // Updates are not supported (can_update == false); insert and remove are no-ops.
    void insert(const key_type *update_list, const smallsize *offsets, size_t size, cudaStream_t stream) {}
    void remove(const key_type *update_list, size_t size, cudaStream_t stream) {}
};

// relevant parameters for launch (from "special_eq_scan" in "rtc1.cpp" and "RTc1_1c_2p32" in "run.py")
//
//    int column_num = 3; // set by "-b 3" argument in "run.py", assigned as "bindex_num", passed along
//    int ray_length = -1; // set by "-a -1" argument in "run.py", passed along
//    int ray_segment_num = 1; // set by "-s 1" argument in "run.py", passed along
//    bool inverse = false; // passed explicitly in "rtc1.cpp"
//    int ray_mode = 0; // the default (not overridden by command line arguments), passed along
//    int direction = 1; // passed explicitly in "rtc1.cpp"
//    PRIMITIVE_TYPE=0 // means "triangle", during compilation in "run.py"

#endif
