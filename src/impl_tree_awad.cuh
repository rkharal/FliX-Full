// =============================================================================
// File: impl_tree_awad.cuh
// Author: Justus Henneberg
// Description: Implements impl_tree_awad     
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef IMPL_TREE_AWAD_CUH
#define IMPL_TREE_AWAD_CUH

#include "gpu_btree.h"

#include <nvtx3/nvtx3.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

using GpuBTree::gpu_blink_tree;

// for nvtx
struct nvtx_tree_awad_domain{ static constexpr char const* name{"tree_awad"}; };


template <typename key_type, typename size_type, typename btree>
GLOBALQUALIFIER
void modified_find_kernel(
    const key_type* keys,
    smallsize* results,
    const size_type keys_count,
    btree tree,
    bool concurrent = false
) {
  auto thread_id  = threadIdx.x + blockIdx.x * blockDim.x;

  auto block = cooperative_groups::this_thread_block();
  auto tile  = cooperative_groups::tiled_partition<btree::branching_factor>(block);

  if ((thread_id - tile.thread_rank()) >= keys_count) { return; }

  auto key     = btree::invalid_key;
  auto value   = btree::invalid_value;
  bool to_find = false;
  if (thread_id < keys_count) {
    key     = keys[thread_id];
    to_find = true;
  }

  using allocator_type = device_allocator_context<typename btree::allocator_type>;
  allocator_type allocator{tree.allocator_, tile};

  auto work_queue = tile.ballot(to_find);
  while (work_queue) {
    auto cur_rank = __ffs(work_queue) - 1;
    auto cur_key  = tile.shfl(key, cur_rank);
    smallsize cur_result;
    cur_result = tree.cooperative_find(cur_key, tile, allocator, concurrent);
    if (cur_rank == tile.thread_rank()) {
      value   = cur_result;
      to_find = false;
    }
    work_queue = tile.ballot(to_find);
  }

  if (thread_id < keys_count) {
    results[thread_id] = value != std::numeric_limits<uint32_t>::max() ? value : not_found;
  }
}


template <typename key_type, typename size_type, typename btree>
GLOBALQUALIFIER
void modified_range_lookup_kernel(
    const key_type* lower_bounds,
    const key_type* upper_bounds,
    smallsize* results,
    const size_type keys_count,
    btree tree,
    bool concurrent = false
) {
    auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    auto block                = cooperative_groups::this_thread_block();
    auto tile                 = cooperative_groups::tiled_partition<btree::branching_factor>(block);
    auto tile_id              = thread_id / btree::branching_factor;
    auto first_tile_thread_id = tile_id * btree::branching_factor;

    if ((thread_id - tile.thread_rank()) >= keys_count) { return; }

    auto lower_bound = btree::invalid_key;
    auto upper_bound = btree::invalid_key;

    bool to_find    = false;
    if (thread_id < keys_count) {
      lower_bound = lower_bounds[thread_id];
      upper_bound = upper_bounds[thread_id];
      to_find     = true;
    }

    using allocator_type = device_allocator_context<typename btree::allocator_type>;
    allocator_type allocator{tree.allocator_, tile};

    smallsize result = 0;
    auto work_queue = tile.ballot(to_find);
    while (work_queue) {
      smallsize cur_result = 0;
      auto cur_rank        = __ffs(work_queue) - 1;
      auto cur_lower_bound = tile.shfl(lower_bound, cur_rank);
      auto cur_upper_bound = tile.shfl(upper_bound, cur_rank);
      auto local_result = tree.modified_aggregating_cooperative_range_query(
        cur_lower_bound,
        cur_upper_bound,
        tile,
        allocator,
        concurrent);

      cur_result = cooperative_groups::reduce(tile, local_result, cooperative_groups::plus<smallsize>());

      if (cur_rank == tile.thread_rank()) {
        result = cur_result;
        to_find = false;
      }
      work_queue = tile.ballot(to_find);
    }

    if (thread_id < keys_count) {
        results[thread_id] = result;
    }
}


template <typename key_type, typename size_type, typename btree>
GLOBALQUALIFIER
void modified_insert_kernel(
    const key_type* keys,
    const size_type keys_count,
    btree tree
) {
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  auto block     = cooperative_groups::this_thread_block();
  auto tile      = cooperative_groups::tiled_partition<btree::branching_factor>(block);

  if ((thread_id - tile.thread_rank()) >= keys_count) { return; }

  auto key       = btree::invalid_key;
  auto value     = btree::invalid_value;
  bool to_insert = false;
  if (thread_id < keys_count) {
    key       = keys[thread_id];
    value     = thread_id;
    to_insert = true;
  }
  using allocator_type = typename btree::device_allocator_context_type;
  allocator_type allocator{tree.allocator_, tile};

  size_type num_inserted = 1;
  auto work_queue        = tile.ballot(to_insert);
  while (work_queue) {
    auto cur_rank  = __ffs(work_queue) - 1;
    auto cur_key   = tile.shfl(key, cur_rank);
    auto cur_value = tile.shfl(value, cur_rank);

    tree.cooperative_insert(cur_key, cur_value, tile, allocator);

    if (tile.thread_rank() == cur_rank) { to_insert = false; }
    num_inserted++;
    work_queue = tile.ballot(to_insert);
  }
}


// Task codes for the fully concurrent (mixed batch) kernel, one per thread.
enum awad_task : uint8_t {
  awad_task_insert = 0,
  awad_task_lookup = 1,
  awad_task_delete = 2,
  awad_task_range  = 3
};


// Fully concurrent mixed-batch kernel: Algorithm 1 (WCWS) of Awad et al., PPoPP 2019.
// Each thread holds its own key and task; the tile serves one lane at a time in lane order,
// and performTask is a switch over the broadcast task.
//   keys         : key of each thread's operation
//   values       : value to insert (only read for inserts)
//   tasks        : operation of each thread (awad_task: insert, lookup or delete; range unused)
//   results      : per thread: insert/delete -> 1 on success, 0 on failure;
//                  lookup -> found value or not_found
template <typename key_type, typename size_type, typename btree>
GLOBALQUALIFIER
void modified_mixed_kernel(
    const key_type* keys,
    const smallsize* values,
    // const key_type* upper_bounds,   // removed: range queries are not run concurrently
    const uint8_t* tasks,
    smallsize* results,
    const size_type keys_count,
    btree tree
) {
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  auto block     = cooperative_groups::this_thread_block();
  auto tile      = cooperative_groups::tiled_partition<btree::branching_factor>(block);

  if ((thread_id - tile.thread_rank()) >= keys_count) { return; }

  auto key         = btree::invalid_key;
  auto value       = btree::invalid_value;
  uint32_t task    = awad_task_lookup;
  bool is_active   = false;
  if (thread_id < keys_count) {
    key       = keys[thread_id];
    task      = tasks[thread_id];
    is_active = true;
    if (task == awad_task_insert) { value = values[thread_id]; }
  }

  using allocator_type = typename btree::device_allocator_context_type;
  allocator_type allocator{tree.allocator_, tile};

  auto work_queue = tile.ballot(is_active);
  while (work_queue) {
    auto cur_rank = __ffs(work_queue) - 1;
    auto cur_key  = tile.shfl(key, cur_rank);
    auto cur_task = tile.shfl(task, cur_rank);

    // performTask
    smallsize cur_result = 0;
    switch (cur_task) {
      case awad_task_insert: {
        auto cur_value = tile.shfl(value, cur_rank);
        cur_result = tree.cooperative_insert(cur_key, cur_value, tile, allocator) ? 1 : 0;
        break;
      }
      case awad_task_lookup: {
        auto found = tree.cooperative_find(cur_key, tile, allocator, true);
        cur_result = found != std::numeric_limits<uint32_t>::max() ? found : not_found;
        break;
      }
      case awad_task_delete: {
        cur_result = tree.cooperative_erase(cur_key, tile, allocator, true) ? 1 : 0;
        break;
      }
    }

    if (cur_rank == tile.thread_rank()) {
      results[thread_id] = cur_result;
      is_active = false;
    }
    work_queue = tile.ballot(is_active);
  }
}


/*
inline void test_modified_btree() {
    std::vector<uint32_t> keys {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<uint32_t> offsets(keys.size());
    std::vector<uint32_t> probes       {1, 4, 5, 2, 1, 3, 7, 10};
    std::vector<uint32_t> probes_upper {5, 5, 5, 2, 1, 3, 8, 20};
    std::vector<uint32_t> results(probes.size());
    cuda_buffer<uint32_t> kb, ob, pb, pub, rb;
    std::iota(offsets.begin(), offsets.end(), 0);
    kb.alloc_and_upload(keys);
    ob.alloc_and_upload(offsets);
    pb.alloc_and_upload(probes);
    pub.alloc_and_upload(probes_upper);
    rb.alloc_and_upload(results);

    using btree = gpu_blink_tree<uint32_t, uint32_t>;

    btree tree;
    tree.bulk_load(kb, ob, keys.size(), true);
    cudaDeviceSynchronize(); C2EX
    //modified_insert_kernel<<<SDIV(keys.size(), 512), 512>>>(kb, keys.size(), tree);
    //modified_find_kernel<<<SDIV(probes.size(), 512), 512>>>(pb, rb, probes.size(), tree);
    modified_range_lookup_kernel<<<SDIV(probes.size(), 512), 512>>>(pb, pub, rb, probes.size(), tree);
    cudaDeviceSynchronize(); C2EX

    rb.download_and_output(results.size());
}
*/



// -----------------------------------------------------------------------------
// Prefix-sum range aggregation for the B-tree (BTREE_PREFIX_RANGE_QUERIES)
//
// Precompute (all inside range_lookup_sum, so it is timed):
//   1. level-order expansion from the root: for each level, count each node's
//      children, exclusive-scan the counts, and write the children in lane
//      (= key) order. All leaves are at the same depth, so the last frontier is
//      exactly the leaves in key order -- no sort needed;
//   2. per-leaf value sums in that order, exclusive scan, scatter back so that
//      leaf_excl_prefix[node] = sum of all values in leaves before `node`.
// Query (one 16-lane tile per query, like the existing kernels):
//   result = P(hi) - P(lo - 1), P(x) = leaf_excl_prefix[leaf(x)] + values of keys
//   <= x in leaf(x), where leaf(x) is found by an ordinary root-to-leaf descent.
// Sums are 32-bit modular, matching the 32-bit result type.
// -----------------------------------------------------------------------------
namespace btree_prefix {
constexpr uint32_t node_width    = 16;
constexpr uint32_t metadata_lane = node_width - 1;
constexpr uint32_t leaf_bit_mask = 1u << 30;

template <typename size_t_>
GLOBALQUALIFIER void init_frontier_kernel(const size_t_* root_index, uint32_t* frontier) {
    frontier[0] = static_cast<uint32_t>(*root_index);
}

// children per frontier node (0 for leaves); flags the level as leaf level
template <typename pair_t>
GLOBALQUALIFIER void count_children_kernel(const pair_t* nodes, const uint32_t* frontier, uint32_t frontier_size,
                                           uint32_t* child_counts, uint32_t* leaf_level_flag) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= frontier_size) return;
    const pair_t* node = nodes + size_t(frontier[i]) * node_width;
    uint32_t count = 0;
    if (node[metadata_lane].second & leaf_bit_mask) {
        *leaf_level_flag = 1;
    } else {
        for (uint32_t lane = 0; lane < metadata_lane; ++lane)
            count += (node[lane].first != 0xFFFFFFFFu);
    }
    child_counts[i] = count;
}

template <typename pair_t>
GLOBALQUALIFIER void expand_children_kernel(const pair_t* nodes, const uint32_t* frontier, uint32_t frontier_size,
                                            const uint32_t* child_offsets, uint32_t* next_frontier) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= frontier_size) return;
    const pair_t* node = nodes + size_t(frontier[i]) * node_width;
    uint32_t out = child_offsets[i];
    for (uint32_t lane = 0; lane < metadata_lane; ++lane) {
        const auto p = node[lane];
        if (p.first != 0xFFFFFFFFu) next_frontier[out++] = p.second;
    }
}

template <typename pair_t>
GLOBALQUALIFIER void leaf_sums_in_order_kernel(const pair_t* nodes, uint32_t node_count,
                                               const uint32_t* sorted_node_ids, smallsize* sums) {
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= node_count) return;
    const pair_t* node = nodes + size_t(sorted_node_ids[j]) * node_width;
    smallsize sum = 0;
    if (node[metadata_lane].second & leaf_bit_mask) {
        for (uint32_t lane = 0; lane < metadata_lane; ++lane) {
            const auto p = node[lane];
            if (p.first != 0xFFFFFFFFu) sum += p.second;
        }
    }
    sums[j] = sum;
}

GLOBALQUALIFIER void scatter_prefix_kernel(uint32_t node_count, const uint32_t* sorted_node_ids,
                                           const smallsize* excl_sorted, smallsize* leaf_excl_prefix) {
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= node_count) return;
    leaf_excl_prefix[sorted_node_ids[j]] = excl_sorted[j];
}
} // namespace btree_prefix

template <typename key_type, typename size_type, typename btree>
GLOBALQUALIFIER
void modified_prefix_range_lookup_kernel(
    const key_type* lower_bounds,
    const key_type* upper_bounds,
    smallsize* results,
    const size_type keys_count,
    const smallsize* leaf_excl_prefix,
    btree tree
) {
    auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    auto block = cooperative_groups::this_thread_block();
    auto tile  = cooperative_groups::tiled_partition<btree::branching_factor>(block);

    if ((thread_id - tile.thread_rank()) >= keys_count) { return; }

    key_type lower_bound = 0, upper_bound = 0;
    bool to_find = false;
    if (thread_id < keys_count) {
        lower_bound = lower_bounds[thread_id];
        upper_bound = upper_bounds[thread_id];
        to_find     = true;
    }

    using allocator_type = device_allocator_context<typename btree::allocator_type>;
    allocator_type allocator{tree.allocator_, tile};

    smallsize result = 0;
    auto work_queue = tile.ballot(to_find);
    while (work_queue) {
        auto cur_rank  = __ffs(work_queue) - 1;
        auto cur_lower = tile.shfl(lower_bound, cur_rank);
        auto cur_upper = tile.shfl(upper_bound, cur_rank);

        smallsize lane_value = 0;
        if (cur_lower <= cur_upper) {
            lane_value = tree.modified_prefix_sum_upto_lane(cur_upper, tile, allocator, leaf_excl_prefix);
            if (cur_lower > key_type(0)) {
                lane_value -= tree.modified_prefix_sum_upto_lane(
                    static_cast<key_type>(cur_lower - 1), tile, allocator, leaf_excl_prefix);
            }
        }
        const smallsize cur_result =
            cooperative_groups::reduce(tile, lane_value, cooperative_groups::plus<smallsize>());

        if (cur_rank == tile.thread_rank()) {
            result  = cur_result;
            to_find = false;
        }
        work_queue = tile.ballot(to_find);
    }

    if (thread_id < keys_count) {
        results[thread_id] = result;
    }
}

template <typename key_type_, bool bulk_load_full = false>
class tree_awad {
    static_assert(std::is_same<key_type_, key32>::value, "key must be 32 bits wide");

public:
    using key_type = key_type_;

private:
    using tree_type = gpu_blink_tree<key_type, smallsize, bulk_load_full>;
    std::optional<tree_type> wrapped_tree;

    // BTREE_PREFIX_RANGE_QUERIES buffers, sized for the allocator's full node capacity
    size_t prefix_node_capacity = 0;
    cuda_buffer<uint32_t> prefix_sort_keys_in, prefix_sort_keys_out;
    cuda_buffer<uint32_t> prefix_node_ids_in, prefix_node_ids_out;
    cuda_buffer<smallsize> prefix_sums_sorted;
    cuda_buffer<smallsize> prefix_leaf_excl;
    cuda_buffer<uint8_t> prefix_temp;
    cuda_buffer<uint32_t> prefix_flag;

public:
   static constexpr const char* name = "tree_awad";
    static constexpr operation_support can_lookup = operation_support::async;
    static constexpr operation_support can_multi_lookup = operation_support::none;
    static constexpr operation_support can_range_lookup = operation_support::async;
    static constexpr operation_support can_update = operation_support::async;

    static std::string short_description() {
        std::string desc = "tree_awad";
        if constexpr (bulk_load_full) {
            desc += "_full";
        } else {
            desc += "_half";
        }
        return desc;
    }

    static parameters_type parameters() {
        return {
                {"node_size", "16"},
                {"bulk_load", bulk_load_full ? "full" : "half"}
        };
    }

    size_t gpu_resident_bytes() {
        return wrapped_tree.value().compute_memory_usage_bytes() +
               prefix_sort_keys_in.size_in_bytes() + prefix_sort_keys_out.size_in_bytes() +
               prefix_node_ids_in.size_in_bytes() + prefix_node_ids_out.size_in_bytes() +
               prefix_sums_sorted.size_in_bytes() + prefix_leaf_excl.size_in_bytes() +
               prefix_temp.size_in_bytes() + prefix_flag.size_in_bytes();
    }

    void build(const key_type* keys, size_t size, size_t max_size, size_t available_memory_bytes, double* build_time_ms, size_t* build_bytes) {
        cuda_buffer<key_type> sorted_keys_buffer;
        cuda_buffer<smallsize> sorted_offsets_buffer;

        {
            cuda_buffer<uint8_t> temp_buffer;
            cuda_buffer<smallsize> offsets_buffer;
            sorted_keys_buffer.alloc(size);
            offsets_buffer.alloc(size);
            sorted_offsets_buffer.alloc(size);
            init_offsets(offsets_buffer, size, build_time_ms);

            cudaDeviceSynchronize(); C2EX

            size_t temp_storage_bytes = find_pair_sort_buffer_size<key_type, smallsize>(size);
            temp_buffer.alloc(temp_storage_bytes);
            timed_pair_sort(
                temp_buffer.raw_ptr, temp_storage_bytes,
                keys, sorted_keys_buffer.ptr(), offsets_buffer.ptr(), sorted_offsets_buffer.ptr(), size, build_time_ms);

            if (build_bytes) *build_bytes += sorted_keys_buffer.size_in_bytes() + sorted_offsets_buffer.size_in_bytes() + temp_buffer.size_in_bytes() + offsets_buffer.size_in_bytes();

            cudaDeviceSynchronize(); C2EX
        }

        wrapped_tree.emplace(size, max_size); C2EX
#ifdef BTREE_PREFIX_RANGE_QUERIES
        {
            prefix_node_capacity = tree_type::nodes_to_allocate(size, max_size);
            const size_t n = prefix_node_capacity;
            prefix_sort_keys_in.alloc(n); prefix_sort_keys_out.alloc(n);
            prefix_node_ids_in.alloc(n);  prefix_node_ids_out.alloc(n);
            prefix_sums_sorted.alloc(n);  prefix_leaf_excl.alloc(n);
            size_t scan_bytes = 0;
            cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, (const smallsize*)nullptr, (smallsize*)nullptr, n);
            prefix_temp.alloc(scan_bytes);
            prefix_flag.alloc(1);
            C2EX
        }
#endif
        {
            scoped_cuda_timer timer(0, build_time_ms);
            wrapped_tree.value().bulk_load(sorted_keys_buffer, sorted_offsets_buffer, size, true, 0);
            //modified_insert_kernel<<<SDIV(size, 512), 512>>>(keys, size, wrapped_tree.value());
        }
        if (build_bytes) *build_bytes += gpu_resident_bytes();

        cudaDeviceSynchronize(); C2EX
    }

    void lookup(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {
        nvtx3::scoped_range_in<nvtx_tree_awad_domain> launch{"launch"};
        modified_find_kernel<<<SDIV(size, 512), 512, 0, stream>>>(keys, result, size, wrapped_tree.value());
    }

    void multi_lookup_sum(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {}

    void range_lookup_sum(const key_type* lower, const key_type* upper, smallsize* result, size_t size, cudaStream_t stream) {
#ifdef BTREE_PREFIX_RANGE_QUERIES
        {
            auto& tree = wrapped_tree.value();
            // node count lives on the device (bump allocator); one small copy per batch
            const uint32_t node_count = tree.get_num_tree_node();
            if (node_count > prefix_node_capacity) { std::cerr << "btree prefix: node capacity exceeded" << std::endl; std::abort(); }
            const auto* nodes = tree.modified_raw_node_pairs();
            constexpr uint32_t tpb = 256;

            // level-order expansion: the final frontier is the leaves in key order
            uint32_t* frontier = prefix_node_ids_in.ptr();
            uint32_t* next_frontier = prefix_node_ids_out.ptr();
            uint32_t* counts = prefix_sort_keys_in.ptr();
            uint32_t* offsets = prefix_sort_keys_out.ptr();
            uint32_t* d_flag = prefix_flag.ptr();
            btree_prefix::init_frontier_kernel<<<1, 1, 0, stream>>>(tree.modified_root_index_ptr(), frontier);
            uint32_t frontier_size = 1;
            size_t temp_bytes = 0;
            while (true) {
                cudaMemsetAsync(d_flag, 0, sizeof(uint32_t), stream);
                const uint32_t fblocks = SDIV(frontier_size, tpb);
                btree_prefix::count_children_kernel<<<fblocks, tpb, 0, stream>>>(
                    nodes, frontier, frontier_size, counts, d_flag);
                temp_bytes = prefix_temp.size_in_bytes();
                cub::DeviceScan::ExclusiveSum(prefix_temp.raw_ptr, temp_bytes, counts, offsets, frontier_size, stream);
                uint32_t host_vals[3];
                cudaMemcpyAsync(&host_vals[0], d_flag, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(&host_vals[1], offsets + frontier_size - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(&host_vals[2], counts + frontier_size - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                if (host_vals[0]) break; // frontier is the leaf level
                const uint32_t next_size = host_vals[1] + host_vals[2];
                btree_prefix::expand_children_kernel<<<fblocks, tpb, 0, stream>>>(
                    nodes, frontier, frontier_size, offsets, next_frontier);
                std::swap(frontier, next_frontier);
                frontier_size = next_size;
            }
            const uint32_t leaf_count = frontier_size;
            const uint32_t lblocks = SDIV(leaf_count, tpb);
            btree_prefix::leaf_sums_in_order_kernel<<<lblocks, tpb, 0, stream>>>(
                nodes, leaf_count, frontier, prefix_sums_sorted.ptr());
            temp_bytes = prefix_temp.size_in_bytes();
            cub::DeviceScan::ExclusiveSum(prefix_temp.raw_ptr, temp_bytes,
                prefix_sums_sorted.ptr(), prefix_sums_sorted.ptr(), leaf_count, stream);
            btree_prefix::scatter_prefix_kernel<<<lblocks, tpb, 0, stream>>>(
                leaf_count, frontier, prefix_sums_sorted.ptr(), prefix_leaf_excl.ptr());

            modified_prefix_range_lookup_kernel<<<SDIV(size, 512), 512, 0, stream>>>(
                lower, upper, result, size, prefix_leaf_excl.ptr(), tree);
            return;
        }
#endif
        modified_range_lookup_kernel<<<SDIV(size, 512), 512, 0, stream>>>(lower, upper, result, size, wrapped_tree.value());
    }

    void destroy() {
        wrapped_tree.reset();
    }

    void insert(const key_type* update_list, const smallsize* offsets, size_t size, cudaStream_t stream) {
        wrapped_tree.value().insert(update_list, offsets, size, stream);
    }

    void remove(const key_type* update_list, size_t size, cudaStream_t stream) {
        wrapped_tree.value().erase(update_list, size, stream);
    }

    // Fully concurrent mixed batch: one kernel, each thread performs the operation in tasks[]
    // (see awad_task). values is read for inserts; result is written for every
    // operation (insert/delete: 1 success, 0 failure; lookup: value or not_found).
    // void mixed_batch(const key_type* keys, const smallsize* values, const key_type* upper_bounds, const uint8_t* tasks, smallsize* result, size_t size, cudaStream_t stream) {
    void mixed_batch(const key_type* keys, const smallsize* values, const uint8_t* tasks, smallsize* result, size_t size, cudaStream_t stream, size_t round_number = 0) {
        (void)round_number; // Same API as FliX; B-tree does not use round numbers.
        modified_mixed_kernel<<<SDIV(size, 512), 512, 0, stream>>>(keys, values, tasks, result, size, wrapped_tree.value());
    }

     void lookups_successor(const key_type* keys, key_type* result, size_t size, cudaStream_t stream) {
        return;
    }

};

#endif
