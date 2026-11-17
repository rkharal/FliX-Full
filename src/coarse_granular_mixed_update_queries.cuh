// =============================================================================
// File: coarse_granular_mixed_update_queries.cuh
// Author: Rosina Kharal
// Description: Standalone tiled kernels for mixed insert/query and
//              delete/query batches in FliX.
// Copyright (c) 2026 Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef INDEX_PROTOTYPE_COARSE_GRANULAR_MIXED_UPDATE_QUERIES_CUH
#define INDEX_PROTOTYPE_COARSE_GRANULAR_MIXED_UPDATE_QUERIES_CUH

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "coarse_granular_inserts_tiles.cuh"

/*
 * Mixed batch encoding
 * --------------------
 * The most-significant bit identifies an update.  A clear MSB identifies a
 * query.  The remaining bits are the real data-structure key.  The host-side
 * benchmark guarantees that real keys do not use the reserved MSB and sorts
 * the batch by the decoded key.
 */
template <typename key_type>
DEVICEQUALIFIER constexpr key_type mixed_device_operation_bit() {
  static_assert(std::is_integral_v<key_type>,
                "Mixed-operation keys must be integral");
  static_assert(std::is_unsigned_v<key_type>,
                "Mixed-operation keys must be unsigned");
  return static_cast<key_type>(key_type{1} << (sizeof(key_type) * 8 - 1));
}

template <typename key_type>
DEVICEQUALIFIER constexpr key_type mixed_device_payload_mask() {
  return static_cast<key_type>(~mixed_device_operation_bit<key_type>());
}

template <typename key_type>
DEVICEQUALIFIER constexpr key_type
mixed_device_decode_key(key_type encoded_key) {
  return static_cast<key_type>(encoded_key &
                               mixed_device_payload_mask<key_type>());
}

template <typename key_type>
DEVICEQUALIFIER constexpr bool mixed_device_is_update(key_type encoded_key) {
  return (encoded_key & mixed_device_operation_bit<key_type>()) != 0;
}

/*
 * Lower bound over a batch sorted by decoded keys.  The ordinary lower-bound
 * helpers cannot be used here: encoded update values have their MSB set and
 * are therefore not numerically ordered even though their payloads are.
 * Returning operation_count represents "no key greater than or equal to the
 * target".
 */
template <typename key_type>
DEVICEQUALIFIER smallsize
mixed_decoded_lower_bound(const key_type *__restrict__ encoded_operations,
                          smallsize operation_count, key_type target) {
  smallsize left = 0;
  smallsize right = operation_count;

  while (left < right) {
    const smallsize middle = left + ((right - left) >> 1);
    const key_type middle_key =
        mixed_device_decode_key(encoded_operations[middle]);

    if (middle_key < target)
      left = middle + 1;
    else
      right = middle;
  }

  return left;
}

/*
 * Search one node cooperatively.  In addition to membership, this returns the
 * stored offset for a query and the 1-based position needed by shift-right
 * insertion when the key is absent.
 */
template <typename key_type>
DEVICEQUALIFIER bool
mixed_search_node_tile(coop_g::thread_block_tile<TILE_SIZE> tile,
                       const void *node, smallsize node_entry_count,
                       key_type search_key, smallsize &insert_index,
                       smallsize &found_offset) {
  const smallsize lane = tile.thread_rank();

  key_type lane_key = key_type(0);
  smallsize lane_offset = 0;
  const bool lane_live = lane < node_entry_count;

  if (lane_live) {
    lane_key = extract_key_node<key_type>(node, lane + 1);
    // Node layout depends on the key width: 32-bit pairs occupy 8 bytes and
    // 64-bit pairs occupy 12 bytes.  Use key_type when locating the offset.
    lane_offset = extract_offset_node<key_type>(node, lane + 1);
  }

  const bool lane_matches = lane_live && lane_key != key_type(0) &&
                            lane_key != static_cast<key_type>(tombstone) &&
                            lane_key == search_key;

  const unsigned match_mask = tile.ballot(lane_matches);
  if (match_mask != 0) {
    const smallsize winner = static_cast<smallsize>(__ffs(match_mask) - 1);
    insert_index = winner + 1;
    found_offset = tile.shfl(lane_offset, winner);
    return true;
  }

  const bool lane_is_greater = lane_live && lane_key != key_type(0) &&
                               lane_key != static_cast<key_type>(tombstone) &&
                               lane_key > search_key;
  const unsigned greater_mask = tile.ballot(lane_is_greater);

  insert_index = greater_mask != 0
                     ? static_cast<smallsize>(__ffs(greater_mask))
                     : static_cast<smallsize>(node_entry_count + 1);
  found_offset = not_found;
  return false;
}

/*
 * Move the tile's current node forward until its maximum covers search_key.
 * Because the mixed operation array is sorted by decoded key, callers never
 * need to move backward.  prev_node is retained for deletion-time unlinking.
 */
template <typename key_type>
DEVICEQUALIFIER bool
mixed_advance_to_covering_node(coop_g::thread_block_tile<TILE_SIZE> tile,
                               updatable_cg_params *__restrict__ launch_params,
                               key_type search_key, void *&current_node,
                               void *&previous_node, key_type &current_max) {
  const smallsize lane = tile.thread_rank();
  const smallsize node_stride = launch_params->node_stride;
  const smallsize node_size = launch_params->node_size;
  const smallsize allocation_count = launch_params->allocation_buffer_count;
  const smallsize last_position = get_lastposition_bytes<key_type>(node_size);
  auto *allocation_buffer =
      static_cast<uint8_t *>(launch_params->allocation_buffer);

  int valid = 1;
  if (lane == 0) {
    current_max = cg::extract<key_type>(current_node, 0);
    while (current_max < search_key) {
      const smallsize next_pointer =
          cg::extract<smallsize>(current_node, last_position);

      if (next_pointer == 0 || (next_pointer - 1) >= allocation_count) {
        valid = 0;
        break;
      }

      previous_node = current_node;
      current_node = allocation_buffer +
                     static_cast<size_t>(next_pointer - 1) * node_stride;
      current_max = cg::extract<key_type>(current_node, 0);
    }
  }

  valid = tile.shfl(valid, 0);

  uintptr_t current_pointer = reinterpret_cast<uintptr_t>(current_node);
  current_pointer = tile.shfl(current_pointer, 0);
  current_node = reinterpret_cast<void *>(current_pointer);

  uintptr_t previous_pointer = reinterpret_cast<uintptr_t>(previous_node);
  previous_pointer = tile.shfl(previous_pointer, 0);
  previous_node = reinterpret_cast<void *>(previous_pointer);

  current_max = tile.shfl(current_max, 0);
  return valid != 0;
}

#ifdef FLIX_MIXED_REUSE_NODES
// Only the other list is written by deletions during this batch. The caller
// must serialize mixed batches for this index on the same stream.
DEVICEQUALIFIER bool flix_mixed_claim_reused_node(
    updatable_cg_params *params, size_t round_number, smallsize &free_index) {
  if (round_number == 0)
    return false;
  smallsize *count = (round_number & 1) ? &params->reuse_list_count
                                       : &params->temp_reuse_count;
  const smallsize *list = (round_number & 1) ? params->reuse_list
                                             : params->temp_reuse_list;
  // An empty count temporarily underflows; failed reservations restore it.
  // Values above capacity are also failed concurrent empty-list reservations.
  const smallsize old_count = atomicSub(count, smallsize{1});
  if (old_count == 0 || old_count > params->partition_count_with_overflow) {
    atomicAdd(count, smallsize{1});
    return false;
  }
  const smallsize node_link = list[old_count - 1];
  if (node_link == 0 || node_link > params->allocation_buffer_count) {
    printf("FliX mixed reuse: invalid reusable node link %u\n", node_link);
    return false;
  }
  free_index = node_link - 1;
  return true;
}
#endif

/*
 * Split helper for mixed batches. Its data movement and link updates match
 * process_split_tile; a new node is claimed from the allocation buffer.
 */
template <typename key_type>
DEVICEQUALIFIER bool
mixed_process_split_tile(coop_g::thread_block_tile<TILE_SIZE> tile,
                         void *current_node,
                         updatable_cg_params *__restrict__ launch_params
#ifdef FLIX_MIXED_REUSE_NODES
                         , size_t round_number
#endif
                         ) {
  const smallsize lane = tile.thread_rank();
  const smallsize node_stride = launch_params->node_stride;
  const smallsize node_size = launch_params->node_size;
  const smallsize allocation_buffer_count =
      launch_params->allocation_buffer_count;
  const smallsize last_position = get_lastposition_bytes<key_type>(node_size);
  auto *allocation_buffer =
      static_cast<uint8_t *>(launch_params->allocation_buffer);

  smallsize free_index = 0;
  int allocation_succeeded = 0;

  if (lane == 0) {
    free_index = atomicAdd(&launch_params->free_node, smallsize{1});
    if (free_index < allocation_buffer_count) {
      allocation_succeeded = 1;
    }
#ifdef FLIX_MIXED_REUSE_NODES
    else if (flix_mixed_claim_reused_node(launch_params, round_number, free_index)) {
      allocation_succeeded = 1;
    }
#endif
  }

  allocation_succeeded = tile.shfl(allocation_succeeded, 0);
  free_index = tile.shfl(free_index, 0);
  if (allocation_succeeded == 0)
    return false;

  void *linked_node =
      allocation_buffer + static_cast<size_t>(free_index) * node_stride;
  const smallsize half = node_size / 2;
  const smallsize move_count = node_size - half;

  if (lane < move_count) {
    const key_type key =
        extract_key_node<key_type>(current_node, half + 1 + lane);
    const smallsize offset =
        extract_offset_node<key_type>(current_node, half + 1 + lane);
    set_key_node<key_type>(linked_node, lane + 1, key);
    set_offset_node<key_type>(linked_node, lane + 1, offset);
  }
  tile.sync();

  if (lane < move_count) {
    set_key_node<key_type>(current_node, half + 1 + lane, key_type{0});
    set_offset_node<key_type>(current_node, half + 1 + lane, 0);
  }
  tile.sync();

  if (lane == 0) {
    cg::set<smallsize>(current_node, sizeof(key_type), half);
    cg::set<smallsize>(linked_node, sizeof(key_type), move_count);

    const key_type new_current_max =
        extract_key_node<key_type>(current_node, half);
    const key_type old_max = cg::extract<key_type>(current_node, 0);
    cg::set<key_type>(current_node, 0, new_current_max);
    cg::set<key_type>(linked_node, 0, old_max);

    const smallsize old_next =
        cg::extract<smallsize>(current_node, last_position);
    cg::set<smallsize>(linked_node, last_position, old_next);
    cg::set<smallsize>(current_node, last_position, free_index + 1);
  }
  tile.sync();

  return true;
}

// Separate-task mixed API. Keys are ordinary, sorted keys: no bits encode tasks.
// Values, tasks and results use the same sorted operation index as keys.
enum flix_task : uint8_t {
  flix_task_insert = 0,
  flix_task_lookup = 1,
  flix_task_delete = 2
};

// First operation whose key exceeds boundary. Using upper bounds for both
// ends assigns all equal keys to the same bucket, without boundary + 1 overflow.
template <typename key_type>
DEVICEQUALIFIER smallsize flix_mixed_upper_bound(
    const key_type *keys, smallsize count, key_type boundary) {
  smallsize left = 0;
  smallsize right = count;
  while (left < right) {
    const smallsize middle = left + ((right - left) >> 1);
    if (keys[middle] <= boundary)
      left = middle + 1;
    else
      right = middle;
  }
  return left;
}

// All lanes execute each helper with the same request and traversal state.
// No upsert: an existing key returns 0 and keeps its original value.
template <typename key_type>
DEVICEQUALIFIER smallsize flix_mixed_insert_one_tile(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    updatable_cg_params *params, key_type key, smallsize value,
    void *&current_node, void *&previous_node, key_type &current_max
#ifdef FLIX_MIXED_REUSE_NODES
    , size_t round_number
#endif
    ) {
  if (key == key_type{0} || key == static_cast<key_type>(tombstone))
    return 0;

  while (mixed_advance_to_covering_node(
      tile, params, key, current_node, previous_node, current_max)) {
    const smallsize count = cg::extract<smallsize>(current_node, sizeof(key_type));
    smallsize position = 1;
    smallsize old_value = not_found;
    if (mixed_search_node_tile(tile, current_node, count, key, position, old_value))
      return 0;

    if (count < params->node_size) {
      perform_shift_insert_tile<key_type>(
          tile, current_node, position, count, key, value);
      tile.sync();
      return 1;
    }

    // Split cooperatively and retry against the appropriate half of the node.
    if (!mixed_process_split_tile<key_type>(tile, current_node, params
#ifdef FLIX_MIXED_REUSE_NODES
                                           , round_number
#endif
                                           ))
      return 0;
  }
  return 0;
}

#ifdef FLIX_MIXED_REUSE_NODES
// The destination is only consumed by a later batch on the same stream.
template <typename key_type>
DEVICEQUALIFIER bool flix_mixed_retire_empty_node(
    updatable_cg_params *params, void *previous_node, void *current_node,
    size_t round_number) {
  if (round_number == 0 || previous_node == nullptr)
    return false;
  const smallsize link_pos = get_lastposition_bytes<key_type>(params->node_size);
  const smallsize node_index = cg::extract<smallsize>(previous_node, link_pos);
  if (node_index == 0 || node_index > params->allocation_buffer_count ||
      static_cast<uint8_t *>(params->allocation_buffer) +
          static_cast<size_t>(node_index - 1) * params->node_stride != current_node) {
    printf("FliX mixed reuse: invalid overflow node link %u\n", node_index);
    return false;
  }
  smallsize *count = (round_number & 1) ? &params->temp_reuse_count
                                       : &params->reuse_list_count;
  smallsize *list = (round_number & 1) ? params->temp_reuse_list
                                      : params->reuse_list;
  const smallsize slot = atomicAdd(count, smallsize{1});
  if (slot >= params->partition_count_with_overflow) {
    atomicSub(count, smallsize{1});
    printf("FliX mixed reuse: reuse list capacity exceeded\n");
    return false;
  }
  const smallsize next = cg::extract<smallsize>(current_node, link_pos);
  const key_type current_max = cg::extract<key_type>(current_node, 0);
  cg::set<smallsize>(previous_node, link_pos, next);
  cg::set<key_type>(previous_node, 0, current_max);
  list[slot] = node_index;
  return true;
}
#endif

template <typename key_type>
DEVICEQUALIFIER smallsize flix_mixed_delete_one_tile(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    updatable_cg_params *params, key_type key,
    void *&current_node, void *&previous_node, key_type &current_max
#ifdef FLIX_MIXED_REUSE_NODES
    , size_t round_number
#endif
    ) {
  if (key == key_type{0} || key == static_cast<key_type>(tombstone))
    return 0;
  if (!mixed_advance_to_covering_node(
          tile, params, key, current_node, previous_node, current_max))
    return 0;

  const smallsize count = cg::extract<smallsize>(current_node, sizeof(key_type));
  const bool removed = merge_del_steps<key_type>(tile, current_node, count, key);
  tile.sync();

#ifdef FLIX_MIXED_REUSE_NODES
  int retired = 0;
  if (tile.thread_rank() == 0 && removed &&
      cg::extract<smallsize>(current_node, sizeof(key_type)) == 0)
    retired = flix_mixed_retire_empty_node<key_type>(
        params, previous_node, current_node, round_number);
  retired = tile.shfl(retired, 0);
  tile.sync(); // Publish the predecessor's updated link and boundary.
  if (retired) {
    current_node = previous_node;
    previous_node = nullptr;
    current_max = cg::extract<key_type>(current_node, 0);
  }
  tile.sync();
#else
  // Keep empty overflow nodes linked, preserving their key intervals.
  // Retained nodes can be filled by later batches.
#endif
  return removed ? 1 : 0;
}

template <typename key_type>
DEVICEQUALIFIER smallsize flix_mixed_lookup_one_tile(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    updatable_cg_params *params, key_type key,
    void *&current_node, void *&previous_node, key_type &current_max) {
  if (!mixed_advance_to_covering_node(
          tile, params, key, current_node, previous_node, current_max))
    return not_found;
  const smallsize count = cg::extract<smallsize>(current_node, sizeof(key_type));
  smallsize position = 1;
  smallsize value = not_found;
  mixed_search_node_tile(tile, current_node, count, key, position, value);
  return value;
}

// One complete pass over this bucket, loading one request per lane per group.
// The whole tile serves selected lanes in increasing sorted operation order.
template <uint8_t pass_task, typename key_type>
DEVICEQUALIFIER void flix_mixed_bucket_pass_tile(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    updatable_cg_params *params, void *bucket_first_node,
    smallsize first, smallsize end,
    const key_type *keys, const smallsize *values, const uint8_t *tasks,
    smallsize *results
#ifdef FLIX_MIXED_REUSE_NODES
    , size_t round_number
#endif
    ) {
  const smallsize lane = tile.thread_rank();
  // Each pass restarts node traversal as well as the operation-list traversal.
  void *current_node = bucket_first_node;
  void *previous_node = nullptr;
  key_type current_max = key_type{0};

  for (smallsize group = first; group < end;) {
    const smallsize remaining = end - group;
    const smallsize width = remaining < TILE_SIZE ? remaining : TILE_SIZE;
    const bool valid_lane = lane < width;
    const smallsize operation_index = group + (valid_lane ? lane : 0);
    const bool selected = valid_lane && tasks[operation_index] == pass_task;
    const key_type lane_key = selected ? keys[operation_index] : key_type{0};
    smallsize lane_value = 0;
    if constexpr (pass_task == flix_task_insert) {
      if (selected)
        lane_value = values[operation_index];
    }

    unsigned pending = tile.ballot(selected);
    while (pending != 0) {
      const unsigned request_lane = static_cast<unsigned>(__ffs(pending) - 1);
      const key_type key = tile.shfl(lane_key, request_lane);
      smallsize result = not_found;
      if constexpr (pass_task == flix_task_insert) {
        const smallsize value = tile.shfl(lane_value, request_lane);
        result = flix_mixed_insert_one_tile(
            tile, params, key, value, current_node, previous_node, current_max
#ifdef FLIX_MIXED_REUSE_NODES
            , round_number
#endif
            );
      } else if constexpr (pass_task == flix_task_delete) {
        result = flix_mixed_delete_one_tile(
            tile, params, key, current_node, previous_node, current_max
#ifdef FLIX_MIXED_REUSE_NODES
            , round_number
#endif
            );
      } else {
        result = flix_mixed_lookup_one_tile(
            tile, params, key, current_node, previous_node, current_max);
      }
      if (lane == request_lane)
        results[operation_index] = result;
      tile.sync();
      pending &= pending - 1;
    }
    // width also handles a partial last group and avoids uint32 index wraparound.
    group += width;
  }
  tile.sync();
}

// One tile exclusively owns each bucket. Phase order is local to the bucket;
// different buckets may execute different phases concurrently. Do not launch
// another mutating kernel or rebuild against the same index concurrently.
template <typename key_type>
GLOBALQUALIFIER void flix_mixed_batch_kernel_tiled(
    updatable_cg_params *params,
    const key_type *keys, const smallsize *values, const uint8_t *tasks,
    smallsize *results, smallsize operation_count
#ifdef FLIX_MIXED_REUSE_NODES
    , size_t round_number
#endif
    ) {
  auto tile = coop_g::tiled_partition<TILE_SIZE>(coop_g::this_thread_block());
  const size_t bucket_id = static_cast<size_t>(blockIdx.x) *
      (blockDim.x / TILE_SIZE) + tile.meta_group_rank();
  if (operation_count == 0 || bucket_id >= params->partition_count_with_overflow)
    return;

  const auto *max_values = static_cast<const key_type *>(params->maxvalues);
  smallsize first = 0;
  smallsize end = 0;
  if (tile.thread_rank() == 0) {
    if (bucket_id != 0)
      first = flix_mixed_upper_bound(keys, operation_count, max_values[bucket_id - 1]);
    end = flix_mixed_upper_bound(keys, operation_count, max_values[bucket_id]);
  }
  first = tile.shfl(first, 0);
  end = tile.shfl(end, 0);
  if (first == end)
    return;

  void *bucket_first_node = static_cast<uint8_t *>(params->ordered_node_pairs) +
      bucket_id * params->node_stride;
  flix_mixed_bucket_pass_tile<flix_task_insert>(
      tile, params, bucket_first_node, first, end, keys, values, tasks, results
#ifdef FLIX_MIXED_REUSE_NODES
      , round_number
#endif
      );
  flix_mixed_bucket_pass_tile<flix_task_delete>(
      tile, params, bucket_first_node, first, end, keys, values, tasks, results
#ifdef FLIX_MIXED_REUSE_NODES
      , round_number
#endif
      );
  flix_mixed_bucket_pass_tile<flix_task_lookup>(
      tile, params, bucket_first_node, first, end, keys, values, tasks, results
#ifdef FLIX_MIXED_REUSE_NODES
      , round_number
#endif
      );
}

#ifdef FLIX_MIXED_SINGLE_THREADED
// Independent scalar implementation for the separate-task mixed_batch API.
// The tiled and legacy MSB-encoded functions above remain unchanged.
template <typename key_type>
DEVICEQUALIFIER bool flix_mixed_advance_single(
    updatable_cg_params *params, key_type key, void *&current_node
#ifdef FLIX_MIXED_REUSE_NODES
    , void **previous_node = nullptr
#endif
    ) {
  const smallsize link_position = get_lastposition_bytes<key_type>(params->node_size);
  while (cg::extract<key_type>(current_node, 0) < key) {
    const smallsize link = cg::extract<smallsize>(current_node, link_position);
    if (link == 0 || link > params->allocation_buffer_count)
      return false;
#ifdef FLIX_MIXED_REUSE_NODES
    if (previous_node != nullptr)
      *previous_node = current_node;
#endif
    current_node = static_cast<uint8_t *>(params->allocation_buffer) +
        static_cast<size_t>(link - 1) * params->node_stride;
  }
  return true;
}

template <typename key_type>
DEVICEQUALIFIER bool flix_mixed_search_single(
    const void *node, smallsize count, key_type key,
    smallsize &position, smallsize &value) {
  position = 1;
  value = not_found;
  // The existing scalar binary search reads position 1 even for an empty
  // node, and reports count rather than count+1 for an append. Adapt those
  // cases here without changing the existing helper.
  if (count == 0)
    return false;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const bool found = binary_search_in_cuda_buffer<key_type>(
      node, count, key, position, tid);
  if (found && key != key_type{0} && key != static_cast<key_type>(tombstone)) {
    value = extract_offset_node<key_type>(node, position);
    return true;
  }
  if (position == count && extract_key_node<key_type>(node, count) < key)
    position = count + 1;
  return false;
}

// Scalar counterpart of the guarded mixed split: one thread copies the upper
// half, updates boundaries and links, and retries its pending insertion.
template <typename key_type>
DEVICEQUALIFIER bool flix_mixed_split_single(
    updatable_cg_params *params, void *current_node
#ifdef FLIX_MIXED_REUSE_NODES
    , size_t round_number
#endif
    ) {
  smallsize free_index = atomicAdd(&params->free_node, smallsize{1});
  if (free_index >= params->allocation_buffer_count
#ifdef FLIX_MIXED_REUSE_NODES
      && !flix_mixed_claim_reused_node(params, round_number, free_index)
#endif
      )
    return false;

  void *linked_node = static_cast<uint8_t *>(params->allocation_buffer) +
      static_cast<size_t>(free_index) * params->node_stride;
  const smallsize half = params->node_size / 2;
  const smallsize move_count = params->node_size - half;
  const smallsize link_position = get_lastposition_bytes<key_type>(params->node_size);
  for (smallsize i = 0; i < move_count; ++i) {
    set_key_node<key_type>(linked_node, i + 1,
        extract_key_node<key_type>(current_node, half + i + 1));
    set_offset_node<key_type>(linked_node, i + 1,
        extract_offset_node<key_type>(current_node, half + i + 1));
    set_key_node<key_type>(current_node, half + i + 1, key_type{0});
    set_offset_node<key_type>(current_node, half + i + 1, 0);
  }
  cg::set<smallsize>(current_node, sizeof(key_type), half);
  cg::set<smallsize>(linked_node, sizeof(key_type), move_count);
  const key_type old_max = cg::extract<key_type>(current_node, 0);
  cg::set<key_type>(current_node, 0, extract_key_node<key_type>(current_node, half));
  cg::set<key_type>(linked_node, 0, old_max);
  cg::set<smallsize>(linked_node, link_position,
      cg::extract<smallsize>(current_node, link_position));
  cg::set<smallsize>(current_node, link_position, free_index + 1);
  return true;
}

template <typename key_type>
DEVICEQUALIFIER smallsize flix_mixed_insert_one_single(
    updatable_cg_params *params, key_type key, smallsize value,
    void *&current_node
#ifdef FLIX_MIXED_REUSE_NODES
    , size_t round_number
#endif
    ) {
  if (key == key_type{0} || key == static_cast<key_type>(tombstone))
    return 0;
  while (flix_mixed_advance_single(params, key, current_node)) {
    const smallsize count = cg::extract<smallsize>(current_node, sizeof(key_type));
    smallsize position = 1;
    smallsize existing_value = not_found;
    if (flix_mixed_search_single(current_node, count, key, position, existing_value))
      return 0; // Preserve the value of an existing key; no upsert.
    if (count < params->node_size) {
      perform_shift_insert_single<key_type>(current_node, position, count, key, value);
      return 1;
    }
    if (!flix_mixed_split_single<key_type>(params, current_node
#ifdef FLIX_MIXED_REUSE_NODES
                                           , round_number
#endif
                                           ))
      return 0;
  }
  return 0;
}

template <typename key_type>
DEVICEQUALIFIER smallsize flix_mixed_delete_one_single(
    updatable_cg_params *params, key_type key, void *&current_node
#ifdef FLIX_MIXED_REUSE_NODES
    , void *&previous_node, size_t round_number
#endif
    ) {
  if (key == key_type{0} || key == static_cast<key_type>(tombstone) ||
      !flix_mixed_advance_single(params, key, current_node
#ifdef FLIX_MIXED_REUSE_NODES
                                 , &previous_node
#endif
                                 ))
    return 0;
  const smallsize count = cg::extract<smallsize>(current_node, sizeof(key_type));
  smallsize position = 1;
  smallsize old_value = not_found;
  if (!flix_mixed_search_single(current_node, count, key, position, old_value))
    return 0;
  // This existing scalar helper shifts left and clears the tail, but leaves
  // the logical-size decrement to its caller, as in process_deletes().
  perform_delete_shift<key_type>(current_node, position, count, params->partition_size);
  cg::set<smallsize>(current_node, sizeof(key_type), count - 1);
#ifdef FLIX_MIXED_REUSE_NODES
  if (count == 1 && flix_mixed_retire_empty_node<key_type>(
          params, previous_node, current_node, round_number)) {
    current_node = previous_node;
    previous_node = nullptr;
  }
#else
  // Retain empty nodes and their boundaries; never publish new reusable nodes.
#endif
  return 1;
}

template <typename key_type>
DEVICEQUALIFIER smallsize flix_mixed_lookup_one_single(
    updatable_cg_params *params, key_type key, void *&current_node) {
  if (!flix_mixed_advance_single(params, key, current_node))
    return not_found;
  const smallsize count = cg::extract<smallsize>(current_node, sizeof(key_type));
  smallsize position = 1;
  smallsize value = not_found;
  flix_mixed_search_single(current_node, count, key, position, value);
  return value;
}

template <uint8_t pass_task, typename key_type>
DEVICEQUALIFIER void flix_mixed_bucket_pass_single(
    updatable_cg_params *params, void *bucket_first_node,
    smallsize first, smallsize end, const key_type *keys,
    const smallsize *values, const uint8_t *tasks, smallsize *results
#ifdef FLIX_MIXED_REUSE_NODES
    , size_t round_number
#endif
    ) {
  // Sorted requests permit forward-only node traversal within this pass.
  // Both the operation scan and node traversal restart for the next pass.
  void *current_node = bucket_first_node;
#ifdef FLIX_MIXED_REUSE_NODES
  void *previous_node = nullptr;
#endif
  for (smallsize i = first; i < end; ++i) {
    if (tasks[i] != pass_task)
      continue;
    if constexpr (pass_task == flix_task_insert)
      results[i] = flix_mixed_insert_one_single(params, keys[i], values[i], current_node
#ifdef FLIX_MIXED_REUSE_NODES
                                                , round_number
#endif
                                                );
    else if constexpr (pass_task == flix_task_delete)
      results[i] = flix_mixed_delete_one_single(params, keys[i], current_node
#ifdef FLIX_MIXED_REUSE_NODES
                                                , previous_node, round_number
#endif
                                                );
    else
      results[i] = flix_mixed_lookup_one_single(params, keys[i], current_node);
  }
}

template <typename key_type>
GLOBALQUALIFIER void flix_mixed_batch_kernel_single_threaded(
    updatable_cg_params *params,
    const key_type *keys, const smallsize *values, const uint8_t *tasks,
    smallsize *results, smallsize operation_count
#ifdef FLIX_MIXED_REUSE_NODES
    , size_t round_number
#endif
    ) {
  const size_t bucket_id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (operation_count == 0 || bucket_id >= params->partition_count_with_overflow)
    return;
  const auto *max_values = static_cast<const key_type *>(params->maxvalues);
  const smallsize first = bucket_id == 0 ? 0 :
      flix_mixed_upper_bound(keys, operation_count, max_values[bucket_id - 1]);
  const smallsize end = flix_mixed_upper_bound(keys, operation_count, max_values[bucket_id]);
  if (first == end)
    return;
  void *bucket_first_node = static_cast<uint8_t *>(params->ordered_node_pairs) +
      bucket_id * params->node_stride;
  flix_mixed_bucket_pass_single<flix_task_insert>(
      params, bucket_first_node, first, end, keys, values, tasks, results
#ifdef FLIX_MIXED_REUSE_NODES
      , round_number
#endif
      );
  flix_mixed_bucket_pass_single<flix_task_delete>(
      params, bucket_first_node, first, end, keys, values, tasks, results
#ifdef FLIX_MIXED_REUSE_NODES
      , round_number
#endif
      );
  flix_mixed_bucket_pass_single<flix_task_lookup>(
      params, bucket_first_node, first, end, keys, values, tasks, results
#ifdef FLIX_MIXED_REUSE_NODES
      , round_number
#endif
      );
}
#endif // FLIX_MIXED_SINGLE_THREADED


#endif
