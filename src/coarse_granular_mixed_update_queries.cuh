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

/*
 * Pop one deleted overflow node from the shared reuse stack.
 *
 * Links stored inside nodes are one-based: zero means "no next node" and a
 * nonzero value is allocation_index + 1.  The deletion paths preserve that
 * link representation in reuse_list.  Allocation-buffer addressing is
 * zero-based, so conversion happens exactly once here when the entry is
 * consumed.
 *
 * atomicSub returns the value before subtraction, which makes a direct
 * reuse_list[old_count] access off by one and also permits zero to underflow.
 * This compare-and-swap loop claims slot count - 1 only when count is nonzero.
 * Mixed insert/query batches contain no deletions, so reuse-list producers and
 * consumers are also separated by kernel-launch ordering.
 */
DEVICEQUALIFIER bool
mixed_pop_reuse_node_index(updatable_cg_params *__restrict__ launch_params,
                           smallsize allocation_buffer_count,
                           smallsize &free_index) {
  smallsize observed_count =
      atomicAdd(&launch_params->reuse_list_count, smallsize{0});

  while (observed_count != 0) {
    const smallsize claimed_slot = observed_count - 1;
    const smallsize previous_count = atomicCAS(&launch_params->reuse_list_count,
                                               observed_count, claimed_slot);

    if (previous_count == observed_count) {
      const smallsize stored_link = launch_params->reuse_list[claimed_slot];
      if (stored_link == 0 || stored_link > allocation_buffer_count)
        return false;

      free_index = stored_link - 1;
      return true;
    }

    observed_count = previous_count;
  }

  return false;
}

/*
 * Reserve one valid slot before publishing a deleted node link.  Delete/query
 * kernels only push entries, while insert/query kernels only pop them; the
 * kernel boundary separates producers from consumers.  The CAS keeps the
 * count within the actual reuse-buffer capacity under concurrent tile pushes.
 */
DEVICEQUALIFIER bool
mixed_push_reuse_node_link(updatable_cg_params *__restrict__ launch_params,
                           smallsize reuse_capacity,
                           smallsize one_based_node_link) {
  smallsize observed_count =
      atomicAdd(&launch_params->reuse_list_count, smallsize{0});

  while (observed_count < reuse_capacity) {
    const smallsize previous_count = atomicCAS(
        &launch_params->reuse_list_count, observed_count, observed_count + 1);

    if (previous_count == observed_count) {
      launch_params->reuse_list[observed_count] = one_based_node_link;
      return true;
    }

    observed_count = previous_count;
  }

  return false;
}

/*
 * Standalone split helper for mixed insert/query batches.  Its data movement
 * and link updates match process_split_tile, while its reuse-stack pop uses
 * the corrected one-based-link to zero-based-index conversion above.  Keeping
 * this helper separate leaves every existing insertion kernel unchanged.
 */
template <typename key_type>
DEVICEQUALIFIER bool
mixed_process_split_tile(coop_g::thread_block_tile<TILE_SIZE> tile,
                         void *current_node,
                         updatable_cg_params *__restrict__ launch_params) {
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
    } else if (mixed_pop_reuse_node_index(
                   launch_params, allocation_buffer_count, free_index)) {
      allocation_succeeded = 1;
    }
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

template <typename key_type>
DEVICEQUALIFIER void process_mixed_insert_query_tile(
    key_type bucket_max, smallsize first_operation,
    updatable_cg_params *__restrict__ launch_params, void *bucket_first_node,
    coop_g::thread_block_tile<TILE_SIZE> tile,
    const key_type *__restrict__ encoded_operations,
    const smallsize *__restrict__ operation_offsets,
    smallsize *__restrict__ query_results, smallsize operation_count) {
  const smallsize lane = tile.thread_rank();
  const smallsize node_size = launch_params->node_size;
  void *current_node = bucket_first_node;
  void *previous_node = nullptr;
  key_type current_max = key_type(0);

  bool have_cached_query = false;
  key_type cached_query_key = key_type(0);
  smallsize cached_query_result = not_found;

  smallsize operation_index = first_operation;
  while (operation_index < operation_count) {
    const key_type encoded_key = encoded_operations[operation_index];
    const key_type key = mixed_device_decode_key(encoded_key);
    const bool is_insert = mixed_device_is_update(encoded_key);

    if (key > bucket_max)
      break;

    // Consecutive duplicate queries can reuse either a hit or miss result.
    if (!is_insert && have_cached_query && key == cached_query_key) {
      if (lane == 0)
        query_results[operation_index] = cached_query_result;
      ++operation_index;
      continue;
    }

    if (!mixed_advance_to_covering_node(tile, launch_params, key, current_node,
                                        previous_node, current_max)) {
      if (!is_insert && lane == 0)
        query_results[operation_index] = not_found;
      if (is_insert && lane == 0)
        ERROR_INSERTS("Mixed insert could not find a covering node", key,
                      operation_index);
      break;
    }

    const smallsize current_size =
        cg::extract<smallsize>(current_node, sizeof(key_type));
    smallsize insert_index = 1;
    smallsize found_offset = not_found;
    const bool found = mixed_search_node_tile(tile, current_node, current_size,
                                              key, insert_index, found_offset);

    if (!is_insert) {
      cached_query_key = key;
      cached_query_result = found ? found_offset : not_found;
      have_cached_query = true;

      if (lane == 0)
        query_results[operation_index] = cached_query_result;

      ++operation_index;
      continue;
    }

    // Any update invalidates query-result reuse.  The benchmark generates
    // absent insertion keys, but the membership check prevents duplicates
    // if an invalid trace is supplied.
    have_cached_query = false;
    if (found) {
      ++operation_index;
      continue;
    }

    if (current_size < node_size) {
      perform_shift_insert_tile<key_type>(tile, current_node, insert_index,
                                          current_size, key,
                                          operation_offsets[operation_index]);
      tile.sync();
      ++operation_index;
      continue;
    }

    // The node is full.  Split it, then retry this operation without
    // advancing operation_index.  The next traversal selects the correct
    // split node.
    if (!mixed_process_split_tile<key_type>(tile, current_node,
                                            launch_params)) {
      if (lane == 0)
        ERROR_INSERTS("Mixed insert has no node available for split", key,
                      operation_index);
      return;
    }
  }
}

template <typename key_type>
DEVICEQUALIFIER void process_mixed_delete_query_tile(
    key_type bucket_max, smallsize first_operation,
    updatable_cg_params *__restrict__ launch_params, void *bucket_first_node,
    coop_g::thread_block_tile<TILE_SIZE> tile,
    const key_type *__restrict__ encoded_operations,
    smallsize *__restrict__ query_results, smallsize operation_count) {
  const smallsize lane = tile.thread_rank();
  const smallsize node_size = launch_params->node_size;
  const smallsize allocation_buffer_count =
      launch_params->allocation_buffer_count;
  // reuse_list_buffer is allocated with one slot per bucket head, including
  // the overflow bucket.  It is not sized to allocation_buffer_count.
  const smallsize reuse_capacity = launch_params->partition_count_with_overflow;
  const smallsize last_position = get_lastposition_bytes<key_type>(node_size);

  void *current_node = bucket_first_node;
  void *previous_node = nullptr;
  key_type current_max = key_type(0);

  bool have_cached_query = false;
  key_type cached_query_key = key_type(0);
  smallsize cached_query_result = not_found;

  smallsize operation_index = first_operation;
  while (operation_index < operation_count) {
    const key_type encoded_key = encoded_operations[operation_index];
    const key_type key = mixed_device_decode_key(encoded_key);
    const bool is_delete = mixed_device_is_update(encoded_key);

    if (key > bucket_max)
      break;

    if (!is_delete && have_cached_query && key == cached_query_key) {
      if (lane == 0)
        query_results[operation_index] = cached_query_result;
      ++operation_index;
      continue;
    }

    if (!mixed_advance_to_covering_node(tile, launch_params, key, current_node,
                                        previous_node, current_max)) {
      if (!is_delete && lane == 0)
        query_results[operation_index] = not_found;
      if (is_delete && lane == 0)
        ERROR_INSERTS("Mixed delete could not find a covering node", key,
                      operation_index);
      break;
    }

    const smallsize current_size =
        cg::extract<smallsize>(current_node, sizeof(key_type));
    smallsize found_index = 1;
    smallsize found_offset = not_found;
    const bool found = mixed_search_node_tile(tile, current_node, current_size,
                                              key, found_index, found_offset);

    if (!is_delete) {
      cached_query_key = key;
      cached_query_result = found ? found_offset : not_found;
      have_cached_query = true;

      if (lane == 0)
        query_results[operation_index] = cached_query_result;

      ++operation_index;
      continue;
    }

    have_cached_query = false;

    // The benchmark supplies present delete keys.  Retain a defensive
    // membership check so a malformed trace cannot corrupt the node.
    if (!found) {
      ++operation_index;
      continue;
    }

    const bool removed =
        merge_del_steps<key_type>(tile, current_node, current_size, key);
    tile.sync();

    if (!removed) {
      ++operation_index;
      continue;
    }

    const smallsize size_after_delete =
        cg::extract<smallsize>(current_node, sizeof(key_type));

    int node_was_unlinked = 0;
    if (lane == 0 && size_after_delete == 0 && previous_node != nullptr) {
      // Mirror the active tiled deletion path: bypass the empty overflow
      // node, extend the previous node's maximum to the removed node's
      // maximum, and return the removed node's one-based link value to the
      // reuse list.
      const smallsize removed_node_pointer =
          cg::extract<smallsize>(previous_node, last_position);
      const smallsize following_node_pointer =
          cg::extract<smallsize>(current_node, last_position);

      cg::set<smallsize>(previous_node, last_position, following_node_pointer);
      cg::set<key_type>(previous_node, 0, current_max);

      if (removed_node_pointer == 0 ||
          removed_node_pointer > allocation_buffer_count) {
        ERROR_INSERTS("Mixed delete found invalid reusable node link",
                      removed_node_pointer, operation_index);
      } else if (!mixed_push_reuse_node_link(launch_params, reuse_capacity,
                                             removed_node_pointer)) {
        ERROR_INSERTS("Mixed delete reuse_list overflow", reuse_capacity,
                      operation_index);
      }

      node_was_unlinked = 1;
    }

    node_was_unlinked = tile.shfl(node_was_unlinked, 0);
    tile.sync();

    if (node_was_unlinked) {
      // Restarting from the bucket head avoids retaining a dangling
      // current/previous pair after unlinking.  Empty-node removal is
      // uncommon, so this favors correctness over a fragile fast path.
      current_node = bucket_first_node;
      previous_node = nullptr;
      current_max = key_type(0);
    }

    ++operation_index;
  }
}

template <typename key_type>
GLOBALQUALIFIER void
mixed_insert_query_kernel_tiled(updatable_cg_params *__restrict__ launch_params,
                                const key_type *__restrict__ encoded_operations,
                                const smallsize *__restrict__ operation_offsets,
                                smallsize *__restrict__ query_results,
                                smallsize operation_count) {
  if (operation_count == 0)
    return;

  const auto *max_values =
      static_cast<const key_type *>(launch_params->maxvalues);
  const smallsize bucket_count = launch_params->partition_count_with_overflow;

  coop_g::thread_block block = coop_g::this_thread_block();
  coop_g::thread_block_tile<TILE_SIZE> tile =
      coop_g::tiled_partition<TILE_SIZE>(block);

  const smallsize tiles_per_block = blockDim.x / tile.size();
  const smallsize bucket_id =
      blockIdx.x * tiles_per_block + tile.meta_group_rank();
  const smallsize lane = tile.thread_rank();

  if (bucket_id >= bucket_count)
    return;

  const key_type bucket_max = max_values[bucket_id];
  const key_type bucket_min =
      bucket_id > 0 ? static_cast<key_type>(max_values[bucket_id - 1] + 1)
                    : static_cast<key_type>(1);

  smallsize first_operation = operation_count;
  if (lane == 0) {
    first_operation = mixed_decoded_lower_bound(encoded_operations,
                                                operation_count, bucket_min);
  }
  first_operation = tile.shfl(first_operation, 0);

  if (first_operation >= operation_count ||
      mixed_device_decode_key(encoded_operations[first_operation]) > bucket_max)
    return;

  void *bucket_first_node =
      static_cast<uint8_t *>(launch_params->ordered_node_pairs) +
      static_cast<size_t>(bucket_id) * launch_params->node_stride;

  process_mixed_insert_query_tile(
      bucket_max, first_operation, launch_params, bucket_first_node, tile,
      encoded_operations, operation_offsets, query_results, operation_count);
}

template <typename key_type>
GLOBALQUALIFIER void
mixed_delete_query_kernel_tiled(updatable_cg_params *__restrict__ launch_params,
                                const key_type *__restrict__ encoded_operations,
                                smallsize *__restrict__ query_results,
                                smallsize operation_count) {
  if (operation_count == 0)
    return;

  const auto *max_values =
      static_cast<const key_type *>(launch_params->maxvalues);
  const smallsize bucket_count = launch_params->partition_count_with_overflow;

  coop_g::thread_block block = coop_g::this_thread_block();
  coop_g::thread_block_tile<TILE_SIZE> tile =
      coop_g::tiled_partition<TILE_SIZE>(block);

  const smallsize tiles_per_block = blockDim.x / tile.size();
  const smallsize bucket_id =
      blockIdx.x * tiles_per_block + tile.meta_group_rank();
  const smallsize lane = tile.thread_rank();

  if (bucket_id >= bucket_count)
    return;

  const key_type bucket_max = max_values[bucket_id];
  const key_type bucket_min =
      bucket_id > 0 ? static_cast<key_type>(max_values[bucket_id - 1] + 1)
                    : static_cast<key_type>(1);

  smallsize first_operation = operation_count;
  if (lane == 0) {
    first_operation = mixed_decoded_lower_bound(encoded_operations,
                                                operation_count, bucket_min);
  }
  first_operation = tile.shfl(first_operation, 0);

  if (first_operation >= operation_count ||
      mixed_device_decode_key(encoded_operations[first_operation]) > bucket_max)
    return;

  void *bucket_first_node =
      static_cast<uint8_t *>(launch_params->ordered_node_pairs) +
      static_cast<size_t>(bucket_id) * launch_params->node_stride;

  process_mixed_delete_query_tile(bucket_max, first_operation, launch_params,
                                  bucket_first_node, tile, encoded_operations,
                                  query_results, operation_count);
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
    void *&current_node, void *&previous_node, key_type &current_max) {
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
    if (!mixed_process_split_tile<key_type>(tile, current_node, params))
      return 0;
  }
  return 0;
}

template <typename key_type>
DEVICEQUALIFIER smallsize flix_mixed_delete_one_tile(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    updatable_cg_params *params, key_type key,
    void *&current_node, void *&previous_node, key_type &current_max) {
  if (key == key_type{0} || key == static_cast<key_type>(tombstone))
    return 0;
  if (!mixed_advance_to_covering_node(
          tile, params, key, current_node, previous_node, current_max))
    return 0;

  const smallsize count = cg::extract<smallsize>(current_node, sizeof(key_type));
  const bool removed = merge_del_steps<key_type>(tile, current_node, count, key);
  tile.sync();

  // Keep empty overflow nodes linked. Other buckets may still be inserting:
  // mixed_push_reuse_node_link publishes its count before writing the slot and
  // is safe only when producers and consumers are separated by a kernel boundary.
  // Retained nodes preserve their key intervals and can be filled by later batches.
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
    smallsize *results) {
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
            tile, params, key, value, current_node, previous_node, current_max);
      } else if constexpr (pass_task == flix_task_delete) {
        result = flix_mixed_delete_one_tile(
            tile, params, key, current_node, previous_node, current_max);
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
    smallsize *results, smallsize operation_count) {
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
      tile, params, bucket_first_node, first, end, keys, values, tasks, results);
  flix_mixed_bucket_pass_tile<flix_task_delete>(
      tile, params, bucket_first_node, first, end, keys, values, tasks, results);
  flix_mixed_bucket_pass_tile<flix_task_lookup>(
      tile, params, bucket_first_node, first, end, keys, values, tasks, results);
}

#ifdef FLIX_MIXED_SINGLE_THREADED
// Independent scalar implementation for the separate-task mixed_batch API.
// The tiled and legacy MSB-encoded functions above remain unchanged.
template <typename key_type>
DEVICEQUALIFIER bool flix_mixed_advance_single(
    updatable_cg_params *params, key_type key, void *&current_node) {
  const smallsize link_position = get_lastposition_bytes<key_type>(params->node_size);
  while (cg::extract<key_type>(current_node, 0) < key) {
    const smallsize link = cg::extract<smallsize>(current_node, link_position);
    if (link == 0 || link > params->allocation_buffer_count)
      return false;
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
    updatable_cg_params *params, void *current_node) {
  smallsize free_index = atomicAdd(&params->free_node, smallsize{1});
  if (free_index >= params->allocation_buffer_count &&
      !mixed_pop_reuse_node_index(params, params->allocation_buffer_count, free_index))
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
    void *&current_node) {
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
    if (!flix_mixed_split_single<key_type>(params, current_node))
      return 0;
  }
  return 0;
}

template <typename key_type>
DEVICEQUALIFIER smallsize flix_mixed_delete_one_single(
    updatable_cg_params *params, key_type key, void *&current_node) {
  if (key == key_type{0} || key == static_cast<key_type>(tombstone) ||
      !flix_mixed_advance_single(params, key, current_node))
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
  // Retain empty nodes and their boundaries; never publish new reusable nodes.
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
    const smallsize *values, const uint8_t *tasks, smallsize *results) {
  // Sorted requests permit forward-only node traversal within this pass.
  // Both the operation scan and node traversal restart for the next pass.
  void *current_node = bucket_first_node;
  for (smallsize i = first; i < end; ++i) {
    if (tasks[i] != pass_task)
      continue;
    if constexpr (pass_task == flix_task_insert)
      results[i] = flix_mixed_insert_one_single(params, keys[i], values[i], current_node);
    else if constexpr (pass_task == flix_task_delete)
      results[i] = flix_mixed_delete_one_single(params, keys[i], current_node);
    else
      results[i] = flix_mixed_lookup_one_single(params, keys[i], current_node);
  }
}

template <typename key_type>
GLOBALQUALIFIER void flix_mixed_batch_kernel_single_threaded(
    updatable_cg_params *params,
    const key_type *keys, const smallsize *values, const uint8_t *tasks,
    smallsize *results, smallsize operation_count) {
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
      params, bucket_first_node, first, end, keys, values, tasks, results);
  flix_mixed_bucket_pass_single<flix_task_delete>(
      params, bucket_first_node, first, end, keys, values, tasks, results);
  flix_mixed_bucket_pass_single<flix_task_lookup>(
      params, bucket_first_node, first, end, keys, values, tasks, results);
}
#endif // FLIX_MIXED_SINGLE_THREADED

#ifdef FLIX_MIXED_REUSE_NODES
// Separate state keeps the legacy launch-parameter layout and kernels intact.
// Only deletions write temp_reuse_list. Inserts claim tickets against the fixed
// params->reuse_list_count available at launch. The cooperative grid barrier
// separates all claims/retirements from publication and counter resets.
struct flix_mixed_reuse_state {
  smallsize *temp_reuse_list;
  smallsize temp_reuse_count;
  smallsize temp_capacity;
  smallsize reuse_capacity;
  smallsize error; // 1: invalid link; 2: temporary capacity; 4: publication capacity
  smallsize reuse_claim_count; // zero-initialized, then reset after each grid barrier
};

DEVICEQUALIFIER bool flix_mixed_reserve_retired_slot(
    flix_mixed_reuse_state *state, smallsize &slot) {
  const smallsize ticket = atomicAdd(&state->temp_reuse_count, smallsize{1});
  if (ticket < state->temp_capacity) {
    slot = ticket;
    return true;
  }
  // Return failed reservations. Valid (< capacity) slots cannot be reissued:
  // outstanding failed tickets keep the counter at or above capacity.
  atomicSub(&state->temp_reuse_count, smallsize{1});
  atomicOr(&state->error, smallsize{2});
  return false;
}

DEVICEQUALIFIER bool flix_mixed_allocate_reuse_node(
    updatable_cg_params *params, flix_mixed_reuse_state *state,
    smallsize &index) {
  // This count is read-only until the final grid barrier. A separate ticket
  // counter claims its entries without retrying or decrementing an empty stack.
  const smallsize available = params->reuse_list_count;
  if (available != 0) {
    const smallsize ticket = atomicAdd(&state->reuse_claim_count, smallsize{1});
    if (ticket < available) {
      const smallsize link = params->reuse_list[available - 1 - ticket];
      if (link == 0 || link > params->allocation_buffer_count) {
        atomicOr(&state->error, smallsize{1});
        return false;
      }
      index = link - 1;
      return true;
    }
  }
  // One ticket per attempt avoids a globally contended CAS retry loop.
  // A failed ticket is returned without ever addressing outside the buffer.
  // While failed attempts are outstanding, the counter remains >= capacity;
  // returning them cannot reissue any valid (< capacity) allocation index.
  const smallsize fresh = atomicAdd(&params->free_node, smallsize{1});
  if (fresh < params->allocation_buffer_count) {
    index = fresh;
    return true;
  }
  atomicSub(&params->free_node, smallsize{1});
  return false;
}

// Called by the bucket's single owner (scalar thread or tile leader).
template <typename key_type>
DEVICEQUALIFIER void flix_mixed_retire_empty_overflow(
    updatable_cg_params *params, flix_mixed_reuse_state *state,
    void *bucket_head, void *&current, void *&previous) {
  if (previous == nullptr || current == bucket_head ||
      cg::extract<smallsize>(current, sizeof(key_type)) != 0)
    return;
  const smallsize position = get_lastposition_bytes<key_type>(params->node_size);
  const smallsize link = cg::extract<smallsize>(previous, position);
  if (link == 0 || link > params->allocation_buffer_count ||
      static_cast<uint8_t *>(params->allocation_buffer) +
          static_cast<size_t>(link - 1) * params->node_stride != current) {
    atomicOr(&state->error, smallsize{1});
    return;
  }
  smallsize slot = 0;
  // If capacity is unavailable, retain the node rather than lose its storage.
  if (!flix_mixed_reserve_retired_slot(state, slot))
    return;
  cg::set<smallsize>(previous, position, cg::extract<smallsize>(current, position));
  cg::set<key_type>(previous, 0, cg::extract<key_type>(current, 0));
  state->temp_reuse_list[slot] = link;
  // Continue from the live predecessor, which now covers the removed interval.
  // Sorted deletion keys cannot remove its earlier keys. Clear the unknown
  // predecessor-of-predecessor; advancing to the next node establishes it again.
  current = previous;
  previous = nullptr;
}

// Called ONCE by grid thread zero AFTER the cooperative grid barrier.
DEVICEQUALIFIER void flix_mixed_publish_temp_reuse(
    updatable_cg_params *params, flix_mixed_reuse_state *state) {
  const smallsize available = params->reuse_list_count;
  const smallsize claimed = state->reuse_claim_count;
  const smallsize consumed = claimed < available ? claimed : available;
  const smallsize remaining = available - consumed;
  const smallsize retired = state->temp_reuse_count;
  if (remaining > state->reuse_capacity ||
      retired > state->reuse_capacity - remaining) {
    atomicOr(&state->error, smallsize{4});
    return;
  }
  for (smallsize i = 0; i < retired; ++i)
    params->reuse_list[remaining + i] = state->temp_reuse_list[i];
  params->reuse_list_count = remaining + retired;
  state->temp_reuse_count = 0;
  state->reuse_claim_count = 0;
}

#ifdef FLIX_MIXED_PARALLEL_REUSE_COPY
// Every thread in block zero calls this AFTER the cooperative grid barrier.
// Other blocks do no further work. The next ordered kernel launch waits for
// block zero's copies and counter publication as part of kernel completion.
DEVICEQUALIFIER void flix_mixed_publish_temp_reuse_block(
    coop_g::thread_block block, updatable_cg_params *params,
    flix_mixed_reuse_state *state) {
  const smallsize available = params->reuse_list_count;
  const smallsize claimed = state->reuse_claim_count;
  const smallsize consumed = claimed < available ? claimed : available;
  const smallsize remaining = available - consumed;
  const smallsize retired = state->temp_reuse_count;
  // These counts are immutable until the block barrier below, so this branch
  // is uniform across the block, including on capacity errors.
  if (remaining > state->reuse_capacity ||
      retired > state->reuse_capacity - remaining) {
    if (block.thread_rank() == 0)
      atomicOr(&state->error, smallsize{4});
    return;
  }
  // Preserve the unused ready-list prefix and append distinct temporary slots.
  // size_t keeps the strided increment from wrapping a uint32 operation count.
  for (size_t i = block.thread_rank(); i < static_cast<size_t>(retired);
       i += block.size())
    params->reuse_list[static_cast<size_t>(remaining) + i] =
        state->temp_reuse_list[i];
  // All copy writes and metadata reads must finish before counters are reset.
  block.sync();
  if (block.thread_rank() == 0) {
    params->reuse_list_count = remaining + retired;
    state->temp_reuse_count = 0;
    state->reuse_claim_count = 0;
  }
}
#endif // FLIX_MIXED_PARALLEL_REUSE_COPY

template <typename key_type>
DEVICEQUALIFIER bool
flix_mixed_process_split_tile_reuse(coop_g::thread_block_tile<TILE_SIZE> tile,
                         void *current_node,
                         updatable_cg_params *__restrict__ launch_params,
                         flix_mixed_reuse_state *reuse_state) {
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

  if (lane == 0)
    allocation_succeeded = flix_mixed_allocate_reuse_node(
        launch_params, reuse_state, free_index) ? 1 : 0;

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

template <typename key_type>
DEVICEQUALIFIER smallsize flix_mixed_insert_one_tile_reuse(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    updatable_cg_params *params, key_type key, smallsize value,
    void *&current_node, void *&previous_node, key_type &current_max, flix_mixed_reuse_state *reuse_state) {
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
    if (!flix_mixed_process_split_tile_reuse<key_type>(tile, current_node, params, reuse_state))
      return 0;
  }
  return 0;
}

template <typename key_type>
DEVICEQUALIFIER smallsize flix_mixed_delete_one_tile_reuse(
    coop_g::thread_block_tile<TILE_SIZE> tile, updatable_cg_params *params,
    key_type key, void *&current, void *&previous, key_type &maximum,
    void *bucket_head, flix_mixed_reuse_state *state) {
  if (key == key_type{0} || key == static_cast<key_type>(tombstone))
    return 0;
  if (!mixed_advance_to_covering_node(tile, params, key, current, previous, maximum))
    return 0;
  const smallsize count = cg::extract<smallsize>(current, sizeof(key_type));
  const bool removed = merge_del_steps<key_type>(tile, current, count, key);
  tile.sync();
  if (tile.thread_rank() == 0)
    flix_mixed_retire_empty_overflow<key_type>(params, state, bucket_head, current, previous);
  tile.sync();
  // The traversal helper broadcasts the leader's updated cursors on next use.
  return removed ? 1 : 0;
}

template <uint8_t pass_task, typename key_type>
DEVICEQUALIFIER void flix_mixed_bucket_pass_tile_reuse(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    updatable_cg_params *params, void *bucket_first_node,
    smallsize first, smallsize end,
    const key_type *keys, const smallsize *values, const uint8_t *tasks,
    smallsize *results, flix_mixed_reuse_state *reuse_state) {
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
        result = flix_mixed_insert_one_tile_reuse(
            tile, params, key, value, current_node, previous_node, current_max, reuse_state);
      } else if constexpr (pass_task == flix_task_delete) {
        result = flix_mixed_delete_one_tile_reuse(
            tile, params, key, current_node, previous_node, current_max, bucket_first_node, reuse_state);
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

template <typename key_type>
GLOBALQUALIFIER void flix_mixed_batch_kernel_tiled_reuse(
    updatable_cg_params *params, flix_mixed_reuse_state *state,
    const key_type *keys, const smallsize *values, const uint8_t *tasks,
    smallsize *results, smallsize operation_count) {
  auto grid = coop_g::this_grid();
  auto tile = coop_g::tiled_partition<TILE_SIZE>(coop_g::this_thread_block());
  const size_t tile_id = static_cast<size_t>(blockIdx.x) *
      (blockDim.x / TILE_SIZE) + tile.meta_group_rank();
  const size_t tile_count = static_cast<size_t>(gridDim.x) * (blockDim.x / TILE_SIZE);
  const auto *max_values = static_cast<const key_type *>(params->maxvalues);
  // A resident cooperative grid handles arbitrarily many buckets by striding.
  // Empty/inactive tiles MUST still reach the final grid barrier.
  for (size_t bucket = tile_id; bucket < params->partition_count_with_overflow;
       bucket += tile_count) {
    if (operation_count == 0)
      continue;
    smallsize first = 0, end = 0;
    if (tile.thread_rank() == 0) {
      if (bucket != 0)
        first = flix_mixed_upper_bound(keys, operation_count, max_values[bucket - 1]);
      end = flix_mixed_upper_bound(keys, operation_count, max_values[bucket]);
    }
    first = tile.shfl(first, 0);
    end = tile.shfl(end, 0);
    if (first == end)
      continue;
    void *head = static_cast<uint8_t *>(params->ordered_node_pairs) + bucket * params->node_stride;
    flix_mixed_bucket_pass_tile_reuse<flix_task_insert>(
        tile, params, head, first, end, keys, values, tasks, results, state);
    flix_mixed_bucket_pass_tile_reuse<flix_task_delete>(
        tile, params, head, first, end, keys, values, tasks, results, state);
    flix_mixed_bucket_pass_tile_reuse<flix_task_lookup>(
        tile, params, head, first, end, keys, values, tasks, results, state);
  }
  grid.sync();
#ifdef FLIX_MIXED_PARALLEL_REUSE_COPY
  if (blockIdx.x == 0)
    flix_mixed_publish_temp_reuse_block(coop_g::this_thread_block(), params, state);
#else
  if (grid.thread_rank() == 0)
    flix_mixed_publish_temp_reuse(params, state);
#endif
}

#ifdef FLIX_MIXED_SINGLE_THREADED
template <typename key_type>
DEVICEQUALIFIER bool flix_mixed_advance_single_reuse(
    updatable_cg_params *params, key_type key, void *&current, void *&previous) {
  const smallsize position = get_lastposition_bytes<key_type>(params->node_size);
  while (cg::extract<key_type>(current, 0) < key) {
    const smallsize link = cg::extract<smallsize>(current, position);
    if (link == 0 || link > params->allocation_buffer_count)
      return false;
    previous = current;
    current = static_cast<uint8_t *>(params->allocation_buffer) +
        static_cast<size_t>(link - 1) * params->node_stride;
  }
  return true;
}

template <typename key_type>
DEVICEQUALIFIER bool flix_mixed_split_single_reuse(
    updatable_cg_params *params, void *current_node, flix_mixed_reuse_state *reuse_state) {
  smallsize free_index = 0;
  if (!flix_mixed_allocate_reuse_node(params, reuse_state, free_index))
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
DEVICEQUALIFIER smallsize flix_mixed_insert_one_single_reuse(
    updatable_cg_params *params, key_type key, smallsize value,
    void *&current_node, void *&previous_node, flix_mixed_reuse_state *state) {
  if (key == key_type{0} || key == static_cast<key_type>(tombstone))
    return 0;
  while (flix_mixed_advance_single_reuse(params, key, current_node, previous_node)) {
    const smallsize count = cg::extract<smallsize>(current_node, sizeof(key_type));
    smallsize position = 1;
    smallsize existing_value = not_found;
    if (flix_mixed_search_single(current_node, count, key, position, existing_value))
      return 0; // Preserve the value of an existing key; no upsert.
    if (count < params->node_size) {
      perform_shift_insert_single<key_type>(current_node, position, count, key, value);
      return 1;
    }
    if (!flix_mixed_split_single_reuse<key_type>(params, current_node, state))
      return 0;
  }
  return 0;
}

template <typename key_type>
DEVICEQUALIFIER smallsize flix_mixed_delete_one_single_reuse(
    updatable_cg_params *params, key_type key, void *&current,
    void *&previous, void *head, flix_mixed_reuse_state *state) {
  if (key == key_type{0} || key == static_cast<key_type>(tombstone) ||
      !flix_mixed_advance_single_reuse(params, key, current, previous))
    return 0;
  const smallsize count = cg::extract<smallsize>(current, sizeof(key_type));
  smallsize position = 1, value = not_found;
  const bool removed = flix_mixed_search_single(current, count, key, position, value);
  if (removed) {
    perform_delete_shift<key_type>(current, position, count, params->partition_size);
    cg::set<smallsize>(current, sizeof(key_type), count - 1);
  }
  flix_mixed_retire_empty_overflow<key_type>(params, state, head, current, previous);
  return removed ? 1 : 0;
}

template <uint8_t pass_task, typename key_type>
DEVICEQUALIFIER void flix_mixed_bucket_pass_single_reuse(
    updatable_cg_params *params, void *bucket_first_node,
    smallsize first, smallsize end, const key_type *keys,
    const smallsize *values, const uint8_t *tasks, smallsize *results, flix_mixed_reuse_state *state) {
  // Sorted requests permit forward-only node traversal within this pass.
  // Both the operation scan and node traversal restart for the next pass.
  void *current_node = bucket_first_node;
  void *previous_node = nullptr;
  for (smallsize i = first; i < end; ++i) {
    if (tasks[i] != pass_task)
      continue;
    if constexpr (pass_task == flix_task_insert)
      results[i] = flix_mixed_insert_one_single_reuse(params, keys[i], values[i], current_node, previous_node, state);
    else if constexpr (pass_task == flix_task_delete)
      results[i] = flix_mixed_delete_one_single_reuse(params, keys[i], current_node, previous_node, bucket_first_node, state);
    else
      results[i] = flix_mixed_lookup_one_single(params, keys[i], current_node);
  }
}

template <typename key_type>
GLOBALQUALIFIER void flix_mixed_batch_kernel_single_threaded_reuse(
    updatable_cg_params *params, flix_mixed_reuse_state *state,
    const key_type *keys, const smallsize *values, const uint8_t *tasks,
    smallsize *results, smallsize operation_count) {
  auto grid = coop_g::this_grid();
  const size_t thread_id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t thread_count = static_cast<size_t>(gridDim.x) * blockDim.x;
  const auto *max_values = static_cast<const key_type *>(params->maxvalues);
  for (size_t bucket = thread_id; bucket < params->partition_count_with_overflow;
       bucket += thread_count) {
    if (operation_count == 0)
      continue;
    const smallsize first = bucket == 0 ? 0 :
        flix_mixed_upper_bound(keys, operation_count, max_values[bucket - 1]);
    const smallsize end = flix_mixed_upper_bound(keys, operation_count, max_values[bucket]);
    if (first == end)
      continue;
    void *head = static_cast<uint8_t *>(params->ordered_node_pairs) + bucket * params->node_stride;
    flix_mixed_bucket_pass_single_reuse<flix_task_insert>(
        params, head, first, end, keys, values, tasks, results, state);
    flix_mixed_bucket_pass_single_reuse<flix_task_delete>(
        params, head, first, end, keys, values, tasks, results, state);
    flix_mixed_bucket_pass_single_reuse<flix_task_lookup>(
        params, head, first, end, keys, values, tasks, results, state);
  }
  grid.sync();
#ifdef FLIX_MIXED_PARALLEL_REUSE_COPY
  if (blockIdx.x == 0)
    flix_mixed_publish_temp_reuse_block(coop_g::this_thread_block(), params, state);
#else
  if (grid.thread_rank() == 0)
    flix_mixed_publish_temp_reuse(params, state);
#endif
}
#endif // FLIX_MIXED_SINGLE_THREADED

#endif // FLIX_MIXED_REUSE_NODES

#endif
