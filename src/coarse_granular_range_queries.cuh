// file: ordered_range_lookup.cuh
#ifndef COARSE_GRANULAR_RANGE_QUERIES_CUH
#define COARSE_GRANULAR_RANGE_QUERIES_CUH

#include "coarse_granular_lookups_tile_bulk.cuh"


// Parallel per-bucket counterpart of the single-owner naive range-sum walk
// (process_lookup_tile_bulk_ordered_rq_sums_naive): instead of one tile
// owning a query end-to-end and hopping bucket to bucket, every tile here
// independently scans for queries that overlap its own bucket and
// atomically contributes only its own bucket-local partial sum. A query
// spanning many buckets is therefore covered by many tiles running
// concurrently instead of one tile visiting those buckets serially.
//
// For a query that fully spans this bucket (qlow <= bucket_min and
// qhigh >= bucket_max), the bucket's entire contribution is already known --
// it's `bucket_full_sum`, computed once up front by
// precompute_bucket_offset_sums_kernel -- so no node-chain walk is needed at
// all. The node-by-node scan below only ever runs for a bucket that a query's
// lo or hi boundary actually lands inside (at most two buckets per query).
// `results` must be zero-initialized by the caller before this kernel runs.
template <typename key_type>
DEVICEQUALIFIER void process_lookup_tile_bulk_ordered_dup_rq(
    key_type bucket_min,
    key_type bucket_max,
    int minindex,
  //  int maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile,
    const key_type *__restrict__ lowest_list,
    const key_type *__restrict__ highest_list,
    smallsize query_size,
    smallsize *__restrict__ results,
    smallsize bucket_full_sum)
{
    void *allocation_buffer = launch_params->allocation_buffer;
    const smallsize allocation_count = launch_params->allocation_buffer_count;
    const smallsize node_stride = launch_params->node_stride;
    const smallsize node_size = launch_params->node_size;

    const smallsize lastpos_off = get_lastposition_bytes<key_type>(node_size);
    const smallsize lane = tile.thread_rank();
    const smallsize tile_id =
        blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    void *base_node = starting_node;

    for (int qi = minindex; qi < static_cast<int>(query_size); ++qi)
    {
        const key_type qlow = lowest_list[qi];
        const key_type qhigh = highest_list[qi];

        if (qlow > bucket_max)
            break;

        if (qhigh < bucket_min)
            continue;

        // Whole bucket lies inside [qlow, qhigh]: reuse the precomputed
        // full-bucket sum instead of re-walking the node chain.
        if (qlow <= bucket_min && qhigh >= bucket_max)
        {
            if (lane == 0 && bucket_full_sum != 0)
                atomicAdd(results + qi, bucket_full_sum);

            tile.sync();
            continue;
        }

        const key_type search_key = (qlow > bucket_min) ? qlow : bucket_min;
        if (lane == 0)
        {
            key_type base_max = cg::extract<key_type>(base_node, 0);
            while (base_max < search_key)
            {
                const smallsize next_ptr = cg::extract<smallsize>(base_node, lastpos_off);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_count)
                {
                    ERROR_INSERTS(
                        "ERROR: invalid link in node chain during tile bulk ordered range lookup",
                        search_key, lane, tile_id);
                }

                base_node = static_cast<uint8_t *>(allocation_buffer) +
                            static_cast<size_t>(next_ptr - 1) * node_stride;
                base_max = cg::extract<key_type>(base_node, 0);
            }
        }

        uintptr_t p = reinterpret_cast<uintptr_t>(base_node);
        p = tile.shfl(p, 0);
        void *scan_node = reinterpret_cast<void *>(p);

        unsigned long long query_sum = 0;
        while (true)
        {
            key_type curr_max = key_type(0);
            key_type node_min = key_type(0);
            smallsize curr_size = 0;
            smallsize next_ptr = 0;

            if (lane == 0)
            {
                curr_max = cg::extract<key_type>(scan_node, 0);
                curr_size = cg::extract<smallsize>(scan_node, sizeof(key_type));
                next_ptr = cg::extract<smallsize>(scan_node, lastpos_off);
                node_min = (curr_size > 0)
                               ? extract_key_node<key_type>(scan_node, 1)
                               : curr_max;
            }

            curr_max = tile.shfl(curr_max, 0);
            node_min = tile.shfl(node_min, 0);
            curr_size = tile.shfl(curr_size, 0);
            next_ptr = tile.shfl(next_ptr, 0);

            if (qhigh < node_min)
                break;

            key_type my_key = key_type(0);
            smallsize my_offset = 0;
            const bool lane_live = (lane < curr_size);

            if (lane_live)
            {
                my_key = extract_key_node<key_type>(scan_node, lane + 1);
                my_offset = extract_offset_node<key_type>(scan_node, lane + 1);
            }

            smallsize partial_sum = 0;
            if (lane_live && my_key >= qlow && my_key <= qhigh)
                partial_sum = my_offset;

#pragma unroll
            for (int delta = tile.size() / 2; delta > 0; delta >>= 1)
                partial_sum += tile.shfl_down(partial_sum, delta);

            if (lane == 0)
                query_sum += static_cast<unsigned long long>(partial_sum);

            if (curr_max >= bucket_max || curr_max >= qhigh)
                break;

            if (lane == 0)
            {
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_count)
                {
                    ERROR_INSERTS(
                        "ERROR: invalid link in node chain during tile bulk ordered range lookup",
                        qhigh, lane, tile_id);
                }

                scan_node = static_cast<uint8_t *>(allocation_buffer) +
                            static_cast<size_t>(next_ptr - 1) * node_stride;
            }

            p = reinterpret_cast<uintptr_t>(scan_node);
            p = tile.shfl(p, 0);
            scan_node = reinterpret_cast<void *>(p);
            tile.sync();
        }

        if (lane == 0 && query_sum != 0)
            atomicAdd(results + qi, static_cast<smallsize>(query_sum));

        tile.sync();
    }
}

template <typename key_type>
GLOBALQUALIFIER void lookup_kernel_tile_ordered_rq(
    updatable_cg_params *__restrict__ launch_params,
    const key_type *__restrict__ lowest_list,   // sorted by lower bound
    const key_type *__restrict__ highest_list,  // paired with lowest_list, non-decreasing
    const smallsize *__restrict__ bucket_offset_sums,  // per-bucket full sum, from precompute_bucket_offset_sums_kernel
    smallsize *__restrict__ results,            // must be zero-initialized; tiles atomicAdd into it
    smallsize query_size)
{
    const key_type *__restrict__ maxbuf =
        static_cast<const key_type *>(launch_params->maxvalues);
    const smallsize partition_count_with_overflow =
        launch_params->partition_count_with_overflow;

    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile =
        coop_g::tiled_partition<TILE_SIZE>(block);

    const int tiles_per_block = blockDim.x / tile.size();
    const int tile_id = blockIdx.x * tiles_per_block + tile.meta_group_rank();
    const smallsize lane = tile.thread_rank();

    if (tile_id >= partition_count_with_overflow || query_size == 0)
        return;

    const key_type maxkey = maxbuf[tile_id];
    const key_type minkey =
        (tile_id > 0)
            ? (maxbuf[tile_id - 1] + static_cast<key_type>(1))
            : static_cast<key_type>(1);

    // Only queries whose upper bound reaches into this bucket can possibly
    // overlap it. highest_list is non-decreasing (paired with sorted
    // lowest_list via a monotonic per-key transform), so binary search finds
    // the first such query instead of every tile scanning from index 0.
    int minindex = 0;
    if (lane == 0)
    {
        minindex = binarySearchIndex_leftmost_ge_strict<key_type>(
            highest_list, minkey, 0, static_cast<int>(query_size));
    }
    minindex = tile.shfl(minindex, 0);

    if (minindex < 0)
        return; // no query reaches into this bucket

    uint8_t *__restrict__ curr_node =
        static_cast<uint8_t *>(launch_params->ordered_node_pairs) +
        static_cast<size_t>(launch_params->node_stride) *
            static_cast<size_t>(tile_id);

    const smallsize bucket_full_sum =
        (bucket_offset_sums != nullptr) ? bucket_offset_sums[tile_id] : smallsize(0);

    process_lookup_tile_bulk_ordered_dup_rq<key_type>(
        /*bucket_min*/      minkey,
        /*bucket_max*/      maxkey,
        /*minindex*/        minindex,
        // /-*maxindex*-/   maxindex,
                            launch_params,
        /*curr_node*/       curr_node,
        /*tile*/            tile,
        /*lowest_list*/     lowest_list,
        /*highest_list*/    highest_list,
        /*query_size*/      query_size,
        /*writeback*/       results,
        /*bucket_full_sum*/ bucket_full_sum);
}

// Naive range-sum counterpart of process_lookup_tile_bulk_ordered_dup: exactly
// one tile -- the owner of the bucket containing a query's lower bound, found
// the same way lookup_kernel_tile_ordered finds exact-match owners -- computes
// and writes that query's full sum. No other tile may contribute to the same
// query, so no atomics are needed.
//
// For each owned query (processed one at a time), the tile starts at the
// owner bucket's node containing lo and sums node by node via a tile-wide
// shuffle reduction (same reduction shape as precompute_bucket_offset_sums_kernel /
// lookup_kernel_tile_ordered_rq_sums), stopping as soon as a node's own max
// key reaches/exceeds hi. If a bucket's chain is exhausted before hi is
// reached, the tile moves to the next bucket's first node (no skip-search
// needed there -- every key in a later bucket is already >= lo) and repeats.
// The tile then returns to its own bucket for the next owned query.
template <typename key_type>
DEVICEQUALIFIER void *get_bucket_head_node(
    updatable_cg_params *__restrict__ launch_params,
    smallsize bucket_index);

template <typename key_type>
DEVICEQUALIFIER void process_lookup_tile_bulk_ordered_rq_sums_naive(
    key_type bucket_min,
    key_type bucket_max,
    smallsize minindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile,
    const key_type *__restrict__ lower_list,
    const key_type *__restrict__ upper_list,
    smallsize query_size,
    smallsize *__restrict__ results)
{
    (void)bucket_min; // per-key filtering uses lower_list[qi] directly

    void *allocation_buffer = launch_params->allocation_buffer;
    const smallsize allocation_count = launch_params->allocation_buffer_count;
    const smallsize node_stride = launch_params->node_stride;
    const smallsize node_size = launch_params->node_size;
    const smallsize partition_count_with_overflow =
        launch_params->partition_count_with_overflow;

    const smallsize lastpos_off = get_lastposition_bytes<key_type>(node_size);
    const smallsize lane = tile.thread_rank();
    const smallsize owner_bucket_id =
        blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    // Walk every range query owned by this bucket, one at a time.
    for (smallsize qi = minindex; qi < query_size && lower_list[qi] <= bucket_max; ++qi)
    {
        const key_type lo = lower_list[qi];
        const key_type hi = upper_list[qi];

        smallsize current_bucket_id = owner_bucket_id;
        void *curr_node = starting_node;
        unsigned long long query_sum = 0;

        // Skip any leading nodes in the owner bucket that are entirely below lo.
        if (lane == 0)
        {
            key_type node_max = cg::extract<key_type>(curr_node, 0);
            while (node_max < lo)
            {
                const smallsize next_ptr = cg::extract<smallsize>(curr_node, lastpos_off);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_count)
                {
                    ERROR_INSERTS(
                        "ERROR: invalid link in node chain during naive tile range sum start search",
                        lo, lane, owner_bucket_id);
                }
                curr_node = static_cast<uint8_t *>(allocation_buffer) +
                            static_cast<size_t>(next_ptr - 1) * node_stride;
                node_max = cg::extract<key_type>(curr_node, 0);
            }
        }
        {
            uintptr_t p0 = reinterpret_cast<uintptr_t>(curr_node);
            p0 = tile.shfl(p0, 0);
            curr_node = reinterpret_cast<void *>(p0);
        }

        bool range_fully_covered = false;

        while (true)
        {
            // Sum every node in current_bucket_id's chain, node by node,
            // stopping as soon as a node's own max key reaches/exceeds hi --
            // no later node, in this bucket or any later bucket, can still
            // hold a key <= hi.
            while (true)
            {
                key_type curr_max = key_type(0);
                smallsize curr_size = 0;
                smallsize next_ptr = 0;

                if (lane == 0)
                {
                    curr_max = cg::extract<key_type>(curr_node, 0);
                    curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
                    next_ptr = cg::extract<smallsize>(curr_node, lastpos_off);
                }
                curr_max = tile.shfl(curr_max, 0);
                curr_size = tile.shfl(curr_size, 0);
                next_ptr = tile.shfl(next_ptr, 0);

                key_type my_key = key_type(0);
                smallsize my_offset = 0;
                const bool lane_live = (lane < curr_size);
                if (lane_live)
                {
                    my_key = extract_key_node<key_type>(curr_node, lane + 1);
                    my_offset = extract_offset_node<key_type>(curr_node, lane + 1);
                }

                smallsize partial =
                    (lane_live && my_key >= lo && my_key <= hi) ? my_offset : 0;

#pragma unroll
                for (int delta = tile.size() / 2; delta > 0; delta >>= 1)
                    partial += tile.shfl_down(partial, delta);

                if (lane == 0)
                    query_sum += static_cast<unsigned long long>(partial);

                if (curr_max >= hi)
                {
                    range_fully_covered = true;
                    break; // this node already reaches hi -- nothing further needed
                }

                if (next_ptr == 0)
                    break; // this bucket's chain is exhausted, hi not yet reached

                if (lane == 0)
                {
                    if ((next_ptr - 1) >= allocation_count)
                    {
                        ERROR_INSERTS(
                            "ERROR: invalid link in node chain during naive tile range sum",
                            hi, lane, current_bucket_id);
                    }
                    curr_node = static_cast<uint8_t *>(allocation_buffer) +
                                static_cast<size_t>(next_ptr - 1) * node_stride;
                }
                uintptr_t p = reinterpret_cast<uintptr_t>(curr_node);
                p = tile.shfl(p, 0);
                curr_node = reinterpret_cast<void *>(p);
                tile.sync();
            }

            if (range_fully_covered)
                break; // done -- the range is fully covered

            // Otherwise this bucket is exhausted but hi isn't reached yet --
            // move to the next bucket and keep summing from its first node.
            ++current_bucket_id;
            if (current_bucket_id >= partition_count_with_overflow)
                break; // ran out of buckets; nothing further to sum

            curr_node = get_bucket_head_node<key_type>(launch_params, current_bucket_id);
            tile.sync();
        }

        if (lane == 0)
            results[qi] = static_cast<smallsize>(query_sum);

        tile.sync();
    }
}

template <typename key_type>
GLOBALQUALIFIER void lookup_kernel_tile_ordered_rq_sums_naive(
    updatable_cg_params *__restrict__ launch_params,
    const key_type *__restrict__ lower_list,  // sorted by lower bound
    const key_type *__restrict__ upper_list,  // paired with lower_list
    smallsize *__restrict__ results,          // output
    smallsize query_size)
{
    const key_type *__restrict__ maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    const smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;

    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    const int tiles_per_block = blockDim.x / tile.size();
    const int tile_id = blockIdx.x * tiles_per_block + tile.meta_group_rank();
    const smallsize lane = tile.thread_rank();

    if (tile_id >= partition_count_with_overflow)
        return;

    const key_type maxkey = maxbuf[tile_id];
    const key_type minkey = (tile_id > 0) ? (maxbuf[tile_id - 1] + static_cast<key_type>(1)) : static_cast<key_type>(1);

    // Find the range-query subrange owned by this bucket using the lower
    // bound, exactly the way lookup_kernel_tile_ordered finds the exact-match
    // subrange owned by each bucket.
    int minindex = -1;
    int should_process = 0;

    if (lane == 0)
    {
        minindex = binarySearchIndex_leftmost_ge_strict<key_type>(lower_list, minkey, 0, query_size);
        should_process =
            (minindex >= 0 && lower_list[minindex] <= maxkey) ? 1 : 0;
    }
    minindex = tile.shfl(minindex, 0);
    should_process = tile.shfl(should_process, 0);

    if (!should_process)
        return;

    uint8_t *__restrict__ curr_node =
        static_cast<uint8_t *>(launch_params->ordered_node_pairs) +
        static_cast<size_t>(launch_params->node_stride) * static_cast<size_t>(tile_id);

    process_lookup_tile_bulk_ordered_rq_sums_naive<key_type>(
        /*bucket_min*/ minkey,
        /*bucket_max*/ maxkey,
        /*minindex*/   static_cast<smallsize>(minindex),
        launch_params,
        /*curr_node*/  curr_node,
        /*tile*/       tile,
        /*lower_list*/ lower_list,
        /*upper_list*/ upper_list,
        /*query_size*/ query_size,
        /*writeback*/  results);
}

template <typename key_type>
DEVICEQUALIFIER int find_range_bucket_from_maxvalues(
    const key_type *__restrict__ maxbuf,
    key_type key,
    smallsize partition_count_with_overflow)
{
    if (partition_count_with_overflow == 0)
        return -1;

    int lo = 0;
    int hi = static_cast<int>(partition_count_with_overflow);

    while (lo < hi)
    {
        const int mid = lo + ((hi - lo) >> 1);
        if (maxbuf[mid] >= key)
            hi = mid;
        else
            lo = mid + 1;
    }

    if (lo >= static_cast<int>(partition_count_with_overflow))
        return static_cast<int>(partition_count_with_overflow) - 1;

    return lo;
}

template <typename key_type>
DEVICEQUALIFIER void *get_bucket_head_node(
    updatable_cg_params *__restrict__ launch_params,
    smallsize bucket_index)
{
    return static_cast<uint8_t *>(launch_params->ordered_node_pairs) +
           static_cast<size_t>(bucket_index) *
               static_cast<size_t>(launch_params->node_stride);
}

template <typename key_type>
DEVICEQUALIFIER void *get_next_range_node(
    updatable_cg_params *__restrict__ launch_params,
    void *curr_node,
    smallsize lastpos_off,
    smallsize query_index)
{
    const smallsize next_ptr = cg::extract<smallsize>(curr_node, lastpos_off);
    if (next_ptr == 0)
        return nullptr;

    if ((next_ptr - 1) >= launch_params->allocation_buffer_count)
    {
        ERROR_INSERTS(
            "ERROR: invalid link in node chain during single-thread range lookup",
            query_index, next_ptr, launch_params->allocation_buffer_count);
        return nullptr;
    }

    return static_cast<uint8_t *>(launch_params->allocation_buffer) +
           (static_cast<size_t>(next_ptr) - 1) *
               static_cast<size_t>(launch_params->node_stride);
}

template <typename key_type>
DEVICEQUALIFIER unsigned long long sum_offsets_in_bucket_range_single_thread(
    updatable_cg_params *__restrict__ launch_params,
    smallsize bucket_index,
    key_type lower,
    key_type upper,
    smallsize query_index)
{
    const smallsize node_size = launch_params->node_size;
    const smallsize lastpos_off = get_lastposition_bytes<key_type>(node_size);

    void *curr_node = get_bucket_head_node<key_type>(launch_params, bucket_index);

    while (curr_node != nullptr)
    {
        const key_type curr_max = cg::extract<key_type>(curr_node, 0);
        if (curr_max >= lower)
            break;

        curr_node = get_next_range_node<key_type>(
            launch_params, curr_node, lastpos_off, query_index);
    }

    unsigned long long sum = 0;

    while (curr_node != nullptr)
    {
        const key_type curr_max = cg::extract<key_type>(curr_node, 0);
        const smallsize curr_size =
            cg::extract<smallsize>(curr_node, sizeof(key_type));

        for (smallsize slot = 1; slot <= curr_size; ++slot)
        {
            const key_type key = extract_key_node<key_type>(curr_node, slot);

            if (key > upper)
                return sum;

            if (key >= lower)
            {
                sum += static_cast<unsigned long long>(
                    extract_offset_node<key_type>(curr_node, slot));
            }
        }

        if (curr_max >= upper)
            break;

        curr_node = get_next_range_node<key_type>(
            launch_params, curr_node, lastpos_off, query_index);
    }

    return sum;
}

template <typename key_type>
GLOBALQUALIFIER void lookup_kernel_tile_ordered_rq_sums(
    updatable_cg_params *__restrict__ launch_params,
    const key_type *__restrict__ lowest_list,
    const key_type *__restrict__ highest_list,
    const smallsize *__restrict__ bucket_offset_sums,
    smallsize *__restrict__ results,
    smallsize query_size)
{
    const smallsize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= query_size)
        return;

    if (launch_params == nullptr || bucket_offset_sums == nullptr ||
        lowest_list == nullptr || highest_list == nullptr || results == nullptr)
        return;

    const key_type lower = lowest_list[tid];
    const key_type upper = highest_list[tid];

    if (lower > upper)
    {
        results[tid] = 0;
        return;
    }

    const key_type *__restrict__ maxbuf =
        static_cast<const key_type *>(launch_params->maxvalues);
    const smallsize partition_count_with_overflow =
        launch_params->partition_count_with_overflow;

    const int lower_bucket = find_range_bucket_from_maxvalues<key_type>(
        maxbuf, lower, partition_count_with_overflow);
    const int upper_bucket = find_range_bucket_from_maxvalues<key_type>(
        maxbuf, upper, partition_count_with_overflow);

    if (lower_bucket < 0 || upper_bucket < 0)
    {
        results[tid] = 0;
        return;
    }

    unsigned long long sum = 0;

    if (lower_bucket == upper_bucket)
    {
        sum = sum_offsets_in_bucket_range_single_thread<key_type>(
            launch_params,
            static_cast<smallsize>(lower_bucket),
            lower,
            upper,
            tid);
    }
    else
    {
        sum += sum_offsets_in_bucket_range_single_thread<key_type>(
            launch_params,
            static_cast<smallsize>(lower_bucket),
            lower,
            maxbuf[lower_bucket],
            tid);

        for (int bucket = lower_bucket + 1; bucket < upper_bucket; ++bucket)
        {
            sum += static_cast<unsigned long long>(bucket_offset_sums[bucket]);
        }

        const key_type last_bucket_min =
            (upper_bucket > 0)
                ? (maxbuf[upper_bucket - 1] + static_cast<key_type>(1))
                : static_cast<key_type>(1);

        sum += sum_offsets_in_bucket_range_single_thread<key_type>(
            launch_params,
            static_cast<smallsize>(upper_bucket),
            last_bucket_min,
            upper,
            tid);
    }

    results[tid] = static_cast<smallsize>(sum);
}



// path: gpu_range_lookup.cuh


// -----------------------------------------------------------------------------
// Prefix-sum range aggregation (PREFIX_RANGE_QUERIES)
//
// Precompute (one tile per bucket, then one device scan):
//   bucket_prefix[b]      = sum of all offsets in buckets 0..b      (inclusive)
//   node_excl_prefix[n]   = sum of offsets in the nodes that precede node n
//                           in its own bucket's chain               (exclusive)
// Node ids: a bucket's head node has id b; an overflow node at allocation
// index i (next_ptr = i + 1) has id partition_count_with_overflow + i.
//
// Query (one thread per range): with P(x) = sum of offsets of keys <= x,
//   result = P(hi) - P(lo - 1)
//   P(x)   = bucket_prefix[b - 1] + node_excl_prefix[n] + (keys <= x in node n)
// where b is the first bucket whose max >= x (binary search on maxvalues) and
// n is the first node in b's chain whose max >= x (header-only walk). Nodes
// strictly before n are covered by node_excl_prefix, so only one node's keys
// are ever examined, independent of the range width.
//
// All sums are 32-bit (smallsize) modular, matching the 32-bit result type:
// differences of prefixes give the exact sum mod 2^32.
// -----------------------------------------------------------------------------

template <typename key_type>
GLOBALQUALIFIER void precompute_bucket_and_node_prefix_kernel(
    updatable_cg_params *__restrict__ launch_params,
    smallsize *__restrict__ bucket_sums,
    smallsize *__restrict__ node_excl_prefix)
{
    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    const int tiles_per_block = blockDim.x / tile.size();
    const int tile_id = blockIdx.x * tiles_per_block + tile.meta_group_rank();
    const smallsize lane = tile.thread_rank();

    const smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;
    if (tile_id >= partition_count_with_overflow)
        return;

    const smallsize node_stride = launch_params->node_stride;
    const smallsize node_size = launch_params->node_size;
    const smallsize allocation_count = launch_params->allocation_buffer_count;
    const smallsize lastpos_off = get_lastposition_bytes<key_type>(node_size);

    void *curr_node =
        static_cast<uint8_t *>(launch_params->ordered_node_pairs) +
        static_cast<size_t>(tile_id) * static_cast<size_t>(node_stride);
    smallsize node_id = static_cast<smallsize>(tile_id);

    smallsize running = 0; // sum of the nodes before curr_node in this bucket

    while (true)
    {
        const smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        smallsize lane_sum = 0;
        for (smallsize slot = lane + 1; slot <= curr_size; slot += tile.size())
            lane_sum += extract_offset_node<key_type>(curr_node, slot);

        const smallsize node_sum = coop_g::reduce(tile, lane_sum, coop_g::plus<smallsize>());

        smallsize next_ptr = 0;
        if (lane == 0)
        {
            node_excl_prefix[node_id] = running;
            next_ptr = cg::extract<smallsize>(curr_node, lastpos_off);
        }
        running += node_sum;
        next_ptr = tile.shfl(next_ptr, 0);

        if (next_ptr == 0 || (next_ptr - 1) >= allocation_count)
            break;

        curr_node = static_cast<uint8_t *>(launch_params->allocation_buffer) +
                    (static_cast<size_t>(next_ptr) - 1) * static_cast<size_t>(node_stride);
        node_id = partition_count_with_overflow + (next_ptr - 1);
    }

    if (lane == 0)
        bucket_sums[tile_id] = running;
}

// sum of offsets of all keys <= x
template <typename key_type>
DEVICEQUALIFIER smallsize flix_prefix_sum_upto(
    updatable_cg_params *__restrict__ launch_params,
    const key_type *__restrict__ maxbuf,
    const smallsize *__restrict__ bucket_prefix,
    const smallsize *__restrict__ node_excl_prefix,
    smallsize partition_count_with_overflow,
    smallsize lastpos_off,
    key_type x)
{
    const int b = find_range_bucket_from_maxvalues<key_type>(maxbuf, x, partition_count_with_overflow);
    if (b < 0)
        return 0;

    const smallsize node_stride = launch_params->node_stride;
    const smallsize allocation_count = launch_params->allocation_buffer_count;

    const uint8_t *curr_node =
        static_cast<const uint8_t *>(launch_params->ordered_node_pairs) +
        static_cast<size_t>(b) * static_cast<size_t>(node_stride);
    smallsize node_id = static_cast<smallsize>(b);

    // header-only walk to the first node whose max >= x (or the last node)
    while (cg::extract<key_type>(curr_node, 0) < x)
    {
        const smallsize next_ptr = cg::extract<smallsize>(curr_node, lastpos_off);
        if (next_ptr == 0 || (next_ptr - 1) >= allocation_count)
            break;
        curr_node = static_cast<const uint8_t *>(launch_params->allocation_buffer) +
                    (static_cast<size_t>(next_ptr) - 1) * static_cast<size_t>(node_stride);
        node_id = partition_count_with_overflow + (next_ptr - 1);
    }

    smallsize sum = (b > 0 ? bucket_prefix[b - 1] : smallsize(0)) + node_excl_prefix[node_id];

    const smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
    for (smallsize slot = 1; slot <= curr_size; ++slot)
    {
        if (extract_key_node<key_type>((void *)curr_node, slot) <= x)
            sum += extract_offset_node<key_type>((void *)curr_node, slot);
    }
    return sum;
}

template <typename key_type>
GLOBALQUALIFIER void lookup_kernel_rq_prefix_sums(
    updatable_cg_params *__restrict__ launch_params,
    const key_type *__restrict__ lowest_list,
    const key_type *__restrict__ highest_list,
    const smallsize *__restrict__ bucket_prefix,
    const smallsize *__restrict__ node_excl_prefix,
    smallsize *__restrict__ results,
    smallsize query_size)
{
    const smallsize tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= query_size)
        return;

    const key_type lower = lowest_list[tid];
    const key_type upper = highest_list[tid];
    if (lower > upper)
    {
        results[tid] = 0;
        return;
    }

    const key_type *__restrict__ maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    const smallsize pcwo = launch_params->partition_count_with_overflow;
    const smallsize lastpos_off = get_lastposition_bytes<key_type>(launch_params->node_size);

    const smallsize upto_upper = flix_prefix_sum_upto<key_type>(
        launch_params, maxbuf, bucket_prefix, node_excl_prefix, pcwo, lastpos_off, upper);
    const smallsize below_lower = (lower == key_type(0)) ? smallsize(0) : flix_prefix_sum_upto<key_type>(
        launch_params, maxbuf, bucket_prefix, node_excl_prefix, pcwo, lastpos_off, static_cast<key_type>(lower - 1));

    results[tid] = upto_upper - below_lower;
}


// -----------------------------------------------------------------------------
// Naive scan v2 (NAIVE_RANGE_QUERIES_V2): still visits every key in [lo, hi],
// like NAIVE_RANGE_QUERIES, but
//   - one tile per *query* with a ballot work queue (as the LSM/B-tree scans
//     do), instead of one tile per bucket serially processing every query whose
//     lower bound falls in that bucket (load imbalance under skew);
//   - every lane reads the node header itself (same cache line, broadcast)
//     instead of lane 0 reading it and shuffling it out;
//   - per-lane running sums, reduced once per query instead of once per node,
//     and no tile.sync() per node;
//   - tile width RQ_V2_TILE (default 32) independent of TILE_SIZE.
// -----------------------------------------------------------------------------
#ifndef RQ_V2_TILE
#define RQ_V2_TILE 32
#endif

template <typename key_type, int tile_width>
GLOBALQUALIFIER void lookup_kernel_rq_scan_v2(
    updatable_cg_params *__restrict__ launch_params,
    const key_type *__restrict__ lower_list,
    const key_type *__restrict__ upper_list,
    smallsize *__restrict__ results,
    smallsize query_size)
{
    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<tile_width> tile = coop_g::tiled_partition<tile_width>(block);

    const smallsize tid = blockIdx.x * blockDim.x + threadIdx.x;
    const smallsize lane = tile.thread_rank();
    if ((tid - lane) >= query_size)
        return;

    const key_type *__restrict__ maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    const smallsize pcwo = launch_params->partition_count_with_overflow;
    const smallsize node_stride = launch_params->node_stride;
    const smallsize allocation_count = launch_params->allocation_buffer_count;
    const smallsize lastpos_off = get_lastposition_bytes<key_type>(launch_params->node_size);
    const uint8_t *node_base = static_cast<const uint8_t *>(launch_params->ordered_node_pairs);
    const uint8_t *alloc_base = static_cast<const uint8_t *>(launch_params->allocation_buffer);

    key_type my_lo = 0, my_hi = 0;
    bool to_find = false;
    if (tid < query_size)
    {
        my_lo = lower_list[tid];
        my_hi = upper_list[tid];
        to_find = true;
    }
    smallsize my_result = 0;

    auto work_queue = tile.ballot(to_find);
    while (work_queue)
    {
        const int r = __ffs(work_queue) - 1;
        const key_type lo = tile.shfl(my_lo, r);
        const key_type hi = tile.shfl(my_hi, r);
        smallsize acc = 0;

        if (lo <= hi)
        {
            smallsize b = static_cast<smallsize>(find_range_bucket_from_maxvalues<key_type>(maxbuf, lo, pcwo));
            const uint8_t *node = node_base + static_cast<size_t>(b) * node_stride;
            bool done = false;

            // header-only skip to the first node whose max >= lo
            while (true)
            {
                if (cg::extract<key_type>((void *)node, 0) >= lo)
                    break;
                const smallsize next_ptr = cg::extract<smallsize>((void *)node, lastpos_off);
                if (next_ptr != 0 && (next_ptr - 1) < allocation_count)
                {
                    node = alloc_base + static_cast<size_t>(next_ptr - 1) * node_stride;
                }
                else
                {
                    if (++b >= pcwo) { done = true; break; }
                    node = node_base + static_cast<size_t>(b) * node_stride;
                }
            }

            while (!done)
            {
                const key_type node_max = cg::extract<key_type>((void *)node, 0);
                const smallsize node_count = cg::extract<smallsize>((void *)node, sizeof(key_type));
                const smallsize next_ptr = cg::extract<smallsize>((void *)node, lastpos_off);

                for (smallsize slot = lane + 1; slot <= node_count; slot += tile_width)
                {
                    const key_type k = extract_key_node<key_type>((void *)node, slot);
                    if (k >= lo && k <= hi)
                        acc += extract_offset_node<key_type>((void *)node, slot);
                }

                if (node_max >= hi)
                    break;
                if (next_ptr != 0 && (next_ptr - 1) < allocation_count)
                {
                    node = alloc_base + static_cast<size_t>(next_ptr - 1) * node_stride;
                }
                else
                {
                    if (++b >= pcwo) break;
                    node = node_base + static_cast<size_t>(b) * node_stride;
                }
            }
        }

        const smallsize total = coop_g::reduce(tile, acc, coop_g::plus<smallsize>());
        if (lane == static_cast<smallsize>(r))
        {
            my_result = total;
            to_find = false;
        }
        work_queue = tile.ballot(to_find);
    }

    if (tid < query_size)
        results[tid] = my_result;
}


// -----------------------------------------------------------------------------
// Naive scan v3 (NAIVE_RANGE_QUERIES_V3): v2, plus bucket-parallel interiors for
// large ranges. Still visits every key in [lo, hi] (no precomputed sums).
// If the range spans at least RQ_V3_MIN_INTERIOR buckets strictly between the
// buckets of lo and hi, those interior buckets are fully covered: each lane
// walks a different interior bucket's chain (all offsets, no key compares), so
// many independent pointer chases are in flight instead of one serial walk.
// The two boundary buckets use the cooperative, filtered scan. Smaller ranges
// take exactly the v2 path.
// -----------------------------------------------------------------------------
#ifndef RQ_V3_MIN_INTERIOR
#define RQ_V3_MIN_INTERIOR 4
#endif


// First bucket whose max >= key, searching forward from `hint` (a bucket known to
// satisfy maxbuf[hint - 1] < key, e.g. the previous, smaller query's bucket):
// exponential steps then a binary search, so nearby keys cost O(log distance)
// dependent loads instead of O(log #buckets). Falls back to a full search when
// the hint does not apply.
template <typename key_type>
DEVICEQUALIFIER smallsize find_range_bucket_galloping(
    const key_type *__restrict__ maxbuf, key_type key, smallsize hint, smallsize pcwo)
{
    if (hint >= pcwo || (hint > 0 && maxbuf[hint - 1] >= key))
        return static_cast<smallsize>(find_range_bucket_from_maxvalues<key_type>(maxbuf, key, pcwo));
    // invariant: answer >= hint
    if (maxbuf[hint] >= key)
        return hint;
    smallsize lo = hint + 1; // answer in [lo, hi]
    smallsize step = 1;
    smallsize hi = hint + 1;
    while (hi < pcwo && maxbuf[hi] < key)
    {
        lo = hi + 1;
        step <<= 1;
        hi = hint + step;
    }
    if (hi >= pcwo)
        hi = pcwo - 1;
    if (lo > hi) // every bucket max < key: clamp like find_range_bucket_from_maxvalues
        return pcwo - 1;
    while (lo < hi)
    {
        const smallsize mid = lo + ((hi - lo) >> 1);
        if (maxbuf[mid] >= key)
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

template <typename key_type, int tile_width>
DEVICEQUALIFIER smallsize rq_v3_scan_chain_filtered(
    const uint8_t *node, key_type lo, key_type hi, smallsize lane,
    smallsize lastpos_off, smallsize node_stride, smallsize allocation_count,
    const uint8_t *alloc_base)
{
    smallsize acc = 0;
    while (node != nullptr)
    {
        const key_type node_max = cg::extract<key_type>((void *)node, 0);
        const smallsize node_count = cg::extract<smallsize>((void *)node, sizeof(key_type));
        const smallsize next_ptr = cg::extract<smallsize>((void *)node, lastpos_off);
        if (node_max >= lo)
        {
            for (smallsize slot = lane + 1; slot <= node_count; slot += tile_width)
            {
                const key_type k = extract_key_node<key_type>((void *)node, slot);
                if (k >= lo && k <= hi)
                    acc += extract_offset_node<key_type>((void *)node, slot);
            }
            if (node_max >= hi)
                break;
        }
        node = (next_ptr != 0 && (next_ptr - 1) < allocation_count)
                   ? alloc_base + static_cast<size_t>(next_ptr - 1) * node_stride
                   : nullptr;
    }
    return acc;
}

template <typename key_type, int tile_width>
GLOBALQUALIFIER void lookup_kernel_rq_scan_v3(
    updatable_cg_params *__restrict__ launch_params,
    const key_type *__restrict__ lower_list,
    const key_type *__restrict__ upper_list,
    smallsize *__restrict__ results,
    smallsize query_size)
{
    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<tile_width> tile = coop_g::tiled_partition<tile_width>(block);

    const smallsize tid = blockIdx.x * blockDim.x + threadIdx.x;
    const smallsize lane = tile.thread_rank();
    if ((tid - lane) >= query_size)
        return;

    const key_type *__restrict__ maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    const smallsize pcwo = launch_params->partition_count_with_overflow;
    const smallsize node_stride = launch_params->node_stride;
    const smallsize allocation_count = launch_params->allocation_buffer_count;
    const smallsize lastpos_off = get_lastposition_bytes<key_type>(launch_params->node_size);
    const uint8_t *node_base = static_cast<const uint8_t *>(launch_params->ordered_node_pairs);
    const uint8_t *alloc_base = static_cast<const uint8_t *>(launch_params->allocation_buffer);

    key_type my_lo = 0, my_hi = 0;
    bool to_find = false;
    if (tid < query_size)
    {
        my_lo = lower_list[tid];
        my_hi = upper_list[tid];
        to_find = true;
    }
    smallsize my_result = 0;
#ifdef RQ_V3_GALLOP
    smallsize bucket_hint = 0; // queries are processed in sorted order within a tile
#endif

    auto work_queue = tile.ballot(to_find);
    while (work_queue)
    {
        const int r = __ffs(work_queue) - 1;
        const key_type lo = tile.shfl(my_lo, r);
        const key_type hi = tile.shfl(my_hi, r);
        smallsize acc = 0;

        if (lo <= hi)
        {
#ifdef RQ_V3_GALLOP
            smallsize b = find_range_bucket_galloping<key_type>(maxbuf, lo, bucket_hint, pcwo);
            bucket_hint = b;
            const smallsize b_hi = find_range_bucket_galloping<key_type>(maxbuf, hi, b, pcwo);
#else
            smallsize b = static_cast<smallsize>(find_range_bucket_from_maxvalues<key_type>(maxbuf, lo, pcwo));
            const smallsize b_hi = static_cast<smallsize>(find_range_bucket_from_maxvalues<key_type>(maxbuf, hi, pcwo));
#endif

            if (b_hi >= b + 1 + RQ_V3_MIN_INTERIOR)
            {
                // boundary buckets: cooperative filtered scans
                acc += rq_v3_scan_chain_filtered<key_type, tile_width>(
                    node_base + static_cast<size_t>(b) * node_stride, lo, hi, lane,
                    lastpos_off, node_stride, allocation_count, alloc_base);
                acc += rq_v3_scan_chain_filtered<key_type, tile_width>(
                    node_base + static_cast<size_t>(b_hi) * node_stride, lo, hi, lane,
                    lastpos_off, node_stride, allocation_count, alloc_base);

                // interior buckets: one lane per bucket, whole chains, no key compares
                for (smallsize ib = b + 1 + lane; ib < b_hi; ib += tile_width)
                {
                    const uint8_t *node = node_base + static_cast<size_t>(ib) * node_stride;
                    while (true)
                    {
                        const smallsize node_count = cg::extract<smallsize>((void *)node, sizeof(key_type));
                        const smallsize next_ptr = cg::extract<smallsize>((void *)node, lastpos_off);
                        for (smallsize slot = 1; slot <= node_count; ++slot)
                            acc += extract_offset_node<key_type>((void *)node, slot);
                        if (next_ptr == 0 || (next_ptr - 1) >= allocation_count)
                            break;
                        node = alloc_base + static_cast<size_t>(next_ptr - 1) * node_stride;
                    }
                }
            }
            else
            {
                // small range: identical to v2
                const uint8_t *node = node_base + static_cast<size_t>(b) * node_stride;
                bool done = false;
                while (true)
                {
                    if (cg::extract<key_type>((void *)node, 0) >= lo)
                        break;
                    const smallsize next_ptr = cg::extract<smallsize>((void *)node, lastpos_off);
                    if (next_ptr != 0 && (next_ptr - 1) < allocation_count)
                        node = alloc_base + static_cast<size_t>(next_ptr - 1) * node_stride;
                    else
                    {
                        if (++b >= pcwo) { done = true; break; }
                        node = node_base + static_cast<size_t>(b) * node_stride;
                    }
                }
                while (!done)
                {
                    const key_type node_max = cg::extract<key_type>((void *)node, 0);
                    const smallsize node_count = cg::extract<smallsize>((void *)node, sizeof(key_type));
                    const smallsize next_ptr = cg::extract<smallsize>((void *)node, lastpos_off);
                    for (smallsize slot = lane + 1; slot <= node_count; slot += tile_width)
                    {
                        const key_type k = extract_key_node<key_type>((void *)node, slot);
                        if (k >= lo && k <= hi)
                            acc += extract_offset_node<key_type>((void *)node, slot);
                    }
                    if (node_max >= hi)
                        break;
                    if (next_ptr != 0 && (next_ptr - 1) < allocation_count)
                        node = alloc_base + static_cast<size_t>(next_ptr - 1) * node_stride;
                    else
                    {
                        if (++b >= pcwo) break;
                        node = node_base + static_cast<size_t>(b) * node_stride;
                    }
                }
            }
        }

        const smallsize total = coop_g::reduce(tile, acc, coop_g::plus<smallsize>());
        if (lane == static_cast<smallsize>(r))
        {
            my_result = total;
            to_find = false;
        }
        work_queue = tile.ballot(to_find);
    }

    if (tid < query_size)
        results[tid] = my_result;
}

template <typename Tile>
DEVICEQUALIFIER unsigned long long tile_reduce_sum_ull(Tile tile, unsigned long long value)
{
    for (int delta = tile.size() / 2; delta > 0; delta >>= 1)
    {
        value += tile.shfl_down(value, delta);
    }
    return value;
}

template <typename key_type>
GLOBALQUALIFIER void precompute_bucket_offset_sums_kernel(
    updatable_cg_params *__restrict__ launch_params,
    smallsize *__restrict__ bucket_offset_sums)
{
    if (launch_params == nullptr || bucket_offset_sums == nullptr)
        return;

    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile = coop_g::tiled_partition<TILE_SIZE>(block);

    const int tiles_per_block = blockDim.x / tile.size();
    const int tile_id = blockIdx.x * tiles_per_block + tile.meta_group_rank();
    const smallsize lane = tile.thread_rank();

    if (tile_id >= launch_params->partition_count_with_overflow)
        return;

    const smallsize node_stride = launch_params->node_stride;
    const smallsize node_size = launch_params->node_size;
    const smallsize allocation_count = launch_params->allocation_buffer_count;
    const smallsize lastpos_off = get_lastposition_bytes<key_type>(node_size);

    void *curr_node =
        static_cast<uint8_t *>(launch_params->ordered_node_pairs) +
        static_cast<size_t>(tile_id) * static_cast<size_t>(node_stride);

    unsigned long long bucket_sum = 0;

    while (true)
    {
        const smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        unsigned long long lane_sum = 0;
        for (smallsize slot = static_cast<smallsize>(lane) + 1; slot <= curr_size; slot += tile.size())
        {
            lane_sum += static_cast<unsigned long long>(
                extract_offset_node<key_type>(curr_node, slot));
        }

        const unsigned long long node_sum = tile_reduce_sum_ull(tile, lane_sum);

        if (lane == 0)
        {
            bucket_sum += node_sum;
        }

        tile.sync();

        smallsize next_ptr = 0;
        if (lane == 0)
        {
            next_ptr = cg::extract<smallsize>(curr_node, lastpos_off);
        }
        next_ptr = tile.shfl(next_ptr, 0);

        if (next_ptr == 0)
            break;

        if (lane == 0)
        {
            if ((next_ptr - 1) >= allocation_count)
            {
                printf("ERROR: invalid link in precompute_bucket_offset_sums_kernel, tile_id=%d, next_ptr=%u, allocation_count=%u\n",
                       tile_id,
                       static_cast<unsigned>(next_ptr),
                       static_cast<unsigned>(allocation_count));
                curr_node = nullptr;
            }
            else
            {
                curr_node =
                    static_cast<uint8_t *>(launch_params->allocation_buffer) +
                    (static_cast<size_t>(next_ptr) - 1) * static_cast<size_t>(node_stride);
            }
        }

        uintptr_t p = reinterpret_cast<uintptr_t>(curr_node);
        p = tile.shfl(p, 0);
        curr_node = reinterpret_cast<void *>(p);

        if (curr_node == nullptr)
            break;

        tile.sync();
    }

    if (lane == 0)
    {
        bucket_offset_sums[tile_id] = static_cast<smallsize>(bucket_sum);
    }
}

// Separate prefix-sum path. Existing range kernels above remain unchanged.
// Keep totals wide before scanning: widening a 32-bit bucket sum afterwards
// would already have lost any carry. Reuse the existing node traversal and
// cooperative reduction helpers.
template <typename key_type>
GLOBALQUALIFIER void precompute_bucket_offset_sums_u64_kernel(
    updatable_cg_params *__restrict__ launch_params,
    bigsize *__restrict__ bucket_offset_sums)
{
    coop_g::thread_block block = coop_g::this_thread_block();
    coop_g::thread_block_tile<TILE_SIZE> tile =
        coop_g::tiled_partition<TILE_SIZE>(block);
    const size_t bucket = static_cast<size_t>(blockIdx.x) *
                              (blockDim.x / tile.size()) +
                          tile.meta_group_rank();
    if (bucket >= launch_params->partition_count_with_overflow)
        return;

    const smallsize lane = tile.thread_rank();
    const smallsize lastpos_off =
        get_lastposition_bytes<key_type>(launch_params->node_size);
    void *node = get_bucket_head_node<key_type>(
        launch_params, static_cast<smallsize>(bucket));
    unsigned long long bucket_sum = 0;

    while (node != nullptr)
    {
        const smallsize count = cg::extract<smallsize>(node, sizeof(key_type));
        unsigned long long lane_sum = 0;
        for (smallsize slot = lane + 1; slot <= count; slot += tile.size())
            lane_sum += static_cast<unsigned long long>(
                extract_offset_node<key_type>(node, slot));

        const unsigned long long node_sum = tile_reduce_sum_ull(tile, lane_sum);
        if (lane == 0)
            bucket_sum += node_sum;

        // Only the owner loads the next link; all lanes take the same walk.
        if (lane == 0)
            node = get_next_range_node<key_type>(
                launch_params, node, lastpos_off, static_cast<smallsize>(bucket));
        uintptr_t next = reinterpret_cast<uintptr_t>(node);
        next = tile.shfl(next, 0);
        node = reinterpret_cast<void *>(next);
    }

    if (lane == 0)
        bucket_offset_sums[bucket] = static_cast<bigsize>(bucket_sum);
}

// One independent thread per query, with inclusive key bounds [lower, upper].
// prefix[0] = 0; prefix[b + 1] contains the sum of buckets [0, b].
// Thus strictly interior buckets cost two reads, independent of range length.
template <typename key_type>
GLOBALQUALIFIER void lookup_kernel_range_prefix_sums(
    updatable_cg_params *__restrict__ launch_params,
    const key_type *__restrict__ lower_list,
    const key_type *__restrict__ upper_list,
    const bigsize *__restrict__ bucket_prefix_sums,
    bigsize *__restrict__ results,
    smallsize query_size)
{
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= query_size)
        return;

    const key_type lower = lower_list[tid];
    const key_type upper = upper_list[tid];
    const smallsize bucket_count = launch_params->partition_count_with_overflow;
    if (lower > upper || bucket_count == 0)
    {
        results[tid] = 0;
        return;
    }

    const key_type *maxbuf = static_cast<const key_type *>(launch_params->maxvalues);
    const int first = find_range_bucket_from_maxvalues<key_type>(maxbuf, lower, bucket_count);
    const int last = find_range_bucket_from_maxvalues<key_type>(maxbuf, upper, bucket_count);
    const smallsize query_index = static_cast<smallsize>(tid);

    if (first == last)
    {
        results[tid] = static_cast<bigsize>(
            sum_offsets_in_bucket_range_single_thread<key_type>(
                launch_params, static_cast<smallsize>(first), lower, upper, query_index));
        return;
    }

    bigsize sum = static_cast<bigsize>(
        sum_offsets_in_bucket_range_single_thread<key_type>(
            launch_params, static_cast<smallsize>(first), lower, maxbuf[first], query_index));
    sum += bucket_prefix_sums[last] - bucket_prefix_sums[first + 1];
    const key_type last_bucket_min = maxbuf[last - 1] + static_cast<key_type>(1);
    sum += static_cast<bigsize>(
        sum_offsets_in_bucket_range_single_thread<key_type>(
            launch_params, static_cast<smallsize>(last), last_bucket_min, upper, query_index));
    results[tid] = sum;
}

/*
template <typename key_type>
void precompute_bucket_offset_sums(cudaStream_t stream, smallsize partition_count_with_overflow)
{
    const smallsize total_threads_required = partition_count_with_overflow * TILE_SIZE;
    const smallsize max_blocks_required = SDIV(total_threads_required, MAXBLOCKSIZE / DIV_FACTOR);

    precompute_bucket_offset_sums_kernel<key_type>
        <<<max_blocks_required, MAXBLOCKSIZE / DIV_FACTOR, 0, stream>>>(
            launch_params_buffer.ptr());

    CUERR;
}

void range_lookup_sum( launch_params_buffer_type launch_params_buffer,
                      smallsize partition_count_with_overflow, cudaStream_t stream)
{
    const smallsize total_threads_required = partition_count_with_overflow * TILE_SIZE;
    const smallsize max_blocks_required = SDIV(total_threads_required, MAXBLOCKSIZE / DIV_FACTOR);

    precompute_bucket_offset_sums<key_type>(stream);

    printf("RANGE QUERIES LOOKUP SUM TILE ORDERED: div is:%d, max_blocks_required=%d and MAXBLOCKSIZE/DIV %d\n",
           DIV_FACTOR,
           max_blocks_required,
           MAXBLOCKSIZE / DIV_FACTOR);

    lookup_kernel_tile_ordered_rq<key_type>
        <<<max_blocks_required, MAXBLOCKSIZE / DIV_FACTOR, 0, stream>>>(
            launch_params_buffer.ptr(),
            lower,
            upper,
            result,
            size);

    CUERR;
}
*/

#endif
