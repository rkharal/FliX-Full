// =============================================================================
// Concurrent insert/delete/lookup benchmark for FliX, SlabHash and GPU B-tree.
// Copyright (c) 2026 Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================
#ifndef BENCHMARK_UPDATES_CONCURRENT_CUH
#define BENCHMARK_UPDATES_CONCURRENT_CUH

// Reuse the existing uniform key generator, probe generator, configuration,
// CUDA buffers/timer and result collector. No existing benchmark is invoked.
#if !defined(REGKEYGEN) && !defined(DENSEKEYGEN) && !defined(HYBRIDKEYGEN)
#define CONCURRENT_BENCHMARK_TEMP_REGKEYGEN
#define REGKEYGEN
#endif
#include "benchmarks_updates.cuh"
#ifdef CONCURRENT_BENCHMARK_TEMP_REGKEYGEN
#undef REGKEYGEN
#undef CONCURRENT_BENCHMARK_TEMP_REGKEYGEN
#endif
#include "result_collector.h"
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <string_view>

namespace concurrent_benchmark {

#ifdef Insert_mix
inline constexpr int insert_percent = Insert_mix;
#else
inline constexpr int insert_percent = 25;
#endif
#ifdef Del_mix
inline constexpr int delete_percent = Del_mix;
#else
inline constexpr int delete_percent = 25;
#endif
#ifdef Query_mix
inline constexpr int query_percent = Query_mix;
#else
inline constexpr int query_percent = 50;
#endif

// Override these only for smaller validation runs or a later size experiment.
#ifdef CONCURRENT_BATCH_SIZE
inline constexpr size_t batch_size = CONCURRENT_BATCH_SIZE;
#else
inline constexpr size_t batch_size = 100000000;
#endif
#ifdef CONCURRENT_BUILD_SIZE
inline constexpr size_t build_size = CONCURRENT_BUILD_SIZE;
#else
inline constexpr size_t build_size = size_t{1} << 25;
#endif
#ifdef CONCURRENT_ROUNDS
inline constexpr size_t round_count = CONCURRENT_ROUNDS;
#else
inline constexpr size_t round_count = 8;
#endif

static_assert(insert_percent >= 0 && delete_percent >= 0 && query_percent >= 0 &&
              insert_percent + delete_percent + query_percent == 100,
              "Insert_mix, Del_mix and Query_mix must be nonnegative and sum to 100");
static_assert(batch_size > 0 && batch_size <= std::numeric_limits<smallsize>::max(),
              "Concurrent batch size must fit in smallsize");
static_assert(batch_size % 100 == 0, "Use a batch size divisible by 100 for exact percentages");
static_assert(build_size > 0 && round_count >= 1 && round_count <= 16,
              "Use a nonempty build and 1 to 16 concurrent rounds");

inline constexpr size_t inserts = batch_size / 100 * insert_percent;
inline constexpr size_t deletes = batch_size / 100 * delete_percent;
inline constexpr size_t queries = batch_size / 100 * query_percent;
inline constexpr size_t delete_hits = deletes / 2;
inline constexpr size_t delete_misses = deletes - delete_hits;
static_assert(deletes % 2 == 0, "Deletion count must be even for exactly half hits and half misses");
static_assert(delete_hits <= inserts,
              "Successful deletes come from the previous insert batch: need Del_mix <= 2*Insert_mix");
#ifdef CONCURRENT_SMALL_BUILD_ADAPTIVE_DELETES
// Optional small-build workload: reserve at least half the build for hit queries.
// The original fixed-half deletion policy is preserved verbatim in #else.
static_assert(build_size >= 2, "Adaptive deletion mode requires at least two build keys");
inline constexpr size_t first_round_delete_hits =
    delete_hits < build_size / 2 ? delete_hits : build_size / 2;
inline constexpr size_t first_round_delete_misses = deletes - first_round_delete_hits;
inline constexpr size_t immutable_build_size = build_size - first_round_delete_hits;
inline constexpr size_t insertion_pool_end = build_size + round_count * inserts;
// First-round misses are the maximum: reserve that many never-inserted keys.
inline constexpr size_t pool_size = insertion_pool_end + first_round_delete_misses;
inline constexpr size_t peak_live_size =
    build_size + (round_count - 1) * (inserts - delete_hits) + inserts +
    (round_count > 1 ? delete_hits - first_round_delete_hits : 0);
#else
static_assert(delete_hits <= build_size && (queries == 0 || delete_hits < build_size),
              "Build must hold the first successful deletes and an untouched query region");

inline constexpr size_t immutable_build_size = build_size - delete_hits;
inline constexpr size_t insertion_pool_end = build_size + round_count * inserts;
inline constexpr size_t pool_size = insertion_pool_end + delete_misses;
inline constexpr size_t peak_live_size =
    build_size + (round_count - 1) * (inserts - delete_hits) + inserts;
#endif // CONCURRENT_SMALL_BUILD_ADAPTIVE_DELETES
inline constexpr smallsize unwritten = not_found - 1;
static_assert(pool_size < unwritten, "Generated-key offsets must not collide with result sentinels");

// These task values match all three mixed_batch APIs. No range tasks or MSB encoding.
enum task : uint8_t { insert = 0, lookup = 1, erase = 2 };

template <typename Index>
inline constexpr bool is_flix = std::string_view(Index::name) == "cg_rtx_index_updates";
template <typename Index>
inline constexpr bool is_slab = std::string_view(Index::name) == "hashtable_slab";
template <typename Index>
inline constexpr bool supported = is_flix<Index> || is_slab<Index> ||
    std::string_view(Index::name) == "tree_awad";

template <typename Key>
struct operation {
    Key key;
    smallsize value;
    smallsize expected;
    uint8_t kind;
};

// Initial build, every insertion round, and a never-inserted deletion-miss pool
// are disjoint portions of one globally unique, uniformly generated key pool.
// XVAL/YVAL and dense/shifted generator flags deliberately have no effect here.
template <typename Key>
std::vector<Key> make_uniform_pool() {
    if (static_cast<uint64_t>(pool_size) >
        static_cast<uint64_t>(max_usable_key<Key>() - min_usable_key<Key>()) + 1)
        throw std::invalid_argument("Concurrent key pool exceeds the usable key range");
    std::vector<Key> pool;
    generate_keys(pool_size, min_usable_key<Key>(), max_usable_key<Key>(), pool);
    // Randomize pool partitions independently of unordered_set iteration order.
    std::mt19937_64 random(42);
    std::shuffle(pool.begin(), pool.end(), random);
    return pool;
}

// Cache the final shuffled pool, not an intermediate generator representation.
// Increment this format version if the generation/shuffle policy changes.
template <typename Key>
std::array<uint64_t, 14> uniform_cache_header(uint64_t checksum = 0) {
    return {{0x464c4958434f4e43ull, 1, sizeof(Key), build_size, batch_size,
             round_count, insert_percent, delete_percent, query_percent,
             pool_size, 42, min_usable_key<Key>(), max_usable_key<Key>(), checksum}};
}

template <typename Key>
uint64_t uniform_pool_checksum(const std::vector<Key>& pool) {
    uint64_t checksum = 14695981039346656037ull;
    for (const Key key : pool)
        checksum = (checksum ^ static_cast<uint64_t>(key)) * 1099511628211ull;
    return checksum;
}

template <typename Key>
std::filesystem::path uniform_cache_path() {
    namespace fs = std::filesystem;
    fs::path directory;
    const char* configured = std::getenv("FLIX_CONCURRENT_KEY_CACHE");
    if (configured != nullptr && configured[0] != '\0') {
        directory = configured;
    } else {
        const char* user_home = std::getenv("HOME");
        if (user_home == nullptr || user_home[0] == '\0')
            throw std::runtime_error("Set HOME or FLIX_CONCURRENT_KEY_CACHE for the shared key cache");
        // Shared home storage; independent of GPU, index type and build directory.
        // build_and_make clears data_cache/*.*, so do not put this cache there.
        directory = fs::path(user_home) / "OPTIX" / "Concurrent_Key_Cache";
    }
    std::ostringstream filename;
    filename << "uniform_v1_k" << sizeof(Key) * 8 << "_b" << build_size
             << "_n" << batch_size << "_r" << round_count
             << "_i" << insert_percent << "_d" << delete_percent
             << "_q" << query_percent << "_s42.bin";
    return directory / filename.str();
}

template <typename Key>
bool load_uniform_cache(const std::filesystem::path& path, std::vector<Key>& pool) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    const auto expected = uniform_cache_header<Key>();
    auto header = expected;
    const uintmax_t expected_bytes = sizeof(header) +
        static_cast<uintmax_t>(pool_size) * sizeof(Key);
    std::error_code error;
    const uintmax_t file_bytes = std::filesystem::file_size(path, error);
    input.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    if (error || file_bytes != expected_bytes || !input ||
        !std::equal(expected.begin(), expected.end() - 1, header.begin())) {
        std::cout << "[CONCURRENT] Incomplete or incompatible binary cache; regenerating."
                  << std::endl;
        return false;
    }
    std::cout << "[CONCURRENT] Loading " << pool_size << " cached uniform keys..."
              << std::endl;
    pool.resize(pool_size);
    input.read(reinterpret_cast<char*>(pool.data()),
               static_cast<std::streamsize>(pool.size() * sizeof(Key)));
    if (!input || uniform_pool_checksum(pool) != header.back()) {
        std::cout << "[CONCURRENT] Binary cache failed its integrity check; regenerating."
                  << std::endl;
        std::vector<Key>().swap(pool);
        return false;
    }
    std::cout << "[CONCURRENT] Binary key cache loaded; checksum verified." << std::endl;
    return true;
}

template <typename Key>
void save_uniform_cache(const std::filesystem::path& path, const std::vector<Key>& pool) {
    namespace fs = std::filesystem;
    fs::create_directories(path.parent_path());
    const auto header = uniform_cache_header<Key>(uniform_pool_checksum(pool));
    const auto timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    fs::path temporary = path;
    temporary += ".tmp." + std::to_string(timestamp) + "." +
                 std::to_string(std::random_device{}());
    std::cout << "[CONCURRENT] Saving shared binary key cache..." << std::endl;
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("Cannot create binary key cache: " + temporary.string());
        output.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
        output.write(reinterpret_cast<const char*>(pool.data()),
                     static_cast<std::streamsize>(pool.size() * sizeof(Key)));
        output.flush();
        output.close();
        if (!output) throw std::runtime_error("Cannot finish binary key cache: " + temporary.string());
        // Only publish a complete file. Readers never use .tmp files, and a
        // second writer cannot expose a partly written cache to another index.
        fs::rename(temporary, path);
    } catch (...) {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw;
    }
    std::cout << "[CONCURRENT] Shared binary key cache saved: " << path.string()
              << std::endl;
}

template <typename Key>
std::vector<Key> load_or_generate_uniform_pool() {
    const auto path = uniform_cache_path<Key>();
    std::cout << "[CONCURRENT] Binary key cache: " << path.string() << std::endl;
    std::vector<Key> pool;
    if (load_uniform_cache(path, pool)) return pool;
    std::cout << "[CONCURRENT] Generating " << pool_size << " unique uniform keys..."
              << std::endl;
    pool = make_uniform_pool<Key>();
    save_uniform_cache(path, pool);
    return pool;
}

template <typename Key>
std::vector<operation<Key>> make_round(const std::vector<Key>& pool, size_t round) {
    std::vector<operation<Key>> operations;
    operations.reserve(batch_size);
    const size_t insert_start = build_size + round * inserts;
    for (size_t i = 0; i < inserts; ++i) {
        const size_t offset = insert_start + i;
        operations.push_back({pool[offset], static_cast<smallsize>(offset), 1, insert});
    }
    // Round 0: delete a reserved tail of the build. Later: delete from the
    // preceding round's inserts, which completed before this kernel launch.
    const size_t delete_start = round == 0 ? immutable_build_size : insert_start - inserts;
    for (size_t i = 0; i < delete_hits; ++i)
        operations.push_back({pool[delete_start + i], 0, 1, erase});
    // Reusing this never-inserted pool in later rounds always produces misses.
    for (size_t i = 0; i < delete_misses; ++i)
        operations.push_back({pool[insertion_pool_end + i], 0, 0, erase});

    if constexpr (queries > 0) {
        std::vector<Key> probe_keys;
        std::vector<smallsize> expected;
        draw_probes(queries, pool, 0, immutable_build_size, false, probe_keys, expected);
        for (size_t i = 0; i < queries; ++i)
            operations.push_back({probe_keys[i], 0, expected[i], lookup});
    }
    return operations;
}

template <typename Key>
void validate_results(const std::vector<operation<Key>>& operations,
                      const std::vector<smallsize>& actual, bool verify_state) {
    if (actual.size() != operations.size())
        throw std::runtime_error("Concurrent result size mismatch");
    for (size_t i = 0; i < operations.size(); ++i) {
        const auto& op = operations[i];
        smallsize expected = op.expected;
        if (verify_state && op.kind != lookup)
            expected = op.kind == insert ? op.value : not_found;
        if (actual[i] != expected) {
            std::ostringstream message;
            message << (verify_state ? "Concurrent post-state" : "Concurrent operation")
                    << " validation failed at sorted index " << i << ", key=" << op.key
                    << ", task=" << unsigned(op.kind) << ", expected=" << expected
                    << ", received=" << actual[i];
            if (actual[i] == unwritten) message << " (result unwritten)";
            throw std::runtime_error(message.str());
        }
    }
}

template <typename Key>
struct device_batch {
    cuda_buffer<Key> keys;
    cuda_buffer<smallsize> values, results;
    cuda_buffer<uint8_t> tasks;
    device_batch() {
        keys.alloc(batch_size);
        values.alloc(batch_size);
        tasks.alloc(batch_size);
        results.alloc(batch_size);
        C2EX
    }

    uint64_t upload(const std::vector<operation<Key>>& operations) {
        std::vector<Key> host_keys(operations.size());
        std::vector<smallsize> host_values(operations.size());
        std::vector<uint8_t> host_tasks(operations.size());
        uint64_t digest = 14695981039346656037ull;
        for (size_t i = 0; i < operations.size(); ++i) {
            host_keys[i] = operations[i].key;
            host_values[i] = operations[i].value;
            host_tasks[i] = operations[i].kind;
            digest = (digest ^ static_cast<uint64_t>(host_keys[i])) * 1099511628211ull;
            digest = (digest ^ host_values[i]) * 1099511628211ull;
            digest = (digest ^ host_tasks[i]) * 1099511628211ull;
        }
        keys.upload(host_keys.data(), operations.size());
        values.upload(host_values.data(), operations.size());
        tasks.upload(host_tasks.data(), operations.size());
        reset_results(operations.size());
        C2EX
        return digest;
    }

    void reset_results(size_t count) {
        std::vector<smallsize> initial(count, unwritten);
        results.upload(initial.data(), count);
        C2EX
    }
};

// Exactly one insertion batch per benchmark invocation, before measured runs.
// Use build keys in a disposable index: SlabHash's insertPair would otherwise
// add duplicate entries to the real index. Half the sample is initially absent
// here so the warm-up exercises actual insertion, not only duplicate checks.
template <typename Index>
void warm_up_insertions(const std::vector<typename Index::key_type>& pool) {
    using Key = typename Index::key_type;
    const size_t count = std::min(build_size, size_t{10000});
    const size_t seed_count = std::max(size_t{1}, count / 2);
    std::cout << "[CONCURRENT] One-time untimed insertion warm-up: " << count
              << " build keys on a disposable index (" << count - seed_count
              << " insertions, " << seed_count << " reinsertion attempts)." << std::endl;

    cuda_buffer<Key> keys;
    cuda_buffer<smallsize> values, results;
    cuda_buffer<uint8_t> tasks;
    keys.alloc(count);
    values.alloc(count);
    results.alloc(count);
    tasks.alloc(count);
    keys.upload(pool.data(), seed_count);
    C2EX
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    C2EX
    Index temporary_index;
    if constexpr (is_flix<Index>)
        temporary_index.build_bucket_layer_only(
            keys.ptr(), seed_count, 2 * count, free_bytes, nullptr, nullptr);
    else
        temporary_index.build(keys.ptr(), seed_count, 2 * count, free_bytes, nullptr, nullptr);
    cudaDeviceSynchronize();
    C2EX

    std::vector<operation<Key>> operations;
    operations.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        // FliX rejects an existing key without changing its value. B-tree
        // replaces its value and SlabHash inserts a copy; both report success.
        // Use the original value in every case and discard this index afterward.
        const smallsize expected = is_flix<Index> && i < seed_count ? 0 : 1;
        operations.push_back({pool[i], static_cast<smallsize>(i), expected, insert});
    }
    std::stable_sort(operations.begin(), operations.end(),
                     [](const auto& a, const auto& b) { return a.key < b.key; });
    std::vector<Key> host_keys(count);
    std::vector<smallsize> host_values(count), initial_results(count, unwritten);
    std::vector<uint8_t> host_tasks(count, insert);
    for (size_t i = 0; i < count; ++i) {
        host_keys[i] = operations[i].key;
        host_values[i] = operations[i].value;
    }
    keys.upload(host_keys.data(), count);
    values.upload(host_values.data(), count);
    tasks.upload(host_tasks.data(), count);
    results.upload(initial_results.data(), count);
    // Original call: temporary_index.mixed_batch(keys.ptr(), values.ptr(), tasks.ptr(), results.ptr(), count, 0);
    temporary_index.mixed_batch(keys.ptr(), values.ptr(), tasks.ptr(), results.ptr(), count, 0, 0);
    cudaDeviceSynchronize();
    C2EX
    validate_results(operations, results.download(count), false);

    // Untimed value check after insertion, including duplicate-key attempts.
    std::fill(host_tasks.begin(), host_tasks.end(), lookup);
    tasks.upload(host_tasks.data(), count);
    results.upload(initial_results.data(), count);
    // Original call: temporary_index.mixed_batch(keys.ptr(), values.ptr(), tasks.ptr(), results.ptr(), count, 0);
    temporary_index.mixed_batch(keys.ptr(), values.ptr(), tasks.ptr(), results.ptr(), count, 0, 0);
    cudaDeviceSynchronize();
    C2EX
    validate_results(operations, results.download(count), true);
    std::cout << "[CONCURRENT] Insertion warm-up PASS; disposable index discarded; "
                 "measured index will be built fresh." << std::endl;
}

template <typename Index>
void run(rc::result_collector& collector, size_t runs) {
    using Key = typename Index::key_type;
    static_assert(std::is_same_v<Key, key32>,
                  "Use 32-bit keys for the common FliX/SlabHash/GPU B-tree comparison");
    if (runs == 0) throw std::invalid_argument("Concurrent benchmark needs at least one run");
    cudaSetDevice(0);
    cudaFree(0);
    C2EX
    /* Previous compact stderr banner, preserved for comparison:
    std::cerr << "Concurrent benchmark: " << Index::short_description()
              << ", uniform keys, build=" << build_size << ", operations/round=" << batch_size
              << ", rounds=" << round_count << ", mix=" << insert_percent << '/'
              << delete_percent << '/' << query_percent
              << ", delete hits/misses=" << delete_hits << '/' << delete_misses << '\n';
    */
    std::cout << "\n=== CONCURRENT MIXED-BATCH BENCHMARK ===\n"
              << "Index: " << Index::short_description() << '\n'
              << "Keys: 32-bit, uniform distribution\n"
              << "Initial build: " << build_size << " keys\n"
              << "Mixed batch: " << batch_size << " operations per round\n"
              << "Rounds: " << round_count << ", runs: " << runs << '\n'
              << "Mix (insert/delete/query): " << insert_percent << "% / "
              << delete_percent << "% / " << query_percent << "%\n"
              << "Per round: inserts=" << inserts << ", delete hits=" << delete_hits
              << ", delete misses=" << delete_misses << ", query hits=" << queries << '\n'
              << "Successful deletes: build keys in round 1, previous-round insertions thereafter\n"
              << "Timing: mixed kernel only; validation and sorting excluded; rebuild OFF\n"
              << std::endl;
    /* Previous generation-only path (no persistent cache):
              << "[CONCURRENT] Generating " << pool_size << " unique uniform keys..."
              << std::endl;
    const auto pool = make_uniform_pool<Key>();
    std::cout << "[CONCURRENT] Key generation complete." << std::endl;
    */
    const auto pool = load_or_generate_uniform_pool<Key>();

    warm_up_insertions<Index>(pool);

    for (size_t run = 0; run < runs; ++run) {
        std::cout << "[CONCURRENT] Run " << run + 1 << '/' << runs
                  << ": building " << build_size << " initial keys..." << std::endl;
        device_batch<Key> batch;
        cuda_buffer<Key> build_keys;
        build_keys.alloc(build_size);
        build_keys.upload(pool.data(), build_size);
        C2EX
        size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        C2EX
        Index index;
        double build_ms = 0;
        // SlabHash capacity describes peak simultaneous entries and has a 2^28
        // limit. FliX/B-tree retain nodes: reserve for all distinct inserted keys.
        const size_t capacity = is_slab<Index> ? peak_live_size : insertion_pool_end;
        if constexpr (is_flix<Index>)
            index.build_bucket_layer_only(build_keys.ptr(), build_size, capacity,
                                          free_bytes, &build_ms, nullptr);
        else
            index.build(build_keys.ptr(), build_size, capacity, free_bytes, &build_ms, nullptr);
        cudaDeviceSynchronize();
        C2EX

        std::cout << "[CONCURRENT] Initial build complete; running untimed lookup warm-up."
                  << std::endl;
        // Untimed, read-only warm-up of the same mixed API before measuring.
        std::vector<operation<Key>> warmup;
        const size_t warmup_count = std::min(build_size, std::min(batch_size, size_t{1024}));
        for (size_t i = 0; i < warmup_count; ++i)
            warmup.push_back({pool[i], 0, static_cast<smallsize>(i), lookup});
        std::stable_sort(warmup.begin(), warmup.end(),
                         [](const auto& a, const auto& b) { return a.key < b.key; });
        batch.upload(warmup);
        // Original call: index.mixed_batch(batch.keys.ptr(), batch.values.ptr(), batch.tasks.ptr(),
        //                                  batch.results.ptr(), warmup.size(), 0);
        index.mixed_batch(batch.keys.ptr(), batch.values.ptr(), batch.tasks.ptr(),
                          batch.results.ptr(), warmup.size(), 0, 0);
        cudaDeviceSynchronize();
        C2EX
        validate_results(warmup, batch.results.download(warmup.size()), false);

        cuda_timer timer(0);
        double total_ms = 0;
        std::vector<rc::result> validated_rows;
        for (size_t round = 0; round < round_count; ++round) {
            std::cout << "[CONCURRENT] Run " << run + 1 << '/' << runs
                      << ", round " << round + 1 << '/' << round_count
                      << ": preparing and sorting mixed batch..." << std::endl;
            auto operations = make_round(pool, round);
            const auto sort_start = std::chrono::steady_clock::now();
            std::stable_sort(operations.begin(), operations.end(),
                             [](const auto& a, const auto& b) { return a.key < b.key; });
            const double sort_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - sort_start).count();
            const uint64_t digest = batch.upload(operations);
            cudaDeviceSynchronize();
            C2EX
            timer.start();
            // Original call: index.mixed_batch(batch.keys.ptr(), batch.values.ptr(), batch.tasks.ptr(),
            //                                  batch.results.ptr(), operations.size(), 0);
            index.mixed_batch(batch.keys.ptr(), batch.values.ptr(), batch.tasks.ptr(),
                              batch.results.ptr(), operations.size(), 0, round + 1);
            timer.stop();
            const double kernel_ms = timer.time_ms();
            C2EX
            validate_results(operations, batch.results.download(operations.size()), false);

            // Untimed correctness check through the same API: query every
            // operation key to verify insert values, deletion absence and hits.
            // This is validation, not the separate probe-miss benchmark path.
            std::vector<uint8_t> lookup_tasks(operations.size(), lookup);
            batch.tasks.upload(lookup_tasks.data(), lookup_tasks.size());
            batch.reset_results(operations.size());
            // Original call: index.mixed_batch(batch.keys.ptr(), batch.values.ptr(), batch.tasks.ptr(),
            //                                  batch.results.ptr(), operations.size(), 0);
            index.mixed_batch(batch.keys.ptr(), batch.values.ptr(), batch.tasks.ptr(),
                              batch.results.ptr(), operations.size(), 0, round + 1);
            cudaDeviceSynchronize();
            C2EX
            validate_results(operations, batch.results.download(operations.size()), true);
            total_ms += kernel_ms;
            const size_t live_after = build_size + (round + 1) * (inserts - delete_hits);
            /* Previous compact stderr progress line:
            std::cerr << "  run=" << run << " round=" << round + 1
                      << " mixed_kernel_ms=" << kernel_ms << " Mops/s="
                      << batch_size / (kernel_ms * 1000.0)
                      << " live_keys=" << live_after << " validation=PASS\n";
            */
            std::cout << "[CONCURRENT] " << Index::short_description()
                      << " | run=" << run + 1 << '/' << runs
                      << " | round=" << round + 1 << '/' << round_count
                      << " | mixed_kernel_ms=" << kernel_ms
                      << " | throughput_Mops=" << batch_size / (kernel_ms * 1000.0)
                      << " | live_keys=" << live_after
                      << " | operation_results=PASS | post_state=PASS" << std::endl;

            rc::result row;
            row.add_parameter("record", "round")
               .add_parameter("index", Index::short_description())
               .add_parameter("run", run).add_parameter("round", round + 1)
               .add_parameter("key_bits", 32).add_parameter("distribution", "uniform")
               .add_parameter("build_size", build_size).add_parameter("batch_size", batch_size)
               .add_parameter("Insert_mix", insert_percent).add_parameter("Del_mix", delete_percent)
               .add_parameter("Query_mix", query_percent).add_parameter("insert_count", inserts)
               .add_parameter("delete_hit_count", delete_hits)
               .add_parameter("delete_miss_count", delete_misses)
               .add_parameter("query_count", queries).add_parameter("live_keys_after", live_after)
               .add_parameter("capacity", capacity).add_parameter("workload_digest", digest)
               .add_parameter("rebuild_enabled", false).add_parameter("validated", true)
               .add_measurement("mixed_kernel_ms", kernel_ms)
               .add_measurement("throughput_mops", batch_size / (kernel_ms * 1000.0))
               .add_measurement("sort_ms", sort_ms);
            for (const auto& parameter : Index::parameters())
                row.add_parameter("index_" + parameter.first, parameter.second);
            validated_rows.push_back(std::move(row));
        }
        // Publish a run only after every round passed both kinds of validation.
        for (const auto& row : validated_rows) collector.commit(row);
        rc::result summary;
        summary.add_parameter("record", "summary").add_parameter("index", Index::short_description())
               .add_parameter("run", run).add_parameter("rounds", round_count)
               .add_parameter("Insert_mix", insert_percent).add_parameter("Del_mix", delete_percent)
               .add_parameter("Query_mix", query_percent).add_parameter("build_size", build_size)
               .add_parameter("batch_size", batch_size).add_parameter("distribution", "uniform")
               .add_parameter("insert_warmup_count", std::min(build_size, size_t{10000}))
               .add_parameter("insert_warmup_disposable_index", true)
               .add_parameter("validated", true).add_parameter("rebuild_enabled", false)
               .add_measurement("total_mixed_kernel_ms", total_ms)
               .add_measurement("throughput_mops", round_count * batch_size / (total_ms * 1000.0))
               .add_measurement("build_ms", build_ms);
        collector.commit(summary);
        std::cout << "[CONCURRENT] Run " << run + 1 << '/' << runs
                  << " complete | total_mixed_kernel_ms=" << total_ms
                  << " | throughput_Mops=" << round_count * batch_size / (total_ms * 1000.0)
                  << " | all " << round_count << " rounds validated."
                  << std::endl;
    }
}

#ifdef CONCURRENT_SMALL_BUILD_ADAPTIVE_DELETES
// These new functions leave make_round() and run() unchanged.
inline constexpr size_t small_build_delete_hits_for_round(size_t round) {
    return round == 0 ? first_round_delete_hits : delete_hits;
}
inline constexpr size_t small_build_delete_misses_for_round(size_t round) {
    return deletes - small_build_delete_hits_for_round(round);
}
inline constexpr size_t small_build_live_keys_after_round(size_t round) {
    return build_size + (round + 1) * inserts - first_round_delete_hits - round * delete_hits;
}

template <typename Key>
std::vector<Key> load_or_generate_uniform_pool_small_build() {
    // Separate cache identity for the changed key-pool partitioning policy.
    const auto original = uniform_cache_path<Key>();
    const auto path = original.parent_path() /
        (original.stem().string() + "_adaptive_deletes_v1" + original.extension().string());
    std::cout << "[CONCURRENT] Binary key cache: " << path.string() << std::endl;
    std::vector<Key> pool;
    if (load_uniform_cache(path, pool)) return pool;
    std::cout << "[CONCURRENT] Generating " << pool_size << " unique uniform keys..."
              << std::endl;
    pool = make_uniform_pool<Key>();
    save_uniform_cache(path, pool);
    return pool;
}

template <typename Key>
std::vector<operation<Key>> make_round_small_build(const std::vector<Key>& pool, size_t round) {
    const size_t round_delete_hits = small_build_delete_hits_for_round(round);
    const size_t round_delete_misses = small_build_delete_misses_for_round(round);
    std::vector<operation<Key>> operations;
    operations.reserve(batch_size);
    const size_t insert_start = build_size + round * inserts;
    for (size_t i = 0; i < inserts; ++i) {
        const size_t offset = insert_start + i;
        operations.push_back({pool[offset], static_cast<smallsize>(offset), 1, insert});
    }
    // Round 0: delete a reserved tail of the build. Later: delete from the
    // preceding round's inserts, which completed before this kernel launch.
    const size_t delete_start = round == 0 ? immutable_build_size : insert_start - inserts;
    for (size_t i = 0; i < round_delete_hits; ++i)
        operations.push_back({pool[delete_start + i], 0, 1, erase});
    // Reusing this never-inserted pool in later rounds always produces misses.
    for (size_t i = 0; i < round_delete_misses; ++i)
        operations.push_back({pool[insertion_pool_end + i], 0, 0, erase});

    if constexpr (queries > 0) {
        std::vector<Key> probe_keys;
        std::vector<smallsize> expected;
        draw_probes(queries, pool, 0, immutable_build_size, false, probe_keys, expected);
        for (size_t i = 0; i < queries; ++i)
            operations.push_back({probe_keys[i], 0, expected[i], lookup});
    }
    return operations;
}

template <typename Index>
void run_small_build(rc::result_collector& collector, size_t runs) {
    using Key = typename Index::key_type;
    static_assert(std::is_same_v<Key, key32>,
                  "Use 32-bit keys for the common FliX/SlabHash/GPU B-tree comparison");
    if (runs == 0) throw std::invalid_argument("Concurrent benchmark needs at least one run");
    cudaSetDevice(0);
    cudaFree(0);
    C2EX
    /* Previous compact stderr banner, preserved for comparison:
    std::cerr << "Concurrent benchmark: " << Index::short_description()
              << ", uniform keys, build=" << build_size << ", operations/round=" << batch_size
              << ", rounds=" << round_count << ", mix=" << insert_percent << '/'
              << delete_percent << '/' << query_percent
              << ", delete hits/misses=" << delete_hits << '/' << delete_misses << '\n';
    */
    std::cout << "\n=== CONCURRENT MIXED-BATCH BENCHMARK ===\n"
              << "Index: " << Index::short_description() << '\n'
              << "Keys: 32-bit, uniform distribution\n"
              << "Initial build: " << build_size << " keys\n"
              << "Mixed batch: " << batch_size << " operations per round\n"
              << "Rounds: " << round_count << ", runs: " << runs << '\n'
              << "Mix (insert/delete/query): " << insert_percent << "% / "
              << delete_percent << "% / " << query_percent << "%\n"
              << "Per round: inserts=" << inserts << ", query hits=" << queries << '\n'
              << "Adaptive small-build deletion policy ON\n"
              << "Round 1: delete hits=" << first_round_delete_hits
              << ", delete misses=" << first_round_delete_misses << '\n'
              << "Later rounds: delete hits=" << delete_hits
              << ", delete misses=" << delete_misses << '\n'
              << "Untouched query pool: " << immutable_build_size
              << " keys; queries sampled with replacement\n"
              << "Successful deletes: build keys in round 1, previous-round insertions thereafter\n"
              << "Timing: mixed kernel only; validation and sorting excluded; rebuild OFF\n"
              << std::endl;
    /* Previous generation-only path (no persistent cache):
              << "[CONCURRENT] Generating " << pool_size << " unique uniform keys..."
              << std::endl;
    const auto pool = make_uniform_pool<Key>();
    std::cout << "[CONCURRENT] Key generation complete." << std::endl;
    */
    const auto pool = load_or_generate_uniform_pool_small_build<Key>();

    warm_up_insertions<Index>(pool);

    for (size_t run = 0; run < runs; ++run) {
        std::cout << "[CONCURRENT] Run " << run + 1 << '/' << runs
                  << ": building " << build_size << " initial keys..." << std::endl;
        device_batch<Key> batch;
        cuda_buffer<Key> build_keys;
        build_keys.alloc(build_size);
        build_keys.upload(pool.data(), build_size);
        C2EX
        size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        C2EX
        Index index;
        double build_ms = 0;
        // SlabHash capacity describes peak simultaneous entries and has a 2^28
        // limit. FliX/B-tree retain nodes: reserve for all distinct inserted keys.
        const size_t capacity = is_slab<Index> ? peak_live_size : insertion_pool_end;
        if constexpr (is_flix<Index>)
            index.build_bucket_layer_only(build_keys.ptr(), build_size, capacity,
                                          free_bytes, &build_ms, nullptr);
        else
            index.build(build_keys.ptr(), build_size, capacity, free_bytes, &build_ms, nullptr);
        cudaDeviceSynchronize();
        C2EX

        std::cout << "[CONCURRENT] Initial build complete; running untimed lookup warm-up."
                  << std::endl;
        // Untimed, read-only warm-up of the same mixed API before measuring.
        std::vector<operation<Key>> warmup;
        const size_t warmup_count = std::min(build_size, std::min(batch_size, size_t{1024}));
        for (size_t i = 0; i < warmup_count; ++i)
            warmup.push_back({pool[i], 0, static_cast<smallsize>(i), lookup});
        std::stable_sort(warmup.begin(), warmup.end(),
                         [](const auto& a, const auto& b) { return a.key < b.key; });
        batch.upload(warmup);
        // Original call: index.mixed_batch(batch.keys.ptr(), batch.values.ptr(), batch.tasks.ptr(),
        //                                  batch.results.ptr(), warmup.size(), 0);
        index.mixed_batch(batch.keys.ptr(), batch.values.ptr(), batch.tasks.ptr(),
                          batch.results.ptr(), warmup.size(), 0, 0);
        cudaDeviceSynchronize();
        C2EX
        validate_results(warmup, batch.results.download(warmup.size()), false);

        cuda_timer timer(0);
        double total_ms = 0;
        std::vector<rc::result> validated_rows;
        for (size_t round = 0; round < round_count; ++round) {
            std::cout << "[CONCURRENT] Run " << run + 1 << '/' << runs
                      << ", round " << round + 1 << '/' << round_count
                      << ": preparing and sorting mixed batch..." << std::endl;
            auto operations = make_round_small_build(pool, round);
            const auto sort_start = std::chrono::steady_clock::now();
            std::stable_sort(operations.begin(), operations.end(),
                             [](const auto& a, const auto& b) { return a.key < b.key; });
            const double sort_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - sort_start).count();
            const uint64_t digest = batch.upload(operations);
            cudaDeviceSynchronize();
            C2EX
            timer.start();
            // Original call: index.mixed_batch(batch.keys.ptr(), batch.values.ptr(), batch.tasks.ptr(),
            //                                  batch.results.ptr(), operations.size(), 0);
            index.mixed_batch(batch.keys.ptr(), batch.values.ptr(), batch.tasks.ptr(),
                              batch.results.ptr(), operations.size(), 0, round + 1);
            timer.stop();
            const double kernel_ms = timer.time_ms();
            C2EX
            validate_results(operations, batch.results.download(operations.size()), false);

            // Untimed correctness check through the same API: query every
            // operation key to verify insert values, deletion absence and hits.
            // This is validation, not the separate probe-miss benchmark path.
            std::vector<uint8_t> lookup_tasks(operations.size(), lookup);
            batch.tasks.upload(lookup_tasks.data(), lookup_tasks.size());
            batch.reset_results(operations.size());
            // Original call: index.mixed_batch(batch.keys.ptr(), batch.values.ptr(), batch.tasks.ptr(),
            //                                  batch.results.ptr(), operations.size(), 0);
            index.mixed_batch(batch.keys.ptr(), batch.values.ptr(), batch.tasks.ptr(),
                              batch.results.ptr(), operations.size(), 0, round + 1);
            cudaDeviceSynchronize();
            C2EX
            validate_results(operations, batch.results.download(operations.size()), true);
            total_ms += kernel_ms;
            const size_t live_after = small_build_live_keys_after_round(round);
            /* Previous compact stderr progress line:
            std::cerr << "  run=" << run << " round=" << round + 1
                      << " mixed_kernel_ms=" << kernel_ms << " Mops/s="
                      << batch_size / (kernel_ms * 1000.0)
                      << " live_keys=" << live_after << " validation=PASS\n";
            */
            std::cout << "[CONCURRENT] " << Index::short_description()
                      << " | run=" << run + 1 << '/' << runs
                      << " | round=" << round + 1 << '/' << round_count
                      << " | mixed_kernel_ms=" << kernel_ms
                      << " | throughput_Mops=" << batch_size / (kernel_ms * 1000.0)
                      << " | live_keys=" << live_after
                      << " | operation_results=PASS | post_state=PASS" << std::endl;

            rc::result row;
            row.add_parameter("small_build_adaptive_deletes", true)
               .add_parameter("first_round_delete_hit_count", first_round_delete_hits)
               .add_parameter("first_round_delete_miss_count", first_round_delete_misses)
               .add_parameter("untouched_query_key_count", immutable_build_size);
            row.add_parameter("record", "round")
               .add_parameter("index", Index::short_description())
               .add_parameter("run", run).add_parameter("round", round + 1)
               .add_parameter("key_bits", 32).add_parameter("distribution", "uniform")
               .add_parameter("build_size", build_size).add_parameter("batch_size", batch_size)
               .add_parameter("Insert_mix", insert_percent).add_parameter("Del_mix", delete_percent)
               .add_parameter("Query_mix", query_percent).add_parameter("insert_count", inserts)
               .add_parameter("delete_hit_count", small_build_delete_hits_for_round(round))
               .add_parameter("delete_miss_count", small_build_delete_misses_for_round(round))
               .add_parameter("query_count", queries).add_parameter("live_keys_after", live_after)
               .add_parameter("capacity", capacity).add_parameter("workload_digest", digest)
               .add_parameter("rebuild_enabled", false).add_parameter("validated", true)
               .add_measurement("mixed_kernel_ms", kernel_ms)
               .add_measurement("throughput_mops", batch_size / (kernel_ms * 1000.0))
               .add_measurement("sort_ms", sort_ms);
            for (const auto& parameter : Index::parameters())
                row.add_parameter("index_" + parameter.first, parameter.second);
            validated_rows.push_back(std::move(row));
        }
        // Publish a run only after every round passed both kinds of validation.
        for (const auto& row : validated_rows) collector.commit(row);
        rc::result summary;
        summary.add_parameter("small_build_adaptive_deletes", true)
               .add_parameter("first_round_delete_hit_count", first_round_delete_hits)
               .add_parameter("first_round_delete_miss_count", first_round_delete_misses)
               .add_parameter("untouched_query_key_count", immutable_build_size);
        summary.add_parameter("record", "summary").add_parameter("index", Index::short_description())
               .add_parameter("run", run).add_parameter("rounds", round_count)
               .add_parameter("Insert_mix", insert_percent).add_parameter("Del_mix", delete_percent)
               .add_parameter("Query_mix", query_percent).add_parameter("build_size", build_size)
               .add_parameter("batch_size", batch_size).add_parameter("distribution", "uniform")
               .add_parameter("insert_warmup_count", std::min(build_size, size_t{10000}))
               .add_parameter("insert_warmup_disposable_index", true)
               .add_parameter("validated", true).add_parameter("rebuild_enabled", false)
               .add_measurement("total_mixed_kernel_ms", total_ms)
               .add_measurement("throughput_mops", round_count * batch_size / (total_ms * 1000.0))
               .add_measurement("build_ms", build_ms);
        collector.commit(summary);
        std::cout << "[CONCURRENT] Run " << run + 1 << '/' << runs
                  << " complete | total_mixed_kernel_ms=" << total_ms
                  << " | throughput_Mops=" << round_count * batch_size / (total_ms * 1000.0)
                  << " | all " << round_count << " rounds validated."
                  << std::endl;
    }
}

#endif // CONCURRENT_SMALL_BUILD_ADAPTIVE_DELETES

} // namespace concurrent_benchmark

// Include this header and call explicitly after selecting the index type.
// No edits to main.cu or the existing benchmark_updates() are required here.
template <typename index_type>
void benchmark_updates_concurrent(rc::result_collector& rc, size_t runs) {
    static_assert(concurrent_benchmark::supported<index_type>,
                  "Concurrent benchmark supports only FliX, SlabHash and GPU B-tree; "
                  "LSM-tree and WarpCore are unsupported");
    // Original dispatch, retained below when the optional mode is absent:
    // concurrent_benchmark::run<index_type>(rc, runs);
    if constexpr (concurrent_benchmark::supported<index_type>)
#ifdef CONCURRENT_SMALL_BUILD_ADAPTIVE_DELETES
        concurrent_benchmark::run_small_build<index_type>(rc, runs);
#else
        concurrent_benchmark::run<index_type>(rc, runs);
#endif // CONCURRENT_SMALL_BUILD_ADAPTIVE_DELETES
}

// Deliberately disabled in this benchmark:
// range queries; separate probe-miss measurements; successor queries; rebuilds.
// Only the mixed_batch kernel is timed. Preparation, transfers, warm-up and
// validation (including deleted-key absence checks) are outside that timing.
#endif
