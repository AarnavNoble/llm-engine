#pragma once
// Paged KV cache bookkeeping (the vLLM idea): the cache is a pool of
// fixed-size physical blocks; each sequence owns a block table mapping its
// logical blocks to physical ids. Full blocks are content-hashed so
// sequences with a common prefix share physical blocks (refcounted); blocks
// with no owner stay cached in an LRU list until memory pressure evicts them.
//
// This file is pure bookkeeping — no tensors. The CPU and CUDA models read
// block tables to find where a token's K/V live: slot = block * block_size + offset.
#include <cstdint>
#include <list>
#include <optional>
#include <unordered_map>
#include <vector>

namespace engine {

using BlockId = int32_t;
constexpr BlockId kNoBlock = -1;

struct BlockTable {
  std::vector<BlockId> blocks;
  std::vector<uint64_t> hashes;  // parallel to blocks; 0 = not (yet) hashed
  int num_committed = 0;         // leading blocks whose hash publication was attempted

  int num_blocks() const { return static_cast<int>(blocks.size()); }
  bool empty() const { return blocks.empty(); }
  // Physical slot index for logical token position `pos`.
  int64_t slot(int pos, int block_size) const {
    return static_cast<int64_t>(blocks[pos / block_size]) * block_size + pos % block_size;
  }
};

struct KVCacheStats {
  uint64_t prefix_queries = 0;   // prompts that went through allocate_prompt
  uint64_t prefix_hit_blocks = 0;
  uint64_t prefix_total_blocks = 0;  // full prompt blocks looked up
  uint64_t evictions = 0;
  uint64_t allocation_failures = 0;
};

class KVCacheManager {
 public:
  KVCacheManager(int num_blocks, int block_size, bool enable_prefix_caching);

  int block_size() const { return block_size_; }
  int num_total_blocks() const { return num_blocks_; }
  int num_free_blocks() const { return static_cast<int>(free_.size()); }
  int num_cached_blocks() const { return static_cast<int>(lru_.size()); }  // evictable, refcount 0
  int num_available_blocks() const { return num_free_blocks() + num_cached_blocks(); }
  int num_used_blocks() const { return num_blocks_ - num_available_blocks(); }
  bool prefix_caching_enabled() const { return prefix_caching_; }
  const KVCacheStats& stats() const { return stats_; }

  static int blocks_needed(int num_tokens, int block_size) { return (num_tokens + block_size - 1) / block_size; }

  // Allocate blocks for a fresh sequence whose prompt is `tokens`. Returns the
  // number of prompt tokens whose K/V are already in cache (a multiple of
  // block_size, possibly 0), or nullopt if the pool cannot fit the prompt
  // right now (nothing is allocated in that case).
  std::optional<int> allocate_prompt(BlockTable& bt, const std::vector<int32_t>& tokens);

  // Make sure `num_tokens` slots exist (allocates at most one block).
  // Returns false if out of memory (caller should preempt someone).
  bool ensure_slot(BlockTable& bt, int num_tokens);

  // Publish full blocks whose K/V are now completely computed to the prefix
  // cache. A block is only shareable once every one of its slots is written,
  // otherwise a concurrent hit could read K/V that a kernel in the same step
  // is still producing.
  void commit_computed(BlockTable& bt, const std::vector<int32_t>& tokens, int num_computed);

  // Release the sequence's blocks (decrement refs; unowned hashed blocks
  // stay cached in LRU, unhashed ones return to the free list).
  void free(BlockTable& bt);

  // Number of tokens the prompt could reuse without allocating anything.
  int lookup_cached_prefix(const std::vector<int32_t>& tokens) const;

  static uint64_t hash_block(uint64_t prev_hash, const int32_t* tokens, int n);

 private:
  std::optional<BlockId> take_block();  // free list, else evict LRU
  void release_block(BlockId b, uint64_t hash);
  void touch_cached(BlockId b);         // remove from LRU when re-referenced
  void hash_full_block(BlockTable& bt, int block_idx, const std::vector<int32_t>& tokens);

  int num_blocks_, block_size_;
  bool prefix_caching_;
  std::vector<BlockId> free_;
  std::vector<int> ref_count_;
  std::unordered_map<uint64_t, BlockId> hash_to_block_;
  std::vector<uint64_t> block_hash_;                    // per physical block, 0 if none
  std::list<BlockId> lru_;                              // front = least recently released
  std::unordered_map<BlockId, std::list<BlockId>::iterator> lru_pos_;
  KVCacheStats stats_;
};

}  // namespace engine
