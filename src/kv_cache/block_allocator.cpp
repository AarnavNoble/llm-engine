#include <cassert>
#include <stdexcept>

#include "engine/kv_cache.h"

namespace engine {

KVCacheManager::KVCacheManager(int num_blocks, int block_size, bool enable_prefix_caching)
    : num_blocks_(num_blocks), block_size_(block_size), prefix_caching_(enable_prefix_caching),
      ref_count_(num_blocks, 0), block_hash_(num_blocks, 0) {
  if (num_blocks <= 0 || block_size <= 0) throw std::invalid_argument("num_blocks and block_size must be > 0");
  free_.reserve(num_blocks);
  for (BlockId b = num_blocks - 1; b >= 0; b--) free_.push_back(b);  // hand out low ids first
}

uint64_t KVCacheManager::hash_block(uint64_t prev_hash, const int32_t* tokens, int n) {
  // Chaining prev_hash makes the hash position-aware: the same 16 tokens at
  // a different prefix get a different hash. splitmix64-style mixing.
  uint64_t h = prev_hash ^ 0x9E3779B97F4A7C15ull;
  auto mix = [](uint64_t z) { z += 0x9E3779B97F4A7C15ull; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); };
  for (int i = 0; i < n; i++) h = mix(h ^ static_cast<uint64_t>(static_cast<uint32_t>(tokens[i])));
  return h == 0 ? 1 : h;  // 0 is reserved for "unhashed"
}

std::optional<BlockId> KVCacheManager::take_block() {
  if (!free_.empty()) { BlockId b = free_.back(); free_.pop_back(); return b; }
  if (lru_.empty()) return std::nullopt;
  BlockId b = lru_.front();
  lru_.pop_front();
  lru_pos_.erase(b);
  hash_to_block_.erase(block_hash_[b]);
  block_hash_[b] = 0;
  stats_.evictions++;
  return b;
}

void KVCacheManager::touch_cached(BlockId b) {
  auto it = lru_pos_.find(b);
  if (it != lru_pos_.end()) { lru_.erase(it->second); lru_pos_.erase(it); }
}

void KVCacheManager::release_block(BlockId b, uint64_t hash) {
  assert(ref_count_[b] > 0);
  if (--ref_count_[b] > 0) return;
  if (prefix_caching_ && hash != 0 && block_hash_[b] == hash) {
    lru_.push_back(b);
    lru_pos_[b] = std::prev(lru_.end());
  } else {
    if (block_hash_[b] != 0) { hash_to_block_.erase(block_hash_[b]); block_hash_[b] = 0; }
    free_.push_back(b);
  }
}

void KVCacheManager::hash_full_block(BlockTable& bt, int i, const std::vector<int32_t>& tokens) {
  if (!prefix_caching_) return;
  uint64_t prev = i == 0 ? 0 : bt.hashes[i - 1];
  if (i > 0 && prev == 0) return;  // an earlier block was a duplicate; keep the chain unhashed
  uint64_t h = hash_block(prev, tokens.data() + static_cast<size_t>(i) * block_size_, block_size_);
  BlockId b = bt.blocks[i];
  if (hash_to_block_.count(h)) return;  // another block already represents this content
  hash_to_block_[h] = b;
  block_hash_[b] = h;
  bt.hashes[i] = h;
}

int KVCacheManager::lookup_cached_prefix(const std::vector<int32_t>& tokens) const {
  if (!prefix_caching_) return 0;
  int full = static_cast<int>(tokens.size()) / block_size_;
  uint64_t prev = 0; int hit = 0;
  for (int i = 0; i < full; i++) {
    uint64_t h = hash_block(prev, tokens.data() + static_cast<size_t>(i) * block_size_, block_size_);
    if (!hash_to_block_.count(h)) break;
    hit++; prev = h;
  }
  return hit * block_size_;
}

std::optional<int> KVCacheManager::allocate_prompt(BlockTable& bt, const std::vector<int32_t>& tokens) {
  assert(bt.empty());
  const int n = static_cast<int>(tokens.size());
  const int needed = blocks_needed(n, block_size_);
  const int full = n / block_size_;
  stats_.prefix_queries++;
  stats_.prefix_total_blocks += full;

  // Walk cached prefix first: reusing a block never consumes pool capacity.
  int cached = 0; uint64_t prev = 0;
  std::vector<std::pair<uint64_t, BlockId>> hits;
  if (prefix_caching_) {
    for (int i = 0; i < full; i++) {
      uint64_t h = hash_block(prev, tokens.data() + static_cast<size_t>(i) * block_size_, block_size_);
      auto it = hash_to_block_.find(h);
      if (it == hash_to_block_.end()) break;
      hits.emplace_back(h, it->second); prev = h; cached++;
    }
  }
  // Blocks in `hits` with refcount 0 sit in the LRU; they are not "free" so
  // we must not count them as capacity we can also allocate from.
  int lru_hits = 0;
  for (auto& [h, b] : hits) if (ref_count_[b] == 0) lru_hits++;
  if (needed - cached > num_available_blocks() - lru_hits) { stats_.allocation_failures++; return std::nullopt; }

  for (auto& [h, b] : hits) {
    touch_cached(b); ref_count_[b]++;
    bt.blocks.push_back(b); bt.hashes.push_back(h);
  }
  stats_.prefix_hit_blocks += cached;
  for (int i = cached; i < needed; i++) {
    auto b = take_block();
    assert(b.has_value());
    ref_count_[*b] = 1;
    bt.blocks.push_back(*b); bt.hashes.push_back(0);
  }
  bt.num_committed = cached;
  return cached * block_size_;
}

bool KVCacheManager::ensure_slot(BlockTable& bt, int num_tokens) {
  const int needed = blocks_needed(num_tokens, block_size_);
  assert(needed <= bt.num_blocks() + 1);
  if (needed > bt.num_blocks()) {
    auto b = take_block();
    if (!b) { stats_.allocation_failures++; return false; }
    ref_count_[*b] = 1;
    bt.blocks.push_back(*b); bt.hashes.push_back(0);
  }
  return true;
}

void KVCacheManager::commit_computed(BlockTable& bt, const std::vector<int32_t>& tokens, int num_computed) {
  if (!prefix_caching_) return;
  assert(num_computed <= static_cast<int>(tokens.size()));
  while (bt.num_committed < bt.num_blocks() && (bt.num_committed + 1) * block_size_ <= num_computed) {
    hash_full_block(bt, bt.num_committed, tokens);
    bt.num_committed++;
  }
}

void KVCacheManager::free(BlockTable& bt) {
  for (int i = bt.num_blocks() - 1; i >= 0; i--) release_block(bt.blocks[i], bt.hashes[i]);
  bt.blocks.clear(); bt.hashes.clear(); bt.num_committed = 0;
}

}  // namespace engine
