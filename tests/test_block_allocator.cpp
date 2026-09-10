#include <catch2/catch_test_macros.hpp>
#include <numeric>
#include "engine/kv_cache.h"

using namespace engine;

static std::vector<int32_t> toks(int n, int32_t base = 0) { std::vector<int32_t> v(n); std::iota(v.begin(), v.end(), base); return v; }

TEST_CASE("allocator: prompt allocation, exhaustion, free and reuse") {
  KVCacheManager kv(4, 16, /*prefix=*/false);
  BlockTable a, b;
  REQUIRE(kv.allocate_prompt(a, toks(33)) == 0);  // 3 blocks: 16+16+1
  REQUIRE(a.num_blocks() == 3);
  REQUIRE(kv.num_free_blocks() == 1);
  REQUIRE_FALSE(kv.allocate_prompt(b, toks(17)).has_value());  // needs 2, only 1 free
  REQUIRE(b.empty());
  REQUIRE(kv.num_free_blocks() == 1);
  REQUIRE(kv.stats().allocation_failures == 1);
  kv.free(a);
  REQUIRE(a.empty());
  REQUIRE(kv.num_free_blocks() == 4);
  REQUIRE(kv.allocate_prompt(b, toks(17)) == 0);
  REQUIRE(b.num_blocks() == 2);
}

TEST_CASE("allocator: ensure_slot allocates only at block boundaries") {
  KVCacheManager kv(3, 4, false);
  BlockTable bt;
  auto t = toks(4);
  REQUIRE(kv.allocate_prompt(bt, t) == 0);
  REQUIRE(bt.num_blocks() == 1);
  REQUIRE(kv.ensure_slot(bt, 5)); REQUIRE(bt.num_blocks() == 2);
  REQUIRE(kv.ensure_slot(bt, 6)); REQUIRE(bt.num_blocks() == 2);
  REQUIRE(kv.ensure_slot(bt, 8)); REQUIRE(bt.num_blocks() == 2);
  REQUIRE(kv.ensure_slot(bt, 9)); REQUIRE(bt.num_blocks() == 3);
  REQUIRE(kv.num_free_blocks() == 0);
  REQUIRE(kv.ensure_slot(bt, 12));
  REQUIRE(kv.ensure_slot(bt, 13) == false);  // 13 tokens need 4 blocks
  REQUIRE(bt.slot(5, 4) == static_cast<int64_t>(bt.blocks[1]) * 4 + 1);
}

TEST_CASE("prefix cache: shared prompt blocks are reused and refcounted") {
  KVCacheManager kv(8, 4, true);
  auto shared = toks(8, 1000);           // exactly 2 full blocks
  auto p1 = shared; p1.push_back(1); p1.push_back(2);
  auto p2 = shared; p2.push_back(3);
  BlockTable a, b;
  REQUIRE(kv.allocate_prompt(a, p1) == 0);
  REQUIRE(a.num_blocks() == 3);
  REQUIRE(kv.lookup_cached_prefix(p2) == 0);   // not published until computed
  kv.commit_computed(a, p1, 5);
  REQUIRE(kv.lookup_cached_prefix(p2) == 4);   // only block 0 is complete
  kv.commit_computed(a, p1, 10);
  REQUIRE(kv.lookup_cached_prefix(p2) == 8);
  REQUIRE(kv.allocate_prompt(b, p2) == 8);  // 2 blocks cached
  REQUIRE(b.blocks[0] == a.blocks[0]);
  REQUIRE(b.blocks[1] == a.blocks[1]);
  REQUIRE(b.blocks[2] != a.blocks[2]);
  REQUIRE(kv.num_used_blocks() == 4);
  REQUIRE(kv.stats().prefix_hit_blocks == 2);
  REQUIRE(kv.stats().prefix_total_blocks == 4);

  kv.free(a);
  REQUIRE(kv.num_used_blocks() == 3);      // shared blocks still owned by b
  REQUIRE(kv.num_cached_blocks() == 0);
  kv.free(b);
  REQUIRE(kv.num_used_blocks() == 0);
  REQUIRE(kv.num_cached_blocks() == 2);    // full hashed blocks stay cached
  REQUIRE(kv.num_free_blocks() == 6);

  BlockTable c;                            // third request still hits with nobody holding refs
  REQUIRE(kv.allocate_prompt(c, p1) == 8);
  REQUIRE(kv.num_cached_blocks() == 0);
}

TEST_CASE("prefix cache: hashes are position-aware") {
  KVCacheManager kv(8, 4, true);
  auto p = toks(8, 5);
  std::vector<int32_t> q(p.begin() + 4, p.end());  // second block of p, now at position 0
  BlockTable a, b;
  REQUIRE(kv.allocate_prompt(a, p) == 0);
  REQUIRE(kv.lookup_cached_prefix(q) == 0);
  REQUIRE(kv.allocate_prompt(b, q) == 0);
  REQUIRE(b.blocks[0] != a.blocks[1]);
}

TEST_CASE("prefix cache: decode fills blocks that later requests can reuse") {
  KVCacheManager kv(8, 4, true);
  BlockTable a;
  auto t = toks(3);
  REQUIRE(kv.allocate_prompt(a, t) == 0);
  kv.commit_computed(a, t, 3);
  t.push_back(3); REQUIRE(kv.ensure_slot(a, 4)); kv.commit_computed(a, t, 4);   // block 0 full -> published
  t.push_back(4); REQUIRE(kv.ensure_slot(a, 5)); kv.commit_computed(a, t, 5);
  kv.free(a);
  REQUIRE(kv.num_cached_blocks() == 1);
  REQUIRE(kv.num_free_blocks() == 7);                   // the partial block went straight back
  BlockTable b;
  REQUIRE(kv.allocate_prompt(b, toks(5)) == 4);
}

TEST_CASE("prefix cache: LRU eviction under pressure, oldest released first") {
  KVCacheManager kv(4, 2, true);
  BlockTable a, b;
  REQUIRE(kv.allocate_prompt(a, toks(2, 10)) == 0); kv.commit_computed(a, toks(2, 10), 2);
  REQUIRE(kv.allocate_prompt(b, toks(2, 20)) == 0); kv.commit_computed(b, toks(2, 20), 2);
  kv.free(a);            // released first -> LRU front
  kv.free(b);
  REQUIRE(kv.num_cached_blocks() == 2);
  REQUIRE(kv.num_available_blocks() == 4);
  BlockTable c;
  REQUIRE(kv.allocate_prompt(c, toks(6, 30)) == 0); kv.commit_computed(c, toks(6, 30), 6);  // needs 3: 2 free + evict a's block
  REQUIRE(kv.stats().evictions == 1);
  REQUIRE(kv.lookup_cached_prefix(toks(2, 10)) == 0);  // a evicted
  REQUIRE(kv.lookup_cached_prefix(toks(2, 20)) == 2);  // b survives
  kv.free(c);
  BlockTable d;
  REQUIRE(kv.allocate_prompt(d, toks(2, 20)) == 2);
}

TEST_CASE("prefix cache: cannot over-commit against blocks it is about to reuse") {
  KVCacheManager kv(2, 2, true);
  BlockTable a;
  REQUIRE(kv.allocate_prompt(a, toks(4)) == 0); kv.commit_computed(a, toks(4), 4);
  kv.free(a);  // both blocks cached, 0 free
  BlockTable b;
  // prompt = same 2 blocks + 1 more token: needs 3 blocks total, 2 hits; pool has 2 -> impossible
  REQUIRE_FALSE(kv.allocate_prompt(b, toks(5)).has_value());
  REQUIRE(kv.num_cached_blocks() == 2);  // nothing was disturbed
  // same 2 blocks exactly: fits via hits alone
  REQUIRE(kv.allocate_prompt(b, toks(4)) == 4);
}

TEST_CASE("prefix caching disabled: nothing is cached after free") {
  KVCacheManager kv(4, 2, false);
  BlockTable a;
  REQUIRE(kv.allocate_prompt(a, toks(4)) == 0); kv.commit_computed(a, toks(4), 4);
  kv.free(a);
  REQUIRE(kv.num_cached_blocks() == 0);
  REQUIRE(kv.num_free_blocks() == 4);
  BlockTable b;
  REQUIRE(kv.allocate_prompt(b, toks(4)) == 0);
}
