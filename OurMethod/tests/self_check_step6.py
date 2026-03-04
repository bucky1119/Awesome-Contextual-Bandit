"""Step 6 自检 — Prompt-to-Vector Cache 验证。

Done 标准：
  ✓ 重复 key → 命中
  ✓ put 相同 key 覆盖旧值
  ✓ 超容量时 LRU 淘汰
  ✓ protect_flag 阻止淘汰（保护项被跳过，淘汰非保护项）
  ✓ 全部保护时，强制淘汰最旧保护项
  ✓ hit_rate 统计正确
  ✓ invalidate 工作
"""

import sys, os, unittest
_R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _R not in sys.path:
    sys.path.insert(0, _R)

import numpy as np

from OurMethod.cache.prompt_cache import PromptVectorCache, CacheEntry


class TestCacheBasic(unittest.TestCase):
    def setUp(self):
        self.cache = PromptVectorCache(capacity=4)

    def test_put_and_get(self):
        h = np.ones(8)
        self.cache.put("k1", h_t=h)
        entry = self.cache.get("k1")
        self.assertIsNotNone(entry)
        np.testing.assert_array_equal(entry.h_t, h)

    def test_miss_returns_none(self):
        self.assertIsNone(self.cache.get("nonexistent"))

    def test_hit_rate(self):
        self.cache.put("a", h_t=np.zeros(4))
        self.cache.get("a")   # hit
        self.cache.get("a")   # hit
        self.cache.get("b")   # miss
        self.assertAlmostEqual(self.cache.hit_rate, 2 / 3, places=2)

    def test_overwrite(self):
        self.cache.put("k1", h_t=np.ones(4))
        self.cache.put("k1", h_t=np.zeros(4))
        entry = self.cache.get("k1")
        np.testing.assert_array_equal(entry.h_t, np.zeros(4))
        self.assertEqual(self.cache.size, 1)


class TestCacheEviction(unittest.TestCase):
    def test_lru_eviction(self):
        cache = PromptVectorCache(capacity=3)
        cache.put("a", h_t=np.zeros(2))
        cache.put("b", h_t=np.zeros(2))
        cache.put("c", h_t=np.zeros(2))
        # 访问 a 使其变新
        cache.get("a")
        # d 插入触发淘汰 b（最旧未访问）
        cache.put("d", h_t=np.zeros(2))
        self.assertIsNone(cache.get("b"), "b 应被淘汰")
        self.assertIsNotNone(cache.get("a"), "a 不应被淘汰")
        self.assertIsNotNone(cache.get("c"), "c 不应被淘汰")

    def test_protect_prevents_eviction(self):
        cache = PromptVectorCache(capacity=3)
        cache.put("a", h_t=np.zeros(2), protect=True)
        cache.put("b", h_t=np.zeros(2))
        cache.put("c", h_t=np.zeros(2))
        # d 插入触发淘汰，a 受保护所以淘汰 b
        cache.put("d", h_t=np.zeros(2))
        self.assertIsNotNone(cache.get("a"), "保护项 a 不应被淘汰")
        self.assertIsNone(cache.get("b"), "非保护项 b 应被淘汰")

    def test_all_protected_evicts_oldest(self):
        cache = PromptVectorCache(capacity=2)
        cache.put("a", h_t=np.zeros(2), protect=True)
        cache.put("b", h_t=np.zeros(2), protect=True)
        # 超容量：全部保护，强制淘汰最旧保护项 a
        cache.put("c", h_t=np.zeros(2))
        self.assertIsNone(cache.get("a"), "最旧保护项 a 应被强制淘汰")
        self.assertIsNotNone(cache.get("b"))
        self.assertIsNotNone(cache.get("c"))


class TestCacheInvalidate(unittest.TestCase):
    def test_invalidate(self):
        cache = PromptVectorCache(capacity=10)
        cache.put("k1", h_t=np.zeros(2))
        self.assertTrue(cache.invalidate("k1"))
        self.assertIsNone(cache.get("k1"))
        self.assertFalse(cache.invalidate("k1"))

    def test_clear(self):
        cache = PromptVectorCache(capacity=10)
        cache.put("a", h_t=np.zeros(2))
        cache.put("b", h_t=np.zeros(2))
        cache.clear()
        self.assertEqual(cache.size, 0)
        self.assertAlmostEqual(cache.hit_rate, 0.0)


class TestMakeKey(unittest.TestCase):
    def test_deterministic(self):
        k1 = PromptVectorCache.make_key("u1", "m1", "0,1,2", "bucket_a")
        k2 = PromptVectorCache.make_key("u1", "m1", "0,1,2", "bucket_a")
        self.assertEqual(k1, k2)

    def test_different_inputs_different_keys(self):
        k1 = PromptVectorCache.make_key("u1", "m1", "0,1", "a")
        k2 = PromptVectorCache.make_key("u2", "m1", "0,1", "a")
        self.assertNotEqual(k1, k2)


if __name__ == "__main__":
    unittest.main()
