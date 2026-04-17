
#include <ygm/container/detail/lru_cache.hpp>

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);

  // Test local inserts
  {
    size_t                                      size = 8;
    ygm::container::detail::lru_cache<int, int> cache(world, size);

    for (size_t i = 0; i < size + 1; ++i) {
      cache.local_insert(i, i);
    }

    YGM_ASSERT_RELEASE(cache.local_size() == size);
    YGM_ASSERT_RELEASE(cache.local_get(0).has_value() == false);

    for (size_t i = 1; i < size + 1; ++i) {
      YGM_ASSERT_RELEASE(cache.local_get(i).has_value() == true);
    }
  }

  // Test remote inserts
  {
    size_t                                      size = 8;
    ygm::container::detail::lru_cache<int, int> cache(world, size);

    if (world.rank0()) {
      for (size_t i = 0; i < size + 1; ++i) {
        cache.async_insert(1, i, i);
      }
    }

    world.barrier();

    if (world.rank() == 1) {
      YGM_ASSERT_RELEASE(cache.local_size() == size);
      YGM_ASSERT_RELEASE(cache.local_get(0).has_value() == false);

      for (size_t i = 1; i < size + 1; ++i) {
        YGM_ASSERT_RELEASE(cache.local_get(i).has_value() == true);
      }
    } else {
      YGM_ASSERT_RELEASE(cache.local_size() == 0);
    }
  }

  // Test value updates
  {
    size_t size       = 16;
    size_t num_rounds = 16;

    {
      ygm::container::detail::lru_cache<int, int> cache(world, size);

      for (size_t i = 0; i < size; ++i) {
        for (size_t round = 0; round < num_rounds; ++round) {
          cache.local_insert(i, round);
        }
      }

      YGM_ASSERT_RELEASE(cache.local_size() == size);
      for (size_t i = 0; i < size; ++i) {
        YGM_ASSERT_RELEASE((size_t)cache.local_get(i).value() ==
                           num_rounds - 1);
      }
    }
    {
      ygm::container::detail::lru_cache<int, int> cache(world, size);

      for (size_t round = 0; round < num_rounds; ++round) {
        for (size_t i = 0; i < size; ++i) {
          cache.local_insert(i, round);
        }
      }

      YGM_ASSERT_RELEASE(cache.local_size() == size);
      for (size_t i = 0; i < size; ++i) {
        YGM_ASSERT_RELEASE((size_t)cache.local_get(i).value() ==
                           num_rounds - 1);
      }
    }
  }

  // Test local inserts with functor
  {
    size_t                                      size       = 8;
    size_t                                      num_rounds = 16;
    ygm::container::detail::lru_cache<int, int> cache(world, size);

    auto check_equals = [](const int key, [[maybe_unused]] const int old_val,
                           const int new_val) { return (key == new_val); };

    for (size_t i = 0; i < size; ++i) {
      cache.local_insert(i, 0, check_equals);
    }

    YGM_ASSERT_RELEASE(cache.local_size() == size);
    for (size_t i = 1; i < size; ++i) {
      YGM_ASSERT_RELEASE(cache.local_get(i).has_value() == true);
      YGM_ASSERT_RELEASE(cache.local_get(i).value() == 0);
    }

    for (size_t round = 1; round < num_rounds; ++round) {
      for (size_t i = 0; i < size; ++i) {
        cache.local_insert(i, round, check_equals);
      }
    }

    YGM_ASSERT_RELEASE(cache.local_size() == size);
    for (size_t i = 1; i < size; ++i) {
      YGM_ASSERT_RELEASE(cache.local_get(i).has_value() == true);
      YGM_ASSERT_RELEASE((size_t)cache.local_get(i).value() == i);
    }
  }

  // Test remote inserts with functor
  {
    size_t                                      size       = 8;
    size_t                                      num_rounds = 16;
    ygm::container::detail::lru_cache<int, int> cache(world, size);

    auto check_equals = [](const int key, [[maybe_unused]] const int old_val,
                           const int new_val) { return (key == new_val); };

    if (world.rank0()) {
      for (size_t i = 0; i < size; ++i) {
        cache.async_insert(1, i, 0, check_equals);
      }
    }

    world.barrier();

    if (world.rank() == 1) {
      for (size_t i = 0; i < size; ++i) {
        YGM_ASSERT_RELEASE(cache.local_get(i).has_value() == true);
        YGM_ASSERT_RELEASE((size_t)cache.local_get(i).value() == 0);
      }
    } else {
      YGM_ASSERT_RELEASE(cache.local_size() == 0);
    }

    if (world.rank0()) {
      for (size_t i = 0; i < size; ++i) {
        for (size_t round = 1; round < num_rounds; ++round) {
          cache.async_insert(1, i, round, check_equals);
        }
      }
    }

    world.barrier();

    if (world.rank() == 1) {
      for (size_t i = 0; i < size; ++i) {
        YGM_ASSERT_RELEASE(cache.local_get(i).has_value() == true);
        YGM_ASSERT_RELEASE((size_t)cache.local_get(i).value() == i);
      }
    } else {
      YGM_ASSERT_RELEASE(cache.local_size() == 0);
    }
  }

  return 0;
}
