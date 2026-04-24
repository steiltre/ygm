
#include <boost/unordered/unordered_flat_map.hpp>
#include <list>
#include <map>
#include <optional>

#include <ygm/comm.hpp>

#pragma once

namespace ygm::container::detail {

template <typename Key, typename Value>
class lru_cache {
 public:
  using self_type   = lru_cache<Key, Value>;
  using mapped_type = Value;
  using ptr_type    = typename ygm::ygm_ptr<self_type>;
  using key_type    = Key;
  using size_type   = size_t;

 private:
  struct list_item {
    key_type    key;
    mapped_type value;
  };

  using list_type = std::list<list_item>;
  using map_type =
      boost::unordered_flat_map<key_type,
                                typename std::list<list_item>::iterator>;

 public:
  lru_cache() = delete;

  /**
   * @brief lru_cache Constructor
   *
   * @param comm Communicator to use for communication
   * @param capacity Number of elements in cache
   */
  lru_cache(ygm::comm &comm, size_t capacity)
      : m_comm(comm), pthis(this), m_capacity(capacity) {
    m_comm.log(log_level::info, "Creating ygm::container::lru_cache");
    pthis.check(m_comm);
  }

  /**
   * @brief Insert an item into the cache on a given rank
   *
   * @param key Key to insert
   * @param val Associated value to insert
   */
  void async_insert(const int rank, const key_type &key,
                    const mapped_type &val) {
    auto inserter = [](const auto pcache, const key_type &key,
                       const mapped_type &val) {
      pcache->local_insert(key, val);
    };

    m_comm.async(rank, inserter, pthis, key, val);
  }

  /**
   * @brief Insert an item into the cache on a given rank
   *
   * @tparam Function functor type
   * @param key Key to insert
   * @param val Associated value to insert
   * @param fn Functor that returns true if existing value is to be replaced
   */
  template <typename Function>
  void async_insert(const int rank, const key_type &key, const mapped_type &val,
                    Function &&fn) {
    auto inserter = [fn](const auto pcache, const key_type &key,
                         const mapped_type &val) {
      pcache->local_insert(key, val, fn);
    };

    m_comm.async(rank, inserter, pthis, key, val);
  }

  /**
   * @brief Add element to local cache
   *
   * @param key Key to insert
   * @param val Associated value for key
   */
  void local_insert(const key_type &key, const mapped_type &val) {
    if (m_map.contains(key)) {
      m_list.erase(m_map[key]);
    }

    m_list.push_front(list_item(key, val));
    m_map[key] = m_list.begin();

    shrink_to_capacity();
    ++m_cache_inserts;
  }

  /**
   * @brief Add element to local cache with a provided lambda for determining
   * whether to replace the current value
   *
   * @tparam Function functor type
   * @param key Key to insert
   * @param val Associated value for key
   */
  template <typename Function>
  void local_insert(const key_type &key, const mapped_type &val,
                    Function &&fn) {
    bool replace = true;

    if (m_map.contains(key)) {
      const mapped_type &curr_val = m_map.find(key)->second->value;

      replace = fn(key, curr_val, val);
    }

    if (replace) {
      local_insert(key, val);
    }
  }

  /**
   * @brief Retrieve a cached element, if present
   *
   * @param key Key to retrieve cached value for
   * @return std::optional containing value associdated with key, if present
   */
  std::optional<mapped_type> local_get(const key_type &key) const {
    std::optional<mapped_type> to_return;

    if (m_map.contains(key)) {
      to_return = m_map.find(key)->second->value;
      ++m_cache_hits;
    } else {
      ++m_cache_misses;
    }

    return to_return;
  };

  /**
   * @brief Retrieve the size of the local cache
   *
   * @return Size of the cache stored on the local rank
   */
  size_type local_size() {
    YGM_ASSERT_RELEASE(m_list.size() == m_map.size());
    return m_list.size();
  }

  /**
   * @brief Access the ygm_ptr used by the lru_cache
   *
   * @return ygm_ptr used to identify lru_cache in async calls on ygm::comm
   */
  ptr_type get_ygm_ptr() { return pthis; }

  /**
   * @brief Access the ygm_ptr used by the lru_cache
   *
   * @return ygm_ptr used to identify lru_cache in async calls on ygm::comm
   */
  ptr_type get_ygm_ptr() const { return pthis; }

  /**
   * @brief Return the number of local cache hits
   *
   * @return Number of cache hits
   */
  size_t local_cache_hit_count() { return m_cache_hits; }

  /**
   * @brief Return the number of local cache misses
   *
   * @return Number of cache misses
   */
  size_t local_cache_miss_count() { return m_cache_misses; }

  /**
   * @brief Return the number of local cache inserts
   *
   * @return Number of cache inserts
   */
  size_t local_cache_insert_count() { return m_cache_inserts; }

 private:
  /**
   * @brief Shrinks the cache to the specified capacity by removing
   * least-recently used items
   *
   * @return Number of removed items
   */
  size_t shrink_to_capacity() {
    size_t num_removed{0};

    while (m_list.size() > m_capacity) {
      auto to_remove = m_list.back();
      m_map.erase(to_remove.key);
      m_list.pop_back();
      ++num_removed;
    }

    YGM_ASSERT_RELEASE(m_list.size() == m_map.size());
    YGM_ASSERT_RELEASE(m_list.size() <= m_capacity);

    return num_removed;
  }

 private:
  ygm::comm &m_comm;
  ptr_type   pthis;

  size_t m_capacity;

  mutable size_t m_cache_hits    = 0;
  mutable size_t m_cache_misses  = 0;
  mutable size_t m_cache_inserts = 0;

  list_type m_list;
  map_type  m_map;
};

}  // namespace ygm::container::detail
