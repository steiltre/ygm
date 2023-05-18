// Copyright 2019-2021 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once
#include <unordered_map>
#include <vector>
#include <ygm/collective.hpp>
#include <ygm/comm.hpp>
#include <ygm/container/detail/hash_partitioner.hpp>
#include <ygm/detail/ygm_ptr.hpp>
#include <ygm/detail/ygm_traits.hpp>

namespace ygm::container::detail {
template <typename Item, typename Partitioner>
class disjoint_set_impl {
 public:
  class rank_parent_t;
  using self_type         = disjoint_set_impl<Item, Partitioner>;
  using self_ygm_ptr_type = typename ygm::ygm_ptr<self_type>;
  using value_type        = Item;
  using rank_type         = int16_t;
  using parent_map_type   = std::map<value_type, rank_parent_t>;

  Partitioner partitioner;

  class rank_parent_t {
   public:
    rank_parent_t() : m_rank{-1} {}

    rank_parent_t(const rank_type rank, const value_type &parent)
        : m_rank(rank), m_parent(parent) {}

    bool increase_rank(rank_type new_rank) {
      if (new_rank > m_rank) {
        m_rank = new_rank;
        return true;
      } else {
        return false;
      }
    }

    void set_parent(const value_type &new_parent) { m_parent = new_parent; }

    const rank_type   get_rank() const { return m_rank; }
    const value_type &get_parent() const { return m_parent; }

    template <typename Archive>
    void serialize(Archive &ar) {
      ar(m_parent, m_rank);
    }

   private:
    rank_type  m_rank;
    value_type m_parent;
  };

  class hash_cache {
   public:
    class cache_entry {
     public:
      cache_entry() : occupied(false) {}

      cache_entry(const value_type &_item, const rank_parent_t &_item_info)
          : item(_item), item_info(_item_info) {}

      bool          occupied = false;
      value_type    item;
      rank_parent_t item_info;
    };

    hash_cache(const size_t cache_size)
        : m_cache_size(cache_size), m_cache(cache_size) {}

    void add_cache_entry(const value_type    &item,
                         const rank_parent_t &item_info) {
      size_t index = std::hash<value_type>()(item) % m_cache_size;

      auto &current_entry = m_cache[index];

      // Only replace cached value if current slot is empty or if new entry's
      // rank is higher
      if (current_entry.occupied == false ||
          item_info.get_rank() >= current_entry.item_info.get_rank()) {
        current_entry.occupied  = true;
        current_entry.item      = item;
        current_entry.item_info = item_info;
      }
    }

    const cache_entry &get_cache_entry(const value_type &item) {
      size_t index = std::hash<value_type>()(item) % m_cache_size;

      return m_cache[index];
    }

    // private:
    size_t                   m_cache_size;
    std::vector<cache_entry> m_cache;
  };

  disjoint_set_impl(ygm::comm &comm)
      : m_comm(comm), pthis(this), m_cache(1024) {
    pthis.check(m_comm);
    clear_counters();
  }

  ~disjoint_set_impl() { m_comm.barrier(); }

  typename ygm::ygm_ptr<self_type> get_ygm_ptr() const { return pthis; }

  template <typename Visitor, typename... VisitorArgs>
  void async_visit(const value_type &item, Visitor visitor,
                   const VisitorArgs &...args) {
    int  dest          = owner(item);
    auto visit_wrapper = [](auto p_dset, const value_type &item,
                            const VisitorArgs &...args) {
      auto rank_parent_pair_iter = p_dset->m_local_item_parent_map.find(item);
      if (rank_parent_pair_iter == p_dset->m_local_item_parent_map.end()) {
        rank_parent_t new_ranked_item = rank_parent_t(0, item);
        rank_parent_pair_iter =
            p_dset->m_local_item_parent_map
                .insert(std::make_pair(item, new_ranked_item))
                .first;
      }
      Visitor *vis = nullptr;

      ygm::meta::apply_optional(
          *vis, std::make_tuple(p_dset),
          std::forward_as_tuple(*rank_parent_pair_iter, args...));
    };

    m_comm.async(dest, visit_wrapper, pthis, item,
                 std::forward<const VisitorArgs>(args)...);
  }

  void async_union(const value_type &a, const value_type &b) {
    static auto update_parent_lambda = [](auto             &item_info,
                                          const value_type &new_parent) {
      item_info.second.set_parent(new_parent);
    };

    static auto resolve_merge_lambda = [](auto p_dset, auto &item_info,
                                          const value_type &merging_item,
                                          const rank_type   merging_rank) {
      const auto &my_item   = item_info.first;
      const auto  my_rank   = item_info.second.get_rank();
      const auto &my_parent = item_info.second.get_parent();
      ASSERT_RELEASE(my_rank >= merging_rank);

      if (my_rank > merging_rank) {
        return;
      } else {
        ASSERT_RELEASE(my_rank == merging_rank);
        if (my_parent ==
            my_item) {  // Merging new item onto root. Need to increase rank.
          item_info.second.increase_rank(merging_rank + 1);
        } else {  // Tell merging item about new parent
          p_dset->async_visit(
              merging_item,
              [](auto &item_info, const value_type &new_parent) {
                item_info.second.set_parent(new_parent);
              },
              my_parent);
        }
      }
    };

    // Walking up parent trees can be expressed as a recursive operation
    struct simul_parent_walk_functor {
      void operator()(self_ygm_ptr_type                           p_dset,
                      std::pair<const value_type, rank_parent_t> &my_item_info,
                      const value_type                           &my_child,
                      const value_type                           &other_parent,
                      const value_type                           &other_item,
                      const rank_type                             other_rank) {
        // Note: other_item needs rank info for comparison with my_item's
        // parent. All others need rank and item to determine if other_item
        // has been visited/initialized.

        const value_type &my_item   = my_item_info.first;
        const rank_type  &my_rank   = my_item_info.second.get_rank();
        const value_type &my_parent = my_item_info.second.get_parent();

        // Path splitting
        if (my_child != my_item) {
          p_dset->async_visit(my_child, update_parent_lambda, my_parent);
        }

        if (my_parent == other_parent || my_parent == other_item) {
          return;
        }

        if (my_rank > other_rank) {  // Other path has lower rank
          p_dset->async_visit(other_parent, simul_parent_walk_functor(),
                              other_item, my_parent, my_item, my_rank);
        } else if (my_rank == other_rank) {
          if (my_parent == my_item) {  // At a root

            if (my_item < other_parent) {  // Need to break ties in rank before
                                           // merging to avoid cycles of merges
                                           // creating cycles in disjoint set
              // Perform merge
              my_item_info.second.set_parent(
                  other_parent);  // other_parent may be of same rank as my_item
              p_dset->async_visit(other_parent, resolve_merge_lambda, my_item,
                                  my_rank);
            } else {
              // Switch to other path to attempt merge
              p_dset->async_visit(other_parent, simul_parent_walk_functor(),
                                  other_item, my_parent, my_item, my_rank);
            }
          } else {  // Not at a root
            // Continue walking current path
            p_dset->async_visit(my_parent, simul_parent_walk_functor(), my_item,
                                other_parent, other_item, other_rank);
          }
        } else {                       // Current path has lower rank
          if (my_parent == my_item) {  // At a root
            my_item_info.second.set_parent(
                other_parent);  // Safe to attach to other path
          } else {              // Not at a root
            // Continue walking current path
            p_dset->async_visit(my_parent, simul_parent_walk_functor(), my_item,
                                other_parent, other_item, other_rank);
          }
        }
      }
    };

    async_visit(a, simul_parent_walk_functor(), a, b, b, -1);
  }

  template <typename Function, typename... FunctionArgs>
  void async_union_and_execute(const value_type &a, const value_type &b,
                               Function fn, const FunctionArgs &...args) {
    static auto update_parent_and_cache_lambda =
        [](auto p_dset, auto &item_info, const value_type &new_parent,
           const rank_type &new_rank) {
          p_dset->m_cache.add_cache_entry(item_info.second.get_parent(),
                                          rank_parent_t(new_rank, new_parent));

          item_info.second.set_parent(new_parent);
          ++(p_dset->update_parent_lambda_count);
        };

    static auto resolve_merge_lambda = [](auto p_dset, auto &item_info,
                                          const value_type &merging_item,
                                          const rank_type   merging_rank) {
      const auto &my_item   = item_info.first;
      const auto  my_rank   = item_info.second.get_rank();
      const auto &my_parent = item_info.second.get_parent();
      ASSERT_RELEASE(my_rank >= merging_rank);

      if (my_rank > merging_rank) {
        return;
      } else {
        ASSERT_RELEASE(my_rank == merging_rank);
        if (my_parent == my_item) {  // Has not found new parent
          ++(p_dset->roots_visited);
          item_info.second.increase_rank(merging_rank + 1);
        } else {  // Tell merging item about new parent
          p_dset->async_visit(
              merging_item,
              [](auto &item_info, const value_type &new_parent) {
                item_info.second.set_parent(new_parent);
              },
              my_parent);
        }
      }
      ++(p_dset->resolve_merge_lambda_count);
    };

    // Walking up parent trees can be expressed as a recursive operation
    struct simul_parent_walk_functor {
      void operator()(self_ygm_ptr_type                           p_dset,
                      std::pair<const value_type, rank_parent_t> &my_item_info,
                      const value_type &my_child, value_type other_parent,
                      value_type other_item, rank_type other_rank,
                      const value_type &orig_a, const value_type &orig_b,
                      const FunctionArgs &...args) {
        // Note: other_item needs rank info for comparison with my_item's
        // parent. All others need rank and item to determine if other_item
        // has been visited/initialized.
        value_type my_item   = my_item_info.first;
        rank_type  my_rank   = my_item_info.second.get_rank();
        value_type my_parent = my_item_info.second.get_parent();

        bool rank_7 = (my_rank == 7);

        ++(p_dset->simul_parent_walk_functor_count);
        ++(p_dset->walk_visit_ranks)[my_rank];
        if (my_parent == my_item) {
          ++(p_dset->roots_visited);
        }

        std::tie(my_item, my_rank, my_parent) =
            p_dset->walk_cache(my_item, my_rank, my_parent);
        std::tie(other_item, other_rank, other_parent) =
            p_dset->walk_cache(other_item, other_rank, other_parent);

        if (not rank_7 && my_rank == 7) {
          ++(p_dset->cache_rank_7);
        }

        // Path splitting
        if (my_child != my_item) {
          p_dset->async_visit(my_child, update_parent_and_cache_lambda,
                              my_parent, my_rank);
        }

        if (my_parent == other_parent || my_parent == other_item) {
          return;
        }

        ++(p_dset->walk_visit_ranks)[my_rank];

        if (my_rank > other_rank) {  // Other path has lower rank
          p_dset->async_visit(other_parent, simul_parent_walk_functor(),
                              other_item, my_parent, my_item, my_rank, orig_a,
                              orig_b, args...);
        } else if (my_rank == other_rank) {
          if (my_parent == my_item) {  // At a root

            if (my_item < other_parent) {  // Need to break ties in rank before
                                           // merging to avoid cycles of merges
                                           // creating cycles in disjoint set
              // Perform merge
              my_item_info.second.set_parent(
                  other_parent);  // Guaranteed any path through current
                                  // item will find an item with rank >=
                                  // my_rank+1 by going to other_parent

              // Perform user function after merge
              Function *f = nullptr;
              if constexpr (std::is_invocable<decltype(fn), const value_type &,
                                              const value_type &,
                                              FunctionArgs &...>() ||
                            std::is_invocable<decltype(fn), self_ygm_ptr_type,
                                              const value_type &,
                                              const value_type &,
                                              FunctionArgs &...>()) {
                ygm::meta::apply_optional(
                    *f, std::make_tuple(p_dset),
                    std::forward_as_tuple(orig_a, orig_b, args...));
              } else {
                static_assert(
                    ygm::detail::always_false<>,
                    "remote disjoint_set lambda signature must be invocable "
                    "with (const value_type &, const value_type &) signature");
              }

              // return;

              p_dset->async_visit(other_parent, resolve_merge_lambda, my_item,
                                  my_rank);
            } else {
              // Switch to other path to attempt merge
              p_dset->async_visit(other_parent, simul_parent_walk_functor(),
                                  other_item, my_parent, my_item, my_rank,
                                  orig_a, orig_b, args...);
            }
          } else {  // Not at a root
            // Continue walking current path
            p_dset->async_visit(my_parent, simul_parent_walk_functor(), my_item,
                                other_parent, other_item, other_rank, orig_a,
                                orig_b, args...);
          }
        } else {                       // Current path has lower rank
          if (my_parent == my_item) {  // At a root
            my_item_info.second.set_parent(
                other_parent);  // Safe to attach to other path

            // Perform user function after merge
            Function *f = nullptr;
            ygm::meta::apply_optional(
                *f, std::make_tuple(p_dset),
                std::forward_as_tuple(orig_a, orig_b, args...));

            return;

          } else {  // Not at a root
            // Continue walking current path
            p_dset->async_visit(my_parent, simul_parent_walk_functor(), my_item,
                                other_parent, other_item, other_rank, orig_a,
                                orig_b, args...);
          }
        }
      }
    };

    async_visit(a, simul_parent_walk_functor(), a, b, b, -1, a, b, args...);
  }

  void all_compress() {
    struct rep_query {
      value_type              rep;
      std::vector<value_type> local_inquiring_items;
      bool                    returned;
    };

    static rank_type                                 level;
    static std::unordered_map<value_type, rep_query> queries;
    static std::unordered_map<value_type, std::vector<int>>
        held_responses;  // For holding incoming queries while my items are
                         // waiting for their representatives (only needed for
                         // when parent rank is same as mine)

    struct update_rep_functor {
     public:
      void operator()(self_ygm_ptr_type p_dset, const value_type &parent,
                      const value_type &rep) {
        auto local_rep_query     = queries.at(parent);
        local_rep_query.rep      = rep;
        local_rep_query.returned = true;

        for (const auto &local_item : local_rep_query.local_inquiring_items) {
          p_dset->local_set_parent(local_item, rep);

          // Forward rep for any held responses
          auto held_responses_iter = held_responses.find(local_item);
          if (held_responses_iter != held_responses.end()) {
            for (int dest : held_responses_iter->second) {
              p_dset->comm().async(dest, update_rep_functor(), p_dset,
                                   local_item, rep);
            }
            held_responses.erase(held_responses_iter);
          }
        }
        local_rep_query.local_inquiring_items.clear();
      }
    };

    auto query_rep_lambda = [](self_ygm_ptr_type p_dset, const value_type &item,
                               int inquiring_rank) {
      const auto &item_info = p_dset->m_local_item_parent_map[item];

      if (item_info.get_rank() > level) {
        const value_type &rep = item_info.get_parent();

        p_dset->comm().async(inquiring_rank, update_rep_functor(), p_dset, item,
                             rep);
      } else {  // May need to hold because this item is in the current level
        if (queries.count(
                item_info.get_parent())) {  // If query is ongoing for my
                                            // parent, hold response
          held_responses[item].push_back(inquiring_rank);
        } else {
          p_dset->comm().async(inquiring_rank, update_rep_functor(), p_dset,
                               item, item_info.get_parent());
        }
      }
    };

    m_comm.barrier();

    level = max_rank();
    while (level > 0) {
      --level;  // Start at second highest level
      queries.clear();
      held_responses.clear();

      for (const auto &[local_item, item_info] : m_local_item_parent_map) {
        if (item_info.get_rank() == level) {
          auto query_iter = queries.find(item_info.get_parent());
          if (query_iter == queries.end()) {  // Have not queried for parent's
                                              // rep. Begin new query.
            auto &new_query    = queries[item_info.get_parent()];
            new_query.rep      = item_info.get_parent();
            new_query.returned = false;
            new_query.local_inquiring_items.push_back(local_item);

            int dest = owner(item_info.get_parent());
            m_comm.async(dest, query_rep_lambda, pthis, item_info.get_parent(),
                         m_comm.rank());
          } else {
            if (query_iter->second
                    .returned) {  // Query for parent's rep already completed.
              local_set_parent(local_item, query_iter->second.rep);
            } else {  // Query for parent's rep still in progress.
              query_iter->second.local_inquiring_items.push_back(local_item);
            }
          }
        }
      }

      m_comm.barrier();
    }
    m_comm.cout0("Total items: ", size());
    m_comm.cout0("Max rank items: ",
                 m_comm.all_reduce_max(m_local_item_parent_map.size()));
    m_comm.cout0("Min rank items: ",
                 m_comm.all_reduce_min(m_local_item_parent_map.size()));
  }

  template <typename Function>
  void for_all(Function fn) {
    all_compress();

    if constexpr (std::is_invocable<decltype(fn), const value_type &,
                                    const value_type &>()) {
      const auto end = m_local_item_parent_map.end();
      for (auto iter = m_local_item_parent_map.begin(); iter != end; ++iter) {
        const auto &[item, rank_parent_pair] = *iter;
        fn(item, rank_parent_pair.get_parent());
      }
    } else {
      static_assert(ygm::detail::always_false<>,
                    "local disjoint_set lambda signature must be invocable "
                    "with (const value_type &, const value_type &) signature");
    }
  }

  std::map<value_type, value_type> all_find(
      const std::vector<value_type> &items) {
    m_comm.barrier();

    using return_type = std::map<value_type, value_type>;
    return_type          to_return;
    ygm_ptr<return_type> p_to_return(&to_return);

    struct find_rep_functor {
      void operator()(self_ygm_ptr_type pdset, ygm_ptr<return_type> p_to_return,
                      const value_type &source_item, const int source_rank,
                      const value_type &local_item) {
        const auto parent = pdset->local_get_parent(local_item);

        // Found root
        if (parent == local_item) {
          // Send message to update parent of original item
          int dest = pdset->owner(source_item);
          pdset->comm().async(
              dest,
              [](self_ygm_ptr_type pdset, const value_type &source_item,
                 const value_type &root) {
                pdset->local_set_parent(source_item, root);
              },
              pdset, source_item, parent);
          // Send message to store return value
          pdset->comm().async(
              source_rank,
              [](ygm_ptr<return_type> p_to_return,
                 const value_type    &source_item,
                 const value_type &rep) { (*p_to_return)[source_item] = rep; },
              p_to_return, source_item, parent);
        } else {
          int dest = pdset->owner(parent);
          pdset->comm().async(dest, find_rep_functor(), pdset, p_to_return,
                              source_item, source_rank, parent);
        }
      }
    };

    for (size_t i = 0; i < items.size(); ++i) {
      int dest = owner(items[i]);
      m_comm.async(dest, find_rep_functor(), pthis, p_to_return, items[i],
                   m_comm.rank(), items[i]);
    }

    m_comm.barrier();
    return to_return;
  }

  size_t size() {
    m_comm.barrier();
    return m_comm.all_reduce_sum(m_local_item_parent_map.size());
  }

  size_t num_sets() {
    m_comm.barrier();
    size_t num_local_sets{0};
    for (const auto &item_parent_pair : m_local_item_parent_map) {
      if (item_parent_pair.first == item_parent_pair.second.get_parent()) {
        ++num_local_sets;
      }
    }
    return m_comm.all_reduce_sum(num_local_sets);
    return 0;
  }

  int owner(const value_type &item) const {
    auto [owner, rank] = partitioner(item, m_comm.size(), 1024);
    return owner;
  }

  bool is_mine(const value_type &item) const {
    return owner(item) == m_comm.rank();
  }

  const value_type &local_get_parent(const value_type &item) {
    ASSERT_DEBUG(is_mine(item) == true);

    auto itr = m_local_item_parent_map.find(item);

    // Create new set if item is not found
    if (itr == m_local_item_parent_map.end()) {
      m_local_item_parent_map.insert(
          std::make_pair(item, rank_parent_t(0, item)));
      return m_local_item_parent_map[item].get_parent();
    } else {
      return itr->second.get_parent();
    }
    return m_local_item_parent_map[item].get_parent();
  }

  const rank_type local_get_rank(const value_type &item) {
    ASSERT_DEBUG(is_mine(item) == true);

    auto itr = m_local_item_parent_map.find(item);

    if (itr != m_local_item_parent_map.end()) {
      return itr->second.first;
    }
    return 0;
  }

  void local_set_parent(const value_type &item, const value_type &parent) {
    m_local_item_parent_map[item].set_parent(parent);
  }

  rank_type max_rank() {
    rank_type local_max{0};

    for (const auto &local_item : m_local_item_parent_map) {
      local_max = std::max<rank_type>(local_max, local_item.second.get_rank());
    }

    return m_comm.all_reduce_max(local_max);
  }

  size_t count_rank(const rank_type r) {
    size_t count = 0;

    for (const auto &local_item : m_local_item_parent_map) {
      count += (local_item.second.get_rank() == r);
    }

    return ygm::sum(count, m_comm);
  }

  rank_type min_max_cached_rank() {
    rank_type local_max = 0;

    for (auto &c : m_cache.m_cache) {
      if (c.occupied) {
        local_max = std::max<rank_type>(c.item_info.get_rank(), local_max);
      }
    }

    return ygm::min(local_max, m_comm);
  }

  rank_type max_max_cached_rank() {
    rank_type local_max = 0;

    for (auto &c : m_cache.m_cache) {
      if (c.occupied) {
        local_max = std::max<rank_type>(c.item_info.get_rank(), local_max);
      }
    }

    return ygm::max(local_max, m_comm);
  }

  ygm::comm &comm() { return m_comm; }

  void clear_counters() {
    simul_parent_walk_functor_count = 0;
    resolve_merge_lambda_count      = 0;
    update_parent_lambda_count      = 0;
    roots_visited                   = 0;
    cache_rank_7                    = 0;

    walk_visit_ranks.clear();
    walk_visit_ranks.resize(16);
  }

  void print_counters() {
    std::vector<int64_t> walk_visit_sum(16);
    std::vector<int64_t> walk_visit_min(16);
    std::vector<int64_t> walk_visit_max(16);

    MPI_Allreduce(walk_visit_ranks.data(), walk_visit_sum.data(), 16,
                  MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(walk_visit_ranks.data(), walk_visit_min.data(), 16,
                  MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(walk_visit_ranks.data(), walk_visit_max.data(), 16,
                  MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    m_comm.cout0(
        "----Disjoint set counters----", "\nMax rank:\t", max_rank(),
        "\nRank 7s:\t", count_rank(7),
        "\nCache rank 7 visits:\n\tSum: ", ygm::sum(cache_rank_7, m_comm),
        "\n\tMin: ", ygm::min(cache_rank_7, m_comm),
        "\n\tMax: ", ygm::max(cache_rank_7, m_comm),
        "\nMax cached ranks: min: ", min_max_cached_rank(),
        "\t max: ", max_max_cached_rank(),
        "\nsimul_parent_walk_functor_count:\n\tSum: ",
        ygm::sum(simul_parent_walk_functor_count, m_comm),
        "\n\tMin: ", ygm::min(simul_parent_walk_functor_count, m_comm),
        "\n\tMax: ", ygm::max(simul_parent_walk_functor_count, m_comm),
        "\nroots_visited:\n\tSum: ", ygm::sum(roots_visited, m_comm),
        "\n\tMin: ", ygm::min(roots_visited, m_comm),
        "\n\tMax: ", ygm::max(roots_visited, m_comm),
        "\nresolve_merge_lambda_count:\n\tSum: ",
        ygm::sum(resolve_merge_lambda_count, m_comm),
        "\n\tMin: ", ygm::min(resolve_merge_lambda_count, m_comm),
        "\n\tMax: ", ygm::max(resolve_merge_lambda_count, m_comm),
        "\nupdate_parent_lambda_count:\n\tSum: ",
        ygm::sum(update_parent_lambda_count, m_comm),
        "\n\tMin: ", ygm::min(update_parent_lambda_count, m_comm),
        "\n\tMax: ", ygm::max(update_parent_lambda_count, m_comm));

    m_comm.cout0() << "\nWalk visit ranks:\n\t\t";
    for (int i = 0; i < 16; ++i) {
      m_comm.cout0() << "  (" << i << ", " << walk_visit_sum[i] << ", "
                     << walk_visit_min[i] << ", " << walk_visit_max[i] << ")";
    }
    m_comm.cout0();
  }

 private:
  const std::tuple<value_type, rank_type, value_type> walk_cache(
      const value_type &item, const rank_type &r, const value_type &parent) {
    const value_type                       *curr_item = &item;
    typename hash_cache::cache_entry        tmp_cache_entry(item,
                                                            rank_parent_t(r, parent));
    const typename hash_cache::cache_entry *curr_cache_entry = &tmp_cache_entry;
    const typename hash_cache::cache_entry *next_cache_entry =
        &m_cache.get_cache_entry(item);

    while (*curr_item == next_cache_entry->item && next_cache_entry->occupied &&
           *curr_item != next_cache_entry->item) {
      curr_cache_entry = next_cache_entry;
      curr_item        = &curr_cache_entry->item;
      next_cache_entry = &m_cache.get_cache_entry(*curr_item);
    }

    return std::make_tuple(curr_cache_entry->item,
                           curr_cache_entry->item_info.get_rank(),
                           curr_cache_entry->item_info.get_parent());
  }

 protected:
  disjoint_set_impl() = delete;

  ygm::comm         m_comm;
  self_ygm_ptr_type pthis;
  parent_map_type   m_local_item_parent_map;

  hash_cache m_cache;

  int64_t              simul_parent_walk_functor_count;
  int64_t              resolve_merge_lambda_count;
  int64_t              update_parent_lambda_count;
  int64_t              roots_visited;
  std::vector<int64_t> walk_visit_ranks;
  int64_t              cache_rank_7;
};
}  // namespace ygm::container::detail
