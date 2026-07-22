// Copyright 2019-2025 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#undef NDEBUG

#include <filesystem>
#include <ygm/comm.hpp>
#include <ygm/io/ndjson_parser.hpp>

int main(int argc, char** argv) {
  ygm::comm world(&argc, &argv);

  {
    ygm::io::ndjson_parser jsonp(world,
                                 std::vector<std::string>{"data/3.ndjson"});

    //
    // Test for_all
    {
      size_t local_count{0};
      jsonp.for_all(
          [&local_count]([[maybe_unused]] const auto& json) { ++local_count; });

      world.barrier();
      YGM_ASSERT_RELEASE(ygm::sum(local_count, world) == 3);
    }

    //
    // Test iterators
    {
      size_t local_count{0};
      for (const auto& json : jsonp) {
        ++local_count;
      }

      world.barrier();
      YGM_ASSERT_RELEASE(ygm::sum(local_count, world) == 3);
    }
  }

  // Test json with bad lines
  {
    ygm::io::ndjson_parser jsonp(world,
                                 std::vector<std::string>{"data/bad.ndjson"});
    //
    // Test for_all
    {
      size_t local_count{0};
      jsonp.for_all(
          [&local_count]([[maybe_unused]] const auto& json) { ++local_count; });

      world.barrier();
      YGM_ASSERT_RELEASE(ygm::sum(local_count, world) == 3);
      YGM_ASSERT_RELEASE(jsonp.num_invalid_records() == 3);
    }

    //
    // Test iterators
    {
      size_t local_count{0};
      for (const auto& json : jsonp) {
        ++local_count;
      }

      world.barrier();
      YGM_ASSERT_RELEASE(ygm::sum(local_count, world) == 3);
      YGM_ASSERT_RELEASE(jsonp.num_invalid_records() == 3);
    }
  }

  return 0;
}
