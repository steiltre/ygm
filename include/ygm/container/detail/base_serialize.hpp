// Copyright 2019-2025 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <fstream>
#include <ygm/container/detail/base_concepts.hpp>
#include <ygm/utility/boost_json.hpp>

namespace ygm::container::detail {

constexpr int manifest_version = 1;

template <typename T>
concept HasExtendManifest = requires(const T& v) {
  v.extend_manifest(std::declval<boost::json::object&>());
};

template <typename T>
concept HasCustomSerialize = requires(const T& v) { v.custom_serialize(); };

/**
 * @brief Curiously-recurring template pattern struct that provides serialize
 * and deserialize functions for containers
 */
template <typename derived_type, typename for_all_args>
struct base_serialize {
  /**
   * @brief Create a manifest file that contains basic information about
   * serialized container. Containers can update the manifest before it is
   * written.
   *
   * @param path Path for manifest file
   */
  void write_manifest(const std::filesystem::path& manifest_path) {
    derived_type* derived_this = static_cast<derived_type*>(this);

    boost::json::object manifest_obj{};
    manifest_obj["version"]   = manifest_version;
    manifest_obj["comm_size"] = derived_this->comm().size();

    // Allows extending the manifest for container-specific needs
    if constexpr (HasExtendManifest<derived_type>) {
      derived_this->extend_manifest(manifest_obj);
    }

    // Write manifest from rank 0
    if (derived_this->comm().rank0()) {
      std::ofstream ofs(manifest_path);
      ofs << manifest_obj << std::endl;
    }
  }
};
}  // namespace ygm::container::detail
