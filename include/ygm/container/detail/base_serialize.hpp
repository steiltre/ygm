// Copyright 2019-2025 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <fstream>

#include <cereal/archives/binary.hpp>
// #include <cereal/archives/portable_binary.hpp>
// #include <cereal/archives/json.hpp>

#include <ygm/container/detail/base_concepts.hpp>
#include <ygm/detail/collective.hpp>
#include <ygm/utility/boost_json.hpp>
#include <ygm/utility/filesystem.hpp>

namespace ygm::container::detail {

constexpr int manifest_version = 1;

template <typename derived_type, typename for_all_args>
struct base_serialize;

/*
template <typename T>
concept HasExtendManifest = requires(const T& v) {
  v.extend_manifest(std::declval<boost::json::object&>());
};
*/

template <typename T>
concept HasCustomSerialize = requires(const T& v) {
  v.custom_serialize(
      std::declval<std::filesystem::path>(),
      std::declval<base_serialize<T, typename T::for_all_args>::manifest_t>());
};

template <typename T>
concept HasCustomDeserialize = requires(const T& v) {
  v.custom_deserialize(
      std::declval<std::filesystem::path>(),
      std::declval<base_serialize<T, typename T::for_all_args>::manifest_t>());
};

template <typename T>
concept HasSerializePrologue = requires(const T& v) {
  v.serialize_prologue(

      std::declval<std::filesystem::path>(),
      std::declval<base_serialize<T, typename T::for_all_args>::manifest_t>());
};

template <typename T>
concept HasSerializeEpilogue = requires(const T& v) {
  v.serialize_epilogue(
      std::declval<std::filesystem::path>(),
      std::declval<base_serialize<T, typename T::for_all_args>::manifest_t>());
};

template <typename T>
concept HasDeserializePrologue = requires(const T& v) {
  v.deserialize_prologue(
      std::declval<std::filesystem::path>(),
      std::declval<base_serialize<T, typename T::for_all_args>::manifest_t>());
};

template <typename T>
concept HasDeserializeEpilogue = requires(const T& v) {
  v.deserialize_epilogue(
      std::declval<std::filesystem::path>(),
      std::declval<base_serialize<T, typename T::for_all_args>::manifest_t>());
};

/**
 * @brief Curiously-recurring template pattern struct that provides serialize
 * and deserialize functions for containers
 */
template <typename derived_type, typename for_all_args>
struct base_serialize {
  using manifest_t       = boost::json::object;
  using output_archive_t = cereal::BinaryOutputArchive;
  using input_archive_t  = cereal::BinaryInputArchive;

  /**
   * @brief Function to serialize a container
   *
   * @param serialization_path Path to store serialized container
   *
   * @details The serialization logic can be completely overwritten by providing
   * a `custom_serialize()` method in the container class. When using the
   * serialization operation provided in the CRTP class, the serialization steps
   * are
   * 1. create the manifest object in memory
   * 2. complete the `serialize_prologue(path, manifest)` function, if provided
   * 3. serialize container data to files
   * 4. complete the `serialize_epilogue(path, manifest)` function, if provided
   * 5. write the manifest object from memory to a file alongside the serialized
   * data
   */
  void serialize(std::filesystem::path serialization_path) {
    derived_type* derived_this = static_cast<derived_type*>(this);

    serialization_path /= "";  // Append directory separator if not present
    ygm::utility::fs::make_directories(serialization_path);

    manifest_t manifest_obj = create_manifest();

    if constexpr (HasCustomSerialize<derived_type>) {
      derived_this->custom_serialize(serialization_path, manifest_obj);
    } else {
      if constexpr (HasSerializePrologue<derived_type>) {
        derived_this->serialize_prologue(serialization_path, manifest_obj);
      }

      std::filesystem::path rank_path =
          serialization_path /
          ("rank" + std::to_string(derived_this->comm().rank()));
      std::ofstream ofs(rank_path);
      // cereal::PortableBinaryOutputArchive archive(ofs);
      output_archive_t archive(ofs);

      if constexpr (SingleItemTuple<for_all_args>) {
        for (const auto& value : *derived_this) {
          archive(value);
        }
      } else if constexpr (DoubleItemTuple<for_all_args>) {
        for (const auto& [key, value] : *derived_this) {
          archive(key, value);
        }
      }

      derived_this->comm().cf_barrier();

      if constexpr (HasSerializeEpilogue<derived_type>) {
        derived_this->serialize_epilogue(serialization_path, manifest_obj);
      }
    }

    write_manifest(serialization_path / "manifest.json", manifest_obj);
  }

  /**
   * @brief Function to deserialize a container
   *
   * @param serialization_path Path to serialized container
   * @param check_types Optional bool indicating whether types should be checked
   * before deserialization. Defaults to true.
   *
   * @details The deserialization logic can be completely overwritten by
   * providing a `custom_deserialize()` method in the container class. When
   * using the deserialization operation provided in the CRTP class, the
   * deserialization steps are:
   * 1. Read the manifest for the container
   * 2. Check the types (if check_types == true)
   * 3. complete the `deserialize_prologue(path, manifest)` function, if
   * provided
   * 4. deserialize from files to container
   * 5. complete the `deserialize_epilogue(path, manifest)` function, if
   * provided The ability to run without checking types is provided for cases
   * when the typeid for the stored data does not agree with that stored in the
   * manifest. This can occur, for instance, if the serialization and
   * deserialization executables contain the same class that gets serialized,
   * but with each having unique names. This is mainly included as a potential
   * option that can give access to otherwise inaccessible serialized data.
   */
  void deserialize(std::filesystem::path serialization_path,
                   bool                  check_types = true) {
    derived_type* derived_this = static_cast<derived_type*>(this);

    serialization_path /= "";
    manifest_t manifest_obj =
        read_manifest(serialization_path / "manifest.json");

    if constexpr (HasCustomDeserialize<derived_type>) {
      derived_this->custom_serialize(serialization_path, manifest_obj);
    } else {
      if constexpr (HasDeserializePrologue<derived_type>) {
        derived_this->deserialize_prologue(serialization_path, manifest_obj);
      }

      std::filesystem::path rank_path =
          serialization_path /
          ("rank" + std::to_string(derived_this->comm().rank()));
      std::ifstream   ifs(rank_path, std::ios::binary);
      input_archive_t archive(ifs);

      // This check for when the archive is empty only works with binary
      // archives.
      while (ifs.peek() != EOF) {
        if constexpr (SingleItemTuple<for_all_args>) {
          typename std::tuple_element<0, for_all_args>::type val;
          archive(val);

          derived_this->local_insert(val);
        } else if constexpr (DoubleItemTuple<for_all_args>) {
          typename std::tuple_element<0, for_all_args>::type key;
          typename std::tuple_element<1, for_all_args>::type val;
          archive(key, val);

          derived_this->local_insert(key, val);
        }
      }

      if constexpr (HasDeserializeEpilogue<derived_type>) {
        derived_this->deserialize_epilogue(serialization_path, manifest_obj);
      }
    }

    derived_this->comm().barrier();
  }

  /**
   * @brief Create a manifest of basic information about a serialized container
   * for writing to a file alongside serialized data. Containers can update the
   * manifest through a serialization prologue and epilogue methods.
   */
  manifest_t create_manifest() {
    derived_type* derived_this = static_cast<derived_type*>(this);

    manifest_t manifest_obj{};
    manifest_obj["version"]   = manifest_version;
    manifest_obj["comm_size"] = derived_this->comm().size();

    return manifest_obj;
  }

  /**
   * @brief Write a manifest file that contains basic information about
   * serialized container. Containers can update the manifest before it is
   * written.
   *
   * @param manifest_path Path for manifest file
   * @param manifest_obj Manifest container basic container information
   */
  void write_manifest(const std::filesystem::path& manifest_path,
                      const manifest_t&            manifest_obj) {
    derived_type* derived_this = static_cast<derived_type*>(this);

    // Write manifest from rank 0
    if (derived_this->comm().rank0()) {
      std::ofstream ofs(manifest_path);
      ofs << manifest_obj << std::endl;
    }
  }

  /**
   * @brief Read the manifest file with basic information created when
   * serializing a container. Container class can check additional
   * container-specific info added to the manifest.
   *
   * @param path Path to manifest file
   * @param check_types Boolean indicating whether type IDs are to be checked
   * when reading manifests.
   */
  manifest_t read_manifest(const std::filesystem::path& manifest_path) {
    derived_type* derived_this = static_cast<derived_type*>(this);

    manifest_t manifest_obj;

    if (derived_this->comm().rank0()) {
      std::ifstream      ifs(manifest_path);
      std::ostringstream ss;
      ss << ifs.rdbuf();
      std::string manifest_contents = ss.str();

      manifest_obj = boost::json::parse(manifest_contents).as_object();
    }

    ygm::bcast(manifest_obj, 0, derived_this->comm());

    return manifest_obj;
  }

  /**
   * @brief Assign serialized data files to ranks for reading
   *
   * @tparam
   * @param serialization_path Path to directory of serialized data files
   * @return A vector of rank IDs representing the data files for the local rank
   * to open and deserialize
   */
  std::vector<int> assign_serialized_rank_files(
      const std::filesystem::path& serialization_path) {
    std::vector<int> local_read_rank_ids;

    return local_read_rank_ids;
  }
};
}  // namespace ygm::container::detail
