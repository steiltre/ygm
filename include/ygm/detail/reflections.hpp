// Copyright 2019-2025 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <meta>
#include <utility>
#include <vector>

// start 'expand' definition
namespace __impl {
template <auto... vals>
struct replicator_type {
  template <typename F>
  constexpr void operator>>(F body) const {
    (body.template operator()<vals>(), ...);
  }
};

template <auto... vals>
replicator_type<vals...> replicator = {};
}  // namespace __impl

template <typename R>
consteval auto expand(R range) {
  std::vector<std::meta::info> args;
  for (auto r : range) {
    args.push_back(std::meta::reflect_constant(r));
  }
  return substitute(^^__impl::replicator, args);
}
// end 'expand' definition

struct member_descriptor {
  std::size_t     offset;
  std::size_t     size;
  std::meta::info info;
  bool            operator==(member_descriptor const &) const = default;
};

// returns std::array<member_descriptor, N>
template <typename S>
consteval auto get_layout() {
  constexpr auto   ctx = std::meta::access_context::unchecked();
  constexpr size_t N   = [&ctx]() consteval {
    return nonstatic_data_members_of(^^S, ctx).size();
  }();

  std::array<member_descriptor, N> layout;
  [:expand(nonstatic_data_members_of(^^S, ctx)):] >> [&,
                                                      i = 0]<auto e>() mutable {
    layout[i].offset = offset_of(e).bytes;
    layout[i].size   = size_of(e);
    layout[i].info   = e;
    ++i;
  };
  return layout;
}

template <typename S>
consteval size_t count_reference_members() {
  constexpr auto ctx = std::meta::access_context::unchecked();

  size_t result = 0;
  [:expand(std::meta::nonstatic_data_members_of(
        ^^S, ctx)):] >> [&]<auto e>() mutable {
    if constexpr (std::meta::is_reference_type(std::meta::type_of(e))) {
      ++result;
    }
  };

  return result;
}

template <typename S>
struct contained_refs_wrapper_impl;

template <typename S>
consteval auto get_contained_refs() {
  constexpr auto ctx = std::meta::access_context::unchecked();
  // std::vector<std::meta::info> class_members =
  // std::meta::nonstatic_data_members_of(^^S, ctx);
  std::vector<std::meta::info> class_ref_members = {};

  constexpr size_t N = count_reference_members<S>();
  // std::array<std::meta::info, N> class_ref_members;
  //  for (std::meta::info member : class_members) {
  //   if constexpr (std::meta::is_reference_type(std::meta::type_of(member))) {
  //  class_ref_members.push_back(member);
  //  }
  // }

  return class_ref_members;
}

template <typename S>
consteval auto make_contained_refs_wrapper_struct() {
  constexpr auto               ctx = std::meta::access_context::unchecked();
  std::vector<std::meta::info> class_members =
      std::meta::nonstatic_data_members_of(^^S, ctx);
  std::vector<std::meta::info> class_ref_members = {};

  for (std::meta::info member : class_members) {
    // if constexpr (std::meta::is_reference_type(std::meta::type_of(member))) {
    class_ref_members.push_back(member);
    //}
  }

  return std::meta::define_aggregate(^^contained_refs_wrapper_impl<S>,
                                     class_ref_members);
}

// template <typename S>
// using contained_refs_wrapper = [:make_contained_refs_wrapper_struct<S>():];

template <typename T>
struct incomplete;

consteval {
  std::meta::define_aggregate(^^incomplete<int>, {
                                                 });
}

template <typename T>
consteval auto complete_type() {
  return std::meta::define_aggregate(^^incomplete<T>, {
                                                      });
}

consteval auto return_type() { return ^^int; }

template <typename S>
consteval auto return_templated_type() {
  return ^^S;
}

using my_type = [:return_type():];

using my_type2 = [:return_templated_type<int>():];

template <typename S>
using my_templated_type = [:return_templated_type<S>():];

// constexpr auto r        = complete_type<int>();
// using my_completed_type = [:r:];
//  using my_templated_type = [:complete_type<S>():];

/*
consteval {
  using my_complete_type = [:std::meta::define_aggregate(^^incomplete<float>,
                                                         {
                                                         }):];
}
*/

// template <typename
// T> using complete =
// [:complete_type<T>():];

// using my_complete =
// [:std::meta::define_aggregate(^^incomplete,
//{
//^^int, ^^double}):];

template <typename T>
struct my_wrapper_class {
  struct wrapped_type;

  consteval {
    constexpr auto ctx = std::meta::access_context::unchecked();
    // constexpr auto types = std::meta::nonstatic_data_members_of(^^T);
    constexpr auto ref_types = get_contained_refs<T>();
    //    std::meta::define_aggregate(^^wrapped_type, {
    //    ref_types});
    /*
    std::meta::define_aggregate(
        ^^wrapped_type,
        {
            std::meta::data_member_spec(^^int, {
                                                   .name = "x"})});
                                                   */
    std::meta::define_aggregate(^^wrapped_type,
                                {
                                    std::meta::data_member_spec(^^int,
                                                                {
                                                                }),
                                    std::meta::data_member_spec(^^double, {
                                                                          })});
  }
};

template <typename T>
using my_wrapped_type = typename my_wrapper_class<T>::wrapped_type;

template <size_t N>
struct fixed_string {
  char data[N];

  constexpr fixed_string(char const (&s)[N]) { std::copy(s, s + N, data); }

  constexpr auto view() const -> std::string_view { return data; }
};

template <class T, fixed_string Name>
struct pair {
  static constexpr auto name() -> std::string_view { return Name.view(); }
  using type = T;
};

template <class... Tags>
struct named_tuple {
  struct type;

  consteval {
    std::vector<std::meta::info> nsdms{std::meta::data_member_spec(
        dealias(^^typename Tags::type), {.name = Tags::name()})...};
    define_aggregate(^^type, nsdms);
  }
};

template <class... Tags>
using named_tuple_t = typename named_tuple<Tags...>::type;
