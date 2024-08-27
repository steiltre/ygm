
#include <concepts>
#include <iostream>
#include <tuple>

template <typename T>
concept SingleItemTuple = requires(T v) {
  requires std::tuple_size<T>::value == 1;
};

template <typename T>
concept DoubleItemTuple = requires(T v) {
  requires std::tuple_size<T>::value == 2;
};

template <typename T, typename derived_type>
struct base {
  static_assert(sizeof(for_all_args) != sizeof(for_all_args),
                "Unsupported for_all_args");
  // void func() { std::cout << "In base func" << std::endl; }
};

template <SingleItemTuple T, typename derived_type>
struct base<T, derived_type> {
  void func() { std::cout << "In specialized base func" << std::endl; }
};

template <DoubleItemTuple T, typename derived_type>
struct base<T, derived_type> {
  void func() { std::cout << "In DoubleItemTuple func" << std::endl; }
};

template <typename T>
struct derived1 : public base<T, derived1<T>> {
  using base<T, derived1<T>>::func;
  void derived_func() { std::cout << "In derived_func" << std::endl; }
};

template <typename T>
struct derived2 : public base<T, derived2<T>> {
  void derived_func() { std::cout << "In derived_func" << std::endl; }

  using base<T, derived2<T>>::func;
  void func(int a) { std::cout << "Shadowing base func" << std::endl; }
};
