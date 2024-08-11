
#include <concepts>
#include <iostream>

template <typename T> struct base {
  // void func() { std::cout << "In base func" << std::endl; }
};

template <std::integral T> struct base<T> {
  void func() { std::cout << "In specialized base func" << std::endl; }
};

template <typename T> struct derived1 : public base<T> {
  void derived_func() { std::cout << "In derived_func" << std::endl; }
};

template <typename T> struct derived2 : public base<T> {
  void derived_func() { std::cout << "In derived_func" << std::endl; }

  using base<int>::func;
  void func(int a) { std::cout << "Shadowing base func" << std::endl; }
};
