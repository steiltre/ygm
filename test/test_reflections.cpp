#include <ygm/detail/reflections.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <iostream>
#include <memory>
#include <sstream>

struct X {
  char   a = 'd';
  int    b = 1;
  double c = 3.14;
};

struct Y {
  Y(int &b, double &c) : b(b), c(c) {};

  int    &b;
  double &c;
};

struct Z {
  int a = 1;
  int b = 2;

  // Only serialize a, not b
  template <typename Archive>
  void serialize(Archive &ar) {
    ar(a);
  }
};

template <typename T, typename Archive>
concept has_serialize = requires(T t, Archive a) { t.serialize(a); };

template <typename T, typename Archive>
concept has_external_serialize = requires(T t, Archive a) { serialize(a, t); };

template <typename T, typename Archive>
concept has_save_load = requires(T t, Archive a) {
  t.save(a);
  t.load(a);
};

template <typename T, typename Archive>
concept has_external_save_load = requires(T t, Archive a) {
  save(a, t);
  load(a, t);
};

// Specific to JSONOutputArchive
template <typename T, typename Archive>
concept archive_has_save_value = requires(T t, Archive a) { a.saveValue(t); };

// Specific to JSONInputArchive
template <typename T, typename Archive>
concept archive_has_load_value = requires(T t, Archive a) { a.loadValue(t); };

// Specific to BinaryOutputArchive
template <typename T, typename Archive>
concept archive_has_save_binary =
    requires(T t, Archive a, std::streamsize s) { a.saveBinary(&t, s); };

// Specific to BinaryInputArchive
template <typename T, typename Archive>
concept archive_has_load_binary =
    requires(T t, Archive a, std::streamsize s) { a.loadBinary(&t, s); };

template <typename T, typename Archive>
concept serializable =
    cereal::traits::is_output_serializable<T, Archive>::value ||
    cereal::traits::is_input_serializable<T, Archive>::value;

template <typename T>
void serialize_impl(std::stringstream &, T &&);

template <typename T>
void manual_serialize_impl(std::stringstream &ss, T &&t) {
  cereal::BinaryOutputArchive oarchive(ss);

  constexpr auto t_layout = get_layout<std::remove_reference_t<T>>();
  template for (constexpr auto md : t_layout) {
    using member_t = [:std::meta::remove_reference(
                           std::meta::type_of(md.info)):];

    std::byte *addr = reinterpret_cast<std::byte *>(&t) + md.offset;

    if constexpr (not std::meta::is_reference_type(
                      std::meta::type_of(md.info))) {
      member_t *my_member_ptr = reinterpret_cast<member_t *>(addr);
      serialize_impl(ss, *my_member_ptr);
    } else {
      member_t **my_member_ptr = reinterpret_cast<member_t **>(addr);
      serialize_impl(ss, **my_member_ptr);
    }
  }
}

template <typename T>
void auto_serialize_impl(std::stringstream &ss, T &&t) {
  cereal::BinaryOutputArchive oarchive(ss);

  oarchive(t);
}

template <typename T>
void serialize_impl(std::stringstream &ss, T &&t) {
  cereal::BinaryOutputArchive oarchive(ss);

  if constexpr (serializable<std::remove_reference_t<T>, decltype(oarchive)>) {
    auto_serialize_impl(ss, t);
  } else {
    manual_serialize_impl(ss, t);
  }
}

template <typename T>
void deserialize_impl(std::stringstream &, T *);

template <typename T>
void manual_deserialize_impl(std::stringstream &ss, T *p_t) {
  cereal::BinaryInputArchive iarchive(ss);

  constexpr auto t_layout = get_layout<std::remove_reference_t<T>>();
  template for (constexpr auto md : t_layout) {
    using member_t = [:std::meta::remove_reference(
                           std::meta::type_of(md.info)):];

    std::byte *addr = reinterpret_cast<std::byte *>(p_t) + md.offset;

    if constexpr (not std::meta::is_reference_type(
                      std::meta::type_of(md.info))) {
      member_t *my_member_ptr = reinterpret_cast<member_t *>(addr);
      deserialize_impl(ss, my_member_ptr);
    } else {
      member_t **my_member_ptr_ptr = reinterpret_cast<member_t **>(addr);

      // Need to allocate space to hold deserialized object
      // TODO: currently leaks this memory
      *my_member_ptr_ptr = (member_t *)malloc(sizeof(member_t));

      deserialize_impl(ss, *my_member_ptr_ptr);
    }
  }
}

template <typename T>
void auto_deserialize_impl(std::stringstream &ss, T *p_t) {
  cereal::BinaryInputArchive iarchive(ss);

  iarchive(*p_t);
}

template <typename T>
void deserialize_impl(std::stringstream &ss, T *p_t) {
  if constexpr (serializable<std::remove_reference_t<T>,
                             cereal::BinaryInputArchive>) {
    auto_deserialize_impl(ss, p_t);
  } else {
    manual_deserialize_impl(ss, p_t);
  }
}

template <typename T>
std::stringstream serialize(T &&t) {
  std::stringstream ss;

  serialize_impl(ss, t);

  return ss;
}

template <typename T>
void deserialize(std::stringstream &ss, T *p_t) {
  deserialize_impl(ss, p_t);
}

int main() {
  // Test serializing a vector
  {
    {
      constexpr size_t num_refs = count_reference_members<X>();
      std::cout << num_refs << std::endl;
    }

    {
      constexpr size_t num_refs = count_reference_members<Y>();
      std::cout << num_refs << std::endl;
    }

    std::vector<int> c_vec({1, 4, 9});

    auto ss = serialize(c_vec);

    std::vector<int>  vec;
    std::vector<int> *p_vec = &vec;
    deserialize<std::vector<int>>(ss, &vec);
    std::cout << "Deserialized vec: [";
    for (const auto &item : *p_vec) {
      std::cout << item << "\t";
    }
    std::cout << "]\n";
  }

  // Test serializing a class without a serialize function
  {
    X my_x;

    auto ss = serialize(my_x);

    X  new_x;
    X *p_x = &new_x;
    deserialize(ss, p_x);

    std::cout << "Deserialized X: " << p_x->a << "\t" << p_x->b << "\t"
              << p_x->c << std::endl;
  }

  // Test serializing a class that contains references
  {
    int               my_int = 14;
    std::stringstream ss;

    {
      double my_double = 2.7;
      Y      my_y(my_int, my_double);
      ss = serialize(my_y);
    }

    {
      Y *my_new_y_ptr = (Y *)malloc(sizeof(Y));
      deserialize(ss, my_new_y_ptr);
      std::cout << "Deserialized Y: " << my_new_y_ptr->b << "\t"
                << my_new_y_ptr->c << std::endl;
    }
  }

  // Test serializing a class that has a built-in serialize function
  // Ends up with unitialized copy of variables not serialized
  {
    std::stringstream ss;

    {
      Z my_z;
      my_z.a = 3;
      my_z.b = 4;

      ss = serialize(my_z);
    }
    {
      Z *my_new_z_ptr = (Z *)malloc(sizeof(Z));
      deserialize(ss, my_new_z_ptr);

      std::cout << "Deserialized Z: " << my_new_z_ptr->a << "\t"
                << my_new_z_ptr->b << std::endl;
    }
  }

  // Deserializing lambda that captures another lambda
  {
    int              a = 4;
    double           b = 5.5;
    std::vector<int> c_vec({1, 4, 9, 16});

    auto my_first_func = [a, &b, &c_vec]() {
      std::cout << a << "\t" << b << std::endl;
      std::cout << "[" << c_vec[0];
      for (size_t i = 1; i < c_vec.size(); ++i) {
        std::cout << "\t" << c_vec[i];
      }
      std::cout << "]" << std::endl;
    };

    auto my_second_func = [my_first_func]() {
      std::cout << "In second func" << std::endl;
      my_first_func();
    };

    auto ss = serialize(my_first_func);

    decltype(my_second_func) *func_ptr =
        (decltype(my_second_func) *)malloc(sizeof(my_second_func));

    deserialize(ss, func_ptr);

    (*func_ptr)();
  }

  // Playing with contained_refs_wrapper
  {
    // contained_refs_wrapper<X> my_x_ref_wrapper;
    // my_complete m;
    // complete<double> m;

    my_type  m;
    my_type2 m2;

    my_templated_type<int> m3;

    // my_complete_type m4;

    constexpr std::meta::info my_info  = ^^my_type;
    constexpr std::meta::info my_info2 = ^^m2;

    incomplete<int> m4;

    // auto my_tuple = named_tuple_t<std::pair<int, "x">>{.x = 1};
    [[maybe_unused]] auto r =
        named_tuple_t<pair<int, "x">, pair<double, "y">>{.x = 1, .y = 2.0};
  }

  // my_wrapped_type<int> w;
  auto w = my_wrapped_type<int>{};

  return 0;
}
