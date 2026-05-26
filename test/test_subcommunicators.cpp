// Copyright 2019-2026 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#undef NDEBUG
#include <ygm/comm.hpp>
#include <ygm/container/bag.hpp>

int main() {
  YGM_ASSERT_MPI(MPI_Init(nullptr, nullptr));

  ygm::comm world(MPI_COMM_WORLD);

  //
  // Test initializing YGM communicator from subcommunicator
  {
    MPI_Comm self_mpi_comm;
    MPI_Comm_split(MPI_COMM_WORLD, world.rank(), 0, &self_mpi_comm);

    ygm::comm self_comm(self_mpi_comm);

    YGM_ASSERT_RELEASE(self_comm.size() == 1);
    YGM_ASSERT_RELEASE(self_comm.rank() == 0);

    bool flag     = false;
    auto flag_ptr = self_comm.make_ygm_ptr(flag);

    self_comm.async(0, [](auto flag_ptr) { *flag_ptr = true; }, flag_ptr);

    self_comm.barrier();
    YGM_ASSERT_RELEASE(flag == true);
  }

  //
  // Test container creation on subcommunicators
  {
    {
      MPI_Comm self_mpi_comm;
      MPI_Comm_split(MPI_COMM_WORLD, world.rank(), 0, &self_mpi_comm);

      ygm::comm self_comm(self_mpi_comm);

      for (int i = 0; i < world.rank(); ++i) {
        ygm::container::bag<int> b(self_comm);
      }
    }

    // Return to world
    ygm::container::bag<int> b(world);
  }

  return 0;
}
