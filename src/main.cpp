#include "config.h"
#include "mpi.h"
#include "communicator2D.h"
#include "domain2D.h"
#include "cellList2D.h"
#include "rand.h"
#include "run2D.h"
#include "particle2D.h"

using namespace std;

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int my_rank, tot_proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &tot_proc);

  int arg_size = 8;
  if ((argc - 1) % arg_size != 0) {
    std::cout << "Error, argc = " << argc << std::endl;
    exit(1);
  }
  int n_group = (argc - 1) / arg_size;
  int cores_per_group = tot_proc / n_group;
  MPI_Group group, gl_group, root_group;
  MPI_Comm group_comm, root_comm;

  int* ranks = new int[cores_per_group];  // rank of processer belong to the same group
  int* root_ranks = new int[n_group];     // rank of root processer of each group
  int my_group = my_rank / cores_per_group;
  for (int i = 0; i < cores_per_group; i++) {
    ranks[i] = i + my_group * cores_per_group;
  }
  for (int i = 0; i < n_group; i++) {
    root_ranks[i] = i * cores_per_group;
  }
  MPI_Comm_group(MPI_COMM_WORLD, &gl_group);
  MPI_Group_incl(gl_group, cores_per_group, ranks, &group);
  MPI_Group_incl(gl_group, n_group, root_ranks, &root_group);
  MPI_Comm_create(MPI_COMM_WORLD, group, &group_comm);
  MPI_Comm_create(MPI_COMM_WORLD, root_group, &root_comm);

  int idx_beg = my_group * arg_size;
  double Lx = atof(argv[1 + idx_beg]);
  double Ly = atof(argv[2 + idx_beg]);
  double eta = atof(argv[3 + idx_beg]);
  double rho0 = atof(argv[4 + idx_beg]);
  double v0 = atof(argv[5 + idx_beg]);
  int n_step = atof(argv[6 + idx_beg]);
  unsigned long long seed = atoi(argv[7 + idx_beg]);
  std::string ini_mode = argv[8 + idx_beg];

  Vec_2<double> gl_l(Lx, Ly);
  double eps = 0;
  int seed2 = 1200;
  int snap_dt = 10000;
  int field_dt = 10000;
  int field_dx = 8;

  int gl_par_num = int(gl_l.x * gl_l.y * rho0);

  run(gl_par_num, gl_l, eta, eps, v0, n_step, ini_mode,
      seed, seed2, snap_dt, field_dt, field_dx, group_comm, root_comm);
  MPI_Finalize();
}
