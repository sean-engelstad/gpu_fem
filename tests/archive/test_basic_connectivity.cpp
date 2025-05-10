#include "../basic_connectivity.h"
#include "../mesh_reader.h"

int main(int argc, char *argv[]) {
  using T = double;

  std::string mesh_name = "mesh_ONERAM6_inv_ffd.su2";
  SU2MeshReader<T> mesh_reader(mesh_name);

  BasicConnectivity3D *basic_conn = mesh_reader.create_connectivity();

  std::cout << "num elements: " << basic_conn->get_num_elements() << "\n";
  std::cout << "num edges   : " << basic_conn->get_num_edges() << "\n";
  std::cout << "num verts   : " << basic_conn->get_num_vertices() << "\n";
  std::cout << "num faces   : " << basic_conn->get_num_faces() << "\n";

  return 0;
}
