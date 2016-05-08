#include <GASengine/csv_graph_reader.cuh>
//#include <b40c/graph/builder/market.cuh>
//#include <b40c/graph/builder/random.cuh>
//#include <b40c/graph/builder/rmat.cuh>
//#include <b40c/graph/coo_edge_tuple.cuh>
//#include <GASengine/csr_problem.cuh>
#include <string>
#include <fstream>

using namespace b40c;
using namespace graph;
using namespace std;

int main(int argc, char** argv)
{
	if(argc != 2)
	{
		cout << "Wrong args: csv2mtx filename" << endl;
	}
	char* graph_file = argv[1];
	const bool g_stream_from_host = false;
	const bool g_with_value = true;
	typedef int VertexId;
	typedef int Value;
	typedef int SizeT;
	CsrGraph<VertexId, Value, SizeT> csr_graph(g_stream_from_host);
	csv_graph_reader<g_with_value, VertexId, Value, SizeT> reader(graph_file, true);
	reader.read(csr_graph);
	string outfilename = string(graph_file) + ".mtx";
	ofstream ofs(outfilename.c_str(), ofstream::out);
	ofs << csr_graph.nodes << ' ' << csr_graph.nodes << ' ' << csr_graph.edges << endl;
	for(int i=0; i<csr_graph.nodes; i++)
	{
		VertexId src = i;
		for(int j=csr_graph.row_offsets[i]; j<csr_graph.row_offsets[i+1]; j++)
		{
          VertexId dst = csr_graph.column_indices[j];
          Value v = csr_graph.edge_values[j];
          ofs << src << ' ' << dst << ' ' << v << endl;
		}
	}
	ofs.close();
	return 0;
}
