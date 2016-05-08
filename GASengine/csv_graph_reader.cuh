#pragma once

#include <string>
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>

#include <b40c/graph/csr_graph.cuh>
#include <b40c/graph/builder/utils.cuh>
#include <b40c/graph/coo_edge_tuple.cuh>
using namespace std;
using namespace b40c;
using namespace graph;

template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
class csv_graph_reader{
	typedef CooEdgeTuple<VertexId, Value> EdgeTupleType;
	char *filename;
	bool undirected;

	class CSVRow
	{
	public:
		std::string const& operator[](std::size_t index) const
		{
			return m_data[index];
		}
		std::size_t size() const
		{
			return m_data.size();
		}

		void readNextRow(std::istream& str)
		{
			std::string         line;
			std::getline(str,line);

			std::stringstream   lineStream(line);
			std::string         cell;

			m_data.clear();
			while(std::getline(lineStream,cell,','))
			{
				m_data.push_back(cell);
			}
		}
	private:
		std::vector<std::string>    m_data;
	};

//	istream& operator>>(istream& str, CSVRow& data)
//	{
//	    data.readNextRow(str);
//	    return str;
//	}


public:
	csv_graph_reader(char* filename,  bool undirected) :
		filename(filename), undirected(undirected){}

	void read(CsrGraph<VertexId, Value, SizeT>& csr_graph)
	{
		vector<EdgeTupleType> coos;
		ifstream file(filename);
		vector<VertexId> ids;
        ids.reserve(100000000);
        vector<Value> values;
        values.reserve(100000000);
        coos.reserve(100000000);

		CSVRow row;
		int c = 0;
		long long edge_count = 0;
		while(file.good())
		{
			row.readNextRow(file);
			VertexId src = atoi(row[0].c_str());
			ids.push_back(src);
//			std::cout << src << endl;

			for(int i=1; i<row.size(); i++)
			{
				int pos = row[i].find(":");
				VertexId dst = atoi((row[i].substr(0, pos)).c_str());
				ids.push_back(dst);
				Value edge_value = atoi((row[i].substr(pos+1)).c_str());
				values[edge_count++] = edge_value;
				EdgeTupleType e(src, dst, edge_value);
				coos.push_back(e);
//				std::cout << src << " " << dst << " " << w << endl;
			}

			c++;
		}


		thrust::device_vector<VertexId> d_ids(ids.begin(), ids.end());
		thrust::sort(d_ids.begin(), d_ids.end());
		int nodes = thrust::unique(thrust::device, d_ids.begin(), d_ids.end()) - d_ids.begin();
		thrust::copy(d_ids.begin(), d_ids.end(), ids.begin());

		map<VertexId,VertexId> mapping;

//		unordered_map<VertexId, VertexId> mapping;
		for(int i=0; i<nodes; i++)
		{
			mapping[ids[i]] = i;
		}

//		printf("mapping\n");
//		for(int i=0; i < 100; i++)
//		{
//			printf("%d --> %d\n", ids[i], mapping[ids[i]]);
//		}

		for(int i =0; i<coos.size(); i++)
		{
			coos[i].row = mapping[coos[i].row];
			coos[i].col = mapping[coos[i].col];
		}

		if(undirected)
		{
			for(int i=0; i<edge_count; i++)
			{
				EdgeTupleType e(coos[i].col, coos[i].row, coos[i].val);
				coos.push_back(e);
			}
			edge_count *= 2;
		}



		csr_graph.template FromCoo<true>(&coos[0], nodes, edge_count, false);

//		csr_graph.DisplayGraph();

//		printf("nodes=%d, edge_count=%d\n", nodes, coos.size());
//		for(int i=0; i < 100; i++)
//		{
//			printf("%d %d %d\n", coos[i].row, coos[i].col, coos[i].val);
//		}
	}

	void read2(CsrGraph<VertexId, Value, SizeT>& csr_graph)
		{
			vector<EdgeTupleType> coos;
			ifstream file(filename);
			vector<VertexId> ids;
	        ids.reserve(100000000);
	        vector<Value> values;
	        values.reserve(100000000);
	        coos.reserve(100000000);

			CSVRow row;
			int c = 0;
			long long edge_count = 0;
			while(file.good())
			{
				row.readNextRow(file);
				VertexId src = atoi(row[0].c_str());
				ids.push_back(src);
				VertexId dst = atoi(row[1].c_str());
				ids.push_back(dst);
				Value edge_value = atoi(row[2].c_str());
				values[edge_count++] = edge_value;
				EdgeTupleType e(src, dst, edge_value);
				coos.push_back(e);
//	//			std::cout << src << endl;
//
//				for(int i=1; i<row.size(); i++)
//				{
//					int pos = row[i].find(":");
//					VertexId dst = atoi((row[i].substr(0, pos)).c_str());
//					ids.push_back(dst);
//					Value edge_value = atoi((row[i].substr(pos+1)).c_str());
//					values[edge_count++] = edge_value;
//					EdgeTupleType e(src, dst, edge_value);
//					coos.push_back(e);
//	//				std::cout << src << " " << dst << " " << w << endl;
//				}

				c++;
			}


			thrust::device_vector<VertexId> d_ids(ids.begin(), ids.end());
			thrust::sort(d_ids.begin(), d_ids.end());
			int nodes = thrust::unique(thrust::device, d_ids.begin(), d_ids.end()) - d_ids.begin();
			thrust::copy(d_ids.begin(), d_ids.end(), ids.begin());

			map<VertexId,VertexId> mapping;

	//		unordered_map<VertexId, VertexId> mapping;
			for(int i=0; i<nodes; i++)
			{
				mapping[ids[i]] = i;
			}

	//		printf("mapping\n");
	//		for(int i=0; i < 100; i++)
	//		{
	//			printf("%d --> %d\n", ids[i], mapping[ids[i]]);
	//		}

			for(int i =0; i<coos.size(); i++)
			{
				coos[i].row = mapping[coos[i].row];
				coos[i].col = mapping[coos[i].col];
			}

			if(undirected)
			{
				for(int i=0; i<edge_count; i++)
				{
					EdgeTupleType e(coos[i].col, coos[i].row, coos[i].val);
					coos.push_back(e);
				}
				edge_count *= 2;
			}



			csr_graph.template FromCoo<true>(&coos[0], nodes, edge_count, false);

	//		csr_graph.DisplayGraph();

	//		printf("nodes=%d, edge_count=%d\n", nodes, coos.size());
	//		for(int i=0; i < 100; i++)
	//		{
	//			printf("%d %d %d\n", coos[i].row, coos[i].col, coos[i].val);
	//		}
		}
};
