#pragma once
enum SrcVertex
{
  SINGLE, ALL
};

enum GatherEdges
{
  NO_GATHER_EDGES, GATHER_IN_EDGES, GATHER_OUT_EDGES, GATHER_ALL_EDGES
};

enum ExpandEdges
{
  NO_EXPAND_EDGES, EXPAND_IN_EDGES, EXPAND_OUT_EDGES, EXPAND_ALL_EDGES
};

enum ApplyVertices
{
  NO_APPLY_VERTICES, APPLY_ALL, APPLY_FRONTIER
};

enum PostApplyVertices
{
  NO_POST_APPLY_VERTICES, POST_APPLY_ALL, POST_APPLY_FRONTIER
};

namespace GASengine
{
  /**
   * Enumeration of global frontier queue configurations
   */
  enum FrontierType
  {
    VERTEX_FRONTIERS,      // O(n) ping-pong global vertex frontiers
    EDGE_FRONTIERS,        // O(m) ping-pong global edge frontiers
    MIXED_FRONTIERS,       // O(n) global vertex frontier, O(m) global edge frontier
    MULTI_GPU_FRONTIERS,         // O(MULTI_GPU_VERTEX_FRONTIER_SCALE * n) global vertex frontier, O(m) global edge frontier, O(m) global sorted, filtered edge frontier
    MULTI_GPU_VERTEX_FRONTIER_SCALE = 2,
  };
}
