import java.util.*;

public class TriestImpr implements DataStreamAlgo {
		/*
		 * Constructor.
		 * The parameter samsize denotes the size of the sample, i.e., the number of
		 * edges_to_triangles that the algorithm can store.
		 */

	// sample size
	public int m;
	// num triangles
	public double d = 0;
	// time
	public int t = 0;
	// hashmap of each edge to the number of triangles it belongs to
	public HashMap<HashSet<Integer>,Integer> edges_to_triangles = new HashMap<HashSet<Integer>,Integer>();
	// hashmap of each node to its neighbors
	public HashMap<Integer,HashSet<Integer>> nodes_to_neighbors = new HashMap<Integer,HashSet<Integer>>();

	public TriestImpr(int samsize) {
			m = samsize;
			//System.out.println("Sample Size: "+samsize);
	}

	public void handleEdge(Edge edge) {

				//System.out.println("Edge: "+edge);
				HashSet<Integer> new_edge = new HashSet<Integer>();
				int u = edge.u;
				int v = edge.v;
				new_edge.add(u);
				new_edge.add(v);

				// If the edge is new and a valid edge...
        if (!edges_to_triangles.containsKey(new_edge) && u!=v) {

            // Increment time
            t++;

            // If t<=m...
            if (t <= m) {

                // Update the global hashmaps
                updateNeighbors(u,v);
                updateEdges(new_edge,u,v);

            }
						else
						{
								// Increment D by g*n
								int g = calcG(new_edge,u,v);
								double n = calcN();
								double gn = g*n;

								d = d+gn;

								// Do coinflip compared with bias of m/t
                double coinflip = Math.random();
                double bias = (double)m/t;

                // If coinflip<bias...
								if (coinflip < bias) {

										// Get random edge to remove
										List<HashSet<Integer>> keys = new ArrayList<HashSet<Integer>>(edges_to_triangles.keySet());
										Random r = new Random();
										HashSet<Integer> edge_to_remove = keys.get(r.nextInt(keys.size()));

										// Get nodes to remove from neighbors map
										int u_to_remove = -1;
										int v_to_remove = -1;
										int count = 0;
										for (int node: edge_to_remove) {
											if (count == 0) {
												u_to_remove = node;
												count++;
											}
											else v_to_remove = node;
										}

										// Remove edge from edges_to_triangles
										edges_to_triangles.remove(edge_to_remove);

										// Make removed edge's nodes no longer neighbors in neighbors map
										HashSet<Integer> removed_u_neighbors = nodes_to_neighbors.get(u_to_remove);
										HashSet<Integer> removed_v_neighbors = nodes_to_neighbors.get(v_to_remove);
										removed_u_neighbors.remove(v_to_remove);
										removed_v_neighbors.remove(u_to_remove);
										nodes_to_neighbors.put(u_to_remove, removed_u_neighbors);
										nodes_to_neighbors.put(v_to_remove, removed_v_neighbors);

										// Update the global hashmaps
										updateNeighbors(u,v);
										updateEdges(new_edge,u,v);

								}
						}
				}
		}

	public int getEstimate() {
			return (int)d;
	}

	public void updateNeighbors(int u, int v) {

		// Instantiate temporary neighbors sets (values)
    HashSet<Integer> u_neighbors = new HashSet<Integer>();
    HashSet<Integer> v_neighbors = new HashSet<Integer>();

    // New neighbors sets = old neighbors sets if they exist
    if (nodes_to_neighbors.containsKey(u)) u_neighbors = nodes_to_neighbors.get(u);
    if (nodes_to_neighbors.containsKey(v)) v_neighbors = nodes_to_neighbors.get(v);

    // Include each node in the other's neighbors
    u_neighbors.add(v);
    v_neighbors.add(u);

    // Put new neighbors maps back in nodes_to_neighbors
    nodes_to_neighbors.put(u, u_neighbors);
    nodes_to_neighbors.put(v, v_neighbors);
	}

	public void updateEdges(HashSet<Integer> edge, int u, int v) {

		// Instantiate new ArrayList to hold neighbor nodes
    ArrayList<Integer> nodes_to_check = new ArrayList<Integer>();
    int num_triangles = 0;

    // Add u's neighbor nodes to temp ArrayList
    for (int node: nodes_to_neighbors.get(u)) {
      nodes_to_check.add(node);
    }
    // If v has same neighbor node(s), we can increase num triangles
    for (int node: nodes_to_neighbors.get(v)) {
      if (nodes_to_check.contains(node)) num_triangles++;
    }

    // Increment D
    d += num_triangles;

    // Put new edge's triangle values in edges_to_triangles
    edges_to_triangles.put(edge, num_triangles);
	}

	public int calcG(HashSet<Integer> edge, int u, int v) {

		// Instantiate new ArrayList to hold neighbor nodes
		ArrayList<Integer> nodes_to_check = new ArrayList<Integer>();
		int num_triangles = 0;

		// If u and v are already in the neighbors hashmap...
		if(nodes_to_neighbors.containsKey(u) && nodes_to_neighbors.containsKey(v)) {
			// Add u's neighbor nodes to temp ArrayList
			for (int node: nodes_to_neighbors.get(u)) {
				nodes_to_check.add(node);
			}
			// If v has same neighbor node(s), we can increase num triangles
			for (int node: nodes_to_neighbors.get(v)) {
				if (nodes_to_check.contains(node)) num_triangles++;
			}
		}

		return num_triangles;

	}

	public double calcN() {
		// Break equation in two due to java memory constraints
		double left = (t-1)/m;
		double right = (t-2)/(m-1);
		double n = left*right;
		return n;
	}
}
