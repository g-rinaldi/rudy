/**
 * @file rudy.cpp
 *
 * @brief A command-line graph generator using the Stanford GraphBase
 * (SGB) library.
 *
 * @author Giovanni Rinaldi
 *
 * Description: Rudy is a tool for generating various types of graphs
 * and performing operations on them. It parses command-line arguments
 * using Reverse Polish Notation (RPN), where graph generators act as
 * operands and graph modification/combination functions act as
 * operators.
 *
 * Features: 
 * - Graph Generation: Creates standard graphs like cliques, circuits,
 *   grids (planar, toroidal, N-dimensional), leap graphs (chess-like
 *   moves on boards), simplex graphs, random graphs (Erdos-Renyi
 *   style), and planar graphs with specified density.
 * - Graph Operations: Supports unary operations (complement, line
 *   graph, random weight assignment, scalar multiplication/addition
 *   to weights) and binary operations (union, cartesian product,
 *   join).
 * - RPN Stack: Uses a stack to manage intermediate graphs during RPN
 *   evaluation.
 * - Special Generators: Includes standalone options (-spinglass*) to
 *   directly generate and output specific spin glass lattice graphs
 *   (2D/3D, +/-J or Gaussian weights) without using the RPN stack.
 * - Output: Prints the final resulting graph to standard output in a
 *   simple text format: number of nodes, number of edges, followed by
 *   an edge list (u, v, weight), using 1-based vertex indexing.
 * - SGB Integration: Leverages the Stanford GraphBase library for
 *   core graph data structures and generation algorithms.
 *
 * Usage:
 *   rudy <graph_expression>
 *   (Run 'rudy' with no arguments for detailed help on the RPN syntax
 *   and options).
 *
 * Example:
 *   rudy -grid_2D 5 5 -random 1 10 123 -clique 25 +
 *
 *   (Generates a 5x5 grid, assigns random weights [1,10], generates a
 *    K_25, takes the union (requires same size, so this example might
 *    error unless grid is 25 nodes).)
 *
 * Example:
 *   rudy -grid_2D 3 3 -circuit 9 +
 *
 *   (Generates 3x3 grid (9 nodes), generates C_9 (9 nodes), takes
 *   their union).
 *
 * Example:
 *   rudy -spinglass2pm 10 10 50 42
 *
 *   (Directly generates a 10x10 toroidal grid with 50% +/-1 bonds,
 *   seed 42).
 */

#include <cstring> // For strcmp
#include <iostream> // For cout, cerr, endl
#include <iomanip> // For setprecision()
#include <cstdlib> // For atoi, atof, exit, EXIT_SUCCESS, EXIT_FAILURE
#include <vector>

// Include headers from the Stanford GraphBase (SGB) library
#include "gb_graph.h" /* we use the gb_graph data structures */
#include "gb_basic.h" /* we use the basic graph operations */
#include "gb_flip.h"  /* we use the random number generation */
#include "gb_rand.h"  /* we use the random graphs generation */

// Error codes for exception handling
enum Err {
  OK,             // Allright
  NOT_FOUND,      // Element not found
  ERROR,          // Generic or unknown error / unknown command option
  EMPTY_GRAPH,    // An operation requires a non-empty graph operand
  ONE_OP,         // An operation requires exactly one graph operand
                  //   on the stack
  TWO_OP,         // An operation requires exactly two graph operands on
                  //   the stack
  TOO_MANY,       // Stack overflow: trying to push onto a full stack
  DIFF_SIZE,      // Binary operation requires operands (graphs) of the
                  //   same size (node count)
  ILL_PAR,        // Illegal parameter value provided for an operation
  ILL_NUM,        // Incorrect number of parameters provided for an operation
  SIMP_GR,        // Operation requires a <simple_graph> input (e.g.,
                  //   spinglass generators)
  NOT_EMPTY,      // Stack should be empty at the end, but isn't (usually
                  //   means only one graph result expected)
  NO_UTILITY,     // Could not find an unused SGB utility field ('Z' type)
                  //   for output indexing
  PANIC           // SGB function returned NULL unexpectedly, indicating
                  //   an internal error
};

// Declare external C functions from the SGB library
extern "C" {
  // Graph combination/modification functions
  // Graph union
  Graph* gunion(Graph* g, Graph* gg, long multi, long directed);
  // Graph products (e.g., cartesian)
  Graph* product(Graph* g, Graph* gg, long type, long directed);
  // Induced subgraph
  Graph* induced(Graph* g, char* description, long self, long multi,
		 long directed);
  // Graph complement
  Graph* complement(Graph* g, long copy, long self, long directed);
  // Line graph
  Graph* lines(Graph* g, long directed);

  // Graph generation functions
  // Grid/board graphs
  Graph* board(long n1, long n2, long n3, long n4, long piece, long wrap,
	       long directed);
  // Random graphs (Erdos-Renyi style)
  Graph* random_graph(unsigned long n, unsigned long m, long multi,
		      long self, long directed, long* dist_from,
		      long* dist_to, long min_len, long max_len, long seed);
  // Simplex graphs
  Graph* simplex(unsigned long n, long n0, long n1, long n2, long n3,
		 long n4, long directed);

  // Utility functions
  // Free memory associated with an SGB graph
  void gb_recycle(Graph* g);
  // Assign random edge lengths
  long random_lengths(Graph* g, long directed, long min_len, long max_len,
		      long* dist, long seed);
  // Initialize SGB random number generator
  void gb_init_rand(long seed);
  // Generate uniform random integer in [0, m-1]
  long gb_unif_rand(long m);
  // Create a new empty SGB graph
  Graph* gb_new_graph(long n);
  // Add an edge to an SGB graph
  void gb_new_edge(Vertex* u, Vertex* v, long len);
  // Generate Gaussian random number (not from SGB library)
  double gauss();
}

// Use standard namespace
using namespace std;

// Because gb_recycle(r) does not set r to null ..
void my_recycle(Graph*& g) {
  if (g != nullptr) {
    gb_recycle(g);
    g = nullptr;
  }
}

// Forward declaration for board graph generator
Err Board(char** argv, long& argp, int argc, Graph** g, long& stack);
// Forward declaration for custom planar graph generator
Graph* planar(long n, double density, long seed);
// Forward declaration for spin glass graphs
void Assign_binary(long seed_sg, long nedges, long perc, vector<double>& cost);
void Assign_gaussian(long seed_sg, long nedges, vector<double>& cost);
void Output_2D_spinglass(long r, long c, vector<double>& cost);
void Output_3D_spinglass(long r, long c, long l, vector<double>& cost);
// Forward declaration for help message function
void help();

// Maximum number of graphs allowed on the RPN stack
static constexpr long max_stack{10};

static constexpr long VP{10000000}; // Flag to denote a variable parameter
struct Board_type {
  const char* name;
  long p[7];
};

static constexpr long BOARD_SIZE{12};
static constexpr Board_type B[BOARD_SIZE]{
  {"-circuit",          {VP,  0L,  0L, 0L,  1L,  1L, 0L}},
  {"-clique",           {VP,  0L,  0L, 0L, -1L,  0L, 0L}},
  {"-leap_2D",          {VP,  VP,  0L, 0L, -VP,  0L, 0L}},
  {"-leap",             {VP, -VP,  0L, 0L, -VP,  0L, 0L}},
  {"-wrapped_leap_2D",  {VP,  VP,  0L, 0L, -VP, -1L, 0L}},
  {"-wrapped_leap",     {VP, -VP,  0L, 0L, -VP, -1L, 0L}},
  {"-grid_2D",          {VP,  VP,  0L, 0L,  1L,  0L, 0L}},
  {"-grid",             {VP, -VP,  0L, 0L,  1L,  0L, 0L}},
  {"-toroidal_grid_2D", {VP,  VP,  0L, 0L,  1L, -1L, 0L}},
  {"-toroidal_grid",    {VP, -VP,  0L, 0L,  1L, -1L, 0L}},
  {"-simplex",          {VP, -VP,  0L, 0L,  0L,  0L, 0L}},
  {"-bounded_simplex",  {VP,  VP, -VP, 0L,  0L,  0L, 0L}}
};
  
// --- Main Program ---
int main(int argc, char** argv) {

  // Stack to hold graphs during RPN evaluation
  Graph *g[max_stack] = {nullptr};

  // Temporary graph pointers for complex operations
  Graph *gtmp = nullptr, *gtmp1 = nullptr;

  // Index of the top of the stack (-1 means empty)
  long stack = -1;

  // Index for iterating through command line arguments
  long argp = 0;

  // If no arguments are given, print help message and exit
  if (argc == 1) {
    help();
    return(EXIT_SUCCESS);
  }

  // Main processing loop: Parse arguments using Reverse Polish Notation (RPN)
  try {
    while (++argp < argc) { // Iterate through command-line arguments
                            // (argv[0] is program name)

      // --- Binary Operators ---

      if (!strcmp(argv[argp], "+")) { // Graph Union
        // Check stack preconditions
        if (stack <= 0) throw TWO_OP; // Need two operands

	// Operands must have same number of vertices
        if (g[stack - 1]->n != g[stack]->n) throw DIFF_SIZE;

        if (g[stack]->n <= 0) throw EMPTY_GRAPH; // Operands cannot be empty

        // Perform union: g[stack-1] = g[stack-1] U g[stack]
        // gunion(g, gg, multi, directed): multi=0 (no multi-edges),
        // directed=0 (undirected)
        // IMPORTANT: gunion modifies g[stack-1] in place and returns it.

        gtmp = gunion(g[stack - 1], g[stack], 0L, 0L);
        if (gtmp == nullptr) throw PANIC; // Check for SGB errors
        my_recycle(g[stack--]);  // Free the two operands
        my_recycle(g[stack]);
      	g[stack] = gtmp; // assign result to g[stack]
	
        continue; // Move to next argument
      }

      if (!strcmp(argv[argp], "x")) { // Cartesian Product
        // Check stack preconditions
        if (stack <= 0) throw TWO_OP; // Need two operands
	      // Operands cannot be empty
        if (g[stack - 1]->n <= 0 || g[stack]->n <= 0) throw EMPTY_GRAPH;

        // Perform cartesian product: result = g[stack] x g[stack-1]
        // (Note the order) product(g, gg, type, directed):
        // type=cartesian, directed=0 (undirected) product always
        // returns a new graph.
	
        gtmp = product(g[stack], g[stack - 1], cartesian, 0L);
        if (gtmp == nullptr) throw PANIC; // Check for SGB errors
        my_recycle(g[stack--]);  // Free the two operands
        my_recycle(g[stack]);
	g[stack] = gtmp; // assign result to g[stack]

        continue; // Move to next argument
      }

      if (!strcmp(argv[argp], ":")) { // Join Operation (G1 + G2 + K(V1,V2))
        // Check stack preconditions
        if (stack <= 0) throw TWO_OP; // Need two operands
	// Operands cannot be empty
        if (g[stack - 1]->n <= 0 || g[stack]->n <= 0) throw EMPTY_GRAPH;

        // This operation uses the 'induced' SGB function in a clever way.

        // 1. Create a temporary 2-vertex complete graph (K2).
	// board(size=2, piece=-1 => complete graph)
        gtmp = board(2L, 0L, 0L, 0L, -1L, 0L, 0L);
        if (gtmp == nullptr) throw PANIC;

        // 2. Mark the vertices of K2 as "IND_GRAPH" and assign the
        //    operand graphs as the subgraphs to be induced at these
        //    vertices.	
        (gtmp->vertices)->ind = IND_GRAPH; // Vertex 0 represents g[stack-1]
        (gtmp->vertices)->subst = g[stack - 1];
        (gtmp->vertices + 1)->ind = IND_GRAPH;  // Vertex 1 represents g[stack]
        (gtmp->vertices + 1)->subst = g[stack];

        // 3. Call 'induced' to create the final graph. This function:
        //    - Takes the union of the vertex sets of g[stack-1] and g[stack].
        //    - Includes all edges originally within g[stack-1] and g[stack].
        //    - Adds edges between *every* vertex of g[stack-1] and
        //      *every* vertex of g[stack], because there's an edge
        //      between vertex 0 and vertex 1 in the temporary K2.
        // induced(g, desc, self, multi, directed): self=0, multi=0, directed=0
        // induced always returns a new graph.
        gtmp1 = induced(gtmp, nullptr, 0L, 0L, 0L);
        if (gtmp1 == nullptr) throw PANIC;

        // Clean up temporary K2 and original operand graphs
        my_recycle(gtmp);       // Free the K2 template
        // IMPORTANT: induced does NOT recycle the subst graphs
        // (g[stack-1], g[stack])
        my_recycle(g[stack--]); // Free original g[stack-1]
	// Free original g[stack-1] and decrement stack pointer
        my_recycle(g[stack--]);

        // Push the result onto the stack (increment stack first)
        g[++stack] = gtmp1; // Place result at the new top
        // Check for stack overflow after incrementing
        if (g[stack] == nullptr) throw PANIC;

        // Set all edge lengths in the resulting graph to 1 (induced
        // might not guarantee this)
        for (Vertex* v = g[stack]->vertices;
	     v < g[stack]->vertices + g[stack]->n; ++v) {
          for (Arc* a = v->arcs; a; a = a->next) {
            a->len = 1;
          }
        }

        continue; // Move to next argument
      }

      // --- Unary Operators ---

      if (!strcmp(argv[argp], "-complement")) { // Graph Complement
        // Check stack preconditions
        if (stack < 0) throw ONE_OP; // Need one operand
        if (g[stack]->n <= 0) throw EMPTY_GRAPH; // Operand cannot be empty

        // Replace graph on top of stack with its complement
        // complement(g, copy, self, directed): copy=0 (modify in
        // place), self=0 (no loops), directed=0
        gtmp = complement(g[stack], 0L, 0L, 0L);
        if (gtmp == nullptr) throw PANIC;
	my_recycle(g[stack]);
	g[stack] = gtmp;

        continue; // Move to next argument
      }

      if (!strcmp(argv[argp], "-line")) { // Line Graph
        // Check stack preconditions
        if (stack < 0) throw ONE_OP; // Need one operand
	// Operand cannot be empty
        if (g[stack]->n <= 0) throw EMPTY_GRAPH;

        // Replace graph on top of stack with its line graph
        // lines(g, directed): directed=0
        // lines() always creates a new graph.
        gtmp = lines(g[stack], 0L);
        if (gtmp == nullptr) throw PANIC;
	my_recycle(g[stack]);
	g[stack] = gtmp;

        continue; // Move to next argument
      }

      if (!strcmp(argv[argp], "-random")) { // Assign Random Edge Weights
	long lower_weight, upper_weight, seed;
        // Check number of parameters
        if (argp + 3 >= argc) throw ILL_NUM;

        // Parse parameters
        lower_weight = atoi(argv[++argp]);
        upper_weight = atoi(argv[++argp]);
        seed = atoi(argv[++argp]);

        // SGB's random_lengths uses range [min_len, max_len).
	if (lower_weight < upper_weight) upper_weight++; //due to a gb mistake

	if (lower_weight > upper_weight || seed < 0) throw ILL_PAR;

        // Validate parameters
        if (seed < 0) throw ILL_PAR;

        // Check stack preconditions
        if (stack < 0) throw ONE_OP; // Need one operand

        if (g[stack]->n <= 0) throw EMPTY_GRAPH;

        // Assign random lengths to edges of the graph on top of the
        // stack random_lengths(g, directed, min, max, dist, seed):
        // directed=0, dist=NULL It modifies the graph in place and
        // returns the number of edges changed, or -1 on error.
	random_lengths(g[stack], 0L, lower_weight, upper_weight, nullptr, seed);
        if (g[stack] == nullptr) throw PANIC;

        continue; // Move to next argument
      }

      if (!strcmp(argv[argp], "-times")) { // Multiply Edge Weights by Scalar
	long scalar;
        // Check number of parameters
        if (argp + 1 >= argc) throw ILL_NUM;

        // Parse scalar parameter
        scalar = atoi(argv[++argp]);

        // Check stack preconditions
        if (stack < 0) throw ONE_OP; // Need one operand

        if (g[stack]->n <= 0) throw EMPTY_GRAPH;

        // Iterate through all edges and multiply weights SGB graphs
        // are undirected but store two directed arcs per edge.  This
        // loop processes each arc. If weights must be symmetric, this
        // is fine.
        for (Vertex* v = g[stack]->vertices;
	     v < g[stack]->vertices + g[stack]->n; ++v) {
          for (Arc* a = v->arcs; a; a = a->next) {
            a->len *= scalar;
          }
        }
	
        continue; // Move to next argument
      }

      if (!strcmp(argv[argp], "-plus")) { // Add Scalar to Edge Weights
	long scalar;
        // Check number of parameters
        if (argp + 1 >= argc) throw ILL_NUM;

        // Parse scalar parameter
        scalar = atoi(argv[++argp]);

        // Check stack preconditions
        if (stack < 0) throw ONE_OP; // Need one operand
        if (g[stack]->n <= 0) throw EMPTY_GRAPH;

        // Iterate through all edges and add scalar to weights
        for (Vertex* v = g[stack]->vertices;
	     v < g[stack]->vertices + g[stack]->n; ++v) {
          for (Arc* a = v->arcs; a; a = a->next) {
            a->len += scalar;
          }
        }
	
        continue; // Move to next argument
      }

      // --- Graph Generators (Operands) ---
      // These push a new graph onto the stack.

      { // Board graph
	Err status;
	status = Board(argv, argp, argc, g, stack);
	if (status == OK) continue; // Move to next argument
	else if (status != NOT_FOUND) throw status;
      }
      
      if (!strcmp(argv[argp], "-planar")) { // Generate Random Planar Graph
	long size, seed;
	// Density parameter (usually percentage) for random/planar graphs
	double density;
        // Check number of parameters
        if (argp + 3 >= argc) throw ILL_NUM;
        // Parse parameters
        size = atoi(argv[++argp]);
	density = atof(argv[++argp]); // Density as percentage [0, 100]
        seed = atoi(argv[++argp]);
        // Validate parameters
        if (size <= 0 || density < 0.0 || density > 100.0 || seed < 0)
	  throw ILL_PAR;

        // Check stack capacity BEFORE pushing
        if (stack + 1 >= max_stack) throw TOO_MANY;

        // Generate planar graph using the custom planar() function
        // (defined below)
        g[++stack] = planar(size, density, seed);
        // planar() returns null for n<=0 or on allocation error.
        if (g[stack] == nullptr) throw PANIC; // Treat allocation error as PANIC

        continue; // Move to next argument
      }

      // Random Graph (Erdos-Renyi style) G(n, m)
      if (!strcmp(argv[argp], "-rnd_graph")) {
	long seed;
        long n_nodes;
        long n_edges;
        double rn_edges; // Use double for calculation before rounding
	// Density parameter (usually percentage) for random/planar graphs
	double density;
	
        // Check number of parameters
        if (argp + 3 >= argc) throw ILL_NUM;
        // Parse parameters
        n_nodes = atoi(argv[++argp]); // Number of nodes
	density = atof(argv[++argp]); // Edge density percentage [0, 100]
        seed = atoi(argv[++argp]);
        // Validate parameters
        // Allow n_nodes=0? random_graph might handle it. Let's allow n_nodes=0.
        if (n_nodes <= 0 || density < 0.0 || density > 100.0 || seed < 0)
	  throw ILL_PAR;

        // Check stack capacity BEFORE pushing
        if (stack + 1 >= max_stack) throw TOO_MANY;

        // Calculate number of edges based on density
        // Max edges in simple undirected graph = n * (n - 1) / 2
        // Handle n=0, n=1 cases where max edges is 0.
	rn_edges = (n_nodes * (n_nodes - 1.0) * density) / 200.0;

        // Round to nearest integer for number of edges 'm'
        n_edges = rn_edges;
        if (rn_edges - n_edges >= 0.5) n_edges++;

        // Generate random graph using random_graph()
        // random_graph(n, m, multi, self, directed, dist_from, dist_to,
	//              min_len, max_len,
	// seed)
        // n=nodes, m=edges, multi=0, self=0, directed=0, dists=NULL,
	// min/max_len=1
	// (weights=1), seed=seed
        g[++stack] = random_graph(n_nodes, n_edges, 0, 0, 0, nullptr,
				  nullptr, 1, 1, seed);
        // random_graph returns null on error (e.g., n<0, m too large).
        if (g[stack] == nullptr) throw PANIC; // Treat as PANIC if n>0

	continue; // Move to next argument
      }

      // --- Special Spin Glass Generators ---
      // These options bypass the RPN stack, generate output directly, and exit.
      // They require the stack to be empty initially.

      // 2D Toroidal Grid +/-J Weights
      if (!strcmp(argv[argp], "-spinglass2pm")) {
        long r, c, perc, seed_sg; // Use different seed variable name
        long nnodes, nedges;

        // Check preconditions
        if (stack != -1) throw SIMP_GR; // Stack must be empty
        if (argp + 4 >= argc) throw ILL_NUM; // Check number of parameters
        // This check ensures no other arguments follow the spinglass command
        if ((argp + 5) != argc) throw SIMP_GR;

        // Parse parameters
        r = atoi(argv[++argp]); // Rows
        c = atoi(argv[++argp]); // Columns
        perc = atoi(argv[++argp]); // Percentage of negative (-1) bonds
        seed_sg = atoi(argv[++argp]); // Seed for random assignment

        // Basic validation
        if (r <= 0 || c <= 0 || perc < 0 || perc > 100 || seed_sg < 0)
	  throw ILL_PAR;

        nnodes = r * c;
        nedges = 2 * nnodes; // Horizontal + Vertical edges in a 2D grid

	{
	  // Allocate memory for edge costs (+1 because SGB often uses
	  // 1-based indexing internally)
	  std::vector<double> cost(nedges + 1);

	  Assign_binary(seed_sg, nedges, perc, cost);
	  Output_2D_spinglass(r, c, cost);
	}
	
        return(EXIT_SUCCESS);
      }

      // 3D Toroidal Grid +/-J Weights
      if (!strcmp(argv[argp], "-spinglass3pm")) {
        long r, c, l, perc, seed_sg;
        long nnodes, nedges;

        // Check preconditions
        if (stack != -1) throw SIMP_GR;
        if (argp + 5 >= argc) throw ILL_NUM;
        if ((argp + 6) != argc) throw SIMP_GR;

        // Parse parameters
        r = atoi(argv[++argp]); // Rows
        c = atoi(argv[++argp]); // Columns
        l = atoi(argv[++argp]); // Layers
        perc = atoi(argv[++argp]); // Percentage of negative bonds
        seed_sg = atoi(argv[++argp]); // Seed

        // Basic validation
        if (r <= 0 || c <= 0 || l <= 0 || perc < 0 || perc > 100 || seed_sg < 0)
	  throw ILL_PAR;

        nnodes = r * c * l;
        nedges = 3 * nnodes; // Edges in X, Y, Z directions

	{
	  // Use std::vector for costs
	  std::vector<double> cost(nedges + 1);

	  Assign_binary(seed_sg, nedges, perc, cost);
	  Output_3D_spinglass(r, c, l, cost);
	}
	
        return(EXIT_SUCCESS);
      }

      // 2D Toroidal Grid Gaussian Weights
      if (!strcmp(argv[argp], "-spinglass2g")) {
        // Similar structure to spinglass2pm, but uses Gaussian weights
        {
          // Check preconditions
          if (stack != -1) throw SIMP_GR;
          if (argp + 3 >= argc) throw ILL_NUM;
          if ((argp + 4) != argc) throw SIMP_GR;

          // Parse parameters
          long r = atoi(argv[++argp]);
          long c = atoi(argv[++argp]);
          long seed_sg = atoi(argv[++argp]);

          // Basic validation
          if (r <= 0 || c <= 0 || seed_sg < 0) throw ILL_PAR;

          long nnodes = r * c;
          long nedges = 2 * nnodes;

	  {
	    std::vector<double> cost(nedges + 1);

	    Assign_gaussian(seed_sg, nedges, cost);
	    Output_2D_spinglass(r, c, cost);
	  }
	  
          return(EXIT_SUCCESS);
        }
      }

      // 3D Toroidal Grid Gaussian Weights
      if (!strcmp(argv[argp], "-spinglass3g")) {
        {
          // Check preconditions
          if (stack != -1) throw SIMP_GR;
          if (argp + 4 >= argc) throw ILL_NUM;
          if ((argp + 5) != argc) throw SIMP_GR;

          // Parse parameters
          long r = atoi(argv[++argp]);
          long c = atoi(argv[++argp]);
          long l = atoi(argv[++argp]);
          long seed_sg = atoi(argv[++argp]);

          // Basic validation
          if (r <= 0 || c <= 0 || l <= 0 || seed_sg < 0) throw ILL_PAR;

          long nnodes = r * c * l;
          long nedges = 3 * nnodes; // X, Y, Z edges

	  {
	    std::vector<double> cost(nedges + 1);

	    Assign_gaussian(seed_sg, nedges, cost);
	    Output_3D_spinglass(r, c, l, cost);
	  }
	  
	  return(EXIT_SUCCESS);
        }
      }

      // If the argument didn't match any known option
      throw ERROR;

    } // End while loop (parsing arguments)

  } catch (Err e) {
    // --- Error Handling ---
    // Print specific error message based on the caught error code
    switch (e) {
      // Use more descriptive messages matching the help text/context
      case Err::ERROR:
	cerr << "Error: Unknown option or invalid RPN expression near argument "
	     << argp << "." << endl;
	break;
      case EMPTY_GRAPH:
	cerr << "Error: Operation requires a non-empty graph operand near "
	     << "argument " << argp << "." << endl;
	break;
      case Err::ONE_OP:
	cerr << "Error: Operation requires exactly one graph operand on the "
	     << "stack near argument " << argp << "." << endl;
	break;
      case TWO_OP:
	cerr << "Error: Operation requires exactly two graph operands on the "
	     << "stack near argument " << argp << "." << endl;
	break;
      case TOO_MANY:
	cerr << "Error: Stack overflow. Too many graph operands (max "
	     << max_stack << ") near argument " << argp << "." << endl;
	break;
      case DIFF_SIZE:
	cerr << "Error: Binary operation requires operands with the same "
	     << "number of vertices near argument " << argp << "."
	     << endl;
	break;
      case ILL_PAR:
	cerr << "Error: Illegal parameter value provided for operation near "
	     << "argument " << argp << "." << endl;
	break;
      case ILL_NUM:
	cerr << "Error: Incorrect number of parameters provided for operation "
	     << "near argument " << argp << "." << endl;
	break;
      case Err::SIMP_GR:
	cerr << "Error: Standalone generator (" << argv[argp]
	     << ") cannot be used in RPN chain or with existing graphs on "
	     << "stack." << endl;
	break;
      case NOT_EMPTY:
	cerr << "Error: RPN expression resulted in " << (stack + 1)
	     << " graphs on stack; expected exactly one." << endl;
	break;
      case NO_UTILITY:
	cerr << "Error: Could not find an available SGB utility field (u..z) "
	     << "for output indexing." << endl;
	break;
      case Err::PANIC:
	cerr << "Error: Stanford GraphBase library function failed "
	     << "unexpectedly (panic code " << panic_code
	     << ") near argument " << argp << "." << endl;
	break;
      default: // Should not happen
	cerr << "Error: An unexpected error occurred (code " << e
	     << ") near argument " << argp << "." << endl;
	break;
    }
    // Print the command line arguments up to the point of error for context
    cerr << "Command line context: ";
    for (long i = 0; i <= argp && i < argc; ++i) cerr << argv[i] << " ";
    cerr << endl;
    // Clean up any graphs remaining on the stack before exiting
    for (long i = 0; i <= stack; ++i)
      my_recycle(g[i]);
    
    return(EXIT_FAILURE); // Exit with failure status
  }

  // --- Final Output ---
  // If execution reaches here, the RPN expression was parsed successfully.
  try {
    long i; // Node indices for output
    // Index of the SGB utility field to use (0=u, 1=v, ..., 5=z)
    long free_util = -1;

    // Check that exactly one graph remains on the stack
    if (stack != 0) throw NOT_EMPTY; // Should have exactly one result graph
    // Ensure the resulting graph is not null (could be null if size=0
    // was allowed)
    if (g[0] == nullptr) {
        // This likely means an empty graph was generated (e.g., planar 0 ...)
        // Output format for empty graph (0 nodes, 0 edges)
        cout << "0 0" << endl;
        // No need to recycle nullptr
        return(EXIT_SUCCESS);
    }
    // Ensure the graph is not empty (n>0) for standard output
    if (g[0]->n <= 0) {
        // Output format for empty graph (0 nodes, 0 edges)
        cout << "0 0" << endl;
        my_recycle(g[0]); // Clean up the empty graph object
        return(EXIT_SUCCESS);
    }

    // Find an available integer utility field ('Z' type in SGB) in the
    // result graph
    // These fields (u, v, w, x, y, z) are used to assign temporary
    // 1-based indices for output.
    for (free_util = 0; free_util < 6; free_util++) {
      // util_types is an array like "ZZZZZZ" initially. SGB might change chars
      // if a field is used for something else (e.g., 'S' for string).
      // We need 'Z'.
      if (g[0]->util_types[free_util] == 'Z') {
        break; // Found an available field
      }
    }
    // If no 'Z' field is found
    if (free_util >= 6) throw NO_UTILITY;

    // Assign sequential 1-based indices to the vertices using the chosen
    // utility field
    i = 0; // Use 'i' as the 1-based index counter
    Vertex* current_vertex = g[0]->vertices;
    Vertex* const end_vertex = g[0]->vertices + g[0]->n;
    for (; current_vertex < end_vertex; ++current_vertex) {
      // Store the index '++i' into the appropriate utility field
      // (v->u.I, v->v.I, etc.)
      // Using a pointer instead of switch for potentially cleaner
      // code/optimization
      long* util_field_ptr = nullptr;
      switch (free_util) {
        case 0 : util_field_ptr = &(current_vertex->u.I); break;
        case 1 : util_field_ptr = &(current_vertex->v.I); break;
        case 2 : util_field_ptr = &(current_vertex->w.I); break;
        case 3 : util_field_ptr = &(current_vertex->x.I); break;
        case 4 : util_field_ptr = &(current_vertex->y.I); break;
        case 5 : util_field_ptr = &(current_vertex->z.I); break;
        // default case should not be reachable due to the check above
      }
      *util_field_ptr = ++i; // Assign the 1-based index
    }

    // Output the graph header: number of nodes, number of edges
    // (m/2 for undirected)
    // SGB stores directed arcs, so g[0]->m is the total number of arcs
    // (twice the number of undirected edges)
    cout << g[0]->n << " " << g[0]->m / 2 << endl;

    // Output the edges: vertex1 vertex2 weight
    current_vertex = g[0]->vertices; // Reset pointer for iteration
    for (; current_vertex < end_vertex; ++current_vertex) {
      // Retrieve the source vertex index 'i' from its utility field
      long source_index = 0; // Initialize i
      switch (free_util) {
          case 0 : source_index = current_vertex->u.I; break;
          case 1 : source_index = current_vertex->v.I; break;
          case 2 : source_index = current_vertex->w.I; break;
          case 3 : source_index = current_vertex->x.I; break;
          case 4 : source_index = current_vertex->y.I; break;
          case 5 : source_index = current_vertex->z.I; break;
      }
      // Iterate through outgoing arcs from the current vertex
      for (Arc* a = current_vertex->arcs; a; a = a->next) {
        // Retrieve the target vertex index 'j' from the utility field
	// of the arc's tip
        Vertex* target_vertex = a->tip;
        long target_index = 0; // Initialize j
        switch (free_util) {
          case 0 : target_index = target_vertex->u.I; break;
          case 1 : target_index = target_vertex->v.I; break;
          case 2 : target_index = target_vertex->w.I; break;
          case 3 : target_index = target_vertex->x.I; break;
          case 4 : target_index = target_vertex->y.I; break;
          case 5 : target_index = target_vertex->z.I; break;
        }
        // Print edge only once for undirected graphs (e.g., when source
	// index < target index)
        // This avoids printing both (u,v) and (v,u)
        if (source_index < target_index) {
          cout << source_index << " " << target_index << " "
	       << fixed << setprecision(0) << a->len << endl;
        }
      }
    }
  } catch (Err e) {
    // Handle errors specific to the output phase
    switch (e) {
    case Err::NOT_EMPTY:
      cerr << "Error during output: Stack contained " << (stack + 1)
	   << " graphs, expected 1." << endl;
      break;
    case NO_UTILITY:
      cerr << "Error during output: No available SGB utility field found "
	   << "for indexing." << endl;
      break;
      // PANIC during output likely means g[0] became null unexpectedly,
      // but we check earlier.
      // Add a case just in case. 
    case PANIC:
      cerr << "Error during output: Final graph is unexpectedly null." << endl;
      break;
    default:
      cerr << "An unexpected error occurred during output (code " << e
	   << ")." << endl;
      break;
    }
    // Clean up the graph if it exists and wasn't already handled
    if (stack == 0)
      my_recycle(g[0]);

    return(EXIT_FAILURE);
  }

  // Clean up the final graph if it exists
  if (stack == 0)
    my_recycle(g[0]);
  
  // Successful execution
  return(EXIT_SUCCESS);

} // main


Err Board(char** argv, long& argp, int argc, Graph** g, long& stack) { 
  long index;

  for (index = 0; index != BOARD_SIZE; ++ index)
    if (strcmp(argv[argp], B[index].name) == 0) break; 

  if (index == BOARD_SIZE) return Err::NOT_FOUND;

  long p[7];

  for (long i = 0; i != 7; ++i) p[i] = B[index].p[i];

  // count the number of variable parameters

  long n_VP = 0;
  for (long i = 0; i != 7; ++i)
    if (abs(p[i]) == VP) ++n_VP;
    
  // Check number of parameters 
  if (argp + n_VP >= argc) return Err::ILL_NUM;

  // Parse parameters
  for (long i = 0; i != 7; ++i)
    if (abs(p[i]) == VP) {
      long sign = 1;
      long vp = atoi(argv[++argp]);

      // Validate parameters
      if (vp <= 0) return Err::ILL_PAR;

      if (p[i] < 0L) sign = -1;
      p[i] = sign * vp;
    }

  // Check stack capacity BEFORE pushing
  if (stack + 1 >= max_stack) return Err::TOO_MANY;

  if ((strcmp(argv[argp], "-simplex") == 0) ||
      (strcmp(argv[argp], "-bounded_simplex") == 0)) {

    // --- Simplex Graphs ---
    // Vertices are vectors (x1, ..., xd) of non-negative integers.
    // Edges connect vertices if Euclidean distance is sqrt(2). Weights=1.
    
    g[++stack] = simplex(p[0], p[1], p[2], p[3], p[4], p[5], p[6]);

  } else {

    // Generate graph using board()
    // board(n1, n2, n3, n4, piece, wrap, directed) 
    
    g[++stack] = board(p[0], p[1], p[2], p[3], p[4], p[5], p[6]);
  }
  if (g[stack] == nullptr) return PANIC;

  return Err::OK;
}


// --- Helper Classes for Planar Graph Generation ---

// Represents a triangular face (used during triangulation)
class Face {
public:
  long i, j, k; // Indices of the three vertices forming the face
  Face() = default; // Default constructor
};

// Represents an edge (used to store edges before creating the SGB graph)
class Edge {
public:
  long u, v; // Indices of the two vertices forming the edge
  Edge() = default; // Default constructor
};

// --- Custom Planar Graph Generator ---
// Generates a random maximal planar graph (triangulation) and then
// removes edges randomly to achieve the desired density.
// Based on an incremental triangulation algorithm.
Graph* planar(long n, double density, long seed) {
  Graph *new_graph = nullptr; // The SGB graph to be returned
  long f, a, b, c, d, e; // Vertex and face indices
  long n_edges = 0;           // Current number of edges generated
  // Max edges in the initial triangulation (3n-6 for n>=3)
  long max_edges_possible = 0;
  long target_n_edges;        // Desired number of edges based on density

  if (n <= 0) return nullptr; // Cannot create graph with non-positive size

  std::vector<Edge> edge_list;

  if (n == 1) {
    max_edges_possible = 0; // No edges needed
  } else if (n == 2) {
    max_edges_possible = 1;
    target_n_edges = (density * max_edges_possible) / 100.0; // Round
    if (target_n_edges > 0) {
      edge_list.push_back({0, 1}); // Add edge {0, 1}
      n_edges = 1;
    }
  } else { // n >= 3
    gb_init_rand(seed); // Initialize SGB RNG

    // A maximal planar graph with n>=3 vertices has 3n-6 edges and 2n-4 faces
    // (by Euler's formula)
    // Max edges in triangulation
    max_edges_possible = 3 * (n - 2);
    // Handle n=3 case where 3 * 3 - 6 = 3
    if (max_edges_possible < 0) max_edges_possible = 0;

    // Reserve space in edge_list for efficiency
    edge_list.reserve(max_edges_possible);

    // Max faces = 2n - 4.
    // Use std::vector for faces as well
    std::vector<Face> face_list;
    face_list.reserve(2 * n - 4); // Reserve space

    // Start with a triangle (K3) as the base
    face_list.push_back({0, 1, 2}); // First face {0, 1, 2}

    // Add edges of the initial triangle
    edge_list.push_back({0, 1}); n_edges++;
    edge_list.push_back({0, 2}); n_edges++;
    edge_list.push_back({1, 2}); n_edges++;

    // Incrementally add remaining vertices (index d = 3 to n-1)
    for (d = 3; d < n; d++) { // d is the index of the vertex being added
      // Choose a random existing face to insert the new vertex into
      // Number of faces is currently face_list.size()
      // Choose random face index [0, size - 1]
      f = gb_unif_rand(face_list.size()); 

      // Get vertices of the chosen face
      a = face_list[f].i;
      b = face_list[f].j;
      c = face_list[f].k;

      // Add edges connecting the new vertex 'd' to the vertices of the
      // chosen face
      edge_list.push_back({a, d}); n_edges++;
      edge_list.push_back({b, d}); n_edges++;
      edge_list.push_back({c, d}); n_edges++;

      // Update the face list:
      // The chosen face f is replaced by (a, b, d).
      // Two new faces (a, c, d) and (b, c, d) are added.
      Face face1 = {a, c, d}; // New face 1
      Face face2 = {b, c, d}; // New face 2
      face_list[f].k = d;      // Modify face f to become (a, b, d)
      face_list.push_back(face1); // Add new face 1
      face_list.push_back(face2); // Add new face 2
    }
    // face_list goes out of scope here, memory is freed.
  }

  // At this point, 'edge_list' contains the edges of a maximal planar graph
  // (or fewer for n<3). 'n_edges' is the number of edges in this graph.

  // Calculate the target number of edges based on density
  target_n_edges = (density * max_edges_possible) / 100.0; // Round
  target_n_edges = std::max(0L, target_n_edges); // Ensure non-negative

  // Randomly remove edges until the target density is reached
  // This only applies if we started with more edges than the target
  // (typically n>=3)
  if (n_edges > target_n_edges) {
    // Seed should be initialized if n>=3
    while (n_edges > target_n_edges) {
      // Choose a random edge index to remove
      e = gb_unif_rand(n_edges); // Random index [0, n_edges - 1]
      // Replace the chosen edge with the last edge in the vector
      // (fast removal)
      edge_list[e] = edge_list.back();
      edge_list.pop_back(); // Remove the last element
      n_edges--; // Decrease the count of active edges
    }
  }

  // Create the final SGB graph structure
  new_graph = gb_new_graph(n);
  if (new_graph != nullptr) {
    // Set graph ID based on parameters (optional but good practice)
    // Max length for SGB IDs is typically around 80 chars.
    snprintf(new_graph->id, sizeof(new_graph->id), "planar(%ld,%.1f,%ld)",
	     n, density, seed);

    // Add the remaining edges from edge_list to the SGB graph
    for (const auto& edge : edge_list) {
      // gb_new_edge adds arcs in both directions for undirected graphs
      gb_new_edge(new_graph->vertices + edge.u, // Pointer to source vertex u
                  new_graph->vertices + edge.v, // Pointer to target vertex v
                  1L);                          // Edge length/weight set to 1
    }
  } else {
    // If gb_new_graph failed, return nullptr (memory issue).
    // edge_list will be cleaned up automatically by vector destructor.
    return nullptr;
  }

  return new_graph; // Return the generated SGB graph
}

// Assign +/-J weights randomly (spin glass graph)
void Assign_binary(long seed_sg, long nedges, long perc, vector<double>& cost) {
  long k, t;

  gb_init_rand(seed_sg); // Initialize SGB RNG

  // Calculate number of -1 weights, ensuring it doesn't exceed nedges
  long num_neg = (nedges * perc) / 100.0;  // Number of negative bonds

  // Fill the cost array initially
  for (long j = 1; j <= num_neg; ++j) cost[j] = -1.0;
  for (long j = num_neg + 1; j <= nedges; j++) cost[j] = 1.0;

  // Shuffle the costs using Fisher-Yates algorithm
  for (long j = nedges; j > 1; --j) {
    k = gb_unif_rand(j) + 1; // Random index k from 1 to j
    // Swap cost[j] and cost[k]
    t = cost[j];
    cost[j] = cost[k];
    cost[k] = t;
  }
} // End Assign_binary

// Assign Gaussian weights randomly (spin glass graph)
void Assign_gaussian(long seed_sg, long nedges, vector<double>& cost) {
  long k, t;

  // Scale factor to get larger values
  long scalefactor = 100000;

  gb_init_rand(seed_sg); // Initialize RNG

  // Assign scaled Gaussian random numbers to costs
  for (long j = 1; j <= nedges; ++j) {
    // Assuming gauss() returns standard normal N(0,1)
    cost[j] = gauss() * scalefactor;
  }

  // Shuffle costs
  for (long j = nedges; j > 1; j--) {
    k = gb_unif_rand(j) + 1;
    t = cost[j];
    cost[j] = cost[k];
    cost[k] = t;
  }
} // End Assign_gaussian

// Output the 2D toroidal spin glass graph
void Output_2D_spinglass(long r, long c, vector<double>& cost) {
  long nnodes = r * c;
  long nedges = 2 * nnodes;

  cout << nnodes << " " << nedges << endl;

  for (long e = 1; e <= nedges; ++e) {
    long i, j;
    
    if (e <= nnodes) { // Horizontal Edge
      i = e;
      j = i + 1;
      if ((i % c) == 0) j -= c; // Wrap horizontally
    } else { // Vertical Edge
      i = e - nnodes;
      j = i + c;
      if (j > nnodes) j -= nnodes; // Wrap vertically
    }
    cout << i << " " << j << " " << fixed << setprecision(0)
	 << cost[e] << endl;
  }
} // Output_2D_spinglass

// Output the 3D toroidal spin glass graph
void Output_3D_spinglass(long r, long c, long l, vector<double>& cost) {
  // Assume first nnodes costs are for X, next for Y, last for Z.
  long rc = r * c;
  long nnodes = r * c * l;
  long nedges = 3 * nnodes; // X, Y, Z edges

  cout << nnodes << " " << nedges << endl;
	  
  for (long e = 1; e <= nedges; e++) {
    long i, j;
	    
    long layer = (e - 1) / (3 * rc) + 1;
    long rcl1 = rc * (layer - 1);
    long a = 3 * rcl1  + rc; 
    long b    = a + rc;
	    
    if (e <= a) {                       /* Horizontal Edges  */
      i = e - 2 * rcl1;
      j = i + 1;
      if (!(i % c)) j -= c;
    } else if (e <= b) {                   /* Vertical Edge */
      i = e - (2 * rcl1) - rc;
      j = i + c;
      if (j > layer * rc) j -= rc;
    } else {                             /* Layer numbering */
      i = e - 2 * rcl1 - (2 * rc);
      j = i + rc;
      if (j > nnodes) j -= nnodes;
    }
    cout << i << " " << j << " " << fixed << setprecision(0)
	 << cost[e] << endl;
  }
} // Output_3D_spinglass

// --- Help Message ---
// Prints usage instructions to standard output.
void help() {
  // Using raw string literal R"(...)" for cleaner multi-line output
  cout << R"(
  Rudy: A Rudimental Graph Generator by G. Rinaldi
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usage:
  rudy <graph_expression>

Description:
  Generates a graph based on the <graph_expression> and prints it to
  standard output. The expression is evaluated using Reverse Polish
  Notation (RPN). Graph generators act as operands, pushing a graph onto
  an internal stack. Operators act on the graph(s) currently on the stack.
  The maximum stack depth is 10 graphs.

<graph_expression>:
  A sequence of generators and operators in RPN. Parameters for generators
  and operators follow their respective keywords. Parameters are generally
  expected to be positive integers (long) unless otherwise specified.

*** Postfix Unary Operators ***
  (Operate on the single graph G at the top of the stack)

  -complement
      Replaces G with its complement graph ~G. Edges in ~G have weight 1.

  -line
      Replaces G with its line graph L(G). Edges in L(G) have weight 1.

  -random <LOWER> <UPPER> <SEED>
      Assigns random integer weights to the edges of G. Weights are drawn
      uniformly from the inclusive range [<LOWER>..<UPPER>].
      <SEED> is the random number generator seed (>= 0).

  -times <SCALAR>
      Multiplies the weight of every edge in G by the integer <SCALAR>.

  -plus <SCALAR>
      Adds the integer <SCALAR> to the weight of every edge in G.

*** Postfix Binary Operators ***
  (Operate on the top two graphs on the stack: G1 [deep] and G2 [top])

  +   (Union)
      Requires G1 and G2 to have the same number of vertices (|V1|=|V2|).
      Computes the graph union G = (V1, E1 U E2).
      The result G replaces G1 on the stack, and G2 is removed.

  x   (Cartesian Product)
      Computes the Cartesian product G = G2 x G1.
      The result G replaces G1 on the stack, and G2 is removed.
      (Note the order: top graph is the first operand in the product).

  :   (Join)
      Computes the graph join of G1 and G2. Assumes vertex sets V1 and V2
      are disjoint. The result G contains V1 U V2, E1 U E2, and all edges
      connecting every vertex in V1 to every vertex in V2.
      Resulting edges have weight 1.
      The result G replaces G1 and G2 on the stack (pops two, pushes one).

*** Graph Generators (Operands) ***
  (Push a new graph onto the stack)

  * Canonical Graphs:

  -circuit <NUM_NODES>
      Generates a cycle graph C_<NUM_NODES>. Weights are 1.
      (<NUM_NODES> > 0).

  -clique <NUM_NODES>
      Generates a complete graph K_<NUM_NODES>. The weight of each edge 
      (i,j) is |i-j|. (<NUM_NODES> > 0).

  -planar <NUM_NODES> <DENSITY> <SEED>
      Generates a random planar graph with <NUM_NODES> vertices (>= 0).
      <DENSITY> (real) is a percentage specifying the desired edge 
      density relative to a maximal planar graph (which has 3*N-6 edges
      for N>=3). (<NUM_NODES> > 0), (<DENSITY> >= 0.0 and <= 100.0), 
      <SEED> is the random number generator seed (>= 0). Weights are 1.

  -rnd_graph <NUM_NODES> <DENSITY> <SEED>
      Generates a random graph G(N, M) with <NUM_NODES> vertices (>= 0).
      The number of edges M is determined by <DENSITY>, calculated as 
      floor(N*(N-1)/2 * DENSITY/100). (<NUM_NODES> > 0), (<DENSITY> >= 
      0.0 and <= 100.0), <SEED> is the random number generator seed 
      (>= 0). Weights are 1.

  * Board Graphs (Based on grids and moves):

  -grid_2D <HEIGHT> <WIDTH>
      Generates a planar grid graph of <HEIGHT> x <WIDTH>. Weights are 1.
      (<HEIGHT>, <WIDTH> > 0).

  -grid <SIZE> <DIM>
      Generates a <DIM>-dimensional grid graph where each dimension has
      <SIZE> vertices. Weights are 1. (<SIZE>, <DIM> > 0).

  -toroidal_grid_2D <HEIGHT> <WIDTH>
      Generates a toroidal (wrapped) grid graph of <HEIGHT> x <WIDTH>.
      Weights are 1. (<HEIGHT>, <WIDTH> > 0).

  -toroidal_grid <SIZE> <DIM>
      Generates a <DIM>-dimensional toroidal grid graph of size <SIZE>
      in each dimension. Weights are 1. (<SIZE>, <DIM> > 0).

  -leap_2D <HEIGHT> <WIDTH> <MOVE_TYPE>
      Generates a leap graph on a <HEIGHT> x <WIDTH> board. An edge 
      exists between cells if one can be reached from the other by a 
      sequence of t "knight-like" moves defined by <MOVE_TYPE>. 
      <MOVE_TYPE> is the sum of squared coordinate differences for a 
      basic move (e.g., 1=rook, 2=bishop, 5=knight on a chessboard). 
      The weight of the edge is t. (<HEIGHT>, <WIDTH>, <MOVE_TYPE> > 0).

  -leap <SIZE> <DIM> <MOVE_TYPE>
      Generates a leap graph on a <DIM>-dimensional board of size 
      <SIZE> in each dimension. Weights are as in -leap_2D. (<SIZE>, 
      <DIM>, <MOVE_TYPE> > 0).

  -wrapped_leap_2D <HEIGHT> <WIDTH> <MOVE_TYPE>
      Generates a toroidal (wrapped) leap graph on a <HEIGHT> x <WIDTH> 
      board. Weights are as in -leap_2D. (<HEIGHT>, <WIDTH>, 
      <MOVE_TYPE> > 0).

  -wrapped_leap <SIZE> <DIM> <MOVE_TYPE>
      Generates a toroidal leap graph on a <DIM>-dimensional board of 
      size <SIZE>. Weights are as in -leap_2D. (<SIZE>, <DIM>, 
      <MOVE_TYPE> > 0).

  * Simplex Graphs:
    (Vertices represent integer vectors; edges connect vectors at 
    distance sqrt(2))

  -simplex <SUM> <DIM>
      Generates a simplex graph where vertices are <DIM>-dimensional 
      non-negative integer vectors whose components sum to <SUM>.
      Weights are 1. (<SUM>, <DIM> > 0).

  -bounded_simplex <SUM> <DIM> <BOUND>
      Like -simplex, but each vector component must be less than or 
      equal to <BOUND>. Weights are 1. (<SUM>, <DIM>, <BOUND> > 0).

*** Special Standalone Generators ***
  (These options generate a graph directly to standard output and exit.
   They cannot be used with the RPN stack or other operators.)

  -spinglass2pm <ROWS> <COLS> <PERCENT_NEG> <SEED>
      Generates a 2D (<ROWS> x <COLS>) toroidal grid for a +/-1 spin 
      glass model. <PERCENT_NEG>% (long) of edges randomly get weight 
      -1, others get +1. (<ROWS>, <COLS> >0), (<PERC> >= 0 and <= 100), 
      <SEED> is the random seed (>= 0).

  -spinglass3pm <ROWS> <COLS> <LAYERS> <PERCENT_NEG> <SEED>
      Generates a 3D (<ROWS> x <COLS> x <LAYERS>) toroidal grid for a 
      +/-1 spin glass model. <PERCENT_NEG>% (long) of edges randomly 
      get weight -1, others get +1. (<ROWS>, <COLS>, <LAYERS> >0), 
      (<PERC> >= 0 and <= 100), <SEED> is the random seed (>= 0).

  -spinglass2g <ROWS> <COLS> <SEED>
      Generates a 2D (<ROWS> x <COLS>) toroidal grid for a Gaussian 
      spin glass model. Edge weights are drawn from N(0, 1) * 100000 
      (scaled). (<ROWS>, <COLS>, <LAYERS> >0), <SEED> is the random 
      seed (>= 0).

  -spinglass3g <ROWS> <COLS> <LAYERS> <SEED>
      Generates a 3D (<ROWS> x <COLS> x <LAYERS>) toroidal grid for a 
      Gaussian spin glass model. Edge weights are drawn from 
      N(0, 1) * 100000 (scaled). (<ROWS>, <COLS>, <LAYERS> >0),
      <SEED> is the random seed (>= 0).

*** Output Format ***
  The generated graph is printed to standard output in the following 
  format:

  N M
  u1 v1 w1
  u2 v2 w2
  ...

  Where:
    N = number of vertices
    M = number of edges
    u, v = 1-based vertex indices of an edge
    w = weight of the edge (printed as an integer)

*** Examples ***

  # Generate a 5x5 grid, assign random weights [1..10], generate a
  # C_25, then compute the union (requires N=25 for both graphs).
  rudy -grid_2D 5 5 -random 1 10 123 -circuit 25 +

  # Generate a 3x3 toroidal grid (9 nodes) and take its complement.
  rudy -toroidal_grid_2D 3 3 -complement

  # Directly generate a 10x10 +/-1 spin glass grid.
  rudy -spinglass2pm 10 10 50 42

)" << endl;
} // help()
