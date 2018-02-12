#pragma once

#include "Util.h"

#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
using std::vector;
using std::map;
using std::make_pair;

class FactorFunction
{
public:
    virtual double GetValue(int y1, int y2) = 0;
	virtual double GetValue(int y1, int y2, int y3) = 0;
};

class Node
{
public:
    int             id;
    int             num_label;

    vector<Node*>   neighbor;
    vector<double*> belief;
    map<int, int>   neighbor_pos;

    double          *msg;
        
    virtual void Init(int num_label) = 0;
    void BasicInit(int num_label);
    void NormalizeMessage();

    void AddNeighbor(Node* ng)
    {
        neighbor_pos.insert( make_pair(ng->id, neighbor.size()) );
        neighbor.push_back(ng);

        belief.push_back( MatrixUtil::GetDoubleArr(num_label) );
    }

    virtual void BeliefPropagation(double* diff_max, bool labeled_given) = 0;
    virtual void MaxSumPropagation(double* diff_max, bool labeled_given) = 0;

    void GetMessageFrom(int u, double* msgvec, double* diff_max)
    {
        int p = neighbor_pos[u];
        for (int y = 0; y < num_label; y ++)
        {
            if (fabs(belief[p][y] - msgvec[y]) > *diff_max)
                *diff_max = fabs(belief[p][y] - msgvec[y]);
            belief[p][y] = msgvec[y];
        }
    }
    
    virtual ~Node()
    {
        for (int i = 0; i < (int)belief.size(); i ++)
            delete[] belief[i];
        if (msg)
            delete[] msg;
    }
};

class VariableNode : public Node
{
public:
    int     y;
    int     label_type;

    double* state_factor;

    double* marginal;

    VariableNode() { state_factor = NULL; }

    virtual void Init(int num_label);
    virtual void BeliefPropagation(double* diff_max, bool labeled_given);
    virtual void MaxSumPropagation(double* diff_max, bool labeled_given);

    virtual ~VariableNode()
    {
        if (state_factor) delete[] state_factor;
        if (marginal) delete[] marginal;
    }
};

class FactorNode : public Node
{
public:
    FactorFunction  *func;
    double **marginal;
	double ***marginal3d;

    virtual void Init(int num_label);
    virtual void BeliefPropagation(double* diff_max, bool labeled_given);
    virtual void MaxSumPropagation(double* diff_max, bool labeled_given);

    virtual ~FactorNode() 
    {
        if (marginal)
        {
            for (int i = 0; i < num_label; i ++)
                if (marginal[i]) delete[] marginal[i];
            delete[] marginal;
        }
		if (marginal3d)
		{
			for (int i = 0; i < num_label; i ++)
				if (marginal3d[i])
				{
					for (int j = 0; j < num_label; j ++)
						if (marginal3d[i][j])
							delete[] marginal3d[i][j];
					delete[] marginal3d[i];
				}
			delete[] marginal3d;
		}
    }
};

class FactorGraph
{
public:    
    int                 n, m, num_label;
    int                 num_node;

    bool                converged;
    double              diff_max;

    bool                labeled_given;

    VariableNode*       var_node;
    FactorNode*         factor_node;
    Node**              p_node;
    Node**              bfs_node;

    // For each subgraph (connected component), we select one node as entry
    vector<Node*>       entry;

    int                 factor_node_used;

    void InitGraph(int n, int m, int num_label);
    void AddEdge(int a, int b, FactorFunction* func);
	void AddTriangle(int a, int b, int c, FactorFunction *func);
    void GenPropagateOrder();
    void ClearDataForSumProduct();
    void ClearDataForMaxSum();
    
    void SetVariableLabel(int u, int y) { var_node[u].y = y; }
    void SetVariableStateFactor(int u, int y, double v){ var_node[u].state_factor[y] = v; }
    
    // Sum-Product
    void BeliefPropagation(int max_iter);
    void CalculateMarginal();

    // Max-Sum
    void MaxSumPropagation(int max_iter);
    
    void Clean();
    ~FactorGraph(){ Clean(); }
};