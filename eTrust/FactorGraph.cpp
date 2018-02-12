#include "FactorGraph.h"
#include "Constant.h"

#include <cassert>

/**************************************************
 Node
**************************************************/

void Node::BasicInit(int num_label)
{
    this->num_label = num_label;
    msg = new double[num_label];
}

void VariableNode::Init(int num_label)
{
    BasicInit(num_label);

    state_factor = MatrixUtil::GetDoubleArr(num_label);

    marginal = MatrixUtil::GetDoubleArr(num_label);
}

void FactorNode::Init(int num_label)
{
    BasicInit(num_label);

    marginal = new double*[num_label];
    for (int i = 0; i < num_label; i ++)
        marginal[i] = new double[num_label];

	marginal3d = new double** [num_label];
	for (int i = 0; i < num_label; i ++)
	{
		marginal3d[i] = new double* [num_label];
		for (int j = 0; j < num_label; j++)
			marginal3d[i][j] = new double[num_label];
	}
}

void Node::NormalizeMessage()
{
    double s = 0.0;
    for (int y = 0; y < num_label; y ++)
        s += msg[y];
    for (int y = 0; y < num_label; y ++)
        msg[y] /= s;
}

void VariableNode::BeliefPropagation(double* diff_max, bool labeled_given)
{
    double product;
    bool *zeroFlag = new bool[num_label];

    for (int i = 0; i < (int)neighbor.size(); i ++)
    {
        FactorNode* f = (FactorNode*) neighbor[i];
        double minMsg = 1e8;
        for (int y = 0; y < num_label; y ++)
        {
            zeroFlag[y] = false;
        }
        for (int y = 0; y < num_label; y ++)
        {
            if (this->state_factor[y] < 1e-8) {
                zeroFlag[y] = true;
            }
            //fprintf(stderr, "y = %d, state_factor = %.4f\n", y, this->state_factor[y]);
            product = log(this->state_factor[y]);
            for (int j = 0; j < (int)neighbor.size(); j ++)
                if (i != j) {
                    if (this->belief[j][y] < 1e-8) {
                        zeroFlag[y] = true;
                    }
                    //fprintf(stderr, "y = %d, belief[j][y] = %.4f\n", y, this->belief[j][y]);
                    product += log(this->belief[j][y]);
                }
            if (!zeroFlag[y]) {
                msg[y] = product;
                minMsg = std::min(minMsg, msg[y]);
            }
        }

        for (int y = 0; y < num_label; y ++)
        {
            //fprintf(stderr, "before y = %d, zeroFlag = %d, msg = %.4f\n", y, zeroFlag[y], msg[y]);
            if (!zeroFlag[y]) {
                msg[y] = exp(msg[y] - minMsg);
            } else {
                msg[y] = 0;
            }
            //fprintf(stderr, "after  y = %d, zeroFlag = %d, msg = %.4f\n", y, zeroFlag[y], msg[y]);
            assert(!isnan(msg[y]));
        }

        NormalizeMessage();
        f->GetMessageFrom(id, msg, diff_max);
    }
    delete[] zeroFlag;
}

void FactorNode::BeliefPropagation(double* diff_max, bool labeled_given)
{
	if (SIZE(neighbor) == 2)
	{
		for (int i = 0; i < 2; i ++)
		{
			if (labeled_given && ((VariableNode*)neighbor[i])->label_type == Enum::KNOWN_LABEL)
			{
				for (int y = 0; y < num_label; y ++)
					msg[y] = 0;
				msg[((VariableNode*)neighbor[i])->y] = 1.0;
			}
			else
			{
				for (int y = 0; y < num_label; y ++)
				{
					double s = 0;
					for (int y1 = 0; y1 < num_label; y1 ++)
						s += func->GetValue(y, y1) * belief[1 - i][y1];
					msg[y] = s;
				}
				NormalizeMessage();
			}

			neighbor[i]->GetMessageFrom(id, msg, diff_max);
		}
	}
	else
	{
		for (int i = 0; i < 3; i ++)
		{
			if (labeled_given && ((VariableNode*)neighbor[i])->label_type == Enum::KNOWN_LABEL)
			{
				for (int y = 0; y < num_label; y ++)
					msg[y] = 0;
				msg[((VariableNode*)neighbor[i])->y] = 1.0;
			}
			else
			{
				for (int y = 0; y < num_label; y ++)
				{
					double s = 0;
					for (int y1 = 0; y1 < num_label; y1 ++)
						for (int y2 = 0; y2 < num_label; y2 ++)
						{
							double b = belief[(i == 2) ? 0 : (i + 1)][y1] * belief[(i == 0) ? 2 : (i - 1)][y2];
							double g = (i == 0) ? func->GetValue(y, y1, y2) : ((i == 1) ? func->GetValue(y2, y, y1) : func->GetValue(y1, y2, y));
							s += g * b;
						}
					msg[y] = s;
				}
				NormalizeMessage();
			}

			neighbor[i]->GetMessageFrom(id, msg, diff_max);
		}
	}
}

void VariableNode::MaxSumPropagation(double* diff_max, bool labeled_given)
{
    double product;
    bool *zeroFlag = new bool[num_label];

    for (int i = 0; i < (int)neighbor.size(); i ++)
    {
        FactorNode* f = (FactorNode*) neighbor[i];
        double minMsg = 1e8;
        for (int y = 0; y < num_label; y ++)
        {
            zeroFlag[y] = false;
        }
        for (int y = 0; y < num_label; y ++)
        {
            if (this->state_factor[y] < 1e-8) {
                zeroFlag[y] = true;
            }
            product = log(this->state_factor[y]);
            for (int j = 0; j < (int)neighbor.size(); j ++)
                if (i != j) {
                    if (this->belief[j][y] < 1e-8) {
                        zeroFlag[y] = true;
                    }
                    product += log(this->belief[j][y]);
                }
            if (!zeroFlag[y]) {
                msg[y] = product;
                minMsg = std::min(minMsg, msg[y]);
            }
        }

        for (int y = 0; y < num_label; y ++)
        {
            if (!zeroFlag[y]) {
                msg[y] = exp(msg[y] - minMsg);
            } else {
                msg[y] = 0;
            }
            assert(!isnan(msg[y]));
        }

        NormalizeMessage();
        f->GetMessageFrom(id, msg, diff_max);
    }
    delete[] zeroFlag;
}

void FactorNode::MaxSumPropagation(double* diff_max, bool labeled_given)
{
	if (SIZE(neighbor) == 2)
	{
		for (int i = 0; i < 2; i ++)    
		{
			if (labeled_given && ((VariableNode*)neighbor[i])->label_type == Enum::KNOWN_LABEL)
			{
				for (int y = 0; y < num_label; y ++)
					msg[y] = 0;
				msg[((VariableNode*)neighbor[i])->y] = 1.0;
			}
			else
			{
				for (int y = 0; y < num_label; y ++)
				{
					double s = -1e200;
					for (int y1 = 0; y1 < num_label; y1 ++)
					{
						double t = func->GetValue(y, y1) * belief[1 - i][y1];
						if (t > s) 
							s = t;
					}
					msg[y] = s;
				}
				NormalizeMessage();
			}

			neighbor[i]->GetMessageFrom(id, msg, diff_max);
		}
	}
	else
	{
		for (int i = 0; i < 3; i ++)    
		{
			if (labeled_given && ((VariableNode*)neighbor[i])->label_type == Enum::KNOWN_LABEL)
			{
				for (int y = 0; y < num_label; y ++)
					msg[y] = 0;
				msg[((VariableNode*)neighbor[i])->y] = 1.0;
			}
			else
			{
				for (int y = 0; y < num_label; y ++)
				{
					double s = -1e200;
					for (int y1 = 0; y1 < num_label; y1 ++)
						for (int y2 = 0; y2 < num_label; y2++)
						{
							double b = belief[(i == 2) ? 0 : (i + 1)][y1] * belief[(i == 0) ? 2 : (i - 1)][y2];
							double g = (i == 0) ? func->GetValue(y, y1, y2) : ((i == 1) ? func->GetValue(y2, y, y1) : func->GetValue(y1, y2, y));
							double t = g * b;
							if (t > s)
								s = t;
						}
					msg[y] = s;
				}
				NormalizeMessage();
			}

			neighbor[i]->GetMessageFrom(id, msg, diff_max);
		}
	}
}


/**************************************************
 FactorGraph
**************************************************/

void FactorGraph::InitGraph(int n, int m, int num_label)
{
    this->labeled_given = false;
    this->n = n;
    this->m = m;
    this->num_label = num_label;
    this->num_node = n + m;
    
    var_node = new VariableNode[n];
    factor_node = new FactorNode[m];

    int p_node_id = 0;

    p_node = new Node*[n + m];
    for (int i = 0; i < n; i ++)
    {
        var_node[i].id = p_node_id;
        p_node[p_node_id ++] = &var_node[i];
        
        var_node[i].Init( num_label );
    }

    for (int i = 0; i < m; i ++)
    {
        factor_node[i].id = p_node_id;
        p_node[p_node_id ++] = &factor_node[i];

        factor_node[i].Init( num_label );
    }

    factor_node_used = 0;
}

void FactorGraph::AddEdge(int a, int b, FactorFunction* func)
{
    // AddEdge can be called at most m times
    if (factor_node_used == m) return;

    factor_node[factor_node_used].func = func;
        
    factor_node[factor_node_used].AddNeighbor( &var_node[a] );
    factor_node[factor_node_used].AddNeighbor( &var_node[b] );

    var_node[a].AddNeighbor( &factor_node[factor_node_used] );
    var_node[b].AddNeighbor( &factor_node[factor_node_used] );

    factor_node_used ++;
}

void FactorGraph::AddTriangle(int a, int b, int c, FactorFunction *func)
{
	//AddTriangle
	if (factor_node_used == m) return;

	factor_node[factor_node_used].func = func;

	factor_node[factor_node_used].AddNeighbor( &var_node[a] );
	factor_node[factor_node_used].AddNeighbor( &var_node[b] );
	factor_node[factor_node_used].AddNeighbor( &var_node[c] );

	var_node[a].AddNeighbor( &factor_node[factor_node_used] );
	var_node[b].AddNeighbor( &factor_node[factor_node_used] );
	var_node[c].AddNeighbor( &factor_node[factor_node_used] );

	factor_node_used ++;
}

void FactorGraph::ClearDataForSumProduct()
{   
    for (int i = 0; i < n; i ++)
    {
        MatrixUtil::DoubleArrFill(var_node[i].state_factor, num_label, 1.0 / num_label);          
    }

    for (int i = 0; i < num_node; i ++)
    {
        for (int t = 0; t < (int)p_node[i]->neighbor.size(); t ++)
        {
            MatrixUtil::DoubleArrFill(p_node[i]->belief[t], num_label, 1.0 / num_label);
        }
    }
}

void FactorGraph::ClearDataForMaxSum()
{
    for (int i = 0; i < n; i ++)
    {
        MatrixUtil::DoubleArrFill(var_node[i].state_factor, num_label, 1.0 / num_label);
    }
    for (int i = 0; i < num_node; i ++)
    {
        for (int t = 0; t < (int)p_node[i]->neighbor.size(); t ++)
        {
            for (int y = 0; y < num_label; y ++)
                p_node[i]->belief[t][y] = 1.0 / num_label;
        }
    }
}

void FactorGraph::GenPropagateOrder()
{
    bool* mark = new bool[num_node];
    bfs_node = new Node*[num_node];

    for (int i = 0; i < num_node; i ++)
        mark[i] = false;

    int head = 0, tail = -1;
    for (int i = 0; i < num_node; i ++)
    {
        if (! mark[i])
        {
            entry.push_back( p_node[i] );
            bfs_node[ ++ tail ] = p_node[i];
            mark[ p_node[i]->id ] = 1;

            while (head <= tail)
            {
                Node* u = bfs_node[head ++];
                for (vector<Node*>::iterator it = u->neighbor.begin(); it != u->neighbor.end(); it ++)
                    if (! mark[(*it)->id] )
                    {
                        bfs_node[ ++ tail ] = *it;
                        mark[ (*it)->id ] = 1;
                    }
            }
        }
    }

    delete[] mark;
}

void FactorGraph::BeliefPropagation(int max_iter)
{    
    int start, end, dir;

    converged = false;
    for (int iter = 0; iter < max_iter; iter ++)
    {
        diff_max = 0.0;

        if (iter % 2 == 0)
            start = num_node - 1, end = -1, dir = -1;
        else
            start = 0, end = num_node, dir = +1;

        for (int p = start; p != end; p += dir)
        {
            bfs_node[p]->BeliefPropagation(&diff_max, this->labeled_given);
        }

        if (diff_max < 1e-6) break;
    }
}

void FactorGraph::CalculateMarginal()
{
    bool *zeroFlag = new bool[num_label];
    for (int i = 0; i < n; i ++)
    {
        double sum_py = 0.0;

        double minMarginal = 1e8;
        for (int y = 0; y < num_label; y ++)
        {
            zeroFlag[y] = false;
        }

        for (int y = 0; y < num_label; y ++)
        {
            if (var_node[i].state_factor[y] < 1e-8) {
                zeroFlag[y] = true;
            }
            var_node[i].marginal[y] = log(var_node[i].state_factor[y]);
            for (int t = 0; t < (int)var_node[i].neighbor.size(); t ++) {
                if (var_node[i].belief[t][y] < 1e-8) {
                    zeroFlag[y] = true;
                }
                // fprintf(stderr, "belief = %.4f\n", var_node[i].belief[t][y]);
                var_node[i].marginal[y] += log(var_node[i].belief[t][y]);
            }
            // fprintf(stderr, " zeroFlag = %d, marginal = %.4f\n", zeroFlag[y], var_node[i].marginal[y]);
            if (!zeroFlag[y]) {
                minMarginal = std::min(minMarginal, var_node[i].marginal[y]);
            }
        }

        for (int y = 0; y < num_label; y ++)
        {
            // fprintf(stderr, " zeroFlag = %d, marginal = %.4f\n", zeroFlag[y], var_node[i].marginal[y]);
            if (!zeroFlag[y]) {
                var_node[i].marginal[y] = exp(var_node[i].marginal[y] - minMarginal);
            } else {
                var_node[i].marginal[y] = 0;
            }
            // fprintf(stderr, " zeroFlag = %d, marginal = %.4f\n", zeroFlag[y], var_node[i].marginal[y]);
            sum_py += var_node[i].marginal[y];
        }

        // fprintf(stderr, " sum_py = %.4f\n", sum_py);

        for (int y = 0; y < num_label; y ++)
        {
            var_node[i].marginal[y] /= sum_py;
        }
    }

    delete []zeroFlag;

    for (int i = 0; i < m; i ++)
    {
        double sump = 0.0;
		if (SIZE(factor_node[i].neighbor) == 2)
		{
			for (int a = 0; a < num_label; a ++)
				for (int b = 0; b < num_label; b ++)
				{
					factor_node[i].marginal[a][b] =
							factor_node[i].belief[0][a]
						  * factor_node[i].belief[1][b]
						  * factor_node[i].func->GetValue(a, b);
					sump += factor_node[i].marginal[a][b];
				}
			for (int a = 0; a < num_label; a ++)
				for (int b = 0; b < num_label; b ++)
					factor_node[i].marginal[a][b] /= sump;
		}
		else
		{
			for (int a = 0; a < num_label; a ++)
				for (int b = 0; b < num_label; b ++)
					for (int c = 0; c < num_label; c ++)
					{
						factor_node[i].marginal3d[a][b][c] =
								factor_node[i].belief[0][a]
							  * factor_node[i].belief[1][b]
							  * factor_node[i].belief[2][c]
							  * factor_node[i].func->GetValue(a, b, c);
						sump += factor_node[i].marginal3d[a][b][c];
					}
			for (int a = 0; a < num_label; a ++)
				for (int b = 0; b < num_label; b ++)
					for (int c = 0; c < num_label; c ++)
						factor_node[i].marginal3d[a][b][c] /= sump;
		}
    }
}

void FactorGraph::MaxSumPropagation(int max_iter)
{
    int start, end, dir;

    converged = false;
    for (int iter = 0; iter < max_iter; iter ++)
    {
        diff_max = 0;

        if (iter % 2 == 0)
            start = num_node - 1, end = -1, dir = -1;
        else
            start = 0, end = num_node, dir = +1;

        for (int p = start; p != end; p += dir)
        {
            bfs_node[p]->MaxSumPropagation(&diff_max, labeled_given);
        }

        if (diff_max < 1e-6) break;
    }
}

void FactorGraph::Clean()
{
    if (var_node) delete[] var_node;
    if (factor_node) delete[] factor_node;
    if (p_node) delete[] p_node;
    if (bfs_node) delete[] bfs_node;
}
