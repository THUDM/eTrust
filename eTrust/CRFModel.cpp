#include "CRFModel.h"
//#include "Transmitter.h"
#include "Constant.h"
#include <iostream>
#include <cstdio>
#include <cstring>

using namespace std;

/*
extern "C" {
    // interface to LBFGS optimization written in FORTRAN
    extern void lbfgs_(int * n, int * m, double * x, double * f, double * g,
		       int * diagco, double * diag, int * iprint, double * eps,
		       double * xtol, double * w, int * iflag);		       
}
*/

void CRFModel::InitTrain(Config* conf, DataSet* train_data)
{
    this->conf = conf;
    this->train_data = train_data;

    num_sample = train_data->num_sample;
    num_label = train_data->num_label;
    num_attrib_type = train_data->num_attrib_type;
    num_edge_type = train_data->num_edge_type;
	num_triangle_type = 3;

    GenFeature();
    lambda = new double[num_feature];
    // Initialize parameters
    for (int i = 0; i < num_feature; i ++)
        lambda[i] = 0.0;
    SetupFactorGraphs();
}

void CRFModel::GenFeature()
{
    num_feature = 0;

    // state feature: f(y, x)
    num_attrib_parameter = num_label * num_attrib_type;
    num_feature += num_attrib_parameter;

    // edge feature: f(edge_type, y1, y2)
    edge_feature_offset.clear();
    int offset = 0;
    for (int y1 = 0; y1 < num_label; y1 ++)
        for (int y2 = y1; y2 < num_label; y2 ++)
        {
            edge_feature_offset[y1 * num_label + y2] = offset;
			edge_feature_offset[y2 * num_label + y1] = offset;
            offset ++;
        }
    num_edge_feature_each_type = offset;
    num_feature += num_edge_type * num_edge_feature_each_type;

	// triangle feature: f(triangle_type, y1, y2, y3)
	for (int i = 0; i < num_triangle_type; i++)
		triangle_feature_offset[i].clear();
	offset = 0;
	for (int y1 = 0; y1 < num_label; y1 ++)
		for (int y2 = y1; y2 < num_label; y2 ++)
			for (int y3 = y2; y3 < num_label; y3 ++)
			{
				triangle_feature_offset[0][y1 * num_label * num_label + y2 * num_label + y3] = offset;
				triangle_feature_offset[0][y1 * num_label * num_label + y3 * num_label + y2] = offset;
				triangle_feature_offset[0][y2 * num_label * num_label + y1 * num_label + y3] = offset;
				triangle_feature_offset[0][y2 * num_label * num_label + y3 * num_label + y1] = offset;
				triangle_feature_offset[0][y3 * num_label * num_label + y1 * num_label + y2] = offset;
				triangle_feature_offset[0][y3 * num_label * num_label + y2 * num_label + y1] = offset;				
				offset++;
			}
	num_triangle_feature[0] = offset;
	offset = 0;
	for (int y1 = 0; y1 < num_label; y1 ++)
		for (int y2 = 0; y2 < num_label; y2 ++)
			for (int y3 = 0; y3 < num_label; y3 ++)
			{
				triangle_feature_offset[1][y1 * num_label * num_label + y2 * num_label + y3] = offset;
				offset++;
			}
	num_triangle_feature[1] = offset;
	offset = 0;
	for (int y1 = 0; y1 < num_label; y1 ++)
		for (int y2 = y1; y2 < num_label; y2 ++)
			for (int y3 = y2; y3 < num_label; y3 ++)
			{
				triangle_feature_offset[2][y1 * num_label * num_label + y2 * num_label + y3] = offset;
				triangle_feature_offset[2][y1 * num_label * num_label + y3 * num_label + y2] = offset;
				triangle_feature_offset[2][y2 * num_label * num_label + y1 * num_label + y3] = offset;
				triangle_feature_offset[2][y2 * num_label * num_label + y3 * num_label + y1] = offset;
				triangle_feature_offset[2][y3 * num_label * num_label + y1 * num_label + y2] = offset;
				triangle_feature_offset[2][y3 * num_label * num_label + y2 * num_label + y1] = offset;				
				offset++;
			}
	num_triangle_feature[2] = offset;
	for (int i = 0; i < num_triangle_type; i++)
		num_feature += num_triangle_feature[i];
}

void CRFModel::SetupFactorGraphs()
{
    double* p_lambda = lambda + num_attrib_parameter;
    edge_func_list = new EdgeFactorFunction*[ num_edge_type ];
    for (int i = 0; i < num_edge_type; i ++)
    {
        edge_func_list[i] = new EdgeFactorFunction(num_label, p_lambda, &edge_feature_offset);
        p_lambda += num_edge_feature_each_type;
    }

	triangle_func_list = new EdgeFactorFunction*[ num_triangle_type ];
	for (int i = 0; i < num_triangle_type; i ++)
	{
		triangle_func_list[i] = new EdgeFactorFunction(num_label, p_lambda, &triangle_feature_offset[i]);
		p_lambda += num_triangle_feature[i];
	}

    sample_factor_graph = new FactorGraph[num_sample];
    for (int s = 0; s < num_sample; s ++)
    {
        DataSample* sample = train_data->sample[s];

		//sample->num_edge = 0;
		//sample->num_triangle = 0;

        int n = sample->num_node;
        int m = sample->num_edge;
		int o = sample->num_triangle;

		sample_factor_graph[s].InitGraph(n, m + o, num_label);

        // Add node info
        for (int i = 0; i < n; i ++)
        {
            sample_factor_graph[s].SetVariableLabel(i, sample->node[i]->label);
            sample_factor_graph[s].var_node[i].label_type = sample->node[i]->label_type;
        }   
        
        // Add edge info
        for (int i = 0; i < m; i ++)
        {
			int a = sample->edge[i]->a;
			int b = sample->edge[i]->b;
			sample_factor_graph[s].AddEdge(a, b, edge_func_list[sample->edge[i]->edge_type]);
        }

		// Add triangle info
		for (int i = 0; i < o; i++)
		{
			int a = sample->triangle[i]->a;
			int b = sample->triangle[i]->b;
			int c = sample->triangle[i]->c;
			sample_factor_graph[s].AddTriangle(a, b, c, triangle_func_list[sample->triangle[i]->triangle_type]);
		}

        sample_factor_graph[s].GenPropagateOrder();
    }
}

bool isLastIter = false;

void CRFModel::Train()
{    
    double* gradient;
    double  f;          // log-likelihood

    gradient = new double[num_feature + 1];

    // Master node
    if (conf->my_rank == 0)
    {
        ///// Initilize all info

        // Data Varible         
        double  old_f = 0.0;

        // Variable for lbfgs optimization
        int     m_correlation = 3;
        double* work_space = new double[num_feature * (2 * m_correlation + 1) + 2 * m_correlation];
        int     diagco = 0;
        double* diag = new double[num_feature];
        int     iprint[2] = {-1, 0}; // do not print anything
        double  eps = conf->eps;
        double  xtol = 1.0e-16;
        int     iflag = 0;
    
        // Other Variables
        int     num_iter;
        double  *tmp_store = new double[num_feature + 1];
        
        // Main-loop of CRF
        // Paramater estimation via L-BFGS
        num_iter = 0;
        do {
            num_iter ++;

            isLastIter = (num_iter == conf->max_iter);
                        
            // Step A. Send lambda to all procs
            //Transmitter::Master_SendDoubleArray(lambda, num_feature, conf->num_procs);

            // Step B. Calc gradient and log-likehood of the local datas
            f = CalcGradient(gradient);

            // Step C. Collect gradient and log-likehood from all procs
            //Transmitter::Master_CollectGradientInfo(gradient, &f, num_feature, tmp_store, conf->num_procs);

            // Step 4. Opitmization by L-BFGS
            printf("[Iter %3d] log-likelihood : %.8lf\n", num_iter, f);
            fflush(stdout);

            // If diff of log-likelihood is small enough, break.
            if (fabs(old_f - f) < eps) break;
            old_f = f;
		
		    // Negate f and gradient vector because the LBFGS optimization below minimizes the ojective function while we would like to maximize it
            f *= -1;
            for (int i = 0; i < num_feature; i ++)
                gradient[i] *= -1;

            // Invoke L-BFGS

            if (conf->optimization_method == LBFGS)
            {
                //lbfgs_(&num_feature, &m_correlation, lambda, &f, gradient, &diagco, diag, iprint, &eps, &xtol, work_space, &iflag);

                // Checking after calling LBFGS
                if (iflag < 0) // LBFGS error
		        {
		            fprintf(stderr, "LBFGS routine encounters an error\n");
		            break;
		        }
            }
            else
            {
                // Normalize Graident
                double g_norm = 0.0;
                for (int i = 0; i < num_feature; i ++)
                    g_norm += gradient[i] * gradient[i];
                g_norm = sqrt(g_norm);
                
                if (g_norm > 1e-8)
                {
                    for (int i = 0; i < num_feature; i ++)
                        gradient[i] /= g_norm;
                }

                for (int i = 0; i < num_feature; i ++)
                    lambda[i] -= gradient[i] * conf->gradient_step;
                iflag = 1;
            }

            if (conf->eval_each_iter && conf->my_rank == 0)
            {
                //SelfEvaluate();
            }
        } while (iflag != 0 && num_iter < conf->max_iter);

        //Transmitter::Master_SendQuit(conf->num_procs);

        delete[] tmp_store;

        delete[] work_space;
        delete[] diag;
    }
    else
    {
	/*
        bool done;

        while (1)
        {
            done = Transmitter::Slave_RecvDoubleArray(lambda, num_feature);            
            if (done) break;

            f = CalcGradient(gradient);

            Transmitter::Slave_SendGradientInfo(gradient, &f, num_feature);
        }
	*/
    }

    delete[] gradient;
}

double CRFModel::CalcGradient(double* gradient)
{
    double  f;
        
    // Initialize

    // If there is a square penalty, gradient should be initialized with (- lambda[i] / sigma^2). 
    // f should be accordingly modified as : -||lambda||^2/ (2*sigma^2)
    // note : should be added only in one procs (master)

    f = 0.0;
    for (int i = 0; i < num_feature; i ++)
    {
        //gradient[i] = - lambda[i] / conf->penalty_sigma_square;
        gradient[i] = 0; // no penalty
    }

    // Calculation
    for (int i = 0; i < num_sample; i ++)
    {
        // double t = CalcGradientForSample(train_data->sample[i], &sample_factor_graph[i], gradient);
        double t = CalcPartialLabeledGradientForSample(train_data->sample[i], &sample_factor_graph[i], gradient);
        f += t;
    }
    
    return f;
}

DataSample *cmp_sample;
int *cmp_inf_label;
double **cmp_prob;
int cmp_label;
// bool trust_cmp(int i, int j) {
//     bool a1 = (cmp_sample->node[i]->label_type == Enum::KNOWN_LABEL);
//     bool a2 = (cmp_sample->node[j]->label_type == Enum::KNOWN_LABEL);
//     if (a1 != a2)
//         return a1 < a2;
//     bool b1 = (cmp_sample->node[i]->label == -1);
//     bool b2 = (cmp_sample->node[j]->label == -1);
//     if (b1 != b2)
//         return b1 < b2;
//     bool c1 = (cmp_inf_label[i] == 0);
//     bool c2 = (cmp_inf_label[j] == 0);
//     if (c1 != c2)
//         return c1 < c2;
//     return (cmp_sample->node[i]->u < cmp_sample->node[j]->u) || (cmp_sample->node[i]->u == cmp_sample->node[j]->u && cmp_trust_value[i] > cmp_trust_value[j]);
// }

bool trust_cmp_new(int i, int j) {
    bool a1 = (cmp_sample->node[i]->label_type == Enum::KNOWN_LABEL);
    bool a2 = (cmp_sample->node[j]->label_type == Enum::KNOWN_LABEL);
    if (a1 != a2)
        return a1 < a2;
    bool b1 = (cmp_sample->node[i]->label == -1);
    bool b2 = (cmp_sample->node[j]->label == -1);
    if (b1 != b2)
        return b1 < b2;
    return cmp_prob[i][cmp_label] > cmp_prob[j][cmp_label];
}

double CRFModel::CalcPartialLabeledGradientForSample(DataSample* sample, FactorGraph* factor_graph, double* gradient)
{   
    int n = sample->num_node;
    int m = sample->num_edge;

    //****************************************************************
    // Belief Propagation 1: labeled data are given.
    //****************************************************************

    factor_graph->labeled_given = true;
    factor_graph->ClearDataForSumProduct();

    // Set state_factor
    for (int i = 0; i < n; i ++)
    {
        double* p_lambda = lambda;
        for (int y = 0; y < num_label; y ++)
        {
            if (sample->node[i]->label_type == Enum::KNOWN_LABEL && y != sample->node[i]->label
			 || !same_type_label(num_label, y, sample->node[i]->label))
            {
                factor_graph->SetVariableStateFactor(i, y, 0);
            }
            else
            {
                double v = 1;
                for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                    v *= exp( p_lambda[ sample->node[i]->attrib[t] ] * sample->node[i]->value[t] );                
                factor_graph->SetVariableStateFactor(i, y, v);
            }
            p_lambda += num_attrib_type;
        }
    }

    factor_graph->BeliefPropagation(conf->max_bp_iter);
    factor_graph->CalculateMarginal();    

    /***
     * Gradient = E_{Y|Y_L} f_i - E_{Y} f_i
     */

    // calc gradient part : + E_{Y|Y_L} f_i
    for (int i = 0; i < n; i ++)
    {
        for (int y = 0; y < num_label; y ++)
        {
            for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                gradient[ GetAttribParameterId(y, sample->node[i]->attrib[t]) ] += sample->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
        }
    }

    for (int i = 0; i < factor_graph->m; i ++)
    {
		if (SIZE(factor_graph->factor_node[i].neighbor) == 2)
		{
			for (int a = 0; a < num_label; a ++)
				for (int b = 0; b < num_label; b ++)
				{
					gradient[ GetEdgeParameterId(sample->edge[i]->edge_type, a, b) ] += factor_graph->factor_node[i].marginal[a][b];
				}
		}
		else
		{
			for (int a = 0; a < num_label; a ++)
				for (int b = 0; b < num_label; b ++)
					for (int c = 0; c < num_label; c ++)
					{
						gradient[ GetTriangleParameterId(sample->triangle[i - m]->triangle_type, a, b, c) ] += factor_graph->factor_node[i].marginal3d[a][b][c];
					}
		}
    }

    //****************************************************************
    // Belief Propagation 2: labeled data are not given.
    //****************************************************************

    factor_graph->ClearDataForSumProduct();
    factor_graph->labeled_given = false;

    for (int i = 0; i < n; i ++)
    {
        double* p_lambda = lambda;
        for (int y = 0; y < num_label; y ++)
        {
            double v = 1;
            for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                v *= exp( p_lambda[ sample->node[i]->attrib[t] ] * sample->node[i]->value[t] );
            factor_graph->SetVariableStateFactor(i, y, v);
            p_lambda += num_attrib_type;
        }
    }    

    factor_graph->BeliefPropagation(conf->max_bp_iter);
    factor_graph->CalculateMarginal();
        
    // calc gradient part : - E_{Y} f_i
    for (int i = 0; i < n; i ++)
    {
        for (int y = 0; y < num_label; y ++)
        {
            for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                gradient[ GetAttribParameterId(y, sample->node[i]->attrib[t]) ] -= sample->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
        }
    }
    for (int i = 0; i < factor_graph->m; i ++)
    {
		if (SIZE(factor_graph->factor_node[i].neighbor) == 2)
		{
			for (int a = 0; a < num_label; a ++)
				for (int b = 0; b < num_label; b ++)
				{
					gradient[ GetEdgeParameterId(sample->edge[i]->edge_type, a, b) ] -= factor_graph->factor_node[i].marginal[a][b];
				}
		}
		else
		{
			for (int a = 0; a < num_label; a ++)
				for (int b = 0; b < num_label; b ++)
					for (int c = 0; c < num_label; c ++)
					{
						gradient[ GetTriangleParameterId(sample->triangle[i - m]->triangle_type, a, b, c) ] -= factor_graph->factor_node[i].marginal3d[a][b][c];
					}
		}
    }
    
    // Calculate gradient & log-likelihood
    double f = 0.0, Z = 0.0;

    // \sum \lambda_i * f_i
    for (int i = 0; i < n; i ++)
    {
        int y = sample->node[i]->label;
        if (y == -1) {
            continue;
        }
        for (int t = 0; t < sample->node[i]->num_attrib; t ++)
            f += lambda[this->GetAttribParameterId(y, sample->node[i]->attrib[t])] * sample->node[i]->value[t];
    }

	for (int i = 0; i < m; i ++)
	{
		int a = sample->node[ sample->edge[i]->a ]->label;
		int b = sample->node[ sample->edge[i]->b ]->label;
        if (a == -1 || b == -1) {
            continue;
        }      
		f += lambda[this->GetEdgeParameterId(sample->edge[i]->edge_type, a, b)];
	}

	for (int i = 0; i < sample->num_triangle; i ++)
	{
		int a = sample->node[ sample->triangle[i]->a ]->label;
		int b = sample->node[ sample->triangle[i]->b ]->label;
		int c = sample->node[ sample->triangle[i]->c ]->label;
        if (a == -1 || b == -1 || c == -1) {
            continue;
        }
		f += lambda[this->GetTriangleParameterId(sample->triangle[i]->triangle_type, a, b, c)];
	}

    //fprintf(stderr, "  f = %.4f\n", f);

    // calc log-likelihood
    //  using Bethe Approximation
    for (int i = 0; i < n; i ++)
    {
        for (int y = 0; y < num_label; y ++)
        {
            for (int t = 0; t < sample->node[i]->num_attrib; t ++) {
                Z += lambda[this->GetAttribParameterId(y, sample->node[i]->attrib[t])] * sample->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
                //fprintf(stderr, "node: %.4f %.4f %.4f\n", lambda[this->GetAttribParameterId(y, sample->node[i]->attrib[t])], sample->node[i]->value[t], factor_graph->var_node[i].marginal[y]);
            }
        }
        //fprintf(stderr, "node:  Z = %.4f\n", Z);
    }
    for (int i = 0; i < factor_graph->m; i ++)
    {
		if (SIZE(factor_graph->factor_node[i].neighbor) == 2)
		{
			for (int a = 0; a < num_label; a ++)
				for (int b = 0; b < num_label; b ++)
				{
					Z += lambda[this->GetEdgeParameterId(sample->edge[i]->edge_type, a, b)] * factor_graph->factor_node[i].marginal[a][b];
				}
            //fprintf(stderr, "edge:  Z = %.4f\n", Z);
		}
		else
		{
			for (int a = 0; a < num_label; a++)
				for (int b = 0; b < num_label; b++)
					for (int c = 0; c < num_label; c++)
					{
						Z += lambda[this->GetTriangleParameterId(sample->triangle[i - m]->triangle_type, a, b, c)] * factor_graph->factor_node[i].marginal3d[a][b][c];
					}
            //fprintf(stderr, "triangle:  Z = %.4f\n", Z);
		}
    }

    //fprintf(stderr, "  Z = %.4f\n", Z);

    // Edge entropy
    for (int i = 0; i < factor_graph->m; i ++)
    {
        double h_e = 0.0;
		if (SIZE(factor_graph->factor_node[i].neighbor) == 2)
		{
			for (int a = 0; a < num_label; a ++)
				for (int b = 0; b < num_label; b ++)
				{
					if (factor_graph->factor_node[i].marginal[a][b] > 1e-10)
						h_e += - factor_graph->factor_node[i].marginal[a][b] * log(factor_graph->factor_node[i].marginal[a][b]);
				}
		}
		else
		{
			for (int a = 0; a < num_label; a++)
				for (int b = 0; b < num_label; b++)
					for (int c = 0; c < num_label; c++)
					{
						if (factor_graph->factor_node[i].marginal3d[a][b][c] > 1e-10)
							h_e += - factor_graph->factor_node[i].marginal3d[a][b][c] * log(factor_graph->factor_node[i].marginal3d[a][b][c]);
					}
		}
        Z += h_e;
    }
    // Node entroy
    for (int i = 0; i < n; i ++)
    {
        double h_v = 0.0;
        for (int a = 0; a < num_label; a ++)
            if (fabs(factor_graph->var_node[i].marginal[a]) > 1e-10)
                h_v += - factor_graph->var_node[i].marginal[a] * log(factor_graph->var_node[i].marginal[a]);
        Z -= h_v * ((int)factor_graph->var_node[i].neighbor.size() - 1);
    }

    f -= Z;
    
    // Let's take a look of current accuracy

    factor_graph->ClearDataForMaxSum();
    factor_graph->labeled_given = true;

    for (int i = 0; i < n; i ++)
    {
        double* p_lambda = lambda;

        for (int y = 0; y < num_label; y ++)
        {
            double v = 1.0;
            for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                v *= exp( p_lambda[ sample->node[i]->attrib[t] ] * sample->node[i]->value[t] );
            factor_graph->SetVariableStateFactor(i, y, v);

            p_lambda += num_attrib_type;
        }
    }    

    factor_graph->MaxSumPropagation(conf->max_bp_iter);

    int* inf_label = new int[n];
    double** inf_prob = new double*[n];
    for (int i = 0; i < n; i ++)
    {
        inf_prob[i] = new double[num_label];

        int ybest = -1;
        double vbest, v;

        double exp_sum = 0.0;
        for (int y = 0; y < num_label; y ++)
        {
            v = factor_graph->var_node[i].state_factor[y];
            for (int t = 0; t < (int)factor_graph->var_node[i].neighbor.size(); t ++)
                v *= factor_graph->var_node[i].belief[t][y];
            if (same_type_label(num_label, y, sample->node[i]->label) && (ybest < 0 || v > vbest))
                ybest = y, vbest = v;
            
            inf_prob[i][y] = v;

            // exp_sum += exp(v);
            exp_sum += v;
        }
        inf_label[i] = ybest;
        for (int y = 0; y < num_label; y ++)
        {
            // inf_prob[i][y] = exp(inf_prob[i][y]) / exp_sum;
            inf_prob[i][y] /= exp_sum;
        }
    }

    int hit = 0, miss = 0;
    int hitu = 0, missu = 0;

	int cnt[50][50];
	int ucnt[50][50];

	memset(cnt, 0, sizeof(cnt));
	memset(ucnt, 0, sizeof(ucnt));

	//FILE *pred_output = fopen("prediction.txt", "w");

    for (int i = 0; i < n; i ++)
    {
		//fprintf(pred_output, "%d %d\n", sample->node[i]->label_type, inf_label[i]);
        if (sample->node[i]->label != -1) {
            if (inf_label[i] == sample->node[i]->label)
                hit ++;
            else
                miss ++;

            cnt[ inf_label[i] ][ sample->node[i]->label ] ++;

            if (sample->node[i]->label_type == Enum::UNKNOWN_LABEL)
            {
                if (inf_label[i] == sample->node[i]->label)
                    hitu ++;
                else
                    missu ++;

                ucnt[ inf_label[i] ][ sample->node[i]->label ] ++;
            }
        }
    }

    //printf("HIT = %4d, MISS = %4d, All_Accuracy = %.5lf Unknown_Accuracy = %.5lf\n", hit, miss, (double)hit / (hit + miss), (double)hitu / (hitu + missu));
	
	printf("A_HIT  = %4d, U_HIT = %4d\n", hit, hitu);
	printf("A_MISS = %4d, U_MISS = %4d\n", miss, missu);
/*
	for (int i = 0; i < num_label; ++i)
		printf("%s ", g_dataset->label_dict.GetKeyWithId(i).c_str());
	printf("\n");
*/

	printf("A_Accuracy  = %.4lf\n", (double)hit / (hit + miss));
	for (int k = 0; k < num_label; ++k)
	{
		double srow = 0, scolumn = 0;
		for (int i = 0; i < num_label; ++i)
		{
			srow += cnt[k][i];
			scolumn += cnt[i][k];
		}
		double a_precision = (double)cnt[k][k] / srow;
		double a_recall = (double)cnt[k][k] / scolumn;
		double a_fmeasure = 2.0 * a_precision * a_recall / (a_precision + a_recall);
		printf("%d   %.4lf   %.4lf   %.4lf\n", k, a_precision, a_recall, a_fmeasure);
	}
	printf("U_Accuracy  = %.4lf\n", (double)hitu / (hitu + missu));
	for (int k = 0; k < num_label; ++k)
	{
		double srow = 0, scolumn = 0;
		for (int i = 0; i < num_label; ++i)
		{
			srow += ucnt[k][i];
			scolumn += ucnt[i][k];
		}
		double u_precision = (double)ucnt[k][k] / srow;
		double u_recall = (double)ucnt[k][k] / scolumn;
		double u_fmeasure = 2.0 * u_precision * u_recall / (u_precision + u_recall);
		printf("%d   %.4lf   %.4lf   %.4lf\n", k, u_precision, u_recall, u_fmeasure);
	}

    // std::cout << "Calc Performance" << std::endl;

    int accuracy = 0, total = 0;
    for (int i = 0; i < n; ++i) {
        if (sample->node[i]->label_type == Enum::KNOWN_LABEL || sample->node[i]->label == -1)
            continue;
        if (sample->node[i]->label == inf_label[i]) {
            ++accuracy;
        }
        ++total;
    }
    fprintf(stdout, "hit = %d, total = %d\n", accuracy, total);
    fprintf(stdout, "accuracy = %.2f%%\n", accuracy * 100.0 / total);

    cmp_sample = sample;
    cmp_inf_label = inf_label;
    cmp_prob = inf_prob;

    char fileName[50];
    for (int z = 1; z < num_label; ++z) {
        sprintf(fileName, "P@_fgm_%d.txt", z);

        vector<int> index;
        for (int i = 0; i < n; ++i) {
            index.push_back(i);
        }
        cmp_label = z;
        sort(index.begin(), index.end(), trust_cmp_new);

        int counts[30];
        double precision[30];
        int total = 0;
        for (int i = 0; i < n; ++i) {
            total += (sample->node[i]->label_type == Enum::UNKNOWN_LABEL && sample->node[i]->label != -1);
        }
        for (int i = 1; i < 10; ++i) {
            counts[i] = (int)(total * 0.01 * i);
        }
        for (int i = 10; i < 20; ++i) {
            counts[i] = (int)(total * 0.1 * (i - 9));
        }

        int num = 0, pos_num = 0, iter = 1;
        for (int j = 0; j < n; ++j) {
            int i = index[j];
            if (sample->node[i]->label_type == Enum::KNOWN_LABEL || sample->node[i]->label == -1)
                break;
            // fprintf(stdout, "%d %d %.4f\n", j, sample->node[i]->label,inf_prob[i][z]);
            ++num;
            if (sample->node[i]->label == z) {
                ++pos_num;
            }
            if (num == counts[iter]) {
                precision[iter] = pos_num * 1.0 / num;
                ++iter;
            }
        }

        FILE *file = fopen(fileName, "w");
        // for (int i = 0; i < n; ++i) {
        //     if (sample->node[i]->label_type == Enum::KNOWN_LABEL || sample->node[i]->label == -1)
        //         continue;
        //     fprintf(file, "%d %.6f\n", sample->node[i]->label == z, inf_prob[i][z]);
        // }
        for (int i = 1; i < 20; ++i) {
            fprintf(file, "%.4f\n", precision[i]);
        }
        fclose(file);
    }
    

    double MAP = 0.0, AP;
    int num, pos_num;

    for (int z = 0; z < num_label; ++z)
    {
        // if (num_label == 2 && z == 0) {
        if (z == 0) {
            continue;
        }
        vector<int> index;
        for (int i = 0; i < n; ++i) {
            index.push_back(i);
        }
        cmp_label = z;
        sort(index.begin(), index.end(), trust_cmp_new);

        AP = 0.0;
        num = pos_num = 0;
        for (int j = 0; j < n; ++j) {
            int i = index[j];
            if (sample->node[i]->label_type == Enum::KNOWN_LABEL || sample->node[i]->label == -1) {
                break;
            }
            ++num;
            if (sample->node[i]->label == z ){
                ++pos_num;
                AP += pos_num * 1.0 / num;
            }
        }
        AP /= pos_num;
        MAP += AP;
    }
    // MAP /= (num_label - (num_label == 2));
    MAP /= (num_label - 1);
    //fprintf(stdout, "total = %d, MAP = %.2f%%\n", test_total, MAP * 100);
    fprintf(stdout, "MAP = %.2f%%\n", MAP * 100);

    double mean_auc = 0.0, auc;
    for (int z = 0; z < num_label; ++z)
    {
        // if (num_label == 2 && z == 0) {
        if (z == 0) {
            continue;
        }
        vector<int> index;
        for (int i = 0; i < n; ++i) {
            index.push_back(i);
        }
        cmp_label = z;
        sort(index.begin(), index.end(), trust_cmp_new);

        num = pos_num = 0;
        auc = 0.0;
        for (int j = 0; j < n; ++j) {
            int i = index[j];
            if (sample->node[i]->label_type == Enum::KNOWN_LABEL || sample->node[i]->label == -1) {
                break;
            }
            if (sample->node[i]->label == z) {
                ++pos_num;
                auc -= j;
            }
            ++num;
        }
        auc -= pos_num * (pos_num + 1) / 2;
        auc += (num * pos_num);
        auc /= (pos_num * (num - pos_num));

        mean_auc += auc;
    }
    // mean_auc /= (num_label - (num_label == 2));
    mean_auc /= (num_label - 1);
    fprintf(stdout, "AUC = %.2f%%\n", mean_auc * 100);

    double macroP, macroR, macroF1;
    macroP = macroR = macroF1 = 0.0;
    for (int z = 0; z < num_label; ++z) {
        // if (num_label == 2 && z == 0) {
        if (z == 0) {
            continue;
        }
        int A = 0, B = 0, C = 0;
        for (int i = 0; i < n; i ++)
        {
            if (sample->node[i]->label_type == Enum::UNKNOWN_LABEL && sample->node[i]->label != -1) {
                if (inf_label[i] == z) {
                    if (sample->node[i]->label == z) {
                        ++A;
                    } else {
                        ++B;
                    }
                } else if (sample->node[i]->label == z) {
                    ++C;
                }
            }
        }
        double P = A * 1.0 / (A + B);
        double R = A * 1.0 / (A + C);
        double F1 = 2 * P * R / (P + R);

        macroP += P;
        macroR += R;
        macroF1 += F1;
    }
    // macroP /= (num_label - (num_label == 2));
    // macroR /= (num_label - (num_label == 2));
    // macroF1 /= (num_label - (num_label == 2));
    macroP /= (num_label - 1);
    macroR /= (num_label - 1);
    macroF1 /= (num_label - 1);
	fprintf(stdout, "P = %.2f%%  R = %.2f%%  F1 = %.2f%%\n", macroP * 100, macroR * 100, macroF1 * 100);
	fprintf(stdout, "\n");

    if (isLastIter) {
        FILE *result_file = fopen(conf->result_file.c_str(), "a");
        fprintf(result_file, "%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\n", accuracy * 100.0 / total, macroP * 100, macroR * 100, macroF1 * 100, MAP * 100, mean_auc * 100);
        fclose(result_file);
    }

	//fclose(pred_output);
    fflush(stdout);

	delete[] inf_label;
    for (int i = 0; i < n; ++i) {
        delete[] inf_prob[i];
    }
    delete[] inf_prob;
    return f;
}

double CRFModel::CalcGradientForSample(DataSample* sample, FactorGraph* factor_graph, double* gradient)
{
    factor_graph->ClearDataForSumProduct();

    // Set state_factor
    int n = sample->num_node;
    int m = sample->num_edge;

    for (int i = 0; i < n; i ++)
    {
        double* p_lambda = lambda;
        for (int y = 0; y < num_label; y ++)
        {
            double v = 1;
            for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                v *= exp( p_lambda[ sample->node[i]->attrib[t] ] * sample->node[i]->value[t] );
            factor_graph->SetVariableStateFactor(i, y, v);
            p_lambda += num_attrib_type;
        }
    }    

    factor_graph->BeliefPropagation(conf->max_bp_iter);
    factor_graph->CalculateMarginal();
    
    // Calculate gradient & log-likelihood
    double f = 0.0, Z = 0.0;

    // \sum \lambda_i * f_i
    for (int i = 0; i < n; i ++)
    {
        int y = sample->node[i]->label;
        for (int t = 0; t < sample->node[i]->num_attrib; t ++)
            f += lambda[this->GetAttribParameterId(y, sample->node[i]->attrib[t])] * sample->node[i]->value[t];
    }
    for (int i = 0; i < m; i ++)
    {
        int a = sample->node[ sample->edge[i]->a ]->label;
        int b = sample->node[ sample->edge[i]->b ]->label;        
        f += lambda[this->GetEdgeParameterId(sample->edge[i]->edge_type, a, b)];
    }

    // calc log-likelihood
    //  using Bethe Approximation
    for (int i = 0; i < n; i ++)
    {
        for (int y = 0; y < num_label; y ++)
        {
            for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                Z += lambda[this->GetAttribParameterId(y, sample->node[i]->attrib[t])] * sample->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
        }
    }
    for (int i = 0; i < m; i ++)
    {
        for (int a = 0; a < num_label; a ++)
            for (int b = 0; b < num_label; b ++)
            {
                Z += lambda[this->GetEdgeParameterId(sample->edge[i]->edge_type, a, b)] * factor_graph->factor_node[i].marginal[a][b];
            }
    }
    // Edge entropy
    for (int i = 0; i < m; i ++)
    {
        double h_e = 0.0;
        for (int a = 0; a < num_label; a ++)
            for (int b = 0; b < num_label; b ++)
            {
                if (factor_graph->factor_node[i].marginal[a][b] > 1e-10)
                    h_e += - factor_graph->factor_node[i].marginal[a][b] * log(factor_graph->factor_node[i].marginal[a][b]);
            }
        Z += h_e;
    }
    // Node entroy
    for (int i = 0; i < n; i ++)
    {
        double h_v = 0.0;
        for (int a = 0; a < num_label; a ++)
            if (fabs(factor_graph->var_node[i].marginal[a]) > 1e-10)
                h_v += - factor_graph->var_node[i].marginal[a] * log(factor_graph->var_node[i].marginal[a]);
        Z -= h_v * ((int)factor_graph->var_node[i].neighbor.size() - 1);
    }

    f -= Z;
    fflush(stdout);

    // calc gradient
    for (int i = 0; i < n; i ++)
        for (int t = 0; t < sample->node[i]->num_attrib; t ++)
            gradient[ GetAttribParameterId(sample->node[i]->label, sample->node[i]->attrib[t]) ] += sample->node[i]->value[t];
    for (int i = 0; i < m; i ++)
        gradient[ GetEdgeParameterId(sample->edge[i]->edge_type, 
                                     sample->node[sample->edge[i]->a]->label, 
                                     sample->node[sample->edge[i]->b]->label) ] += 1.0;

    // - expectation
    for (int i = 0; i < n; i ++)
    {
        for (int y = 0; y < num_label; y ++)
        {
            for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                gradient[ GetAttribParameterId(y, sample->node[i]->attrib[t]) ] -= sample->node[i]->value[t] * factor_graph->var_node[i].marginal[y];
        }
    }
    for (int i = 0; i < m; i ++)
    {
        for (int a = 0; a < num_label; a ++)
            for (int b = 0; b < num_label; b ++)
            {
                gradient[ GetEdgeParameterId(sample->edge[i]->edge_type, a, b) ] -= factor_graph->factor_node[i].marginal[a][b];
            }
    }

    return f;
}

void CRFModel::SelfEvaluate()
{
    int ns = train_data->num_sample;
    int tot, hit;

    tot = hit = 0;
    for (int s = 0; s < ns; s ++)
    {
        DataSample* sample = train_data->sample[s];
        FactorGraph* factor_graph = &sample_factor_graph[s];
        
        int n = sample->num_node;
        int m = sample->num_edge;
        
        factor_graph->InitGraph(n, m, num_label);
        // Add edge info
        for (int i = 0; i < m; i ++)
        {
            factor_graph->AddEdge(sample->edge[i]->a, sample->edge[i]->b, edge_func_list[sample->edge[i]->edge_type]);
        }        
        factor_graph->GenPropagateOrder();

        factor_graph->ClearDataForMaxSum();

        for (int i = 0; i < n; i ++)
        {
            double* p_lambda = lambda;

            for (int y = 0; y < num_label; y ++)
            {
                double v = 1.0;
                for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                    v *= exp( p_lambda[ sample->node[i]->attrib[t] ] * sample->node[i]->value[t] );
                factor_graph->SetVariableStateFactor(i, y, v);

                p_lambda += num_attrib_type;
            }
        }    

        factor_graph->MaxSumPropagation(conf->max_bp_iter);

        int* inf_label = new int[n];
        for (int i = 0; i < n; i ++)
        {
            int ybest = -1;
            double vbest, v;

            for (int y = 0; y < num_label; y ++)
            {
                v = factor_graph->var_node[i].state_factor[y];
                for (int t = 0; t < (int)factor_graph->var_node[i].neighbor.size(); t ++)
                    v *= factor_graph->var_node[i].belief[t][y];
                if (ybest < 0 || v > vbest)
                    ybest = y, vbest = v;
            }

            inf_label[i] = ybest;
        }

        int curt_tot, curt_hit;
        curt_tot = curt_hit = 0;
        for (int i = 0; i < n; i ++)
        {   
            curt_tot ++;
            if (inf_label[i] == sample->node[i]->label) curt_hit ++;
        }
        
        printf("Accuracy %4d / %4d : %.6lf\n", curt_hit, curt_tot, (double)curt_hit / curt_tot);
        hit += curt_hit;
        tot += curt_tot;

        delete[] inf_label;
    }

    printf("Overall Accuracy %4d / %4d : %.6lf\n", hit, tot, (double)hit / tot);
}

void CRFModel::InitEvaluate(Config* conf, DataSet* test_data)
{
    this->conf = conf;
    this->test_data = test_data;
}

void CRFModel::Evalute()
{
    int ns = test_data->num_sample;
    int tot, hit;

    tot = hit = 0;


    FILE* fout = fopen(conf->pred_file.c_str(), "w");

    for (int s = 0; s < ns; s ++)
    {
        DataSample* sample = test_data->sample[s];
        FactorGraph* factor_graph = new FactorGraph();
        
		int n = sample->node.size();
		int m = sample->edge.size();
		int o = sample->triangle.size();

		factor_graph->InitGraph(n, m + o, num_label);

        // Add edge info
        for (int i = 0; i < m; i ++)
        {
			int a = sample->edge[i]->a;
			int b = sample->edge[i]->b;
			factor_graph->AddEdge(a, b, edge_func_list[sample->edge[i]->edge_type]);
        }

		// Add triangle info
		for (int i = 0; i < o; i++)
		{
			int a = sample->triangle[i]->a;
			int b = sample->triangle[i]->b;
			int c = sample->triangle[i]->c;
			factor_graph->AddTriangle(a, b, c, triangle_func_list[sample->triangle[i]->triangle_type]);
		}
        
        factor_graph->GenPropagateOrder();

        factor_graph->ClearDataForMaxSum();

        for (int i = 0; i < n; i ++)
        {
            double* p_lambda = lambda;

            for (int y = 0; y < num_label; y ++)
            {
                double v = 1.0;
                for (int t = 0; t < sample->node[i]->num_attrib; t ++)
                    v *= exp( p_lambda[ sample->node[i]->attrib[t] ] * sample->node[i]->value[t] );
                factor_graph->SetVariableStateFactor(i, y, v);

                p_lambda += num_attrib_type;
            }
        }    

        factor_graph->MaxSumPropagation(conf->max_bp_iter);

        int* inf_label = new int[n];
        for (int i = 0; i < n; i ++)
        {
            int ybest = -1;
            double vbest, v;

            for (int y = 0; y < num_label; y ++)
            {
                v = factor_graph->var_node[i].state_factor[y];
                for (int t = 0; t < (int)factor_graph->var_node[i].neighbor.size(); t ++)
                    v *= factor_graph->var_node[i].belief[t][y];
                if (same_type_label(num_label, y, sample->node[i]->label) && (ybest < 0 || v > vbest))
                    ybest = y, vbest = v;
            }

            inf_label[i] = ybest;
        }

        int curt_tot, curt_hit;
        curt_tot = curt_hit = 0;
		int cnt[50][50];
		memset(cnt, 0, sizeof(cnt));
        for (int i = 0; i < n; i ++)
        {   
            curt_tot ++;
            if (inf_label[i] == sample->node[i]->label) curt_hit ++;
			++cnt[inf_label[i]][sample->node[i]->label];
        }
        
        printf("Accuracy %4d / %4d : %.6lf\n", curt_hit, curt_tot, (double)curt_hit / curt_tot);
		for (int k = 0; k < num_label; ++k)
		{
			double srow = 0, scolumn = 0;
			for (int i = 0; i < num_label; ++i)
			{
				srow += cnt[k][i];
				scolumn += cnt[i][k];
			}
			double precision = (double)cnt[k][k] / srow;
			double recall = (double)cnt[k][k] / scolumn;
			double fmeasure = 2 * precision * recall / (precision + recall);
			printf("%d   %.4lf   %.4lf   %.4lf\n", k, precision, recall, fmeasure);
			/*
			printf("G_Accuracy  = %.4lf\n", (double)curt_hit / curt_tot);
			printf("G_Precision = %.4lf\n", precision);
			printf("G_Recall    = %.4lf\n", recall);
			printf("G_F1        = %.4lf\n", fmeasure);
			*/
			hit += curt_hit;
			tot += curt_tot;
		}

        // to zz: just print inf_labe[0]
        for (int i = 0; i < n; i ++)
        {
            fprintf(fout, "%d\n", inf_label[i]);
        }

        delete[] inf_label;

		delete factor_graph;
    }

    //printf("Overall Accuracy %4d / %4d : %.6lf\n", hit, tot, (double)hit / tot);
    fclose(fout);
}

void CRFModel::Clean()
{
    if (lambda) delete[] lambda;
    if (sample_factor_graph) delete[] sample_factor_graph;

    for (int i = 0; i < num_edge_type; i ++)
        delete edge_func_list[i];
    delete[] edge_func_list;
}

void CRFModel::SaveModel(const char* file_name)
{
    FILE* fout = fopen(file_name, "w");
    fprintf(fout, "%d\n", num_feature);
    for (int i = 0; i < num_feature; i ++)
        fprintf(fout, "%.6lf\n", lambda[i]);
    fclose(fout);
}

void CRFModel::LoadModel(const char* file_name)
{
    FILE* fin = fopen(file_name, "r");
    int num_feature_saved;
    fscanf(fin, "%d", &num_feature_saved);
    if (num_feature_saved != num_feature)
    {
        fprintf(stderr, "[error] model from %s not match\n", file_name);
        return;
    }
    for (int i = 0; i < num_feature; i ++)
        fscanf(fin, "%lf", &lambda[i]);
    fclose(fin);
}