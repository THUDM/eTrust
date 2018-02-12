#pragma once

#include "Config.h"
#include "DataSet.h"
#include "FactorGraph.h"

class EdgeFactorFunction: public FactorFunction
{
public:
    int     num_label;
    double* lambda;
    map<int, int>* feature_offset;

    EdgeFactorFunction(int num_label, double* p_lambda, map<int,int>* feature_offset)
    {
        this->num_label = num_label;
        this->lambda = p_lambda;
        this->feature_offset = feature_offset;
    }
    
    virtual double GetValue(int y1, int y2)
    {
        int i = (*feature_offset)[ y1 * num_label + y2 ];
        return exp ( lambda[i] );
    }

	virtual double GetValue(int a, int b, int c)
	{
		int i = (*feature_offset)[ a * num_label * num_label + b * num_label + c];
		return exp ( lambda[i] );
	}
};

class CRFModel
{
public:
    Config*     conf;
    DataSet*    train_data;
    DataSet*    test_data;

    int         num_sample;
    int         num_label;
    int         num_attrib_type;
    int         num_edge_type;
	int         num_triangle_type;
            
    int         num_feature;

    double      *lambda;
    FactorGraph *sample_factor_graph;

    int             num_attrib_parameter;
    int             num_edge_feature_each_type;
	int				num_triangle_feature[3];
    map<int, int>   edge_feature_offset;
	map<int, int>   triangle_feature_offset[3];
    EdgeFactorFunction**   edge_func_list;
	EdgeFactorFunction**   triangle_func_list;
	

    CRFModel(){}

    void InitTrain(Config* conf, DataSet* train_data);
    void GenFeature();
    void SetupFactorGraphs();

    void Train();
    double CalcGradient(double* gradient);
    double CalcGradientForSample(DataSample* sample, FactorGraph* factor_graph, double* gradient);
    double CalcPartialLabeledGradientForSample(DataSample* sample, FactorGraph* factor_graph, double* gradient);

    void SelfEvaluate();
    
    void InitEvaluate(Config* conf, DataSet* test_data);
    void Evalute();

    int GetAttribParameterId(int y, int x){ return y * num_attrib_type + x; }

    int GetEdgeParameterId(int edge_type, int a, int b)
    { 
        int offset = edge_feature_offset[ a * num_label + b ];
        return num_attrib_parameter + edge_type * num_edge_feature_each_type + offset;
    }

	int GetTriangleParameterId(int triangle_type, int a, int b, int c)
	{
		int offset = triangle_feature_offset[triangle_type][a * num_label * num_label + b * num_label + c];
		int id = num_attrib_parameter + num_edge_type * num_edge_feature_each_type + offset;
		for (int i = 0; i < triangle_type; ++i)
			id += num_triangle_feature[i];
		return id;
	}

    ~CRFModel() { Clean(); }
    void Clean();

    void SaveModel(const char* file_name);
    void LoadModel(const char* file_name);
};