#include "Config.h"
#include "DataSet.h"
#include "CRFModel.h"
#include "Constant.h"

#include <sys/time.h>
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <cstring>

using namespace std;

void MakeEvaluate(Config* conf, GlobalDataSet* g_dataset, CRFModel* model)
{
    if (conf->my_rank == 0)
    {
        GlobalDataSet* g_testdata = new GlobalDataSet();
        DataSet* testdata = new DataSet();        

        g_testdata->LoadDataWithDict(conf->test_file.c_str(), conf, g_dataset->label_dict, g_dataset->attrib_dict, g_dataset->edge_type_dict);
        testdata->sample = g_testdata->sample;
        testdata->num_sample = g_testdata->sample.size();
        testdata->num_edge_type = g_testdata->num_edge_type;
        testdata->num_attrib_type = g_testdata->num_attrib_type;
        testdata->num_label = g_testdata->num_label;

        for (int i = 0; i < testdata->num_sample; i ++)
        {        
            testdata->sample[i]->num_node = testdata->sample[i]->node.size();
            testdata->sample[i]->num_edge = testdata->sample[i]->edge.size();        

            for (int t = 0; t < testdata->sample[i]->num_node; t ++)
                testdata->sample[i]->node[t]->num_attrib = testdata->sample[i]->node[t]->attrib.size();
        }


        model->InitEvaluate(conf, testdata);
        model->Evalute();
    }
}

void WriteInt(char* mem, int& mem_p, int v)
{
    *((int*)(mem + mem_p)) = v;
    mem_p += sizeof(int);
}

void WriteDouble(char* mem, int& mem_p, double v)
{
    *((double*)(mem + mem_p)) = v;
    mem_p += sizeof(double);
}

int ReadInt(char* mem, int& mem_p)
{
    int v = *((int*)(mem + mem_p));
    mem_p += sizeof(int);
    return v;
}

double ReadDouble(char* mem, int& mem_p)
{
    double v = *((double*)(mem + mem_p));
    mem_p += sizeof(double);
    return v;
}


DataSet* AssignJobs(GlobalDataSet* g_dataset, int num_procs)
{
    int tot_sample = g_dataset->sample.size();    
    int avg_sample = tot_sample / num_procs;

    // Assign Jobs
    int* start = new int[num_procs + 1];
    for (int i = 0; i < num_procs; i ++)
        start[i] = i * avg_sample;
    start[num_procs] = tot_sample;

    // Calc the maximum buffer size
    int max_space = 0;
    int curt_space;
    for (int p = 1; p < num_procs; p ++)
    {
        curt_space = 0;
        for (int i = start[p]; i < start[p + 1]; i ++)
            curt_space += g_dataset->sample[i]->GetSize();
        if (curt_space > max_space)
            max_space = curt_space;
    }
    max_space += 1024;

    // Send the buffer size

    char *mem = new char[max_space];
    int mem_p;

    for (int p = 1; p < num_procs; p ++)
    {
        mem_p = 0;
        WriteInt(mem, mem_p, start[p + 1] - start[p]);        

        WriteInt(mem, mem_p, g_dataset->num_label);
        WriteInt(mem, mem_p, g_dataset->num_attrib_type);
        WriteInt(mem, mem_p, g_dataset->num_edge_type);

        for (int i = start[p]; i < start[p + 1]; i ++)
        {
            DataSample* p_sample = g_dataset->sample[i];

            p_sample->num_node = p_sample->node.size();
            p_sample->num_edge = p_sample->edge.size();
            WriteInt(mem, mem_p, p_sample->num_node);
            WriteInt(mem, mem_p, p_sample->num_edge);

            for (int t = 0; t < p_sample->num_node; t ++)
            {
                DataNode* p_node = p_sample->node[t];
                p_node->num_attrib = p_node->attrib.size();
                WriteInt(mem, mem_p, p_node->label);
                WriteInt(mem, mem_p, p_node->num_attrib);

                for (int k = 0; k < (int)p_node->attrib.size(); k ++)
                {
                    WriteInt(mem, mem_p, p_node->attrib[k]);
                    WriteDouble(mem, mem_p, p_node->value[k]);
                }
            }

            for (int t = 0; t < p_sample->num_edge; t ++)
            {
                DataEdge* p_edge = p_sample->edge[t];
                WriteInt(mem, mem_p, p_edge->a);
                WriteInt(mem, mem_p, p_edge->b);
                WriteInt(mem, mem_p, p_edge->edge_type);
            }

            delete p_sample;
            g_dataset->sample[i] = NULL;
        }

    }

    DataSet* dataset = new DataSet();
    dataset->num_sample = start[1];    
    dataset->num_label = g_dataset->num_label;
    dataset->num_attrib_type = g_dataset->num_attrib_type;
    dataset->num_edge_type = g_dataset->num_edge_type;

    dataset->sample.resize( dataset->num_sample );
    for (int i = 0; i < start[1]; i ++)
    {
        dataset->sample[i] = g_dataset->sample[i];
        dataset->sample[i]->num_node = dataset->sample[i]->node.size();
        dataset->sample[i]->num_edge = dataset->sample[i]->edge.size();        

        for (int t = 0; t < dataset->sample[i]->num_node; t ++)
            dataset->sample[i]->node[t]->num_attrib = dataset->sample[i]->node[t]->attrib.size();
    }

    return dataset;
}

const int FOLD_NUM = 5;

void Estimate(Config* conf)
{
    GlobalDataSet* g_dataset;
    DataSet* dataset;

    // cout << "??1" << endl;

    if (conf->my_rank == 0) // master
    {
        g_dataset = new GlobalDataSet();
        g_dataset->LoadData(conf->train_file.c_str(), conf);
        g_dataset->label_dict.SaveMappingDict(conf->dict_file.c_str());

		dataset = AssignJobs(g_dataset, conf->num_procs);
    }

    // cout << "??2" << endl;

    printf("num_label = %d\n", dataset->num_label);
    printf("num_sample = %d\n", dataset->num_sample);
    printf("num_edge_type = %d\n", dataset->num_edge_type);
    printf("num_attrib_type = %d\n", dataset->num_attrib_type);

    // for (int i = 1; i < dataset->sample[0]->node.size(); ++i) {
    //     int j = rand() % i;
    //     swap(dataset->sample[0]->node[i], dataset->sample[0]->node[j]);
    // }

    // for (int group = 0; group < FOLD_NUM; ++group)
    {
        // for (int i = 0, j = 0; i < dataset->sample[0]->node.size(); ++i) {
        //     if (dataset->sample[0]->node[i]->label == -1) {
        //         continue;
        //     }
        //     dataset->sample[0]->node[i]->label_type = (j % FOLD_NUM == group ? Enum::KNOWN_LABEL : Enum::UNKNOWN_LABEL);
        //     ++j;
        // }

        CRFModel *model = new CRFModel();
        
        // cout << "InitTrain" << endl;

        model->InitTrain(conf, dataset);

        // cout << "Train" << endl;

        model->Train();
        
        if (conf->my_rank == 0)
            model->SaveModel(conf->dst_model_file.c_str());
        
        delete model;
    }
      
    //MakeEvaluate(conf, g_dataset, model);
}

void EstimateContinue(Config* conf)
{
    GlobalDataSet* g_dataset;
    DataSet* dataset;

    if (conf->my_rank == 0) // master
    {
        g_dataset = new GlobalDataSet();
        g_dataset->LoadData(conf->train_file.c_str(), conf);
        g_dataset->label_dict.SaveMappingDict(conf->dict_file.c_str());

		dataset = AssignJobs(g_dataset, conf->num_procs);
    }

    //DEBUG__AddLinearEdge(dataset);

    printf("num_label = %d\n", dataset->num_label);
    printf("num_sample = %d\n", dataset->num_sample);
    printf("num_edge_type = %d\n", dataset->num_edge_type);
    printf("num_attrib_type = %d\n", dataset->num_attrib_type);

    CRFModel *model = new CRFModel();
    
    model->InitTrain(conf, dataset);

    if (conf->my_rank == 0)
        model->LoadModel(conf->src_model_file.c_str());

    model->Train();

    if (conf->my_rank == 0)
        model->SaveModel(conf->dst_model_file.c_str());
    
    //MakeEvaluate(conf, g_dataset, model);
}

void Inference(Config* conf)
{
    GlobalDataSet* g_dataset;
    DataSet* dataset;

    if (conf->my_rank == 0) // master
    {
        g_dataset = new GlobalDataSet();
        g_dataset->LoadData(conf->train_file.c_str(), conf);
        g_dataset->label_dict.SaveMappingDict(conf->dict_file.c_str());

		dataset = AssignJobs(g_dataset, conf->num_procs);
    }


    printf("num_label = %d\n", dataset->num_label);
    printf("num_sample = %d\n", dataset->num_sample);
    printf("num_edge_type = %d\n", dataset->num_edge_type);
    printf("num_attrib_type = %d\n", dataset->num_attrib_type);

    CRFModel *model = new CRFModel();
    
    model->InitTrain(conf, dataset);
    if (conf->my_rank == 0)
        model->LoadModel(conf->src_model_file.c_str());
    
    MakeEvaluate(conf, g_dataset, model);
}

int main(int argc, char* argv[])
{
    int         my_rank;
    int         num_procs;

    // Initialize MPI environment
	my_rank = 0;
	num_procs = 1;

    // Load Configuartion
    Config* conf = new Config();
    if (! conf->LoadConfig(my_rank, num_procs, argc, argv))
    {
        conf->ShowUsage();
        exit( 0 );
    }

    struct timeval startTime, endTime;
    gettimeofday(&startTime, NULL);

    if (conf->task == "-est")
    {
        Estimate(conf);
    }
    else if (conf->task == "-estc")
    {
        EstimateContinue(conf);
    }
    else if (conf->task == "-inf")
    {
        Inference(conf);
    }
    else
    {
        Config::ShowUsage();
    }

    gettimeofday(&endTime, NULL);

    double diffTime = (endTime.tv_sec - startTime.tv_sec) + (double)(endTime.tv_usec - startTime.tv_usec) / 1000000;

    fprintf(stdout, "total time = %.4fs\n", diffTime);

    return 0;
}