#include "Config.h"

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>

void Config::SetDefault()
{
    max_iter = 50;
    max_bp_iter = 10;

    this->task = "-est";
    this->train_file = "train.txt";
    this->test_file = "test.txt";

    this->dict_file = "dict.txt";
    this->pred_file = "pred.txt";

    //this->train_file = "scene_data\\train.txt";
    //this->test_file = "scene_data\\test.txt";

    this->train_file = "zz_data\\run\\train.txt";
    this->test_file = "zz_data\\run\\test.txt";

    this->result_file = "result.txt";
    
    //this->train_file = "relation_graph\\train.txt";
    //this->test_file = "relation_graph\\test.txt";

    //this->train_file = "relation_ind\\train.txt";
    //this->test_file = "relation_ind\\test.txt";

    //this->train_file = "new_adv_data\\parta.dat";
    //this->test_file  = "new_adv_data\\partb.dat";
    
    //this->has_attrib_value = false;
    this->has_attrib_value = true;
    this->eps = 1e-3;

    //this->optimization_method = LBFGS;
    this->optimization_method = GradientDescend;
    this->gradient_step = 0.1;

    this->dst_model_file = "model-final.txt";

    this->eval_each_iter = true;

    this->penalty_sigma_square = 0.0001;
}

bool Config::LoadConfig(int my_rank, int num_procs, int argc, char* argv[])
{
    this->my_rank = my_rank;
    this->num_procs = num_procs;    

    if (argc == 1) return 0;

    int i = 1;
    if (strcmp(argv[1], "-est")==0 || strcmp(argv[1], "-estc")==0 || strcmp(argv[1], "-inf")==0)
    {
        this->task = argv[1];
        i ++;
    }
    else return 0;

    while (i < argc)
    {
        //std::cerr << i << " " << argv[i] << std::endl;
        if (strcmp(argv[i], "-niter") == 0)
        {
            this->max_iter = atoi(argv[++i]); ++i;
        }
        else if (strcmp(argv[i], "-nbpiter") == 0)
        {
            this->max_bp_iter = atoi(argv[++i]); ++i;
        }
        else if (strcmp(argv[i], "-srcmodel") == 0)
        {
            this->src_model_file = argv[++i]; ++i;
        }
        else if (strcmp(argv[i], "-dstmodel") == 0)
        {
            this->dst_model_file = argv[++i]; ++i;
        }
        else if (strcmp(argv[i], "-method") == 0)
        {
            if (argv[++i][0] == 'l') this->optimization_method = LBFGS;
            else this->optimization_method = GradientDescend;
            ++ i;
        }
        else if (strcmp(argv[i], "-gradientstep") == 0)
        {
            this->gradient_step = atof(argv[++i]); ++ i;
        }
        else if (strcmp(argv[i], "-hasvalue") == 0)
        {
            this->has_attrib_value = true; ++ i;
        }
        else if (strcmp(argv[i], "-novalue") == 0)
        {
            this->has_attrib_value = false; ++ i;
        }
        else if (strcmp(argv[i], "-trainfile") == 0)
        {
            this->train_file = argv[++i]; ++ i;
        }
        else if (strcmp(argv[i], "-resultfile") == 0)
        {
            this->result_file = argv[++i]; ++i;
        }
        else if (strcmp(argv[i], "-testfile") == 0)
        {
            this->test_file = argv[++i]; ++i;
        }
        else ++ i;
    }
    
    return 1;
}

void Config::ShowUsage()
{
    printf("OpenCRF v0.4                                                 \n");
    printf("     by Sen Wu, Tsinghua University                     \n");
    printf("                                                             \n");
    printf("Usage: mpiexec -n NUM_PROCS OpenCRF <task> [options]         \n");
    printf(" Options:                                                    \n");
    printf("   task: -est                                                \n");    
    printf("\n");
    printf("   -niter int           : number of iterations                               \n");
    printf("   -nbpiter int         : number of iterations in belif propgation           \n");
    printf("   -srcmodel string     : (for -estc) the model to load                      \n");
    printf("   -dstmodel string     : model file to save                                 \n");
    printf("   -method string       : methods (lbfgs/gradient), default: gradient        \n");
    printf("   -gradientstep double : learning rate                                      \n");
    printf("   -hasvalue            : [default] attributes with values (format: attr:val)\n");
    printf("   -novalue             : attributes without values (format: attr)           \n");
    printf("   -trainfile string    : train file                                         \n");
    printf("   -testfile string     : test file                                          \n");

    exit( 0 );
}
