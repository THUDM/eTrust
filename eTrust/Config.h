#pragma once

#include <string>

using std::string;

#define LBFGS 0
#define GradientDescend 1

class Config
{
public:
    int         my_rank;
    int         num_procs;

    string      task;

    string      train_file;
    string      test_file;
    string      pred_file;
    string      result_file;

    string      dict_file;
    string      src_model_file;
    string      dst_model_file;

    double      eps;

    int         max_iter;
    int         max_bp_iter;

    double      gradient_step;

    bool        has_attrib_value;
    int         optimization_method;

    bool        eval_each_iter;

    double      penalty_sigma_square;
    

    Config(){ SetDefault(); }
    void SetDefault();

    // false => parameter wrong
    bool LoadConfig(int my_rank, int num_procs, int argc, char* argv[]);
    static void ShowUsage();
};