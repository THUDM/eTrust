#pragma once

#include "DataSet.h"


#define MSG_SPACE_SIZE 1
#define MSG_DATA 2
#define MSG_DOUBLE_ARR 3
#define MSG_QUIT 4

class Transmitter
{
public:
    static DataSet* AssignJobs(GlobalDataSet* g_dataset, int num_procs);
    static DataSet* GetJobs();

    static void Master_SendDoubleArray(double* arr, int n, int num_procs);
    static void Master_CollectGradientInfo(double* gradient, double* f, int num_feature, double* tmp_store, int num_procs);
    static void Master_SendQuit(int num_procs);

    static bool Slave_RecvDoubleArray(double* arr, int n);
    static void Slave_SendGradientInfo(double* gradient, double* f, int num_feature);

    static void WriteInt(char* mem, int& mem_p, int v)
    {
        *((int*)(mem + mem_p)) = v;
        mem_p += sizeof(int);
    }

    static void WriteDouble(char* mem, int& mem_p, double v)
    {
        *((double*)(mem + mem_p)) = v;
        mem_p += sizeof(double);
    }

    static int ReadInt(char* mem, int& mem_p)
    {
        int v = *((int*)(mem + mem_p));
        mem_p += sizeof(int);
        return v;
    }

    static double ReadDouble(char* mem, int& mem_p)
    {
        double v = *((double*)(mem + mem_p));
        mem_p += sizeof(double);
        return v;
    }
};