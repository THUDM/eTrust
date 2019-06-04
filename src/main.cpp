#include <sys/time.h>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <map>
#include <cassert>
#include <algorithm>

using namespace std;

const int MAX_LINE_LEN = 5000;
const int MAX_N = 250000;
const int MAX_M = 1250000;
const int FOLD_NUM = 5;
const int TOP_K = 10;

namespace StringHelper {
    void replace(string &str, const string &origin, const string &res) {
        size_t found = str.find(origin);
        int len = origin.length();
        while (found != string::npos) {
            str.replace(found, len, res);
            found = str.find(origin, found);
        }
    }
    vector<string> split(string str, const string &partition) {
        size_t last = -1, cur;
        vector<string> res;
        for (;;) {
            cur = str.find(partition, last + 1);
            if (cur != -1)
                res.push_back(str.substr(last + 1, cur - last - 1));
            else {
                res.push_back(str.substr(last + 1));
                break;
            }
            last = cur;
        }
        return res;
    }
    string lstrip(string str) {
        int p = -1;
        for (int i = 0; i < str.length(); ++i)
            if (str[i] != ' ' && str[i] != '\r' && str[i] != '\n' && str[i] != '\t') {
                p = i;
                break; 
            }
        if (p == -1)
            return "";
        else
            return str.substr(p);
    }
    string rstrip(string str) {
        int p = -1;
        for (int i = (int)str.length() - 1; i >= 0; --i)
            if (str[i] != ' ' && str[i] != '\r' && str[i] != '\n' && str[i] != '\t') {
                p = i;
                break;
            }
        if (p == -1)
            return "";
        else
            return str.substr(0, p + 1);
    }
    string tolower(string str) {
        for (int i = 0; i < str.length(); ++i) {
            if (!isalpha(str[i]))
                continue;
            if (str[i] >= 'A' && str[i] <= 'Z')
                str[i] += 'a' - 'A';
        }
        return str;
    }
}

using namespace StringHelper;

struct Node {
    bool known;
    bool valid;
    bool test;
    bool selected;
    int u, v;
    int group;
    int origin_label;
    int predict_label;
    double *prob;
    double value;
    double valsum;
    double *features;

    // double *triangle_features;
    // vector<string> inputs;
    // vector<pair<int, double> > trans;

    int GetLabel() const {
        // return origin_label;
        
        return known ? origin_label : predict_label;

        // if (known) {
        //     return origin_label;
        // }
        // if (selected) {
        //     return predict_label;
        // }
        // return -1;
    }
};

struct Triangle {
    int u, v, w;

    Triangle(int u, int v, int w) {
        this->u = u;
        this->v = v;
        this->w = w;
    }
};


int n, m, r = 0;
bool use_top_k = false;
vector<Node> nodes;
vector<Triangle> triangles;
vector<int> adjs[MAX_M];
vector<int> triangle_index[MAX_M];
vector<pair<int, int> > edges[MAX_N];
string datafile, edgefile;
int label_total = 0;
int feature_total = 0;
int triangle_features_total = 0;
int edge_features_total = 0;
int unsupervised_features_total = 0;

map<string, int> feature_mapping;

bool binary = false;
bool analyze = false;
bool label_propagation = false;
bool label_spreading = false;
bool iterative_training = false;
bool common_neighbor = false;
bool adamic_adar = false;
bool jaccard_coefficient = false;
bool tidal_trust = false;
bool trust_propagation = false;
bool deep_walk = false;
bool logistic_regression = false;
bool print_precision = false;
bool not_use_edge_feature = false;
bool not_use_triangle_feature = false;
bool use_unsupervised_method = false;

struct LogisticRegression {
    int feature_num;
    int label_num;
    double **weights;
    double **delta_weights;

    LogisticRegression(int feature_num, int label_num) {
        this->feature_num = feature_num;
        this->label_num = label_num;
        weights = new double*[feature_num];
        delta_weights = new double*[feature_num];
        for (int i = 0; i < feature_num; ++i) {
            weights[i] = new double[label_num];
            delta_weights[i] = new double[label_num];
            for (int j = 0; j < label_num; ++j) {
                weights[i][j] = 1.0;
            }
        }
    }
    ~LogisticRegression() {
        for (int i = 0; i < feature_num; ++i) {
            delete[] weights[i];
            delete[] delta_weights[i];
        }
        delete[] weights;
        delete[] delta_weights;
    }

    void init_model() {
        for (int i = 0; i < feature_num; ++i)
            for (int j = 0; j < label_num; ++j)
                weights[i][j] = 1.0;
    }

    double train_batch(double **X, int *y, int batch_size, double alpha) {
        if (batch_size == 0) {
            return 0.0;
        }
        double lambda = 0.0;
        for (int i = 0; i < feature_num; ++i)
            for (int j = 0; j < label_num; ++j)
                delta_weights[i][j] = 0.0;
        double loss = 0.0;
        if (label_num == 2) {
            for (int i = 0; i < batch_size; ++i) {
                double wx = 0.0;
                for (int j = 0; j < feature_num; ++j)
                    wx += X[i][j] * weights[j][0];
                wx = 1.0 / (1.0 + exp(-wx));
                for (int j = 0; j < feature_num; ++j) {
                    delta_weights[j][0] += (y[i] - wx) * X[i][j];
                }
                loss += (y[i] - wx) * (y[i] - wx) / 2.0;
            }
        } else {
            for (int k = 0; k < label_num; ++k) {
                for (int i = 0; i < batch_size; ++i) {
                    double wx = 0.0;
                    for (int j = 0; j < feature_num; ++j)
                        wx += X[i][j] * weights[j][k];
                    wx = 1.0 / (1.0 + exp(-wx));
                    for (int j = 0; j < feature_num; ++j) {
                        delta_weights[j][k] += ((y[i] == k) - wx) * X[i][j];
                    }
                    loss += ((y[i] == k) - wx) * ((y[i] == k) - wx) / 2.0;
                }
            }
        }
        for (int i = 0; i < feature_num; ++i)
            for (int j = 0; j < label_num; ++j)
                weights[i][j] += alpha * (delta_weights[i][j] - lambda * weights[i][j]) / batch_size;
        loss /= batch_size;
        if (label_num > 2) {
            loss /= label_num;
        }
        return loss;
    }

    double calc_triangle_conf(int i) {
        double conf = 0.0;
        int tri_size = (int)triangle_index[i].size();
        for (int p = 0; p < tri_size; ++p) {
            int q = triangle_index[i][p];
            int u = triangles[q].u;
            int v = triangles[q].v;
            int w = triangles[q].w;

            conf += nodes[u].value + nodes[v].value + nodes[w].value;
        }
        if (tri_size > 0) {
            return conf / (3 * tri_size);
        }
        return nodes[i].value;
    }

    double train(int epoch_num, double alpha = 1.0, double threshold = 0.0, bool useAll = false) {
        int batch_size = 1000;
        double **X = new double*[batch_size];
        for (int i = 0; i < batch_size; ++i) {
            X[i] = new double[feature_num];
        }
        int *y = new int[batch_size];
        vector<int> indices;
        int num_0 = 0, num_1 = 0;
        for (int i = 0; i < n; ++i) {
            if (nodes[i].known or useAll) {
                ++num_1;
                // if (calc_triangle_conf(i) > threshold) {
                if (rand() >= threshold * RAND_MAX) {
                    indices.push_back(i);
                    ++num_0;
                }
            }
        }
        // printf("choose: %d %d %.4f\n", num_0, num_1, num_0 * 1.0 / num_1);
        int sample_num = (int)indices.size();
        int T = sample_num / batch_size;
        double loss;
        for (int epoch = 0; epoch <= epoch_num; ++epoch) {
            loss = 0.0;
            random_shuffle(indices.begin(), indices.end());
            for (int i = 0; i < T; ++i) {
                for (int j = 0; j < batch_size; ++j) {
                    int index = indices[i * batch_size + j];
                    for (int k = 0; k < feature_num; ++k) {
                        X[j][k] = nodes[index].features[k];
                    }
                    // y[j] = nodes[index].origin_label;
                    y[j] = nodes[index].GetLabel();
                }
                loss += this->train_batch(X, y, batch_size, alpha);
            }
            loss /= T;
            // if (epoch % 10 == 0)
            //     cout << "Epoch: " << epoch << ", Loss: " << loss << endl;
        }
        for (int i = 0; i < batch_size; ++i) {
            delete[] X[i];
        }
        delete[] X;
        delete[] y;
        return loss;
    }

    double train_loss() {
        double loss = 0.0;
        if (label_num == 2) {
            for (int i = 0; i < n; ++i) {
                double wx = 0.0;
                for (int j = 0; j < feature_num; ++j)
                    wx += nodes[i].features[j] * weights[j][0];
                wx = 1.0 / (1.0 + exp(-wx));
                loss += (nodes[i].origin_label - wx) * (nodes[i].origin_label - wx) / 2.0;
            }
        } else {
            for (int k = 0; k < label_num; ++k) {
                for (int i = 0; i < n; ++i) {
                    double wx = 0.0;
                    for (int j = 0; j < feature_num; ++j)
                        wx += nodes[i].features[j] * weights[j][k];
                    wx = 1.0 / (1.0 + exp(-wx));
                    loss += ((nodes[i].origin_label == k) - wx) * ((nodes[i].origin_label == k) - wx) / 2.0;
                }
            }
        }
        loss /= n;
        if (label_num > 2) {
            loss /= label_num;
        }
        return loss;
    }

    void predict() {
        if (label_num == 2) {
            for (int i = 0; i < n; ++i) {
                double wx = 0.0;
                for (int j = 0; j < feature_num; ++j) {
                    wx += weights[j][0] * nodes[i].features[j];
                }
                wx = 1.0 / (1.0 + exp(-wx));
                nodes[i].prob[0] = 1 - wx;
                nodes[i].prob[1] = wx;
                nodes[i].value = wx;
                nodes[i].predict_label = (wx >= 0.5);
            }
        } else {
            double *wx = new double[label_num];
            for (int i = 0; i < n; ++i) {
                double exp_sum = 0.0;
                for (int k = 0; k < label_num; ++k) {
                    wx[k] = 0.0;
                    for (int j = 0; j < feature_num; ++j)
                        wx[k] += weights[j][k] * nodes[i].features[j];
                    wx[k] = 1.0 / (1.0 + exp(-wx[k]));
                    exp_sum += exp(wx[k]);
                }
                double max_prob = 0.0;
                for (int k = 0; k < label_num; ++k) {
                    wx[k] = exp(wx[k]) / exp_sum;
                    nodes[i].prob[k] = wx[k];
                    if (wx[k] >= max_prob) {
                        max_prob = wx[k];
                        nodes[i].predict_label = k;
                    }
                }
                nodes[i].value = 1.0 - wx[0];
            }
            delete[] wx;
        }
    }
};


int GetTriangleFeatureIndex(int u, int v) {
    if (u > v)
        swap(u, v);
    return (2 * label_total - u + 1) * u / 2 + (v - u);
}

double PreTrain(LogisticRegression *LR) {

    double loss = LR->train(100, 1.0, 0.0);
    LR->predict();

    return loss;
}

void ResetTriangleFeatures() {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < triangle_features_total; ++j) {
            nodes[i].features[feature_total + j] = 0;
        }
        for (int j = 0; j < edge_features_total; ++j) {
            nodes[i].features[feature_total + triangle_features_total + j] = 0;
        }
    }

    if (!not_use_triangle_feature) {
        for (int i = 0; i < m; ++i) {
            int lu = nodes[triangles[i].u].GetLabel();
            int lv = nodes[triangles[i].v].GetLabel();
            int lw = nodes[triangles[i].w].GetLabel();

            if (lu == -1 or lv == -1 or lw == -1) {
                continue;
            }

            nodes[triangles[i].u].features[feature_total + GetTriangleFeatureIndex(lv, lw)] += 1;
            nodes[triangles[i].v].features[feature_total + GetTriangleFeatureIndex(lu, lw)] += 1;
            nodes[triangles[i].w].features[feature_total + GetTriangleFeatureIndex(lu, lv)] += 1;
        }

        // for (int i = 0; i < n; ++i) {
        //     int sum = 0;
        //     for (int j = 0; j < triangle_features_total; ++j) {
        //         sum += nodes[i].features[feature_total + j];
        //     }
        //     if (sum > 0) {
        //         for (int j = 0; j < triangle_features_total; ++j) {
        //             nodes[i].features[feature_total + j] /= sum * 1.0;
        //         }
        //     }
        // }
    }

    if (!not_use_edge_feature) {
        for (int i = 0; i < r; ++i) {
            int size = (int)edges[i].size();
            for (int j = 0; j < size; ++j)
                for (int k = j + 1; k < size; ++k) {
                    int u = edges[i][j].second;
                    int v = edges[i][k].second;
                    nodes[u].features[feature_total + triangle_features_total + nodes[v].GetLabel()] += 1;
                    nodes[v].features[feature_total + triangle_features_total + nodes[u].GetLabel()] += 1;
                }
        }

        for (int i = 0; i < n; ++i) {
            int sum = 0;
            for (int j = 0; j < edge_features_total; ++j) {
                sum += nodes[i].features[feature_total + triangle_features_total + j];
            }
            if (sum > 0) {
                for (int j = 0; j < edge_features_total; ++j) {
                    nodes[i].features[feature_total + triangle_features_total + j] /= sum * 1.0;
                }
            }
        }
    }

    if (use_unsupervised_method) {
        int total = feature_total + triangle_features_total + edge_features_total;
        for (int i = 0; i < n; ++i) {
            int CN = 0;
            double AA = 0.0;
            int u = nodes[i].u, v = nodes[i].v;
            int deg_u = edges[u].size();
            int deg_v = edges[v].size();
            int j = 0, k = 0;
            while (j < deg_u && k < deg_v) {
                if (edges[u][j].first == edges[v][k].first) {
                    if (nodes[edges[u][j].second].GetLabel() == 0) {
                        ++j;
                        continue;
                    }
                    if (nodes[edges[v][k].second].GetLabel() == 0) {
                        ++k;
                        continue;
                    }
                    ++CN;
                    AA += 1.0 / log(edges[edges[u][j].first].size() + 1);
                    ++j;
                    ++k;
                } else if (edges[u][j].first < edges[v][k].first) {
                    ++j;
                } else if (edges[u][j].first > edges[v][k].first) {
                    ++k;
                }
            }
            nodes[i].features[total + 0] = CN;
            nodes[i].features[total + 1] = AA;
            nodes[i].features[total + 2] = CN * 1.0 / (deg_u + deg_v - CN);
        }
    }

}


double Train(LogisticRegression *LR, bool useAll = false) {
    ResetTriangleFeatures();

    double loss = LR->train(100, 1.0, 0.0, useAll);
    LR->predict();

    return loss;
}

int cmp_label;

bool cmp(int i, int j) {
    if (nodes[i].test != nodes[j].test) 
        return nodes[i].test > nodes[j].test;
    // bool c1 = (nodes[i].predict_label == 0);
    // bool c2 = (nodes[j].predict_label == 0);
    // if (c1 != c2)
    //     return c1 < c2;
    // return (nodes[i].u < nodes[j].u) || (nodes[i].u == nodes[j].u && nodes[i].value > nodes[j].value);
    
    // return nodes[i].value > nodes[j].value;
    return nodes[i].prob[cmp_label] > nodes[j].prob[cmp_label];
}

bool cmp_auc(int i, int j) {
    if (nodes[i].test != nodes[j].test) 
        return nodes[i].test > nodes[j].test;
    // return nodes[i].value > nodes[j].value;
    return nodes[i].prob[cmp_label] > nodes[j].prob[cmp_label];
}



struct Result {
    double P, R, F1, Acc, MAP;
    Result() {
        P = R = F1 = Acc = MAP = 0.0;
    }

    Result& operator += (const Result &result) {
        this->P += result.P;
        this->R += result.R;
        this->F1 += result.F1;
        this->Acc += result.Acc;
        this->MAP += result.MAP;
        return *this;
    }
    Result& operator /= (int x) {
        this->P /= x;
        this->R /= x;
        this->F1 /= x;
        this->Acc /= x;
        this->MAP /= x;
        return *this;
    }
};

Result Evaluate(string name = "", bool unsupervised = false) {
    Result result;

    if (!unsupervised) {
        int accuracy = 0, total = 0;
        for (int i = 0; i < n; ++i) {
            if (!nodes[i].test)
                continue;
            if (nodes[i].predict_label == nodes[i].origin_label) {
                ++accuracy;
            }
            ++total;
        }
        fprintf(stdout, "hit = %d, total = %d\n", accuracy, total);
        fprintf(stdout, "accuracy = %.2f%%\n", accuracy * 100.0 / total);
        result.Acc = accuracy * 100.0 / total;
    }

    if (name != "" && print_precision) {
        char fileName[50];
        for (int z = 1; z < label_total; ++z) {
            sprintf(fileName, "P@_%s_%d.txt", name.c_str(), z);
            FILE *file = fopen(fileName, "w");
            // for (int i = 0; i < n; ++i) {
            //     if (!nodes[i].test) {
            //         continue;
            //     }
            //     fprintf(file, "%d %.6f\n", nodes[i].origin_label == z, nodes[i].prob[z]);
            // }
            
            vector<int> index;
            for (int i = 0; i < n; ++i) {
                index.push_back(i);
            }
            cmp_label = z;
            sort(index.begin(), index.end(), cmp);

            int counts[30];
            double precision[30];
            int total = 0;
            for (int i = 0; i < n; ++i) {
                total += (nodes[i].test == true);
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
                if (!nodes[i].test) {
                    break;
                }
                ++num;
                if (nodes[i].origin_label == z) {
                    ++pos_num;
                }
                if (num == counts[iter]) {
                    precision[iter] = pos_num * 1.0 / num;
                    ++iter;
                }
            }

            for (int i = 1; i < 20; ++i) {
                fprintf(file, "%.4f\n", precision[i]);
            }

            fclose(file);
        }
    }

    int num, pos_num, count;

    double MAP = 0.0, AP;
    for (int z = 0; z < label_total; ++z) {
        // if (label_total == 2 && z == 0) {
        if (z == 0) {
            continue;
        }
        AP = 0.0;
        
        vector<int> index;
        for (int i = 0; i < n; ++i) {
            index.push_back(i);
        }
        cmp_label = z;
        sort(index.begin(), index.end(), cmp);

        num = pos_num = 0;
        for (int j = 0, k; j < n; j = k) {
            int i = index[j];
            if (!nodes[i].test) {
                break;
            }
            k = j + 1;
            while (k < n && nodes[index[k]].test && nodes[i].prob[z] - nodes[index[k]].prob[z] < 1e-8) {
                ++k;
            }
            count = 0;
            for (int q = j; q < k; ++q) {
                ++num;
                if (nodes[index[q]].origin_label == z) {
                    ++pos_num;
                    ++count;
                }
            }
            // printf("%d %d %d %.4f\n", pos_num, num, count, nodes[i].prob[z]);
            AP += pos_num * 1.0 / num * count;

            if (use_top_k && num >= TOP_K) {
                break;
            }
        }
        AP /= pos_num;
        MAP += AP;
    }
    // MAP /= (label_total - (label_total == 2));
    MAP /= (label_total - 1);
    
    result.MAP = MAP;
    if (use_top_k)
        fprintf(stdout, "MAP@%d = %.2f%%\n", TOP_K, MAP * 100);
    else
        fprintf(stdout, "MAP = %.2f%%\n", MAP * 100);

    double mean_auc = 0.0, auc;
    for (int z = 0; z < label_total; ++z) {
        // if (label_total == 2 && z == 0) {
        if (z == 0) {
            continue;
        }
        vector<int> index;
        for (int i = 0; i < n; ++i) {
            index.push_back(i);
        }
        cmp_label = z;
        sort(index.begin(), index.end(), cmp_auc);
        num = pos_num = 0;
        auc = 0.0;
        for (int j = 0, k; j < n; j = k) {
            int i = index[j];
            if (!nodes[i].test) {
                break;
            }
            k = j + 1;
            while (k < n && nodes[index[k]].test && nodes[i].prob[z] - nodes[index[k]].prob[z] < 1e-8) {
                ++k;
            }
            count = 0;
            for (int q = j; q < k; ++q) {
                if (nodes[index[q]].origin_label == z) {
                    ++pos_num;
                    ++count;
                }
                ++num;
            }
            auc -= (j + k - 1) / 2.0 * count;
            // printf("%d %d %d\n", j, k, count);
        }
        auc -= pos_num * (pos_num + 1) / 2;
        auc += (num * pos_num);
        auc /= (pos_num * (num - pos_num));

        // cout << z << " " << auc << endl;

        mean_auc += auc;
    }
    // mean_auc /= (label_total - (label_total == 2));
    mean_auc /= (label_total - 1);
    
    fprintf(stdout, "AUC = %.2f%%\n", mean_auc * 100);

    if (!unsupervised) {

        double macroP, macroR, macroF1;
        macroP = macroR = macroF1 = 0.0;
        for (int z = 0; z < label_total; ++z) {
            // if (label_total == 2 && z == 0) {
            if (z == 0) {
                continue;
            }
            int A = 0, B = 0, C = 0;
            for (int i = 0; i < n; ++i) {
                if (!nodes[i].test)
                    continue;
                if (nodes[i].predict_label == z) {
                    if (nodes[i].origin_label == z) {
                        ++A;
                    } else {
                        ++B;
                    }
                } else if (nodes[i].origin_label == z) {
                    ++C;
                }
            }
            double P = A * 1.0 / (A + B);
            double R = A * 1.0 / (A + C);
            double F1 = 2 * P * R / (P + R);

            if (A + B == 0) {
                P = 0.0;
                F1 = 0.0;
            }

            macroP += P;
            macroR += R;
            macroF1 += F1;
        }
        // macroP /= (label_total - (label_total == 2));
        // macroR /= (label_total - (label_total == 2));
        // macroF1 /= (label_total - (label_total == 2));
        macroP /= (label_total - 1);
        macroR /= (label_total - 1);
        macroF1 /= (label_total - 1);

        result.P = macroP;
        result.R = macroR;
        result.F1 = macroF1;
        fprintf(stdout, "P = %.2f%%  R = %.2f%%  F1 = %.2f%%\n", macroP * 100, macroR * 100, macroF1 * 100);
        fprintf(stdout, "\n");
    }
    return result;
}

void ReadConfig(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-data") == 0) {
            datafile = string(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "-edge") == 0) {
            edgefile = string(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "-b") == 0) {
            binary = true;
        }
        if (strcmp(argv[i], "-a") == 0) {
            analyze = true;
        }
        if (strcmp(argv[i], "-it") == 0) {
            iterative_training = true;
        }
        if (strcmp(argv[i], "-lp") == 0) {
            label_propagation = true;
        }
        if (strcmp(argv[i], "-ls") == 0) {
            label_spreading = true;
        }
        if (strcmp(argv[i], "-lr") == 0) {
            logistic_regression = true;
        }
        if (strcmp(argv[i], "-cn") == 0) {
            common_neighbor = true;
        }
        if (strcmp(argv[i], "-aa") == 0) {
            adamic_adar = true;
        }
        if (strcmp(argv[i], "-jc") == 0) {
            jaccard_coefficient = true;
        }
        if (strcmp(argv[i], "-tt") == 0) {
            tidal_trust = true;
        }
        if (strcmp(argv[i], "-tp") == 0) {
            trust_propagation = true;
        }
        if (strcmp(argv[i], "-dw") == 0) {
            deep_walk = true;
        }
        if (strcmp(argv[i], "-pp") == 0) {
            print_precision = true;
        }
        if (strcmp(argv[i], "-nuef") == 0) {
            not_use_edge_feature = true;
        }
        if (strcmp(argv[i], "-nutf") == 0) {
            not_use_triangle_feature = true;
        }
        if (strcmp(argv[i], "-uum") == 0) {
            use_unsupervised_method = true;
        }
    }
}

void ReadData(string datafile, string edgefile) {
    char buf[MAX_LINE_LEN];
    
    string fileName = datafile;
    FILE *file = fopen(fileName.c_str(), "r");
    if (file == NULL) {
        fprintf(stderr, "File %s doesn't exist\n", fileName.c_str());
        exit(-1);
    }
    while (fgets(buf, MAX_LINE_LEN, file) != NULL) {
        string line = rstrip(string(buf));
        vector<string> items = split(line, " ");
        if (items[0] != "#triangle" && items[0].length() > 1) {
            string label_string = items[0].substr(1);
            int label = atoi(label_string.c_str());
            if (binary && label > 1) {
                label = 1;
            }
            label_total = max(label_total, label + 1);
        }
    }
    fclose(file);

    edge_features_total = label_total;
    triangle_features_total = (label_total + 1) * label_total / 2;
    unsupervised_features_total = 0;

    if (not_use_edge_feature) {
        edge_features_total = 0;
    }
    if (not_use_triangle_feature) {
        triangle_features_total = 0;
    }
    if (use_unsupervised_method) {
        unsupervised_features_total = 3;
    }

    file = fopen(fileName.c_str(), "r");
    
    cout << "read feature file" << endl;
    vector<pair<int, double> > trans;
    while (fgets(buf, MAX_LINE_LEN, file) != NULL) {
        string line = rstrip(string(buf));
        //fprintf(stdout, "len = %d\n", (int)line.length());
        vector<string> items = split(line, " ");
        //fprintf(stdout, "num = %d\n", (int)items.size());
        if (items[0] == "#edge") {

        } else if (items[0] == "#triangle") {
            int u = atoi(items[1].c_str());
            int v = atoi(items[2].c_str());
            int w = atoi(items[3].c_str());
            triangles.push_back(Triangle(u, v, w));
            triangle_index[u].push_back((int)triangles.size() - 1);
            triangle_index[v].push_back((int)triangles.size() - 1);
            triangle_index[w].push_back((int)triangles.size() - 1);
        } else { // node
            Node node;
            node.selected = false;
            node.known = (items[0][0] == '+');
            node.valid = false;
            node.test = (items[0].length() >= 2 && items[0][0] == '?');
            //fprintf(stderr, "node = %d, items[0] = %s\n", nodes.size(), items[0].c_str());
            if (items[0].length() == 1) {
                node.origin_label = -1;
            } else {
                string label_string = items[0].substr(1);
                node.origin_label = atoi(label_string.c_str());
                if (binary && node.origin_label > 1) {
                    node.origin_label = 1;
                }
                // label_total = max(label_total, node.origin_label + 1);
            }

            trans.clear();
            for (int j = 1; j < (int)items.size(); ++j) {
                vector<string> terms = split(items[j], ":");
                if (feature_mapping.find(terms[0]) == feature_mapping.end()) {
                    feature_mapping[terms[0]] = feature_total;
                    feature_total++;
                }
                trans.push_back(make_pair(feature_mapping[terms[0]], atof(terms[1].c_str())));
            }
            sort(trans.begin(), trans.end());

            node.prob = new double[label_total];

            node.features = new double[feature_total + triangle_features_total + edge_features_total + unsupervised_features_total];
            // node.triangle_features = new double[triangle_features_total];

            for (int j = 0; j < feature_total; ++j) {
                // cout << j << " " << trans[j].first << endl;
                assert(j == trans[j].first);
                node.features[j] = trans[j].second;
            }
            for (int j = 0; j < triangle_features_total; ++j) {
                node.features[j + feature_total] = 0;
            }
            for (int j = 0; j < edge_features_total; ++j) {
                node.features[j + feature_total + triangle_features_total] = 0;
            }
            for (int j = 0; j < unsupervised_features_total; ++j) {
                node.features[j + feature_total + triangle_features_total + edge_features_total] = 0;
            }

            nodes.push_back(node);

            // if ((int)nodes.size() % 10 == 0)
            //     printf("%d\n", (int)nodes.size());
        }
    }
    fclose(file);

    n = nodes.size();
    m = triangles.size();

    cout << "read edgelist file" << endl;

    // vector<string> name_split = split(fileName, ".");
    // fileName = name_split[0] + "_edgelist." + name_split[1];
    // fileName = name_split[0] + ".edgelist";
    fileName = edgefile;

    file = fopen(fileName.c_str(), "r");
    if (file == NULL) {
        fprintf(stderr, "File %s doesn't exist\n", fileName.c_str());
        exit(-1);
    }
    int u, v;
    for (int i = 0; i < n; ++i) {
        fscanf(file, "%d %d", &u, &v);
        nodes[i].u = u;
        nodes[i].v = v;
        r = max(r, u + 1);
        r = max(r, v + 1);
        assert(r < MAX_N);
        edges[u].push_back(make_pair(v, i));
        edges[v].push_back(make_pair(u, i));

    }
    for (int i = 0; i < r; ++i) {
        sort(edges[i].begin(), edges[i].end());

        if (label_propagation or label_spreading) {
            int size = (int)edges[i].size();
            for (int j = 0; j < size; ++j)
                for (int z = j + 1; z < size; ++z) {
                    adjs[edges[i][j].second].push_back(edges[i][z].second);
                    adjs[edges[i][z].second].push_back(edges[i][j].second);
                }
        }
    }

    fclose(file);
}

void Analyze() {
    for (int i = 0; i < n; ++i) {
        if (nodes[i].origin_label == -1) {
            continue;
        }
        nodes[i].known = true;
    }
    LogisticRegression *PreLR = new LogisticRegression(feature_total, label_total);
    PreTrain(PreLR);
    Evaluate();

    LogisticRegression *LR = new LogisticRegression(feature_total + triangle_features_total + edge_features_total, label_total);
    Train(LR);
    Evaluate();

    FILE *out = fopen("edge_predict.out", "w");
    for (int i = 0; i < n; ++i) {
        fprintf(out, "%d %d %d\n", nodes[i].u, nodes[i].v, nodes[i].GetLabel());
    }
    fclose(out);

    out = fopen("triangle.out", "w");
    for (int i = 0; i < m; ++i) {
        fprintf(out, "%d %d %d\n", triangles[i].u, triangles[i].v, triangles[i].w);
    }
    fclose(out);

    delete PreLR;
    delete LR;
}

void LabelPropogationSpreading() {
    for (int z = 0; z < label_total; ++z) {
        // if (label_total == 2 && z == 0) {
        //     continue;
        // }
        for (int i = 0; i < n; ++i) {
            if (nodes[i].known) {
                nodes[i].prob[z] = nodes[i].origin_label == z ? 1 : -1;
            } else {
                nodes[i].prob[z] = 0.0;
            }
        }

        double alpha = 0.5;
        
        for (int epoch = 0; epoch < 5; ++epoch) {
            for (int i = 0; i < n; ++i) {
                nodes[i].value = 0.0;
                int size = (int)adjs[i].size();
                double sqrt_size = sqrt(size);
                for (int j = 0; j < size; ++j) {
                    int k = adjs[i][j];

                    if (label_propagation)
                        nodes[i].value += nodes[k].prob[z] * 1.0 / (int)adjs[k].size();
                    else if (label_spreading)
                        nodes[i].value += nodes[k].prob[z] * 1.0 / sqrt((int)adjs[k].size()) / sqrt_size;
                }
            }
            for (int i = 0; i < n; ++i) {
                if (label_propagation) {
                    if (nodes[i].known) {
                        nodes[i].prob[z] = nodes[i].origin_label == z ? 1 : -1;
                    } else {
                        nodes[i].prob[z] = nodes[i].value;
                    }
                } else if (label_spreading) {
                    nodes[i].prob[z] = alpha * nodes[i].value + (1 - alpha) * (nodes[i].known ? (nodes[i].origin_label == z ? 1 : -1) : 0);
                }
            }

            // Evaluate();
        }
    }

    for (int i = 0; i < n; ++i) {
        // double exp_sum = 0.0;
        // for (int j = 0; j < label_total; ++j) {
        //     exp_sum += exp(nodes[i].prob[j]);
        // }
        double max_prob = 0.0;
        for (int j = 0; j < label_total; ++j) {
            // nodes[i].prob[j] = exp(nodes[i].prob[j]) / exp_sum;
            if (nodes[i].prob[j] > max_prob) {
                max_prob = nodes[i].prob[j];
                nodes[i].predict_label = j;
            }
        }
    }

    for (int z = 0; z < label_total; ++z) {
        double min_prob = 1e5, max_prob = -1e5;
        for (int i = 0; i < n; ++i) {
            min_prob = min(min_prob, nodes[i].prob[z]);
            max_prob = max(max_prob, nodes[i].prob[z]);
        }
        for (int i = 0; i < n; ++i) {
            nodes[i].prob[z] = (nodes[i].prob[z] - min_prob) / (max_prob - min_prob);
        }
    }

    if (label_propagation)
        Evaluate("lp");
    else if (label_spreading)
        Evaluate("ls");
}

void UnsupervisedMethod() {
    LogisticRegression *PreLR = new LogisticRegression(feature_total, label_total);
    PreTrain(PreLR);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < label_total; ++j) {
            nodes[i].prob[j] = 0.0;
        }
    }
    for (int i = 0; i < n; ++i) {
        int u = nodes[i].u, v = nodes[i].v;
        int su = (int)edges[u].size();
        int sv = (int)edges[v].size();
        for (int j = 0, k = 0; j < su; ++j) {
            while (k < sv && edges[v][k].first < edges[u][j].first)
                ++k;
            if (k >= sv) {
                break;
            }
            if (edges[u][j].first == edges[v][k].first) {
                if (nodes[edges[u][j].second].known and nodes[edges[v][k].second].known)
                {
                    int lu = nodes[edges[u][j].second].GetLabel();
                    int lv = nodes[edges[v][k].second].GetLabel();
                    if (lu == lv && lu >= 0) {
                        if (common_neighbor or jaccard_coefficient)
                            nodes[i].prob[lu] += 1.0;
                        else if (adamic_adar)
                            nodes[i].prob[lu] += 1.0 / log((int)edges[edges[u][j].first].size() + 1);
                    }
                }
            }
        }
        if (jaccard_coefficient) {
            for (int j = 0; j < label_total; ++j) {
                int count = (int)edges[u].size() + (int)edges[v].size();
                // for (int k = 0; k < su; ++k)
                //     count += nodes[edges[u][k].second].GetLabel() == j;
                // for (int k = 0; k < sv; ++k)
                //     count += nodes[edges[v][k].second].GetLabel() == j;
                nodes[i].prob[j] /= (count - nodes[i].prob[j]);
            }
        }
        double sum = 0.0;
        for (int j = 0; j < label_total; ++j) {
            sum += nodes[i].prob[j];
        }
        if (sum > 0) {
            for (int j = 0; j < label_total; ++j) {
                nodes[i].prob[j] /= sum;
            }
        }
    }
    if (common_neighbor)
        Evaluate("cn", true);
    else if (adamic_adar)
        Evaluate("aa", true);
    else if (jaccard_coefficient)
        Evaluate("jc", true);

    delete PreLR;
}

void TidalTrust() {
    // LogisticRegression *PreLR = new LogisticRegression(feature_total, label_total);
    // PreTrain(PreLR);

    for (int z = 0; z < label_total; ++z) {
        // if (label_total == 2 && z == 0) {
        //     continue;
        // }
        for (int i = 0; i < n; ++i) {
            if (nodes[i].known) {
                nodes[i].prob[z] = nodes[i].origin_label == z ? 1.0 : 0.0;
            } else {
                nodes[i].prob[z] = 0.0;
            }
        }

        double threshold = 0.1;
        
        for (int epoch = 0; epoch < 30; ++epoch) {
            for (int i = 0; i < n; ++i) {
                nodes[i].value = 0.0;
                nodes[i].valsum = 0.0;
            }
            for (int i = 0; i < m; ++i) {
                int u = triangles[i].u;
                int v = triangles[i].v;
                int w = triangles[i].w;

                if (nodes[u].prob[z] > 0) {
                    continue;
                }

                if (nodes[v].prob[z] > threshold) {
                    nodes[u].value += nodes[v].prob[z] * nodes[w].prob[z];
                    nodes[u].valsum += nodes[v].prob[z];
                }
            }
            for (int i = 0; i < n; ++i) {
                if (nodes[i].prob[z] > 0) {
                    continue;
                }
                if (nodes[i].valsum > 0)
                    nodes[i].prob[z] = nodes[i].value / nodes[i].valsum;
            }

            // Evaluate();
        }
    }

    Evaluate("tt", true);

    // delete PreLR;
}

void TrustPropagation() {
    for (int z = 0; z < label_total; ++z) {
        // if (label_total == 2 && z == 0) {
        //     continue;
        // }
        for (int i = 0; i < n; ++i) {
            if (nodes[i].known) {
                nodes[i].prob[z] = nodes[i].origin_label == z ? 1.0 : 0.0;
            } else {
                nodes[i].prob[z] = 0.0;
            }
        }

        double alpha = 0.5;
        
        for (int epoch = 0; epoch < 10; ++epoch) {
            for (int i = 0; i < n; ++i) {
                nodes[i].value = 0.0;
                nodes[i].valsum = 0.0;
            }
            for (int i = 0; i < m; ++i) {
                int u = triangles[i].u;
                int v = triangles[i].v;
                int w = triangles[i].w;

                nodes[u].value += nodes[v].prob[z] * nodes[w].prob[z];
                nodes[u].valsum += 1.0;
            }
            for (int i = 0; i < n; ++i) {
                if (nodes[i].valsum > 1e-8)
                    nodes[i].value = nodes[i].value / nodes[i].valsum;
                nodes[i].prob[z] = alpha * nodes[i].prob[z] + (1 - alpha) * nodes[i].value;
            }

            // Evaluate();
        }
    }

    Evaluate("tp", true);
}

void DeepWalk(string fileName) {
    vector<string> name_split = split(fileName, ".");
    // fileName = name_split[0] + "_vec." + name_split[1];
    fileName = name_split[0] + ".vec";

    FILE *file = fopen(fileName.c_str(), "r");
    if (file == NULL) {
        fprintf(stderr, "File %s doesn't exist\n", fileName.c_str());
        exit(-1);
    }

    int num = 100;
    fscanf(file, "%*d%d", &num);

    double **vec = new double*[r];
    for (int i = 0; i < r; ++i) {
        vec[i] = new double[num];
        for (int j = 0; j < num; ++j)
            vec[i][j] = 0.0;
    }
    char id_str[10];
    int id = 0;
    vector<double> v;
    while (fscanf(file, "%s", id_str) != EOF) {
        if (strcmp(id_str, "</s>") == 0) {
            for (int i = 0; i < num; ++i) {
                fscanf(file, "%*lf");
            }
            continue;
        }
        id = atoi(id_str);
        for (int i = 0; i < num; ++i) {
            fscanf(file, "%lf", &vec[id][i]);
        }
    }

    for (int i = 0; i < n; ++i) {
        int u = nodes[i].u;
        int v = nodes[i].v;

        double sum = 0.0;
        double lu = 0.0;
        double lv = 0.0;
        for (int j = 0; j < num; ++j) {
            sum += vec[u][j] * vec[v][j];
            lu += vec[u][j] * vec[u][j];
            lv += vec[v][j] * vec[v][j];
        }
        // printf("%d %d %.4f\n", u, v, sum / sqrt(lu) / sqrt(lv));
        for (int z = 1; z < label_total; ++z)
            if (lu > 0 and lv > 0)
                nodes[i].prob[z] = sum / sqrt(lu) / sqrt(lv);
            else
                nodes[i].prob[z] = 0.0;
    }

    Evaluate("dw", true);

    for (int i = 0; i < r; ++i) {
        delete[] vec[i];
    }
    delete[] vec;
}

void LR() {
    LogisticRegression *PreLR = new LogisticRegression(feature_total, label_total);
    PreTrain(PreLR);
    Evaluate("lr");
    delete PreLR;
}

void TrustRelationalLearing() {
    LogisticRegression *PreLR = new LogisticRegression(feature_total, label_total);
    PreTrain(PreLR);
    // Evaluate();

    LogisticRegression *LR = new LogisticRegression(feature_total + triangle_features_total + edge_features_total + unsupervised_features_total, label_total);
    Train(LR);
    Evaluate("trl");
    
    // for (int i = 0; i < 100; ++i) {
    //     // LR->init_model();
    //     double loss = Train(LR);
    //     Evaluate();
    // }

    if (iterative_training) {
        int batch_size = 1000;

        double **X = new double*[batch_size];
        for (int i = 0; i < batch_size; ++i) {
            X[i] = new double[feature_total + triangle_features_total + edge_features_total];
        }
        int *y = new int[batch_size];

        vector<int> indices;
        for (int i = 0; i < n; ++i) {
            if (nodes[i].known)
                indices.push_back(i);
        }

        int sample_num = (int)indices.size();
        int T = sample_num / batch_size;

        FILE *file = fopen("convergence.txt", "w");

        for (int epoch = 0; epoch < 100; ++epoch) {

            ResetTriangleFeatures();

            random_shuffle(indices.begin(), indices.end());

            double loss;
            // for (int iter = 0; iter <= 1000; ++iter) {
            for (int i = 0; i < T; ++i) {

                loss = 0.0;
                // int i = rand() % T;

                for (int j = 0; j < batch_size; ++j) {
                    int index = indices[i * batch_size + j];
                    for (int k = 0; k < LR->feature_num; ++k) {
                        X[j][k] = nodes[index].features[k];
                    }
                    // y[j] = nodes[index].origin_label;
                    y[j] = nodes[index].GetLabel();
                }
                loss += LR->train_batch(X, y, batch_size, 1.0);

                // if (iter % 10 == 0)
                //     cout << "Iter: " << iter << ", Loss: " << LR->train_loss() << endl;
                // if (iter % 10 == 0)
                //     Evaluate();
            }
            LR->predict();

            loss = LR->train_loss();

            cout << "Epoch: " << epoch << ", Loss: " << loss << endl;
            Result res = Evaluate();

            fprintf(file, "%.4f %.4f\n", loss, res.Acc);
        }

        fclose(file);

        for (int i = 0; i < batch_size; ++i) {
            delete[] X[i];
        }
        delete[] X;
        delete[] y;
    }

    delete PreLR;
    delete LR;

}

int main(int argc, char *argv[]) {
    srand(951230);

    struct timeval startTime, endTime;
    gettimeofday(&startTime, NULL);

    if (argc < 3) {
        fprintf(stderr, "Usage: ./main -data <datafile> -edge <edgefile>\n");
        return -1;
    } else {
        ReadConfig(argc, argv);
    }

    ReadData(datafile, edgefile);

    if (analyze) {
        Analyze();
        return 0;
    }

    if (common_neighbor or adamic_adar or jaccard_coefficient) {
        UnsupervisedMethod();

    } else if (label_propagation or label_spreading) {
        LabelPropogationSpreading();

    } else if (tidal_trust) {
        TidalTrust();
    } else if (trust_propagation) {
        TrustPropagation();
    } else if (deep_walk) {
        DeepWalk(datafile);
    } else if (logistic_regression) {
        LR();
    } else {
        TrustRelationalLearing();
    }

    for (int i = 0; i < n; ++i) {
        delete[] nodes[i].prob;
        delete[] nodes[i].features;
    }

    gettimeofday(&endTime, NULL);

    double diffTime = (endTime.tv_sec - startTime.tv_sec) + (double)(endTime.tv_usec - startTime.tv_usec) / 1000000;

    fprintf(stdout, "total time = %.4fs\n", diffTime);

    return 0;
}