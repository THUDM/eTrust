#include "Util.h"

#include <algorithm>
#include <sstream>
#include <cstdio>
#include <cstring>

using std::make_pair;
using std::istringstream;

bool same_type_label(int num_labels, int l1, int l2)
{
	if (num_labels != 6)
		return true;
	return (l1 < 4) == (l2 < 4);
}

/****************************************************************
 MappingDict
****************************************************************/
int MappingDict::GetId(const string& key)
{
    map<string, int>::iterator it = dict.find(key);
    if (it != dict.end())
    {
        return it->second;
    }
    else
    {
        if (key == "")
            return -1;

        int id = keys.size();
        keys.push_back(key);
        dict.insert(make_pair(key, id));
        return id;
    }
}

int MappingDict::GetIdConst(const string& key) const
{
    map<string, int>::const_iterator it = dict.find(key);
    if (it == dict.end())
        return -1;

    return it->second;
}

string MappingDict::GetKeyWithId(const int id) const
{
    if (id < 0 || id >= keys.size())
        return "";
    return keys[id];
}

void MappingDict::SaveMappingDict(const char* file_name)
{
    FILE* fout = fopen(file_name, "w");
    for (int i = 0; i < keys.size(); i ++)
        fprintf(fout, "%s %d\n", keys[i].c_str(), i);
    fclose(fout);
}

void MappingDict::LoadMappingDict(const char* file_name)
{
    FILE* fin = fopen(file_name, "r");
    dict.clear();
    keys.clear();
    
    char buf[256];
    int id;

    while (fscanf(fin, "%s%d", buf, id) > 0)
    {
        string str = buf;
        keys.push_back( str );
        dict.insert( make_pair(str, id) );
    }

    fclose(fin);
}

/****************************************************************
 CommonUtil
****************************************************************/
vector<string> CommonUtil::StringTokenize(string line)
{
    istringstream   strin(line);
    vector<string>  result;
    string          token;

    while (strin >> token)
        result.push_back(token);

    return result;
}

vector<string> CommonUtil::StringSplit(string line, char separator)
{
    vector<string>  result;
    line += separator;

    int p = 0;
    for (int i = 0; i < line.length(); i ++)
        if (line[i] == separator)
        {
            if (i - p > 0) result.push_back( line.substr(p, i-p) );
            p = i + 1;
        }

    return result;
}