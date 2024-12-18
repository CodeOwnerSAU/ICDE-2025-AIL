#ifndef GRAPH_H
#define	GRAPH_H
#include <time.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <chrono>
#include "tools.h"
#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include "heap.h"
#include <stdlib.h>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm> 
#include <stack>
#include <bitset>
#include <sys/time.h>
#include <xmmintrin.h>
#include <cmath>
#include <queue>
//#include <boost/heap/fibonacci_heap.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <boost/bind/bind.hpp>

using namespace std;
using namespace benchmark; 

static const int KIND = 4000;	//the number of keywords kind

typedef struct COMPARENODE
{
	pair<int, int> pif;//ID, minCost
	bool operator() (const struct COMPARENODE& a, const struct COMPARENODE& b) const  
	{  
		return a.pif.second > b.pif.second; 
	} 
}compareNode;

struct Edge
{
	int ID1, ID2, length, edgeID, cost; 
};

struct PTNode
{
	int nodeID;
	unordered_map<int, int>	PTRoad;	//childrenID, roadID
	unordered_map<int, int> PTChildren; //childrenID, node Pos
};

class Graph
{
public:
	Graph(){}

    Graph(const char * edgeFile)
	{

			readUSMap(edgeFile);
			//readUSMapCost(filename);

	}

	//True: Only adjList and adjListMap are used
	//False: Also use R versions
	int nodeNum;
	int edgeNum;
    int POINum;
	vector<vector<pair<int, int> > > adjList;		//neighborID, Distance
	vector<vector<pair<int, int> > > adjListR;
	vector<vector<pair<int, int> > > adjListEdge;	//neighbor,edgeID
	vector<vector<pair<int, int> > > adjListEdgeR;

	vector<vector<pair<int, int> > > adjListCost;		//neighborID, cost
	vector<vector<pair<int, int> > > adjListCostR;


	vector<Edge> vEdge;
	vector<Edge> vEdgeR; 

	int readCost(string filename);
	
	
	//Identify the ISO nodes
	vector<bool> vbISOF;	//forward
	vector<bool> vbISOB;	//backward
	vector<bool> vbISOU;	//F & B
	vector<bool> vbISO;		//F | B
	int ISONodes();
	int BFS(int nodeID, bool bF, vector<bool>& vbVisited);
	vector<pair<double, double> > vCoor; //存储经纬度
	unordered_map<string, pair<int, int> > mCoor;
	double minX, minY, maxX, maxY;

	//lon, lat
	int readBeijingMapDirected(string filename); 
	int readUSMap(const char * edgeFile);
	int readUSCost(string filename);
	int readUSMapCost(string filename);
	int readExampleMap(string filename);  
	int readUSCoor(string filename);
    int readEdge(string edgeFile);

	//test.cpp
	void testCSP(string filename);
	void SCPT(int root, vector<int>& vSPTDistance, vector<int>& vSPTCost, vector<int>& vSPTParent, vector<int>& vSPTParentEdge, vector<vector<int> >& vSPT, int C);
	void rCPT(int root, int ID1, int C, vector<int>& vrSPTCost, vector<int>& vrSPTDistance);

	// void contractNode(int threshold);
	int Dijkstra(int ID1, int ID2);
	int DijkstraPath(int ID1, int ID2, vector<int>& vPath, vector<int>& vPathEdge);
	int DijkstraPath2(int ID1, int ID2, unordered_set<int>& sRemovedNode, vector<int>& vPath, vector<int>& vPathEdge);
	int AStar(int ID1, int ID2);
	int AStarPath(int ID1, int ID2, vector<int>& vPath, vector<int>& vPathEdge, string& city); 
	int EuclideanDistance(int ID1, int ID2);
	int EuclideanDistanceAdaptive(int ID1, int ID2, int latU, int lonU);

	//Cache (not use)
	unordered_map<string, unordered_map<int, int> > Cache;
	unordered_map<string, unordered_map<int, int> > Cache_node;

	//set KEYS
	bitset<KIND>Qu;		// all query keywords bit
	bitset<KIND>QueryBit; 	// keywords bit not in ID1 and ID2
	vector<int>QueryWord;	// keywords not in ID1 and ID2
	vector<vector<int>> RSP;	// each keyword contain pois (0 ~ KIND-1) RSP[i] id is from 1
	vector<unordered_map<int, int>>KEYS;	//store each poi's keywords using map, and node start from zero
	vector<bitset<KIND>>NodesBit;	// store each node's keywords bit
	void set_nodeKEYS_NodesBit(string filename);	// init Query and each node keywors using map
	int Clen(int S,int T,bitset<KIND> &query, vector<int> &result);	// constraint len by Optimal shortest source-keyword-destination
    int ALLDeviationNode;
    int PruneDeviationNode;

	// H2H
    const int infinity = INT_MAX;
    const int SIZEOFINT = 4;
    int *toRMQ, *height, **RMQIndex;
    int *belong;
    int root, TreeSize;
    int **rootToRoot, *rootSite;
    int **dis,**dis2, **pos, **pos2;
    int *posSize, *pos2Size;
    int *chSize;
    int ** ch;
    int *LOG2, *LOGD;
    int rootSize;
    int *DFSList, *toDFS;
    int ***BS;
    int *Degree;
    int **Neighbor, **Weight;
    FILE *fin_Index;
    int *EulerSeq;
    long long queryCnt;
    int LCAQuery(int _p, int _q);
    /**
     * when use distanceQuery  p q
     * because in H2H.index is from id 1
     * but in road network id is form 0 ,so id must be id+1
     * @param p
     * @param q
     * @return
     */
    int distanceQuery(int p, int q);
    void readGraph(const char * filename);
    int shortestPathQuery(int p, int q);
    void scanIntArray(int *a, int n);
    int* scanIntVector(int *a);
    void readIndex(const char *file);
    void Init_H2H_Index(const char *index, const char* graph);
    int H2HPath(int ID1, int ID2, vector<int>& vPath, vector<int>& vPathEdge,vector<bitset<KIND>> &H2HPathBit);
		
	// DAmrrp
	int eKSPNew(int ID1, int ID2, int k, vector<int>& query,vector<int>& kResults, vector<vector<int> >& vkPath);
    //void SPT();
	void SPT(int root, vector<int>& vSPTDistance, vector<int>& vSPTParent, vector<int>& vSPTParentEdge, vector<vector<int> >& vSPT);

    int FindNN(int deviation,int t,bitset<KIND> &vPathBit);
	void FindRepeatedPath(vector<vector<int> >& vvPath);
	int PruneRepeatedPoiPath(vector<int>& bestpath, vector<int>& bestpoi);
	int PruneRepeatedPoiPath(vector<int> &bestpath);
	int FindPNN(int deviation,int t, bitset<KIND> &vPathBit, pair<int,int> &PNN );
    int FindNNDij(int deviation,int t,bitset<KIND> &vPathBit);
	int ComputeLB(vector<int> &vpath, bitset<KIND> &vpathposBit, int pos);
	void SPTb(int ID2, vector<int>& vSPTDistance, vector<int>& vSPTParent, vector<int>& vSPTParentEdge, vector<vector<int> >& vSPT);

    int getMaxLB(vector<int> &vpath, bitset<KIND> &vpathBit, int pos,int ID2);
    pair<int,int> getMaxLBNode(vector<int> &vpath, bitset<KIND> &vpathBit, int pos,int ID2);
    bitset<KIND> getempPathBit(vector<int> vector1);

    int getTempPathDistance(vector<int> vector1);
    int readPOIInfo(string nodeName);
    void setInitKeyWords(int ID1,int ID2,vector<int> &queryKey);
    int VDE(int ID1, int ID2, int k, vector<int>& query, vector<int>& kResults, vector<vector<int> >& vkPath);

};


#endif
