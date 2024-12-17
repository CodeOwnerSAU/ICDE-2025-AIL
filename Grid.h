//
// Created by xiongxing on 3/25/24.
//

#ifndef GRIDINDEX_GRID_H
#define GRIDINDEX_GRID_H

//1.5Km
#include "MBR.h"
#include "Graph.h"
#include "Point.h"
using namespace std;
//init is 100m 单位是分米
static const int CELLSIZE=30000;
static const int MINPOINT=4;
class Grid {
public:
    Grid(int _colSize, int _rowSize);
    Grid(int _colSize, int _rowSize, MBR* _extend);
    Grid(MBR* _extend);
    Grid(){};
    Grid(Grid* grid);
    bool isBuilt();
    int numCells();
    ~Grid(){};
    int count;
    vector<vector<int>> nodeMap;
    struct gridNode{
        bitset<KIND> gridNodeBits;
        vector<int> POI;
    };
    vector<gridNode> gridNodeMap;
    int colSize;	// # of columns
    int rowSize;	// # of rows
    int colLength;
    int rowLength;
    MBR *extend;
    vector<int> cellContain;

    void createGridIndex(Graph graph);

    void buildIndex(Graph graph);
    void calculateGridSize(const int i);

    int getCell(Point &point);

    void get_POI_Straight_From_StoT(Graph &graph, int s, int t, vector<int> &cell);

    int rightOf(int cid); // return the right cell of given cell
    int upperOf(int cid); // return the upper cell of given cell
    int belowOf(int cid); // return the below cell of given cell
    int leftOf(int cid); // return the left cell of given cell
};

#endif //GRIDINDEX_GRID_H
