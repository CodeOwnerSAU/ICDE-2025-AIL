//
// Created by xiongxing on 9/21/24.
//
#include "Graph.h"
#include <cmath>
#include <sys/param.h>
inline int Graph::rightOf(int cid)
{
    if(cid  == -1)
        return -1;
    else
    {
        int result = cid + 1;
        return (result%colSize == 0) ? -1 : result;
    }
}

inline int Graph::leftOf(int cid)
{
    if(cid  == -1)
        return -1;
    else
    {
        int result = cid - 1;
        return (cid%colSize == 0) ? -1 : result;
    }
}

inline int Graph::upperOf(int cid)
{
    if(cid  == -1)
        return -1;
    else
    {
        int result = cid + colSize;
        return (result >= rowSize*colSize) ? -1 : result;
    }
}

inline int Graph::belowOf(int cid)
{
    if(cid  == -1)
        return -1;
    else
    {
        int result = cid - colSize;
        return (result<0) ? -1 : result;
    }
}
double degreesToRadians(double degrees) {
    return degrees * M_PI / 180.0;
}
double EuclideanDistance(double lon1 ,double lat1,double lon2,double lat2){
    lat1 = degreesToRadians(lat1);
    lon1 = degreesToRadians(lon1);
    lat2 = degreesToRadians(lat2);
    lon2 = degreesToRadians(lon2);
    double lat_diff = lat2 - lat1;
    double lon_diff = lon2 - lon1;

    double a = std::sin(lat_diff / 2) * std::sin(lat_diff / 2) +
               std::cos(lat1) * std::cos(lat2) *
               std::sin(lon_diff / 2) * std::sin(lon_diff / 2);
    double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1 - a));
    return 63710000.0* c;
}
int Graph::getGridID(Node &node){
    int col=abs(node.x-minx)/colLength;
    int row=abs(node.y-miny)/rowLength;
    return row*colSize+col;
}
void Graph::InitGridIndex(){
    //get Grid num
    for(int i=0;i<nodeNum;i++){
        minx= MIN(minx,NodeList[i].x);
        miny= MIN(miny,NodeList[i].y);
        maxx=MAX(maxx,NodeList[i].x);
        maxy=MAX(maxy,NodeList[i].y);
    }
    Node  leftDown;
    Node leftUp;
    Node rightDown;
    leftDown.x=minx;
    leftDown.y=miny;
    leftUp.x=minx;
    leftUp.y=maxy;
    rightDown.x=maxx;
    rightDown.y=miny;
    double div = 1000000.0;
    double xLength= EuclideanDistance(minx/div,miny/div,maxx/div,miny/div);
    double yLength= EuclideanDistance(minx/div,miny/div,minx/div,maxy/div);
    colLength=(maxx-minx)*(double)CELLSIZE/xLength;
    rowLength=(maxy-miny)*(double)CELLSIZE/yLength;
    colSize= ceil((maxx-minx)/colLength);
    rowSize= ceil((maxy-miny)/rowLength);
    int cellNum=colSize*rowSize;
    cellContain.resize(cellNum);
    cout<<"Init Grid info OK"<<endl;
}

void Graph::updateCellInfo(){
    vector<gridNode> gridNodeMapTemp(colSize*rowSize);
    fill(cellContain.begin(),cellContain.end(),0);
    for(auto nodeID:POIList){
        Node  POINode=NodeList[nodeID];
        int gridID= getGridID(POINode);
        gridNodeMapTemp[gridID].gridNodeBits|=NodesBit[nodeID];
        gridNodeMapTemp[gridID].POI.push_back(nodeID);
        cellContain[gridID]++;
    }
    gridNodeMap=gridNodeMapTemp;
}
vector<int> Graph::GridExtend(int s, int t, bitset<KIND> uncover) {
    int dx=NodeList[t].x-NodeList[s].x;
    int dy=NodeList[t].y-NodeList[s].y;
    if(dx==0||dy==0){
        if(dx==0){
            dx=1;
        }else if(dy==0){
            dy=1;
        }
    }
    int increx=dx/abs(dx);
    int increy=dy/abs(dy);
    vector<int> tmp;
    bitset<KIND> cellbit;
    vector<int> cellList;
    double k=(dy*1.0)/(dx*1.0);
    double b=-k*(NodeList[t].x*1.0)+NodeList[t].y*1.0;
    for(int i=0;i<abs(dx);i+=colLength/3){
        int x=NodeList[s].x+(increx*i);
        int y=(int)(k*x+b);
        int col=((x-minx))/colLength;
        int row=((y-miny))/rowLength;
        int cell=row*colSize+col;
        if(cell<0){
            cout<<cell<<endl;
        }
        tmp.push_back(cell);
        if(cellContain[cell]>0){
            //this means that cell contain POI that have keywords from queryKey
            if((gridNodeMap[cell].gridNodeBits&uncover).count()>0){
                cellbit|=gridNodeMap[cell].gridNodeBits&uncover;
                cellList.push_back(cell);
            }
        }
    }
    for(int i=0;i<abs(dy);i+=rowLength/3){
        int y=NodeList[s].y+(increy*i);
        int x=(int)((y-NodeList[s].y)/k)+NodeList[s].x;
        int col=((x-minx))/colLength;
        int row=((y-miny))/rowLength;
        int cell=row*colSize+col;
        tmp.push_back(cell);
        if(cellContain[cell]>0){
            //this means that cell contain POI that have keywords from queryKey
            if((gridNodeMap[cell].gridNodeBits&uncover).count()>0){
                cellbit|=gridNodeMap[cell].gridNodeBits&uncover;
                cellList.push_back(cell);
            }
        }
    }
    //uncoverbit^=cellbit;
    sort(tmp.begin(),tmp.end());
    tmp.erase(unique(tmp.begin(),tmp.end()),tmp.end());

    bitset<KIND> uncoverSecond=((cellbit^uncover)&uncover);
    vector<int> temp;
    int i=0;
    while(uncoverSecond.count()!=0&&(cellbit^uncover).count()!=0){
        temp.clear();
        for(int i=0;i<tmp.size();i++){
            //below
            int cell= belowOf(tmp[i]);
            if(cell!=-1){
                temp.push_back(cell);
                if(cellContain[cell]>0){
                    if((gridNodeMap[cell].gridNodeBits&uncoverSecond).count()>0){
                        cellbit|=gridNodeMap[cell].gridNodeBits&uncoverSecond;
                        cellList.push_back(cell);
                    }
                }
            }
            //upper
            cell= upperOf(tmp[i]);
            if(cell!=-1){
                temp.push_back(cell);
                if(cellContain[cell]>0){
                    if((gridNodeMap[cell].gridNodeBits&uncoverSecond).count()>0){
                        cellbit|=gridNodeMap[cell].gridNodeBits&uncoverSecond;
                        cellList.push_back(cell);
                    }
                }
            }
            //left
            cell= leftOf(tmp[i]);
            if(cell!=-1){
                temp.push_back(cell);
                if(cellContain[cell]>0){
                    if((gridNodeMap[cell].gridNodeBits&uncoverSecond).count()>0){
                        cellbit|=gridNodeMap[cell].gridNodeBits&uncoverSecond;
                        cellList.push_back(cell);
                    }
                }
            }
            //right
            cell= rightOf(tmp[i]);
            if(cell!=-1){
                temp.push_back(cell);
                if(cellContain[cell]>0){
                    if((gridNodeMap[cell].gridNodeBits&uncoverSecond).count()>0){
                        cellbit|=gridNodeMap[cell].gridNodeBits&uncoverSecond;
                        cellList.push_back(cell);
                    }
                }
            }
        }
        //cout<<i++<<endl;
        sort(temp.begin(),temp.end());
        //delete same cell
        temp.erase(unique(temp.begin(),temp.end()),temp.end());
        tmp.clear();
        //change tmp ,tmp=temp in order to next iterator
        tmp.insert(tmp.begin(),temp.begin(),temp.end());
        sort(cellList.begin(),cellList.end());
        cellList.erase(unique(cellList.begin(),cellList.end()),cellList.end());
    }
    sort(cellList.begin(),cellList.end());
    cellList.erase(unique(cellList.begin(),cellList.end()),cellList.end());
    vector<int> POICoordinate;
    POICoordinate.clear();
    //save all poi node in CellList
    vector<int> POIInCellList;
    for(int i=0;i<cellList.size();i++){
        for(auto node:gridNodeMap[cellList[i]].POI){
            if((NodesBit[node]&uncover).count()>0){
                POICoordinate.push_back(node);
            }
        }
    }
    return POICoordinate;
}
int Graph::firstRoundExpand(int s,int t,bitset<KIND> uncover,vector<int> &POICandidate){
    POICandidate=GridExtend(s,t,uncover);
    vector<int> temp;
    map<int,vector<int>> meetQueryPOI;
    for(int i=0;i<QueryWord.size();i++){
        meetQueryPOI.insert({QueryWord[i],temp});
    }
    for(auto q:QueryWord){
        for(int POI:POICandidate){
            if(NodesBit[POI].test(q)){
                meetQueryPOI[q].push_back(POI);
            }
        }
    }
    vector<pair<int,int>> mustNode;
    for(auto map:meetQueryPOI){
        if(map.second.size()==1){
            mustNode.push_back(make_pair(
                    distanceQuery(s+1,map.second[0]+1)+ distanceQuery(map.second[0]+1,t+1),map.first));
        }
    }
    if(mustNode.empty()){
        return -1;
    } else{
        std::sort(mustNode.begin(), mustNode.end());
        return meetQueryPOI[mustNode[0].second][0];
    }
}
void Graph::SearchPOICoordinate(int s, int t, bitset<KIND> uncover, vector<int> POISet) {
   POISet=GridExtend(s,t,uncover);
}
int Graph::buildGreedyPath(int startNode,int endNode,vector<int> POICandidate,vector<int> &Gpath){
    int pathDis=0;
    bitset<KIND> uncover=QueryBit;
    int currentNode=startNode;
    Gpath.clear();
    Gpath.push_back(startNode);
    int c=0;
    if(!POICandidate.empty()){
        while (uncover.count()>0){
            int NNdis=INT_MAX;
            int NNPOI;
            for(auto i=POICandidate.begin();i!=POICandidate.end();i++){
                if(std::find(Gpath.begin(), Gpath.end(),*i)==Gpath.end()&&
                   (uncover&NodesBit[*i]).count()>0){
                    int dis= distanceQuery(currentNode+1,*i+1);
                    if(dis<NNdis){
                        NNdis=dis;
                        NNPOI=*i;
                    }
                }
            }
            pathDis+= distanceQuery(currentNode+1,NNPOI+1);
            Gpath.push_back(NNPOI);
            uncover^=(QueryBit&NodesBit[NNPOI]&uncover);
            currentNode=NNPOI;
            //cout<<uncover.count()<<endl;
        }
    }
    pathDis+= distanceQuery(currentNode+1,endNode+1);
    Gpath.push_back(endNode);
    //cout<<"finsh"<<endl;
    return pathDis;
}
int Graph::buildGreedyMSPath(int s,int t,vector<int> POICandidate,vector<int> &Gpath){
    map<int,vector<pair<int,int>>> kwMap;
    int MAXover=0;
    for(auto node:POICandidate){
        int count=(NodesBit[node]&QueryBit).count();
        if(count>MAXover){
            MAXover=count;
        }
        kwMap[count].
                push_back(make_pair(distanceQuery(s+1,node+1)+ distanceQuery(node+1,t+1),node));
    }
    for (auto& [key, vec] : kwMap) {
        sort(vec.begin(), vec.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
            return a.first < b.first;
        });
    }
    vector<int> CPOI;
    bitset<KIND> uncover=QueryBit;
    Gpath.push_back(s);
    int flag=0;
    while(uncover.count()!=0){
        for(auto data:kwMap[MAXover]){
            if((NodesBit[data.second]&uncover).count()>0){
                uncover^=NodesBit[data.second]&QueryBit&uncover;
                CPOI.push_back(data.second);
                if(uncover.count()==0){
                    break;
                }
            }
        }
        if(uncover.count()<MAXover){
            MAXover=uncover.count();
        } else{
            MAXover--;
        }
//        if(MAXover==1){
//            //change NN to expand
//            cout<<"1"<<endl;
//            break;
//        }
    }
    int resultPathLength1,resultPathLength2;
    vector<int> resultPath1;
    vector<int> resultPath2;
    resultPathLength1=buildGreedyPath(s,t,CPOI,resultPath1);
    resultPathLength2=buildGreedyPath(t,s,CPOI,resultPath2);
    int newPath;
    if(resultPathLength1<resultPathLength2){
        RouteReArrangement(s,t,resultPath1,resultPathLength1,Gpath,newPath);
        Gpath=resultPath1;
        //return resultPathLength1;
        return resultPathLength1;
        //resultPathLength=resultPathLength1;
    } else{
        RouteReArrangement(t,s,resultPath2,resultPathLength2,Gpath,newPath);
        Gpath=resultPath2;
        //Gpath=resultPath2;
        //return resultPathLength2;
        return resultPathLength2;
        //resultPathLength=resultPathLength2;
    }
    //return resultPathLength1;


}
int Graph::GOD_NN(int s,int t,vector<int> QueryList,vector<int> &resultPath,int &resultPathLength){
    bitset<KIND> uncover=QueryBit;
    vector<int> CPOI;
    set<int> POISet;
    int mustPOI=firstRoundExpand(s,t,uncover,CPOI);
    int resultPathLength1,resultPathLength2;
    vector<int> resultPath1;
    vector<int> resultPath2;
    if(mustPOI==-1){
        //
        resultPathLength1=buildGreedyPath(s,t,CPOI,resultPath1);
        resultPathLength2=buildGreedyPath(t,s,CPOI,resultPath2);
        if(resultPathLength1<resultPathLength2){
            resultPath=resultPath1;
            resultPathLength=resultPathLength1;
        } else{
            resultPath=resultPath2;
            resultPathLength=resultPathLength2;
        }
    } else {
        bitset<KIND> SMBit = SPBit(s, mustPOI);
        bitset<KIND> MTBit = SPBit(mustPOI, t);
        if (((SMBit | MTBit) & QueryBit).count() == QueryBit.count()) {
            resultPath.push_back(s);
            resultPath.push_back(mustPOI);
            resultPathLength += distanceQuery(s + 1, mustPOI + 1);
            resultPathLength += distanceQuery(mustPOI + 1, t + 1);
            resultPath.push_back(t);
            return resultPathLength;
        }
        POISet.insert(mustPOI);
        uncover ^= QueryBit & NodesBit[mustPOI];
        vector<int> SMPOI;
        vector<int> MTPOI;
        SMPOI = GridExtend(s, mustPOI, uncover);
        MTPOI = GridExtend(mustPOI, t, uncover);
        for (auto node: SMPOI) {
            POISet.insert(node);
        }
        for (auto node: MTPOI) {
            POISet.insert(node);
        }
        vector<int> POIALL(POISet.begin(), POISet.end());
        resultPathLength1 = buildGreedyPath(s, t, POIALL, resultPath1);
        resultPathLength=resultPathLength1;
        resultPathLength2 = buildGreedyPath(t, s, POIALL, resultPath2);
        if (resultPathLength1 < resultPathLength2) {
            resultPathLength = resultPathLength1;
        } else {
            resultPathLength = resultPathLength2;
        }
    }
    return resultPathLength;
}
int Graph::GOD_MS(int s,int t,vector<int> QueryList,vector<int> &resultPath,int &resultPathLength){
    bitset<KIND> uncover=QueryBit;
    vector<int> CPOI;
    set<int> POISet;
    int mustPOI=firstRoundExpand(s,t,uncover,CPOI);
    if(mustPOI==-1){
        resultPathLength=buildGreedyMSPath(s,t,CPOI,resultPath);
        //resultPathLength=buildGreedyPath(s,t,CPOI,resultPath);
    } else{
        POISet.insert(mustPOI);
        uncover^=QueryBit&NodesBit[mustPOI];
        vector<int> SMPOI;
        vector<int> MTPOI;
        SMPOI=GridExtend(s,mustPOI,uncover);
        MTPOI= GridExtend(mustPOI,t,uncover);
        for(auto node:SMPOI){
            POISet.insert(node);
        }
        for(auto node:MTPOI){
            POISet.insert(node);
        }
        vector<int> POIALL(POISet.begin(),POISet.end());
        resultPathLength=buildGreedyMSPath(s,t,POIALL,resultPath);
    }
}