#include "graph.h"
#include "MinHeap.h"

//
// Created by xiongxing on 6/27/24.
//

void Graph::GetNN(int topPathID,int findNode,bitset<KIND> topPathBit,vector<vector<int>> &pathXNNID,vector<vector<int>> &pathXNNLength){
    vector<pair<int,int>> nodeList;
    //benchmark::heap<2, int, int> queue(nodeNum);  //size
    for(auto key:QueryWord){
        if(!topPathBit.test(key)){
            for(int i=0;i<RSP[key].size();i++){
                nodeList.push_back(make_pair(distanceQuery(findNode+1,RSP[key][i]+1),RSP[key][i]));
            }
        }
    }
    sort(nodeList.begin(),nodeList.end());
    for(int i=0;i<nodeList.size();i++){
        //queue.extract_min(XNN,XNNLength);
        pathXNNID[topPathID].push_back(nodeList[i].second);
        pathXNNLength[topPathID].push_back(nodeList[i].first);
    }
}
bool textDominated(vector<vector<int>> pathList,vector<int> pathLength,vector<bitset<KIND>> pathListBit,vector<int> newPath,int newPathLength,bitset<KIND> newPathBit){
    for(int i=0;i<pathList.size();i++){
        if(pathList[i].back()!=newPath.back())
            continue;
        if(pathList[i].back()==newPath.back()&&pathLength[i]<newPathLength&&pathListBit[i].count()>=newPathBit.count()){
            cout<<"new path is dominated"<<endl;
            return true;
        } else{
            return false;
        }
    }
    return false;
}
bool textDominatedNew(vector<vector<int>> pathList,vector<int> &nodePathList,vector<int> pathLength,vector<bitset<KIND>> pathListBit,
                      vector<int> newPath,int newPathLength,bitset<KIND> newPathBit){
    //cout<<"path size "<<pathList.size()<<endl;
    //cout<<"node path size "<<nodePathList.size()<<endl;
    for(auto i:nodePathList){
        if(pathList[i].back()!=newPath.back()){
            continue;
        }
        if(pathList[i].back()==newPath.back()&&pathLength[i]<=newPathLength&&pathListBit[i].count()>=newPathBit.count()){
            //cout<<"new path is dominated"<<endl;
            return true;
        }
    }
    return false;
}
int Graph::readPOIInfo(string nodeFile) {
    QueryWord.clear();
    RSP.clear();
    KEYS.clear();
    NodesBit.clear();
    NodeList.clear();
    POINum=0;
    //NodeList.resize(nodeNum);
    for(int i = 0;i < KIND; i++){
        vector<int>tmp;
        RSP.push_back(tmp);
    }
    ifstream inNode(nodeFile);
    cout << "Reading " << nodeFile << endl;
    long lon, lat;
    int id;
    string keywords;
    string L;
    int v=0;
    int k=0;
    while(!inNode.eof())
    {
        inNode>>id>>lon>>lat>>keywords;
        vector<string> keys;
        boost::split(keys,keywords,boost::is_any_of(","),boost::token_compress_on);
        vector<int> temp;
        bitset<KIND> bitemp;
        Node node={id-1,lon,lat};
        node.keyword.reset();
        if(keys.size()==1){
            NodesBit.push_back(bitemp);
        } else{
            POINum++;
            for(int i=0;i<keys.size();i++) {
                if (stoi(keys[i]) < 0) {
                    continue;
                }
                int key = stoi(keys[i]);
                if (key == -1)
                    break;
                bitemp.set(key);
                RSP[key].push_back(id-1);
            }
            NodesBit.push_back(bitemp);
        }
        //cout<<id-1<<endl;
        keys.clear();
    }
    cout<<"Node Lon Lat read finished"<<endl;
    cout<<"--------Graph loading finished-----!!!"<<endl;
    inNode.close();
}
