#include "graph.h"
#include "BinaryHeap.h"
struct ComparePair {
    bool operator()(const pair<vector<int>, int>& p1, const pair<vector<int>, int>& p2) {
        return p1.second > p2.second; // 按照pair.second越小的优先级越高
    }
};
int Graph::DA(int ID1, int ID2, int k, vector<int>& query, vector<int>& kResults, vector<vector<int> >& vkPath){
	std::chrono::high_resolution_clock::time_point t1;
	std::chrono::high_resolution_clock::time_point t2;
	std::chrono::duration<double> time_span;
	std::chrono::duration<double> time_lb;
	std::chrono::duration<double> time_pnn;
    //BinaryHeap heap;r
	// Initialize global var
//	QueryWord.clear();// not include query keywords in both ID1 and ID2 vector<int>
//	QueryBit.reset();// not include query keywords in both ID1 and ID2  bitset
//	Qu.reset();		// include query keywords in both ID1 and ID2
//	Cache.clear();
//	//set QueryBit and QueryWord
//	for(int k:query){
//		Qu.set(k);
//		if( !NodesBit[ID1].test(k) && !NodesBit[ID2].test(k) ){
//			QueryWord.push_back(k);
//            QueryBit.set(k);
//		}
//	}
//	QueryBit ^= (QueryBit & ( NodesBit[ID1] | NodesBit[ID2] ));//delete keywords from ID1 and ID2
	//Shortest Path Tree Info
    t1 = std::chrono::high_resolution_clock::now();
    vector<int> vSPTDistance(nodeNum, INF); //record dist from node to root ID2
	vector<int> vSPTParent(nodeNum, -1);  //record node's prenode in shorest path
	vector<int> vSPTParentEdge(nodeNum, -1);
	vector<int> vTmp;
	vector<vector<int> > vSPT(nodeNum, vTmp); //Tree from root  ID2
	vector<bitset<KIND> > vSPTBit(NodesBit);  //path binary from node to root ID2
	SPT(ID2, vSPTDistance, vSPTParent, vSPTParentEdge, vSPT);
	// SPTree for compute path binary from node to ID2
    queue<int> SPTree;
    SPTree.push(ID2);
    while(!SPTree.empty()){
        int node = SPTree.front();
        if(vSPTParent[node] != -1){
            vSPTBit[node] |= vSPTBit[vSPTParent[node]];
        }
        SPTree.pop();
        if(vSPT[node].size() > 0){
            for(int i = 0 ;i < vSPT[node].size(); i++){
                SPTree.push(vSPT[node][i]);
            }
        }
    }
	//maintain the path in PT tree
 	vector<int> vDistance;		//distance
     vector<int> vPreDistance;
	vector<int> vPathParent;	//current path parent path id
	vector<int> vPathParentPos;	//Deviated Pos from Parent(index)
 	vector< vector<int> > vvPathCandidate;	 //nodes
	vector< vector<int> > vvPathCandidateEdge;//edges
	vector< bitset<KIND> > vPathBit;  //curpath keywords binary
	vector< bitset<KIND> > vPathParentBit;  //curpath keywords binary
	vector<int> vPathLB;
	vector<int> vPath;
	vector<int> vPathEdge;
	vector<int> best;	//return result path
	int bestcost;	//result path cost
	//push the shortest path into queue
	vPath.push_back(ID1);
	int p = vSPTParent[ID1];
	int e = vSPTParentEdge[ID1];
    //对应于从当前点作为起点 随后从最短路径树里依次找路径
	while(p != -1)
	{
		vPath.push_back(p);
		vPathEdge.push_back(e);
		e = vSPTParentEdge[p];
		p = vSPTParent[p];
	}
    vector<int> a,b;
    vector<bitset<KIND>> c;
	bitset<KIND>testBit(QueryBit);
	testBit &= vSPTBit[ID1];
    //sp has cover all keywords
	if(QueryBit.count() == 0 || QueryBit.count() == testBit.count()){
        //sp(s,t) has cover all Query keywords
		vkPath.push_back(vPath);
		kResults.push_back(vSPTDistance[ID1]);
        return vkPath.size();
	}
    benchmark::heap<2, int, int> qPath(nodeNum);
    vvPathCandidate.push_back(vPath);
    vvPathCandidateEdge.push_back(vPathEdge);
    //total distance from ID1 to ID2
    vDistance.push_back(vSPTDistance[ID1]);
    vPreDistance.push_back(0);
    vPathParent.push_back(ID1);
    vPathParentPos.push_back(0);
    vPathBit.push_back( QueryBit & (NodesBit[ID1] | NodesBit[ID2]));
    int lb=0;
    vPathLB.push_back(lb);
    vector<vector<int>> pathPOISet(nodeNum);
    for(int i=0;i<nodeNum;i++){
        vector<int> t;
        pathPOISet.push_back(t);
    }
    qPath.update(vvPathCandidate.size()-1,vSPTDistance[ID1]+vPathLB[0]);
    int topPathID,topPathDistance;
    int UB=0;//upper bound to prune current partial path which is greater than UB
    int it=0;
    //deviation start until find k path
    while(kResults.size()<k&&qPath.size()>0){
        t2 = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
        if(time_span.count()>15.0){
            kResults.push_back(-1);
            break;
        }
        //get a partial that have min cost
        qPath.extract_min(topPathID,topPathDistance);
        if(qPath.empty()&&topPathID!=0){
            cout<<"empty"<<endl;
        }
        //test topPath keywords Set whether cover all query kw
        bitset<KIND> topPathBit(vPathBit[topPathID]);
        //connect SPT bit
        topPathBit|=((vSPTBit[vPathParent[topPathID]]|topPathBit)&QueryBit);
        if(topPathBit.count()==QueryBit.count()&&qPath.size()>=0){
            int newPathLength= PruneRepeatedPoiPath(vvPathCandidate[topPathID]);
            if(std::find(kResults.begin(), kResults.end(),newPathLength)!=kResults.end()){
                continue;
            }
//            if(topPathID!=0&&newPathLength==kResults.back()){
//                continue;
//            }
            kResults.push_back(newPathLength);
            vkPath.push_back(vvPathCandidate[topPathID]);
            if(newPathLength>UB){
                    UB=newPathLength;
                    continue;
            }
        }
        //---------------------------------------------
        //UB pruned
        if(vPreDistance[topPathID]+vPathLB[topPathID]>UB&&UB!=0){
            //cout<<"UB pruned"<<endl;
            continue;//upper bound prune
        }
        //---------------------------------------------
        int pos=vPathParentPos[topPathID];
        bitset<KIND> tempBit(vPathBit[topPathID]);
        int lastNN=-1;
        vector<int> isPrune(nodeNum,1);
        for(int i=pos;i<vvPathCandidate[topPathID].size()-1;i++){
            //deviation pruned-----------------
            if(i > (vvPathCandidate[topPathID].size()-1-pos)/2 && pos < (vvPathCandidate[topPathID].size()-1)/3 )//&& devflag)
            {
                //cout<<"deviation pruned"<<endl;
                break;
            }
            //--------------------------------
            vPath.clear();
            int deviation=vvPathCandidate[topPathID][i];
            int lastCount=tempBit.count();
            tempBit|=(QueryBit&NodesBit[deviation]);
            if(isPrune[deviation]==0){
                continue;
            }
            vector<int> tmPath(vvPathCandidate[topPathID].begin(),vvPathCandidate[topPathID].begin()+i+1);
            int PNN=-1;
            PNN=FindNN(deviation,ID2,tempBit);
            pathPOISet[topPathID].push_back(PNN);
            //NN pruned
            if(PNN==lastNN&&tempBit.count()==lastCount||PNN==-1){
                //cout<<"NN pruned"<<endl;
                continue;
            }
            lastNN=PNN;
            vector<int> vPathPNN;
            vector<int> vPathPNNEdge;
            vector<bitset<KIND>> H2HPathBit;
            int NNdis=H2HPath(deviation,PNN,vPathPNN,vPathEdge,H2HPathBit);
            for(int i=1;i<vPathPNN.size()-1;i++){
                if(find(vvPathCandidate[topPathID].begin(), vvPathCandidate[topPathID].end(),vPathPNN[i])!=vvPathCandidate[topPathID].end()){
                    //confirm that path from deviation to NN have a node ,this node is in topPath
                    if((QueryBit&NodesBit[vPathPNN[i]]|tempBit).count()<=tempBit.count()){
                        //if this node can not give another query keywords,this node is pruned
                        //cout<<"path pruned"<<endl;
                        isPrune[vPathPNN[i]]=0;
                    }
                } else if(i==1){
                    break;
                }
            }
            if(NNdis!=0){
                //insert path from deviation node to NN
                tmPath.insert(tmPath.end(),vPathPNN.begin()+1,vPathPNN.end());
            }
            bitset<KIND> newPathBit(tempBit);
            newPathBit|=(NodesBit[PNN]|H2HPathBit.back())&QueryBit;
            int tmpos=tmPath.size()-1;
            //connect sp from PNN to ID2
            int p=vSPTParent[PNN];
            //tmPath.push_back(PNN);
            //int da=vPreDistance[topPathID]+NNdis;
            int da= PruneRepeatedPoiPath(tmPath);
            //insert node from NN to ID2 by SPT
            while(p!=-1){
                tmPath.push_back(p);
                p=vSPTParent[p];
            }
            int tmpDist=da+vSPTDistance[PNN];
            int tmpLB=getMaxLB(tmPath,newPathBit,tmPath[tmpos],ID2);
            vvPathCandidate.push_back(tmPath);
            vPathParent.push_back(PNN);
            vPathParentPos.push_back(tmpos);//tmpos is NN index
            vPathBit.push_back(newPathBit);
            vDistance.push_back(tmpDist);
            vPreDistance.push_back(da);
            vPathLB.push_back(tmpLB);
            qPath.update(vvPathCandidate.size()-1,da+tmpLB);
        }
    }
    return qPath.size();
}

int Graph::DA_Prune(int ID1, int ID2, int k, vector<int>& query, vector<int>& kResults, vector<vector<int> >& vkPath){
    ALLDeviationNode=0;
    PruneDeviationNode=0;
    kResults.clear();
    vkPath.clear();
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::duration<double> time_span;
    std::chrono::duration<double> time_lb;
    std::chrono::duration<double> time_pnn;
    t1 = std::chrono::high_resolution_clock::now();
    //maintain the path in PT tree
    vector<int> vDistance;		//distance
    vector<int> vPreDistance;
    vector<int> vPathParent;	//current path parent path id
    vector<int> vPathParentPos;	//Deviated Pos from Parent(index)
    vector< vector<int> > vvPathCandidate;	 //nodes
    vector< vector<int> > vvPathCandidateEdge;//edges
    vector< bitset<KIND> > vPathBit;  //curpath keywords binary
    vector< bitset<KIND> > vPathParentBit;  //curpath keywords binary
    vector<int> vPathLB;
    vector<int> vPath;
    vector<int> vPathEdge;
    vector<bitset<1000000>> vvPathNodeB;
    vector<int> best;	//return result path
    int bestcost;	//result path cost
    //push the shortest path into queue
    vPath.push_back(ID1);
    vector<int> h2hPath;
    vector<int> h2hEdge;
    vector<bitset<KIND>> h2hBist;
    int Dis=H2HPath(ID1,ID2,h2hPath,h2hEdge,h2hBist);
    if((h2hBist.back()&QueryBit).count()==QueryBit.count()){
        vkPath.push_back(h2hPath);
        kResults.push_back(Dis);
        return vkPath.size();
    }
    benchmark::heap<2, int, int> qPath(nodeNum*100);
    vvPathCandidate.push_back(h2hPath);
    //vvPathCandidateEdge.push_back(vPathEdge);
    //total distance from ID1 to ID2
    vDistance.push_back(Dis);
    vPreDistance.push_back(0);
    vPathParent.push_back(ID1);
    vPathParentPos.push_back(0);
    bitset<1000000> pathB;
    pathB.set(ID1);
    vvPathNodeB.push_back(pathB);
    vPathBit.push_back( QueryBit & (NodesBit[ID1] | NodesBit[ID2]));
    int lb=0;
    vPathLB.push_back(lb);
    qPath.update(vvPathCandidate.size()-1,Dis+vPathLB[0]);
    int topPathID,topPathDistance;
    int UB=0;//upper bound to prune current partial path which is greater than UB
    int it=0;
    //deviation start until find k path
    while(kResults.size()<k&&qPath.size()>0){
        t2 = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
        if(time_span.count()>15.0){
            kResults.push_back(-1);
            break;
        }
        //get a partial that have min cost
        qPath.extract_min(topPathID,topPathDistance);
        if(qPath.empty()&&topPathID!=0){
            cout<<"empty"<<endl;
        }
        //test topPath keywords Set whether cover all query kw
        bitset<KIND> topPathBit(vPathBit[topPathID]);
        h2hPath.clear();
        h2hEdge.clear();
        h2hBist.clear();
        H2HPath(vPathParent[topPathID],ID2,h2hPath,h2hEdge,h2hBist);
        //connect SPT bit
        topPathBit|=((h2hBist.back()|topPathBit)&QueryBit);
        if(topPathBit.count()==QueryBit.count()&&qPath.size()>=0){
            int newPathLength= PruneRepeatedPoiPath(vvPathCandidate[topPathID]);
            if(std::find(kResults.begin(), kResults.end(),newPathLength)!=kResults.end()){
                continue;
            }
//            if(topPathID!=0&&newPathLength==kResults.back()){
//                continue;
//            }
            kResults.push_back(newPathLength);
            vkPath.push_back(vvPathCandidate[topPathID]);
            if(newPathLength>UB){
                UB=newPathLength;
                continue;
            }
        }
        //---------------------------------------------
        //UB pruned
        if(vPreDistance[topPathID]+vPathLB[topPathID]>UB&&UB!=0){
            //cout<<"UB pruned"<<endl;
            continue;//upper bound prune
        }
        //---------------------------------------------
        int pos=vPathParentPos[topPathID];
        bitset<KIND> tempBit(vPathBit[topPathID]);
        int lastNN=-1;
        vector<int> isPrune(nodeNum,1);
        //bitset<1000000> vpathB(vvPathNodeB[topPathID]);
        for(int i=pos;i<vvPathCandidate[topPathID].size()-1;i++){
            ALLDeviationNode++;
            //deviation pruned-----------------
            if(i > (vvPathCandidate[topPathID].size()-1-pos)/2 && pos < (vvPathCandidate[topPathID].size()-1)/3 )//&& devflag)
            {
                //cout<<"deviation pruned"<<endl;
                break;
            }
            //--------------------------------
            vPath.clear();
            int deviation=vvPathCandidate[topPathID][i];
            int lastCount=tempBit.count();
            tempBit|=(QueryBit&NodesBit[deviation]);
            if(isPrune[deviation]==0){
                continue;
            }
            vector<int> tmPath(vvPathCandidate[topPathID].begin(),vvPathCandidate[topPathID].begin()+i+1);
            //vpathB.set(vvPathCandidate[topPathID][i]);
            int PNN=-1;
            PNN=FindNN(deviation,ID2,tempBit);
            //pathPOISet[topPathID].push_back(PNN);
            //NN pruned
            if(PNN==lastNN&&tempBit.count()==lastCount||PNN==-1){
                //cout<<"NN pruned"<<endl;
                continue;
            }
            lastNN=PNN;
            vector<int> vPathPNN;
            vector<int> vPathPNNEdge;
            vector<bitset<KIND>> H2HPathBit;
            int NNdis=H2HPath(deviation,PNN,vPathPNN,vPathEdge,H2HPathBit);
            for(int i=1;i<vPathPNN.size()-1;i++){
                if(find(vvPathCandidate[topPathID].begin(), vvPathCandidate[topPathID].end(),vPathPNN[i])!=vvPathCandidate[topPathID].end()){
                    //confirm that path from deviation to NN have a node ,this node is in topPath
                    if((QueryBit&NodesBit[vPathPNN[i]]|tempBit).count()<=tempBit.count()){
                        //if this node can not give another query keywords,this node is pruned
                        //cout<<"path pruned"<<endl;
                        isPrune[vPathPNN[i]]=0;
                        PruneDeviationNode++;
                    }
                } else if(i==1){
                    break;
                }
            }

            if(NNdis!=0){
                tmPath.insert(tmPath.end(),vPathPNN.begin()+1,vPathPNN.end());
            }
            bitset<KIND> newPathBit(tempBit);
            newPathBit|=(NodesBit[PNN]|H2HPathBit.back())&QueryBit;
            int tmpos=tmPath.size()-1;
            //connect sp from PNN to ID2
            h2hPath.clear();
            h2hEdge.clear();
            h2hBist.clear();
            int da= PruneRepeatedPoiPath(tmPath);
            Dis=H2HPath(vPathPNN.back(),ID2,h2hPath,h2hEdge,h2hBist);
            tmPath.insert(tmPath.end(),h2hPath.begin()+1,h2hPath.end());
            int tmpLB=getMaxLB(tmPath,newPathBit,tmPath[tmpos],ID2);
            vvPathCandidate.push_back(tmPath);
            vPathParent.push_back(PNN);
            vPathParentPos.push_back(tmpos);//tmpos is NN index
            vPathBit.push_back(newPathBit);
            vDistance.push_back(da+Dis);
            vPreDistance.push_back(da);
            vPathLB.push_back(tmpLB);
            qPath.update(vvPathCandidate.size()-1,da+tmpLB);
        }
    }
    return qPath.size();
}
int Graph::DA_Prune_NoPrune(int ID1, int ID2, int k, vector<int>& query, vector<int>& kResults, vector<vector<int> >& vkPath){
    ALLDeviationNode=0;
    PruneDeviationNode=0;
    kResults.clear();
    vkPath.clear();
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::duration<double> time_span;
    std::chrono::duration<double> time_lb;
    std::chrono::duration<double> time_pnn;
    t1 = std::chrono::high_resolution_clock::now();
    //maintain the path in PT tree
    vector<int> vDistance;		//distance
    vector<int> vPreDistance;
    vector<int> vPathParent;	//current path parent path id
    vector<int> vPathParentPos;	//Deviated Pos from Parent(index)
    vector< vector<int> > vvPathCandidate;	 //nodes
    vector< vector<int> > vvPathCandidateEdge;//edges
    vector< bitset<KIND> > vPathBit;  //curpath keywords binary
    vector< bitset<KIND> > vPathParentBit;  //curpath keywords binary
    vector<int> vPathLB;
    vector<int> vPath;
    vector<int> vPathEdge;
    vector<bitset<1000000>> vvPathNodeB;
    vector<int> best;	//return result path
    int bestcost;	//result path cost
    //push the shortest path into queue
    vPath.push_back(ID1);
    vector<int> h2hPath;
    vector<int> h2hEdge;
    vector<bitset<KIND>> h2hBist;
    int Dis=H2HPath(ID1,ID2,h2hPath,h2hEdge,h2hBist);
    if((h2hBist.back()&QueryBit).count()==QueryBit.count()){
        vkPath.push_back(h2hPath);
        kResults.push_back(Dis);
        return vkPath.size();
    }
    benchmark::heap<2, int, int> qPath(nodeNum*100);
    vvPathCandidate.push_back(h2hPath);
    //vvPathCandidateEdge.push_back(vPathEdge);
    //total distance from ID1 to ID2
    vDistance.push_back(Dis);
    vPreDistance.push_back(0);
    vPathParent.push_back(ID1);
    vPathParentPos.push_back(0);
    bitset<1000000> pathB;
    pathB.set(ID1);
    vvPathNodeB.push_back(pathB);
    vPathBit.push_back( QueryBit & (NodesBit[ID1] | NodesBit[ID2]));
    int lb=0;
    vPathLB.push_back(lb);
    qPath.update(vvPathCandidate.size()-1,Dis+vPathLB[0]);
    int topPathID,topPathDistance;
    int UB=0;//upper bound to prune current partial path which is greater than UB
    int it=0;
    //deviation start until find k path
    while(kResults.size()<k&&qPath.size()>0){
        t2 = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
        if(time_span.count()>15.0){
            kResults.push_back(-1);
            break;
        }
        //get a partial that have min cost
        qPath.extract_min(topPathID,topPathDistance);
        if(qPath.empty()&&topPathID!=0){
            cout<<"empty"<<endl;
        }
        //test topPath keywords Set whether cover all query kw
        bitset<KIND> topPathBit(vPathBit[topPathID]);
        h2hPath.clear();
        h2hEdge.clear();
        h2hBist.clear();
        H2HPath(vPathParent[topPathID],ID2,h2hPath,h2hEdge,h2hBist);
        //connect SPT bit
        topPathBit|=((h2hBist.back()|topPathBit)&QueryBit);
        if(topPathBit.count()==QueryBit.count()&&qPath.size()>=0){
            int newPathLength= PruneRepeatedPoiPath(vvPathCandidate[topPathID]);
            if(std::find(kResults.begin(), kResults.end(),newPathLength)!=kResults.end()){
                continue;
            }
//            if(topPathID!=0&&newPathLength==kResults.back()){
//                continue;
//            }
            kResults.push_back(newPathLength);
            vkPath.push_back(vvPathCandidate[topPathID]);
            if(newPathLength>UB){
                UB=newPathLength;
                continue;
            }
        }
        //---------------------------------------------
        //UB pruned
        if(vPreDistance[topPathID]+vPathLB[topPathID]>UB&&UB!=0){
            //cout<<"UB pruned"<<endl;
            continue;//upper bound prune
        }
        //---------------------------------------------
        int pos=vPathParentPos[topPathID];
        bitset<KIND> tempBit(vPathBit[topPathID]);
        int lastNN=-1;
        //bitset<1000000> vpathB(vvPathNodeB[topPathID]);
        for(int i=pos;i<vvPathCandidate[topPathID].size()-1;i++){
            ALLDeviationNode++;
            //deviation pruned-----------------
//            if(i > (vvPathCandidate[topPathID].size()-1-pos)/2 && pos < (vvPathCandidate[topPathID].size()-1)/3 )//&& devflag)
//            {
//                //cout<<"deviation pruned"<<endl;
//                break;
//            }
            //--------------------------------
            vPath.clear();
            int deviation=vvPathCandidate[topPathID][i];
            int lastCount=tempBit.count();
            tempBit|=(QueryBit&NodesBit[deviation]);
            vector<int> tmPath(vvPathCandidate[topPathID].begin(),vvPathCandidate[topPathID].begin()+i+1);
            //vpathB.set(vvPathCandidate[topPathID][i]);
            int PNN=-1;
            PNN=FindNN(deviation,ID2,tempBit);
            //pathPOISet[topPathID].push_back(PNN);
            //NN pruned
            if(PNN==lastNN&&tempBit.count()==lastCount||PNN==-1){
                //cout<<"NN pruned"<<endl;
                continue;
            }
            lastNN=PNN;
            vector<int> vPathPNN;
            vector<int> vPathPNNEdge;
            vector<bitset<KIND>> H2HPathBit;
            int NNdis=H2HPath(deviation,PNN,vPathPNN,vPathEdge,H2HPathBit);
//            for(int i=1;i<vPathPNN.size()-1;i++){
//                if(vvPathNodeB[topPathID].test(vPathPNN[i])==1&&
//                (QueryBit&NodesBit[vPathPNN[i]]|tempBit).count()<=tempBit.count()){
//                    isPrune[vPathPNN[i]]=0;
//                    //cout<<"prune"<<endl;
//                } else if(i==1){
//                    break;
//                }
//            }


            if(NNdis!=0){
                //insert path from deviation node to NN
//                for(auto node:vPathPNN){
//                    vpathB[node]=true;
//                }
                tmPath.insert(tmPath.end(),vPathPNN.begin()+1,vPathPNN.end());
            }
            bitset<KIND> newPathBit(tempBit);
            newPathBit|=(NodesBit[PNN]|H2HPathBit.back())&QueryBit;
            int tmpos=tmPath.size()-1;
            //connect sp from PNN to ID2
            h2hPath.clear();
            h2hEdge.clear();
            h2hBist.clear();
            int da= PruneRepeatedPoiPath(tmPath);
            Dis=H2HPath(vPathPNN.back(),ID2,h2hPath,h2hEdge,h2hBist);
            tmPath.insert(tmPath.end(),h2hPath.begin()+1,h2hPath.end());
            //int p=vSPTParent[PNN];
            //tmPath.push_back(PNN);
            //int da=vPreDistance[topPathID]+NNdis;
            //insert node from NN to ID2 by SPT
            //int tmpDist=da+Dis;
            int tmpLB=getMaxLB(tmPath,newPathBit,tmPath[tmpos],ID2);
            //vvPathNodeB.push_back(vpathB);
            vvPathCandidate.push_back(tmPath);
            vPathParent.push_back(PNN);
            vPathParentPos.push_back(tmpos);//tmpos is NN index
            vPathBit.push_back(newPathBit);
            vDistance.push_back(da+Dis);
            vPreDistance.push_back(da);
            vPathLB.push_back(tmpLB);
            qPath.update(vvPathCandidate.size()-1,da+tmpLB);
        }
    }
    return qPath.size();
}
int Graph::DA_Dij(int ID1, int ID2, int k, vector<int>& query, vector<int>& kResults, vector<vector<int> >& vkPath){
    kResults.clear();
    vkPath.clear();
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::duration<double> time_span;
    std::chrono::duration<double> time_lb;
    std::chrono::duration<double> time_pnn;
    t1 = std::chrono::high_resolution_clock::now();
    vector<int> vDistance;		//distance
    vector<int> vPreDistance;
    vector<int> vPathParent;	//current path parent path id
    vector<int> vPathParentPos;	//Deviated Pos from Parent(index)
    vector< vector<int> > vvPathCandidate;	 //nodes
    vector< vector<int> > vvPathCandidateEdge;//edges
    vector< bitset<KIND> > vPathBit;  //curpath keywords binary
    vector< bitset<KIND> > vPathParentBit;  //curpath keywords binary
    vector<int> vPathLB;
    vector<int> vPath;
    vector<int> vPathEdge;
    vector<int> best;	//return result path
    int bestcost;	//result path cost
    //push the shortest path into queue
    vPath.push_back(ID1);
    vector<int> h2hPath;
    vector<int> h2hEdge;
    vector<bitset<KIND>> h2hBist;
    int Dis=H2HPath(ID1,ID2,h2hPath,h2hEdge,h2hBist);
    if((h2hBist.back()&QueryBit).count()==QueryBit.count()){
        vkPath.push_back(h2hPath);
        kResults.push_back(Dis);
        return vkPath.size();
    }
    benchmark::heap<2, int, int> qPath(nodeNum*100);
    vvPathCandidate.push_back(h2hPath);
    //vvPathCandidateEdge.push_back(vPathEdge);
    //total distance from ID1 to ID2
    vDistance.push_back(Dis);
    vPreDistance.push_back(0);
    vPathParent.push_back(ID1);
    vPathParentPos.push_back(0);
    vPathBit.push_back( QueryBit & (NodesBit[ID1] | NodesBit[ID2]));
    int lb=0;
    vPathLB.push_back(lb);
//    vector<vector<int>> pathPOISet(nodeNum);
//    for(int i=0;i<nodeNum;i++){
//        vector<int> t;
//        pathPOISet.push_back(t);
//    }
    qPath.update(vvPathCandidate.size()-1,Dis+vPathLB[0]);
    int topPathID,topPathDistance;
    int UB=0;//upper bound to prune current partial path which is greater than UB
    int it=0;
    //deviation start until find k path
    while(kResults.size()<k&&qPath.size()>0){
        t2 = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
        if(time_span.count()>15.0){
            kResults.push_back(-1);
            break;
        }
        //get a partial that have min cost
        qPath.extract_min(topPathID,topPathDistance);
        if(qPath.empty()&&topPathID!=0){
            cout<<"empty"<<endl;
        }
        //test topPath keywords Set whether cover all query kw
        bitset<KIND> topPathBit(vPathBit[topPathID]);
        h2hPath.clear();
        h2hEdge.clear();
        h2hBist.clear();
        H2HPath(vPathParent[topPathID],ID2,h2hPath,h2hEdge,h2hBist);
        //connect SPT bit
        topPathBit|=((h2hBist.back()|topPathBit)&QueryBit);
        if(topPathBit.count()==QueryBit.count()&&qPath.size()>=0){
            int newPathLength= PruneRepeatedPoiPath(vvPathCandidate[topPathID]);
            if(std::find(kResults.begin(), kResults.end(),newPathLength)!=kResults.end()){
                continue;
            }
//            if(topPathID!=0&&newPathLength==kResults.back()){
//                continue;
//            }
            kResults.push_back(newPathLength);
            vkPath.push_back(vvPathCandidate[topPathID]);
            if(newPathLength>UB){
                UB=newPathLength;
                continue;
            }
        }
        //---------------------------------------------
        //UB pruned
        if(vPreDistance[topPathID]+vPathLB[topPathID]>UB&&UB!=0){
            //cout<<"UB pruned"<<endl;
            continue;//upper bound prune
        }
        //---------------------------------------------
        int pos=vPathParentPos[topPathID];
        bitset<KIND> tempBit(vPathBit[topPathID]);
        int lastNN=-1;
        vector<int> isPrune(nodeNum,1);
        for(int i=pos;i<vvPathCandidate[topPathID].size()-1;i++){
            //deviation pruned-----------------
//            if(i > (vvPathCandidate[topPathID].size()-1-pos)/2 && pos < (vvPathCandidate[topPathID].size()-1)/3 )//&& devflag)
//            {
//                //cout<<"deviation pruned"<<endl;
//                break;
//            }
            //--------------------------------
            vPath.clear();
            int deviation=vvPathCandidate[topPathID][i];
            int lastCount=tempBit.count();
            tempBit|=(QueryBit&NodesBit[deviation]);
            if(isPrune[deviation]==0){
                continue;
            }
            vector<int> tmPath(vvPathCandidate[topPathID].begin(),vvPathCandidate[topPathID].begin()+i+1);
            int PNN=-1;
            PNN=FindNNDij(deviation,ID2,tempBit);
            //pathPOISet[topPathID].push_back(PNN);
            //NN pruned
            if(PNN==lastNN&&tempBit.count()==lastCount||PNN==-1){
                //cout<<"NN pruned"<<endl;
                continue;
            }
            lastNN=PNN;
            vector<int> vPathPNN;
            vector<int> vPathPNNEdge;
            vector<bitset<KIND>> H2HPathBit;
            int NNdis=H2HPath(deviation,PNN,vPathPNN,vPathEdge,H2HPathBit);
            for(int i=1;i<vPathPNN.size()-1;i++){
                if(find(vvPathCandidate[topPathID].begin(), vvPathCandidate[topPathID].end(),vPathPNN[i])!=vvPathCandidate[topPathID].end()){
                    //confirm that path from deviation to NN have a node ,this node is in topPath
                    if((QueryBit&NodesBit[vPathPNN[i]]|tempBit).count()<=tempBit.count()){
                        //if this node can not give another query keywords,this node is pruned
                        //cout<<"path pruned"<<endl;
                        isPrune[vPathPNN[i]]=0;
                    }
                } else if(i==1){
                    break;
                }
            }
            if(NNdis!=0){
                //insert path from deviation node to NN
                tmPath.insert(tmPath.end(),vPathPNN.begin()+1,vPathPNN.end());
            }
            bitset<KIND> newPathBit(tempBit);
            newPathBit|=(NodesBit[PNN]|H2HPathBit.back())&QueryBit;
            int tmpos=tmPath.size()-1;
            //connect sp from PNN to ID2
            h2hPath.clear();
            h2hEdge.clear();
            h2hBist.clear();
            int da= PruneRepeatedPoiPath(tmPath);
            Dis=H2HPath(vPathPNN.back(),ID2,h2hPath,h2hEdge,h2hBist);
            tmPath.insert(tmPath.end(),h2hPath.begin()+1,h2hPath.end());
            //int p=vSPTParent[PNN];
            //tmPath.push_back(PNN);
            //int da=vPreDistance[topPathID]+NNdis;
            //insert node from NN to ID2 by SPT
            //int tmpDist=da+Dis;
            int tmpLB=getMaxLB(tmPath,newPathBit,tmPath[tmpos],ID2);
            vvPathCandidate.push_back(tmPath);
            vPathParent.push_back(PNN);
            vPathParentPos.push_back(tmpos);//tmpos is NN index
            vPathBit.push_back(newPathBit);
            vDistance.push_back(da+Dis);
            vPreDistance.push_back(da);
            vPathLB.push_back(tmpLB);
            qPath.update(vvPathCandidate.size()-1,da+tmpLB);
        }
    }
    return qPath.size();
}