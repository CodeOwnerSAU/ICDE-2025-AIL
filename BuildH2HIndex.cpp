#include<cstdio>
#include<cstring>
#include<iostream>
#include<fstream>
#include<cstdlib>
#include<vector>
#include<set>
#include<map>
#include<queue>
#include<algorithm>
#include<ctime>
#include <chrono>

using namespace std;

clock_t ct;
int cnt, tree_width = 0;
const int INF = 999999999;
struct Graph{
	int n, m;
	vector<int> V;
	vector< map< int, int > > E;
	vector< vector< pair<int, int> > > Edge;
	vector<int> D;
	Graph(){
		n = m = 0;
		V.clear();
		E.clear();
	}
	Graph(char *file){
		Graph();
        ifstream ifs(file);
        ifs>>n>>m;
		for (int i = 0; i <= n; i++){
			map< int, int > v;
			v.clear();
			E.push_back(v);
		}
		for (int i = 0; i < m; i++){
            char c;
			int x, y, z = 0;
            ifs>>c>>x>>y>>z;
            if(z==0){
                z=1;
            }
			if (x > n || y > n) 
				continue;
			if (E[x].find(y) != E[x].end()){
				if (E[x][y] > z){
					E[x][y] = z;
					E[y][x] = z;
				}
			}
			else{
				E[x].insert(make_pair(y, z));
				E[y].insert(make_pair(x, z));
			}
		}
		D.clear();
		D.push_back(0);
		for (int i = 1; i <= n; i++)
			D.push_back(E[i].size());
        cout<<"finished"<<endl;
	}
	void EdgeInitialize(){
		Edge.clear();
		for (int i = 0; i <= n; i++){
			vector< pair<int, int> > Ed;
			Ed.clear();
			for (map<int, int>::iterator it = E[i].begin(); it != E[i].end(); it++){
				Ed.push_back(*it);
			}
			Edge.push_back(Ed);
		}
	}
	bool isEdgeExist(int u, int v){
		if (E[u].find(v) == E[u].end())
			return false;
		else return true;
	}
	void insertEdge(int u, int v, int k){
		if (E[u].find(v) != E[u].end()) return;
		E[u].insert(make_pair(v, k));
		E[v].insert(make_pair(u, k));
		D[u]++;
		D[v]++;
	}
	void deleteEdge(int u, int v){
		if (E[u].find(v) == E[u].end()) return;
		E[u].erase(E[u].find(v));
		E[v].erase(E[v].find(u));
		D[u]--;
		D[v]--;
	}
};


int *DD, *DD2, *NUM;
int *_DD, *_DD2;
bool *changed;
struct SelEle{
	int x;
	SelEle();
	SelEle(int _x){
		x = _x;
	}
	bool operator< (const SelEle se)const{
		if (DD[x] != DD[se.x])
			return DD[x] < DD[se.x];
		if (DD2[x] != DD2[se.x])
			return DD2[x] < DD2[se.x];
		return x < se.x;	
	}
};	
struct Node{
	vector<int> vert, VL, pos, pos2, dis;
	vector<int> ch;
	int height;
	int pa;
	int uniqueVertex;
	Node(){
		vert.clear();
		VL.clear();
		pos.clear();
		dis.clear();
		ch.clear();
		pa = -1;
		uniqueVertex = -1;
		height = 0;
	}
};
struct Tree_Decomposition{

	FILE *fout, *fin;
	Graph G, H;
	set<SelEle> deg;
	int maxSize;
	Tree_Decomposition(){
	}
	vector< vector<int> > neighbor, length;
	vector<int> ord;
	int heightMax;
	void reduce(){
		deg.clear();
		neighbor.clear();
		length.clear();
		vector<int> vectmp;
		vectmp.clear();
		for (int i = 0; i <= G.n; i++){
			neighbor.push_back(vectmp);
			length.push_back(vectmp);
		}
		DD = (int *)malloc(sizeof(int) * (G.n + 1));
		DD2 = (int *)malloc(sizeof(int) * (G.n + 1));
		_DD = (int *)malloc(sizeof(int) * (G.n + 1));
		_DD2 = (int *)malloc(sizeof(int) * (G.n + 1));
		NUM = (int *)malloc(sizeof(int) * (G.n + 1));
		changed = (bool*)malloc(sizeof(bool) * (G.n + 1));
		for (int i = 0; i <= G.n; i++)
			NUM[i] = i;
		for (int i = 1; i <= G.n; i++){
			int j = rand() % G.n + 1;
			int x = NUM[i];
			NUM[i] = NUM[j];
			NUM[j] = x;
		}
		for (int i = 1; i <= G.n; i++){
			DD[i] = G.D[i];
			DD2[i] = G.D[i];
			_DD[i] = G.D[i];
			_DD2[i] = G.D[i];
			changed[i] = false;
			deg.insert(SelEle(i));
		}
		bool *exist;
		exist = (bool*)malloc(sizeof(bool) * (G.n + 1));
		for (int i = 1; i <= G.n; i++)
			exist[i] = true;
		ord.clear(); 
		int cnt = 0;
		while (!deg.empty()){
			cnt++;
		//	if (cnt % 10000 == 0) 
		//		cout << cnt << endl;
			int x = (*deg.begin()).x;
			while (true){
				if (changed[x]){
					deg.erase(SelEle(x));
					DD[x] = _DD[x];
					DD2[x] = _DD2[x];
					deg.insert(SelEle(x));
					changed[x] = false;
					x = (*deg.begin()).x;
				}
				else break;
			}
			ord.push_back(x);
			deg.erase(deg.begin());
			exist[x] = false;
			vector<int> neigh, leng;
			neigh.clear();
			leng.clear();
			for (map<int,int>::iterator it = G.E[x].begin(); it != G.E[x].end(); it++){
				int y = (*it).first;
				if (exist[y]){
					neigh.push_back(y);
					leng.push_back((*it).second);
				}	
			}

			int k = -1;
			for (int i = 0; i < neigh.size(); i++){
				int y = neigh[i];

				G.deleteEdge(x, y);
				_DD[y] = G.D[y];
				changed[y] = true;

			}
			for (int pu = 0; pu < neigh.size(); pu++){
				for (int pv = 0; pv < neigh.size(); pv++)
					if (pu != pv){
						int u = neigh[pu], v = neigh[pv];
						if (G.isEdgeExist(u, v)){
							if (G.E[u][v] > leng[pu] + leng[pv])
								G.E[u][v] = leng[pu] + leng[pv];
							if (G.E[v][u] > leng[pu] + leng[pv])
								G.E[v][u] = leng[pu] + leng[pv];
							
						}
						else {
						//	deg.erase(SelEle(u));
						//	deg.erase(SelEle(v));
							G.insertEdge(u, v, leng[pu] + leng[pv]);
							_DD[u] = G.D[u];
							_DD[v] = G.D[v];
							++_DD2[u];
							++_DD2[v];
							changed[u] = true;
							changed[v] = true;
						//	deg.insert(SelEle(u));
						//	deg.insert(SelEle(v));
						}
					}	
			}
			if (neigh.size() > tree_width)
				tree_width = neigh.size();
			neighbor[x] = neigh;
			length[x] = leng;
		}
		free(DD);
		free(DD2);
		free(exist);
	}
	int match(int x, vector<int> & neigh){
		int nearest = neigh[0];
		for (int i = 1; i < neigh.size(); i++)
			if (rank[neigh[i]] > rank[nearest])
				nearest = neigh[i];
		int p = belong[nearest];
		vector<int> a = Tree[p].vert;
		if (Tree[p].uniqueVertex >= 0){
			a.push_back(Tree[p].uniqueVertex);
		}
		sort(a.begin(), a.end());
		int i, j = 0;
		for (; (i < neigh.size()) && (j < a.size()); ){
			if (neigh[i] == a[j]){
				i++; j++;
			}
			else if (neigh[i] < a[j])
				break;
				else j++;
		}
		if (i >= neigh.size()) {
			return p;
		}
		printf("no match!\n");
	}
	vector<Node> Tree;
	int root;
	int * belong, *rank;
	void makeTree(){	
		belong = (int*) malloc(sizeof(int) * (H.n + 1));
		rank = (int*) malloc(sizeof(int) * (H.n + 1));
		int len = ord.size() - 1;
		Node rootn;
		Tree.clear();
		heightMax = 0;
		
		int x = ord[len];//??????
		rootn.vert = neighbor[x];//??????
		rootn.VL = length[x];
		rootn.uniqueVertex = x;
		rootn.pa = -1;
		rootn.height = 1;
		rank[x] = 0;
		belong[x] = 0;
		Tree.push_back(rootn);
		len--;
		
		for (; len >= 0; len--){//??????????
			int x = ord[len];
			Node nod;
			nod.vert = neighbor[x];
			nod.VL = length[x];
			nod.uniqueVertex = x;
			int pa = match(x, neighbor[x]);
			Tree[pa].ch.push_back(Tree.size());
			nod.pa = pa;
			nod.height = Tree[pa].height + 1;
			if (nod.height > heightMax)
				heightMax = nod.height;
			rank[x] = Tree.size();
			belong[x] = Tree.size();
			Tree.push_back(nod);

		}

		root = 0;
	}
	struct PT{
	   	int dis;
		int x;
		PT(){
		}
		PT(int _dis, int _x){
			dis = _dis;
			x = _x;
		}
		// ?????? 
		bool operator < (const PT _pt) const{
			if (dis == _pt.dis)
				return x > _pt.x;
			return dis > _pt.dis;
		}
	};
	void calculateDistanceAll(int start, int * a){
		for (int i = 0; i <= G.n; i++)
			a[i] = INF;
		a[start] = 0;
		priority_queue<PT> Q;
	//	heap Q;
		while (!Q.empty())
			Q.pop();
		Q.push(PT(0, start));
		while (!Q.empty()){
			int distance = Q.top().dis;
			int x = Q.top().x;
			Q.pop();
			if (distance > a[x]) continue;
			for (int i = 0; i < H.Edge[x].size(); i++){
				int y = H.Edge[x][i].first, z = H.Edge[x][i].second;
				if (a[y] > distance + z){
					a[y] = distance + z;
					Q.push(PT(a[y], y));
				}
			}
		}
	}

	int *toRMQ;
	vector<int> EulerSeq;
	vector< vector<int> > RMQIndex;
	void makeRMQDFS(int p, int height){
		toRMQ[p] = EulerSeq.size();
		EulerSeq.push_back(p);
		for (int i = 0; i < Tree[p].ch.size(); i++){
			makeRMQDFS(Tree[p].ch[i], height + 1);
			EulerSeq.push_back(p);
		}
	}
	void makeRMQ(){
		EulerSeq.clear();
		toRMQ = (int*)malloc(sizeof(int) * (G.n + 1));
		makeRMQDFS(root, 1);
		RMQIndex.clear();
		RMQIndex.push_back(EulerSeq);
		int m = EulerSeq.size();
		for (int i = 2, k = 1; i < m; i = i * 2, k++){
			vector<int> tmp;
			tmp.clear();
			tmp.resize(EulerSeq.size());
			for (int j = 0; j < m - i; j++){
				int x = RMQIndex[k - 1][j], y = RMQIndex[k - 1][j + i / 2];
				if (Tree[x].height < Tree[y].height)
					tmp[j] = x;
				else tmp[j] = y;
			}
			RMQIndex.push_back(tmp);
		}
	}
	int LCAQuery(int _p, int _q){//?????????
		int p = toRMQ[_p], q = toRMQ[_q];
		if (p > q){
			int x = p;
			p = q;
			q = x;
		}
		int len = q - p + 1;
		int i = 1, k = 0;
		while (i * 2 < len){
			i *= 2;
			k++;
		}
		q = q - i + 1;
		if (Tree[RMQIndex[k][p]].height < Tree[RMQIndex[k][q]].height)
			return RMQIndex[k][p];
		else return RMQIndex[k][q]; 
	}
	int distanceQueryAncestorToPosterity(int p, int q){
		if (p == q) return 0;
		int x = belong[p], y = belong[q];
		return Tree[y].dis[Tree[x].pos[Tree[x].pos.size() - 1]];
	}
	void calculateIndexSizeDFS(int p, int pre, int tmp, long long &result){
		for (int i = 0; i < Tree[p].ch.size(); i++){
			calculateIndexSizeDFS(Tree[p].ch[i], pre + 1, (pre + 1) + tmp, result);
		}
		if (tmp + (pre + 1) > result) result = tmp + (pre + 1);
//		result += pre;
	}
	long long calculateIndexSize(){
		long long res = Tree[root].vert.size();
		for (int i = 0; i < Tree[root].ch.size(); i++){
			calculateIndexSizeDFS(Tree[root].ch[i], Tree[root].vert.size(), Tree[root].vert.size(), res);
		}
		return res;
	}
	void makeIndexDFS(int p, vector<int> &list, int *toList){//???????index
	//	cnt++;
	//	cout << endl << cnt << endl;
		Tree[p].pos.resize(Tree[p].vert.size() + 1);
		Tree[p].dis.resize(list.size());
	//	printf("step1");
		for (int i = 0; i < Tree[p].vert.size(); i++){
			int j;
			for (j = 0; j < list.size(); j++)
				if (list[j] == Tree[p].vert[i])
					break;
			Tree[p].pos[i] = j;//pos???????????????????
		}
		Tree[p].pos[Tree[p].vert.size()] = list.size();
	//	printf("step2");
		for (int i = 0; i < list.size(); i++){
			Tree[p].dis[i] = INF;
		}
		priority_queue<PT> Q;
		while (!Q.empty()) Q.pop();
		int x = Tree[p].uniqueVertex;

		//priority_queue -> ??????? 
		//set -> array 
	//	printf("step3");
	//	cout << endl;
	//	for (int i = 0; i < Tree[p].vert.size(); i++)
	//		cout << Tree[p].vert[i] << " ";
	//	cout << endl;
		for (int i = 0; i < Tree[p].vert.size(); i++){
			if (Tree[p].dis[toList[Tree[p].vert[i]]] > Tree[p].VL[i])
				Tree[p].dis[toList[Tree[p].vert[i]]] = Tree[p].VL[i];
			int x = Tree[p].vert[i];
			int k;
			for (k = 0; k < list.size(); k++)
				if (list[k] == x) break;
			for (int j = 0; j < list.size(); j++){
				int y = list[j];
				int z;
				if (k < j)
					z = distanceQueryAncestorToPosterity(x, y);
				else z = distanceQueryAncestorToPosterity(y, x);
				if (Tree[p].dis[toList[y]] > z + Tree[p].dis[toList[Tree[p].vert[i]]])
					Tree[p].dis[toList[y]] = z + Tree[p].dis[toList[Tree[p].vert[i]]];
			}
		}
	//	ct += clock() - t;
	//	printf("step4");
		toList[Tree[p].uniqueVertex] = list.size();
		list.push_back(Tree[p].uniqueVertex);
		for (int i = 0; i < Tree[p].ch.size(); i++){
			makeIndexDFS(Tree[p].ch[i], list, toList);
		}
		list.pop_back();
	//	printf("step5");
		//---
		Tree[p].pos2 = Tree[p].pos;
		for (int i = Tree[p].vert.size() - 1; i >= 0; i--){
			if (Tree[p].VL[i] > Tree[p].dis[Tree[p].pos2[i]]){
				Tree[p].pos2.erase(Tree[p].pos2.begin() + i);	
			}
		}
		//---
		
		sort(Tree[p].pos.begin(), Tree[p].pos.end());
		sort(Tree[p].pos2.begin(), Tree[p].pos2.end());

		fwrite(&p, SIZEOFINT, 1, fout);
		printIntVector(Tree[p].pos);
		printIntVector(Tree[p].dis);

		
		vector<int> v1, v2;
		v1.clear();
		v2.clear();
	//	Tree[p].pos.swap(v1);
		Tree[p].dis.swap(v2);
	}
	void makeIndex(){
		makeRMQ();

		H.EdgeInitialize();
		
	}
	void makeIndex2(){
		vector<int> list;
		list.clear();
		int *toList;
		toList = (int*)malloc(sizeof(int) * (G.n + 1));
		Tree[root].pos.clear();
		
		// ????????????dijstra ---- map -> vector 
		toList[Tree[root].uniqueVertex] = 0;
		list.push_back(Tree[root].uniqueVertex);
		Tree[root].pos.push_back(0);
	//	cout << "list size: " << list.size() << endl;
	//	for (int i = 0; i < list.size(); i++)
	//		cout << list[i] << " ";
	//	cout << endl;
		
		fwrite(&root, SIZEOFINT, 1, fout);
		printIntVector(Tree[root].pos);
		printIntVector(Tree[root].dis);
		cnt = 0;
		for (int i = 0; i < Tree[root].ch.size(); i++)
			makeIndexDFS(Tree[root].ch[i], list, toList);
		free(toList);
	}
	void reducePosDFS(int p){
		//----
		if (Tree[p].ch.size() == 2){
			int t = Tree[p].ch[0];
			if (Tree[Tree[p].ch[0]].pos.size() > Tree[Tree[p].ch[1]].pos.size())
				t = Tree[p].ch[1];
			Tree[p].pos = Tree[t].pos;
			Tree[p].pos.erase(Tree[p].pos.begin() + (Tree[p].pos.size() - 1));
		}
		//----
		for (int i = 0; i < Tree[p].ch.size(); i++)
			reducePosDFS(Tree[p].ch[i]);
		fwrite(&p, SIZEOFINT, 1, fout);
		printIntVector(Tree[p].pos);
	}
	void reducePos(){
		reducePosDFS(root);
	}
	void cntSize(ofstream &result){
		long long s_nonroot = 0;
		long long s_size = 0;
		
		long long s_dis = 0;
		for( int i = 0; i < Tree.size(); ++i ) {
			s_nonroot += Tree[i].height - 1;
			s_size += Tree[i].vert.size();
			s_dis += Tree[i].height;
		}
		long long s_root = (long long) Tree[0].vert.size() * (long long) G.n;
		printf( "tree width: %d\n", tree_width);
		printf( "nonroot idx size = %0.3lfGB, avg node size=%0.3lf, avg label size=%0.3lf\n",
				s_nonroot * 4.0 / 1000000000.0, s_size * 1.0 / G.n, s_dis * 1.0 / G.n);
        result<<"nonroot idx size = "<<s_nonroot * 4.0 / 1000000000.0<<"GB, avg node size= "<<s_size * 1.0 / G.n<<" ,avg label size= "<<s_dis * 1.0 / G.n<<endl;
	}
	static const int SIZEOFINT = 4;
	void printIntArray(int * a, int n){
		fwrite(a, SIZEOFINT, n, fout);
	}
	void printIntVector(vector<int> &a){
		if (a.size() == 0){
			int x = 0;
			fwrite(&x, SIZEOFINT, 1, fout);
			return;
		}
		int x = a.size();
		fwrite(&x, SIZEOFINT, 1, fout);
		for (int i = 0; i < a.size(); i++){
			fwrite(&a[i], SIZEOFINT, 1, fout);
		}
	}
	void print(){
		//G.n
		fwrite(&G.n, SIZEOFINT, 1, fout); 
		//Tree.size() Tree.height
		int x = Tree.size();
		fwrite(&x, SIZEOFINT, 1, fout);
		for (int i = 0; i < Tree.size(); i++){
			fwrite(&Tree[i].height, SIZEOFINT, 1, fout);
		}
		//belong
		printIntArray(belong, H.n + 1);
		//LCA - toRMQ - RMQIndex
		printIntArray(toRMQ, H.n + 1);
		x = RMQIndex.size();
		fwrite(&x, SIZEOFINT, 1, fout);
		x = EulerSeq.size();
		fwrite(&x, SIZEOFINT, 1, fout);
		for (int i = 0; i < RMQIndex.size(); i++)
			printIntVector(RMQIndex[i]);
		//rootDistance
		fwrite(&root, SIZEOFINT, 1, fout);
		
		//
		for (int i = 0; i < Tree.size(); i++){
			int t = Tree[i].ch.size();
			fwrite(&t, SIZEOFINT, 1, fout);
			for (int j = 0; j < t; j++)
				fwrite(&Tree[i].ch[j], SIZEOFINT, 1, fout);
		}
		//		

	}
	void print2(){
				//Tree.pos dis
		for (int i = 0; i < Tree.size(); i++){
			printf("%d\n", i);
			printIntVector(Tree[i].pos);
			printIntVector(Tree[i].dis);
		}
	}
};

// floyd deal with vertices which is deleted
int main()
{
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::high_resolution_clock::time_point t3;
    std::chrono::high_resolution_clock::time_point t4;
    std::chrono::duration<double> time_span;
    std::chrono::duration<double> IG_base_time_totol;
	srand(time(0));
	int operation = 1;
//	cout << "Operation: ";
//	scanf("%d", &operation);
//	operation = 2;
	if (operation == 1){ 
		string filest;
		char *file, *fileout;
		int i;
         char *edgeFile="/home/xiongxing/reviewProject/MAP/Example/USA-road-d.Example.gr";
         char *indexFile="/home/xiongxing/reviewProject/MAP/Example/H2H/Example.index";
		file = edgeFile;
		cout << "file: " << file<<endl;
		fileout = indexFile;
		Tree_Decomposition td;
		td.fout = fopen(fileout, "wb");
		td.G = Graph(file);
		td.H = td.G;
		
		clock_t start = clock();
        t1 = std::chrono::high_resolution_clock::now();
		td.reduce();
		td.makeTree();
		td.makeIndex();
		cout << "MakeIndex time: " << (double)(clock() - start) / CLOCKS_PER_SEC << endl; 
		td.print();
		td.makeIndex2();
        t2 = std::chrono::high_resolution_clock::now();
	td.reducePos();
        time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
        cout<<"time :"<<time_span.count()<<endl;
		cout << "MakeIndex time: " << (double)(clock() - start) / CLOCKS_PER_SEC << endl;
		td.cntSize(result);
		fclose(stdout);

	}

}
