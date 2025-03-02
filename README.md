# Welcome to AIL

## 🚀 Welcome to the repo of AIL

The source code for the ICDE 2025 paper under review: *When the Shortest Path Collects Spatial Keywords: A Journey to Reconstruct the Optimal Algorithms.*

## 🏠 Overview  

we build an Algorithm Integration Library (AIL) to tailor the optimal algorithm for route planning based on the current specific *ORCSK* query, considering both response time and path distance.

![AIL framework](https://github.com/CodeOwnerSAU/ICDE-2025-AIL/blob/main/AILframework.png)

## 📦 Data set

We used four datasets of real road networks, which are located in the Map CO stands for point file, which contains latitude and longitude information as well as point ID The gr file represents an edge file. For example, v 1-121904167 41974556 indicates that the latitude and longitude of the point are 121.904167 41.974556.

The POI file stores keyword data under different parameters, attached after the POI point.

The Query folder stores 100000 random queries tested under each road network, including starting and ending points as well as a list of query keywords。

## 1️⃣ Index File

Due to the large size of the index file (>1GB), we do not directly provide the index file, but instead provide the source code for building the index. This can also be used to build indexes locally.

BuildH2HIndex.cpp has built an H2H index for conducting shortest distance queries.
BuildIGTreeIndex.cpp has built an IGTree index.

Please note that these two files are separate C++ projects that can be run directly

##  2️⃣ Predictive model

We used four models, which are in the folder /Model.

The code for some of the models is referenced  at  "*<u>Why do tree-based models still outperform deep learning on tabular data?* (Advances in Neural Information Processing Systems 2022)</u> " https://github.com/LeoGrin/tabular-benchmark/tree/main.

To accommodate the multi-class classification model in this paper, we modified the original code.

*SVM*: We use the One-vs-One strategy and then train binary SVM classifiers in parallel.

*MLP*: The Mish activation function is used to replace the traditional ReLU, a learnable feature scaling layer is introduced to automatically learn the weights for each feature dimension, and a sigmoid gating mechanism is added after each activation layer.

*XGBoost*: We add a custom Focal Loss to reduce the weight of easily classified samples, and scipy.sparse.csr_matrix is utilized to handle high-dimensional sparse features.

## 3️⃣ Train

 In the AIL_peak_train.csv file, each row of the table represents a query that contains features.

```
Query_ID,Start_Point,End_Point,Road_Distance,Keyword_Count,POI_Density,Query_Density,POI_Type,POI_Contain,Execution_Time,Path_Distance,OptimalID
1,93035,234870,5497208,2,2,0.010546,500,5,0.0002031560,5657805,7
```

OptimalID represents the optimal algorithm that has been assigned to the current query.

Then, preprocessing is carried out using preprocess.py, and the model is trained based on the hyperparameters in HYPERPARAMETERS.py.

