# Welcome to AIL

## data set

We used four datasets of real road networks, which are located in the Map CO stands for point file, which contains latitude and longitude information as well as point ID The gr file represents an edge file. For example, v 1-121904167 41974556 indicates that the latitude and longitude of the point are 121.904167 41.974556.

The POI file stores keyword data under different parameters, attached after the POI point.

The Query folder stores 100000 random queries tested under each road network, including starting and ending points as well as a list of query keywords。

## Index File

Due to the large size of the index file (>1GB), we do not directly provide the index file, but instead provide the source code for building the index. This can also be used to build indexes locally.

BuildH2HIndex.cpp has built an H2H index for conducting shortest distance queries.
BuildIGTreeIndex.cpp has built an IGTree index.

Please note that these two files are separate C++ projects that can be run directly

 