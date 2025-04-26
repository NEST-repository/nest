#include <iostream>
#include <fstream>
#include <set>
#include <unordered_set>
#include <limits>
#include <sstream>
#include <iomanip>
#include <set>
#include <Python.h>

#include "nlohmann/json.hpp"

using namespace std;
using json = nlohmann::json;

// begin trivial helper stuff
ostream& dbg = cerr;

void fail (const string &s) {
    cout << "FAIL: " << s << endl;
    dbg << "FAIL: " << s << endl;
    exit(1);
}

void warn (const string &s) {
    dbg << "WARNING: " << s << endl;
}

#define DBG(vari) cerr<<"["<<__LINE__<<"] "<<#vari<<" = "<<(vari)<<endl;

template <typename T>
ostream& operator << (ostream &s, const vector<T> &v) {
    for (const T &x : v) {
        s << x << " ";
    }
    return s;
}

template <typename T>
string to_string (const vector<T> &v) {
    stringstream ss;
    ss << v;
    return ss.str();
}

template <typename T>
void append (vector<T> &v, const vector<T> &w) {
    v.insert(v.end(), w.begin(), w.end());
}

template <typename T>
inline void minify (T &x, const T &y) {
    x = min(x,y);
}

int ceildiv (int x, int y) {
    assert(y > 0);
    return (x + y - 1) / y;
}

constexpr double INFTY = 1e30;

vector<int> vectorOfSetBits (const vector<bool> &v) {
    vector<int> res;
    for (int i = 0; i < v.size(); ++i) {
        if (v[i]) {
            res.push_back(i);
        }
    }
    return res;
}

// end trivial helper stuff


constexpr int DOWNSETS_LIMIT = 40'000;
constexpr int DOWNSETS_EXPLORATION_LIMIT = 200'000;
constexpr int DEVICES_LIMIT = 10'000; // some loose upper bound on number of devices there can be in any reasonable input
constexpr bool DATA_PARALLEL_DEGREE_MUST_DIVIDE_BATCH_SIZE = false;
constexpr bool DATA_PARALLEL_DEGREE_MUST_DIVIDE_NUM_DEVICES = false;
bool isZero = true;


struct Node {
    // Node represents a layer,
    // in a graph where the TMPC width t is *already fixed*
    int id; // v
    double parameterSize; // size of weights
    double activationSize; // sum of ALL (also intermediate) activation sizes
    double optimalLatencyFw; // computed in the per-layer optimization problem (ILP etc.)
    double optimalLatencyBw;
    bool isTensorParallelized; // if YES, the node represents a layer *slice*
};

void from_json (const json &j, Node &n) {
    j.at("id").get_to(n.id);
    j.at("parameterSize").get_to(n.parameterSize);
    j.at("activationSize").get_to(n.activationSize);
    j.at("optimalLatencyFw").get_to(n.optimalLatencyFw);
    j.at("optimalLatencyBw").get_to(n.optimalLatencyBw);
    j.at("isTensorParallelized").get_to(n.isTensorParallelized);
}

struct Edge {
    int sourceId; // u
    int destId; // v
    double communicationCost; // c(u,v), in bytes
};

void from_json (const json &j, Edge &e) {
    j.at("sourceId").get_to(e.sourceId);
    j.at("destId").get_to(e.destId);
    j.at("communicationCost").get_to(e.communicationCost);
}

struct Instance {
    double maxMemoryPerDevice;
    int maxDevices;
    double bandwidth;
    vector<double> bws;
    vector<double> bws_dp;
    vector<double> latency_per_level;
    vector<double> acc_latency;
    double frequency;
    int num_levels;
    vector<int> devices_per_level;
    int mbsInBatch;
    map<int, vector<Node>> nodes; // nodes[t] are for graph of TMPC width t
    map<int, vector<Edge>> edges; // edges[t] are for graph of TMPC width t
    bool activationRecomputation;
    string optimizerAlgorithm; // SGD or Adam (in the latter case the memory usage becomes 3* rather than 2*parameterSize)
    int fixedDPStrategy[3];
    int numTransformerLayers;

    // filled with renumber()
    unordered_map<int,int> newNumber;
    vector<int> oldNumber;

    void checkInputCorrectness() const;
    vector<int> isDAG() const;
    void renumber();    
};

void from_json (const json &j, Instance &ii) {
    j.at("maxMemoryPerDevice").get_to(ii.maxMemoryPerDevice);
    j.at("maxDevices").get_to(ii.maxDevices);
    j.at("bandwidth").get_to(ii.bandwidth);
    j.at("bws").get_to(ii.bws);
    j.at("latency_per_level").get_to(ii.latency_per_level);
    j.at("devices_per_level").get_to(ii.devices_per_level);
    j.at("frequency").get_to(ii.frequency);
    ii.num_levels = ii.bws.size();
    j.at("mbsInBatch").get_to(ii.mbsInBatch);
    j.at("fixedPP").get_to(ii.fixedDPStrategy[0]);
    j.at("fixedTMPC").get_to(ii.fixedDPStrategy[1]);
    j.at("fixedDP").get_to(ii.fixedDPStrategy[2]);
    j.at("numTransformerLayers").get_to(ii.numTransformerLayers);
    map<string, vector<Node>> nodes;
    j.at("nodes").get_to(nodes);
    for (const auto &p : nodes) {
        ii.nodes[stoi(p.first)] = p.second;
    }
    map<string, vector<Edge>> edges;
    j.at("edges").get_to(edges);
    for (const auto &p : edges) {
        ii.edges[stoi(p.first)] = p.second;
    }
    j.at("activationRecomputation").get_to(ii.activationRecomputation);
    j.at("optimizerAlgorithm").get_to(ii.optimizerAlgorithm);

    ii.checkInputCorrectness();
    ii.renumber();
    ii.checkInputCorrectness();

    ii.acc_latency.resize(ii.latency_per_level.size(), 0);

    //acumulate latency
    double sum = 0.0;
    for (int i = 0; i < ii.latency_per_level.size(); i++) {
        sum += ii.latency_per_level[i];
        ii.acc_latency[i] = sum;
    }

    //calculate bandwidth for pipeline parallel
    ii.bws_dp = ii.bws;
    for (size_t i = 1; i < ii.bws.size(); ++i) {
        if (ii.bws[i] > ii.bws[i - 1]) {
            ii.bws[i] = ii.bws[i - 1];
        }
    }
}

void Instance::checkInputCorrectness() const {
    if (maxDevices < 1 || maxDevices > DEVICES_LIMIT) {
        fail("wrong number of devices");
    }
    if (bandwidth < 1e-9) {
        fail("wrong bandwidth");
    }
    if (maxMemoryPerDevice < 1e-9) {
        fail("wrong maxMemoryPerDevice");
    }
    if (mbsInBatch < 1) {
        fail("wrong mbsInBatch");
    }
    if (optimizerAlgorithm != "SGD" && optimizerAlgorithm != "Adam") {
        fail("optimizerAlgorithm should be SGD or Adam");
    }
    if (nodes.empty()) {
        fail("no graphs/TMPCwidths in input");
    }
    set<int> tmpcWidths;
    for (const auto &p : nodes) {
        tmpcWidths.insert(p.first);
        if (p.first > DEVICES_LIMIT) {
            fail("TMPC width clearly too large");
        }
        if (p.first < 1) {
            fail("TMPC width < 1?");
        }
    }
    set<int> edgeTmpcWidths;
    for (const auto &p : edges) {
        edgeTmpcWidths.insert(p.first);
    }
    if (tmpcWidths != edgeTmpcWidths) {
        fail("graphs and edges have different TMPC width sets");
    }
    set<int> nodeIds;
    for (const Node &n : nodes.begin()->second) {
        nodeIds.insert(n.id);
    }
    if (nodeIds.empty()) {
        fail("no nodes in graph");
    }
    if (nodeIds.size() != nodes.begin()->second.size()) {
        fail("node ids are not unique");
    }
    for (const auto &p : nodes) {
        set<int> nodeIdsThisTmpc;
        for (const Node &n : p.second) {
            if (n.parameterSize < 0) {
                fail("parameterSize < 0");
            }
            if (n.activationSize < 0) {
                fail("activationSize < 0");
            }
            if (n.optimalLatencyFw < 0) {
                fail("optimalLatencyFw < 0");
            }
            if (n.optimalLatencyBw < 0) {
                fail("optimalLatencyBw < 0");
            }
            if (p.first == 1 && n.isTensorParallelized) {
                fail("TMPC width 1 but node isTensorParallelized");
            }
            nodeIdsThisTmpc.insert(n.id);
        }
        if (nodeIdsThisTmpc != nodeIds) {
            fail("node (layer) ids are not the same for all TMPC widths");
        }
    }
    set<pair<int,int>> edgeIds;
    for (const Edge &e : edges.begin()->second) {
        edgeIds.insert({e.sourceId, e.destId});
    }
    if (edgeIds.size() != edges.begin()->second.size()) {
        fail("parallel edges exist (maybe not a problem but let's rather contract them)");
    }
    for (const auto &p : edges) {
        set<pair<int,int>> edgeIdsThisTmpc;
        for (const Edge &e : p.second) {
            if (nodeIds.count(e.sourceId) == 0) {
                fail("edge sourceId not in nodeIds");
            }
            if (nodeIds.count(e.destId) == 0) {
                fail("edge destId not in nodeIds");
            }
            if (e.sourceId == e.destId) {
                fail("edge sourceId == destId (self-loop)");
            }
            if (e.communicationCost < 0) {
                fail("communicationCost < 0");
            }
            bool inserted = edgeIdsThisTmpc.insert({e.sourceId, e.destId}).second;
            if (inserted == false) {
                fail("parallel edges exist (maybe not a problem but let's rather contract them)");
            }
        }
        if (edgeIds != edgeIdsThisTmpc) {
            fail("edges are not the same for all TMPC widths");
        }
    }
    if (isDAG().empty()) {
        fail("graph is not a DAG");
    }
}


// returns empty vector if not a DAG, otherwise topological order
vector<int> Instance::isDAG() const {
    unordered_map<int,int> indegree;
    unordered_map<int,vector<int>> outgoingEdges;
    for (const Edge &e : edges.begin()->second) {
        ++indegree[e.destId];
        outgoingEdges[e.sourceId].push_back(e.destId);
    }
    vector<int> deg0vertices;
    for (const Node &n : nodes.begin()->second) {
        if (indegree[n.id] == 0) {
            deg0vertices.push_back(n.id);
        }
    }
    vector<int> verticesInTopologicalOrder;
    while (!deg0vertices.empty()) {
        int v = deg0vertices.back();
        deg0vertices.pop_back();
        verticesInTopologicalOrder.push_back(v);
        for (int w : outgoingEdges[v]) {
            --indegree[w];
            if (indegree[w] == 0) {
                deg0vertices.push_back(w);
            }
        }
    }
    if (verticesInTopologicalOrder.size() != nodes.begin()->second.size()) {
        return vector<int>();
    } else {
        return verticesInTopologicalOrder;
    }
}


void Instance::renumber () {
    // renumber nodes as 0,1,2,... in a topological order
    assert(oldNumber.empty());
    // build oldNumber and newNumber
    oldNumber = isDAG();
    if (oldNumber.empty()) {
        fail("graph is not a DAG");
    }
    for (int i = 0; i < oldNumber.size(); ++i) {
        newNumber[oldNumber[i]] = i;
    }
    // now replace old ids with new ids everywhere
    for (auto &it : nodes) {
        for (Node &n : it.second) {
            n.id = newNumber[n.id];
        }
    }
    for (auto &it : edges) {
        for (Edge &e : it.second) {
            e.sourceId = newNumber[e.sourceId];
            e.destId = newNumber[e.destId];
        }
    }
}

//networkSkeleton structure
struct networkSkeleton {
    //skeleton of the network
    unordered_map<int, vector<int>> fatTree;

    //number of devices in the network
    int placeNodes(const Instance &ins, unordered_map<int, vector<int>>& fatTree, vector<int> nodes, int num_devices);
    //returns the device index that contains a given graph node in the fatTree skeleton
    int findDevice(unordered_map<int, vector<int>>& fatTree, int node);
    //returns the minimum device index - required for Data Parallel degree determination
    int findDevice_min(unordered_map<int, vector<int>>& fatTree, int node);
    //returns a vector of a devices indices that represent a series of 'a' devices 
    //that are closest to the device that contains the previous graph node
    vector<int> nearestDevices(unordered_map<int, vector<int>>& fatTree, int prevDevice, int a);
    //duplicates the fatTree based on data parallel degree
    unordered_map<int, vector<int>> duplicateFatTree(const unordered_map<int, vector<int>>& fatTree, int d);
    //returns the load index of the device that contains the previous graph node
    int getLoadIdx(const Instance &ins, unordered_map<int, vector<int>>& fatTree, int prev_node, int next_node);
};

struct ResultStage {
    vector<int> nodes; // (new ids, unless renumberResultBack was run)
    int devicesForStage;
};

void to_json(json &j, const ResultStage &s) {
    j = json{{"nodes", s.nodes},
             {"devicesForStage", s.devicesForStage}
    };
}

struct Result {
    vector<ResultStage> stages;
    int dataParallelDegree;
    int tensorParallelDegree;
    networkSkeleton netSkel;
    networkSkeleton netSkel_dp;
};

void to_json(json &j, const Result &r) {
    j = json{{"stages", r.stages},
             {"dataParallelDegree", r.dataParallelDegree},
             {"tensorParallelDegree", r.tensorParallelDegree}
    };
}

struct LoadOfStage {
    double fw_bw_latency_with_recompute; // max_device (fw + (fw + bw))
    double fw_bw_latency_wo_recompute; // max_device (fw + (just bw))
    double parameter_size; // max_device (parameter_size)
    int max_s_memory_feasible; // max s for which each device has enough memory
};

struct CommLevel {
    int groups;         // Number of groups at this level
    int procs_per_group;
    double alpha;       // Latency
    double beta;        // Bandwidth
};

struct Graph {
    // this is already for a fixed TMPC width t!
    const int tmpcWidth;
    const Instance &ins; // already renumbered (nodes 0,1,2,...)
    const int boundOnS; // set as min(maxDevices, # nodes)

    vector<vector<pair<int,double>>> incomingEdges; // v -> vector of {u, c(u,v)}
    vector<vector<pair<int,double>>> outgoingEdges; // v -> vector of {w, c(v,w)}
    vector<const Node*> node; // node[v] = pointer to node with (new) id v

    // downsets, represented as indicator vectors
    unordered_map<vector<bool>,int> downsetToId; // map downset to its ID
    vector<vector<bool>> downsets; // maps ID to downset
    vector<int> downsetsSortedBySize; // IDs of downsets, sorted by size

    // pairs of downsets (that induce contiguous sets)
    vector<vector<int>> immediateSubDownsets;
    // immediateSubDownsets[id] = IDs of downsets that are immediate subsets of the downset with ID id
    vector<vector<int>> subDownsets;
    // subDownsets[id] = IDs of downsets that are subsets of the downset with ID id
    // (this takes O(numberOfDownsetPairs) space; could be done on the fly in the DP perhaps,
    //  but if one can't afford this memory then probably one can't afford the DP alg timewise)
    long long numberOfDownsetPairs;

    Graph (const Instance &_ins, int _tmpcWidth);
    void generateDownsets();
    void growDownset(const vector<bool> &downset, int myId);
    void prepareSubDownsets();

    //analytical model for AllReduce in a tree
    double getDataParallelResyncCostTree (int d, double parameterSize, vector<int>& node_ids) const;
    //creates a partial network with the given nodes
    pair<int, int> create_partial_network(const vector<int>& node_ids, vector<int> devices_per_level) const;

    vector<bool> getContiguousSet (int id, int subId) const;
    
    // a -> loadOfStage(downsets[id] \ downsets[subId], a many devices)
    // (only some a will appear, namely those that yield smaller load than a-1)
    // (e.g. if there is no branching in the layer graph, then only a=1 or a=t can make sense)

    vector<map<int,LoadOfStage>> getLoadOfStage (PyObject* pModule, int id, int subId) const; // wrapper
    vector<map<int,LoadOfStage>> getLoadOfStage (PyObject* pModule, const vector<int> &nodes) const; // wrapper, used in reconstruction
    vector<map<int,LoadOfStage>> getLoadOfStage (PyObject* pModule, const vector<int> &nodes, const vector<bool> &nodesVB) const;
    void getLoadOfStageForA (PyObject* pModule, const vector<int> &nodes, const vector<bool> &nodesVB,
                             int a, vector<map<int,LoadOfStage>> &resultMap) const;
    //ZeRO optimization
    int getMemoryFeasibilityZero (const vector<int> &nodes, const vector<bool> &nodesVB,
        int a, int zeroDeg, int zeroType, vector<map<int,LoadOfStage>> &resultMap) const;
    int getZeroLoadOfStageForA (PyObject* pModule, const vector<int> &nodes, const vector<bool> &nodesVB,
                             int a, int zeroDeg, int zeroType, vector<map<int,LoadOfStage>> &resultMap) const;

    vector<vector<vector<vector<pair<double, networkSkeleton>>>>> dp;  //dp table
    // dp[level][downset id][num devices][num stages] -> minimal max-load of a device
    int idOfFullSet;
    Result runDP(PyObject* pModule);

    void renumberResultBack (Result &r) const;

    double getTimePerBatchForResult (PyObject* pModule, const Result &r, networkSkeleton final_ns) const;
};


Graph::Graph (const Instance &_ins, int _tmpcWidth)
  : tmpcWidth(_tmpcWidth),
    ins(_ins),
    boundOnS(min(ins.maxDevices, (int)ins.nodes.at(tmpcWidth).size())),
    numberOfDownsetPairs(0)
{
    // build incomingEdges, outgoingEdges, node
    incomingEdges.resize(ins.nodes.at(tmpcWidth).size());
    outgoingEdges.resize(ins.nodes.at(tmpcWidth).size());
    for (const Edge &e : ins.edges.at(tmpcWidth)) {
        incomingEdges[e.destId].push_back({e.sourceId, e.communicationCost});
        outgoingEdges[e.sourceId].push_back({e.destId, e.communicationCost});
    }
    node.resize(ins.nodes.at(tmpcWidth).size());
    for (const Node &n : ins.nodes.at(tmpcWidth)) {
        node[n.id] = &n;
    }
    // generate downsets
    generateDownsets();
    // immediateSubDownsets is prepared. now prepare subDownsets
    prepareSubDownsets();
}


void Graph::generateDownsets () {
    if (!downsets.empty()) {
        fail("downsets not empty. generating downsets twice?");
    }

    // start with empty set
    const vector<bool> emptySet(ins.nodes.at(tmpcWidth).size(), false);
    downsetToId[emptySet] = 0;
    downsets.push_back(emptySet);
    immediateSubDownsets.emplace_back();
    growDownset(emptySet, 0);

    dbg << "generated " << downsets.size() << " downsets" << endl;
    if (downsets.size() > DOWNSETS_LIMIT) {
        fail("too many downsets (current limit set at " + to_string(DOWNSETS_LIMIT) + "); this isn't going to work...");
    }

    idOfFullSet = downsetToId.at(vector<bool>(ins.nodes.at(tmpcWidth).size(), true));

    // prepare downsetsSortedBySize
    vector<pair<int,int>> sorter; // {<size, downset id>}
    for (int i = 0; i < downsets.size(); ++i) {
        sorter.emplace_back(count(downsets[i].begin(), downsets[i].end(), true), i);
    }
    sort(sorter.begin(), sorter.end());
    for (auto &p : sorter) {
        downsetsSortedBySize.push_back(p.second);
    }
    assert(downsetsSortedBySize[0] == 0);
}


void Graph::growDownset (const vector<bool> &downset, int myId) {
    // try to add every vertex
    for (int v = 0; v < node.size(); ++v) {
        if (!downset[v]) {
            // try downset + {v} as a new downset
            // check if valid: do all v's successors belong to downset?
            bool valid = true;
            for (const pair<int,double> &p : outgoingEdges[v]) {
                // edge v -> p.first
                if (!downset[p.first]) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                vector<bool> newDownset = downset;
                newDownset[v] = true;
                // check if newDownset had already been generated
                if (!downsetToId.count(newDownset)) {
                    // new downset
                    int newId = downsets.size();
                    downsetToId[newDownset] = newId;
                    downsets.push_back(newDownset);
                    if (downsets.size() >= DOWNSETS_EXPLORATION_LIMIT) {
                        fail("already over " + to_string(DOWNSETS_EXPLORATION_LIMIT) + " downsets. this isn't going to work...");
                    }
                    immediateSubDownsets.emplace_back();
                    growDownset(newDownset, newId);
                }
                immediateSubDownsets[downsetToId[newDownset]].push_back(myId);
            }
        }
    }
}


void Graph::prepareSubDownsets () {
    // subDownsets = transitive closure of immediateSubDownsets

    if (numberOfDownsetPairs != 0) {
        fail("prepareSubDownsets() called twice?");
    }
    subDownsets.resize(downsets.size());

    for (int id = 0; id < downsets.size(); ++id) {
        // we will generate subIdeals[id] using some BFS/DFS
        vector<int> queue = {id};
        unordered_set<int> enqueuedDownsets = {id};
        while (!queue.empty()) {
            int subId = queue.back();
            queue.pop_back();

            // now visiting subId
            if (subId != id) {
                subDownsets[id].push_back(subId);
                ++numberOfDownsetPairs;
            }

            // expand further from subId
            for (int subSubId : immediateSubDownsets[subId]) {
                if (enqueuedDownsets.insert(subSubId).second == true) {
                    // subSubId was not in enqueuedIdeals before
                    queue.push_back(subSubId);
                }
            }
        }
    }

    dbg << "numberOfDownsetPairs = " << numberOfDownsetPairs << endl;
}


// returns the difference downsets[id] \ downsets[subId] as vector<bool>
vector<bool> Graph::getContiguousSet (int id, int subId) const {
    vector<bool> downset = downsets[id], subDownset = downsets[subId];
    for (int v = 0; v < ins.nodes.at(tmpcWidth).size(); ++v) {
        if (subDownset[v]) {
            downset[v] = false;
        }
    }
    return downset;
}

//creates a partial network with the given nodes
//returns the source index (common switch to all devices) and the level of the source
pair<int, int> Graph::create_partial_network(const std::vector<int>& node_ids, vector<int> devices_per_level) const{
    vector<int> dimension = devices_per_level;
    dimension.push_back(1);

    std::vector<std::vector<int>> all_level_ids;

    for (size_t i = 0; i < node_ids.size(); ++i) {
        std::vector<int> level_ids(dimension.size() - 1, -1);

        for (size_t j = 0; j < level_ids.size(); ++j) {
            if (j == 0) {
                level_ids[j] = node_ids[i] / dimension[j];
            } else {
                level_ids[j] = level_ids[j - 1] / dimension[j];
            }
        }

        all_level_ids.push_back(level_ids);
    }

    int source_idx = -1;
    int source_level = -1;

    for (size_t i = 0; i < all_level_ids[0].size(); ++i) {
        bool all_equal = true;

        for (size_t lst_idx = 0; lst_idx < all_level_ids.size(); ++lst_idx) {
            if (all_level_ids[lst_idx][i] != all_level_ids[0][i]) {
                all_equal = false;
                break;
            }
        }

        if (all_equal) {
            source_idx = all_level_ids[0][i];
            source_level = static_cast<int>(i);
            break;
        }
    }

    if (source_level != -1) {
        source_level += 1;
    }

    return {source_idx, source_level};
}

double hierarchical_allreduce(const vector<CommLevel>& levels, double data_per_proc, const vector<int>& node_ids) {
    int num_leaf_nodes = node_ids.size();  // only level 0 nodes do computation
    int stages = static_cast<int>(log2(num_leaf_nodes));
    double total_cost = 0.0;

    // Halving-Doubling at level 0
    for (int i = 0; i < stages; ++i) {
        double data = data_per_proc / pow(2, i);
        total_cost += levels[0].alpha + levels[0].beta * data;
    }

    double lower_comm_cost = 0.0;

    // Cost of data going up and down the tree through switches
    for (int level = 1; level < levels.size(); ++level) {
        // Each level transmits full data once up and once down
        total_cost += (2 * (levels[level].alpha + levels[level].beta * data_per_proc)) + lower_comm_cost;
        lower_comm_cost += total_cost;
    }

    return total_cost;
}

//Data Parallel Resync Cost - AllReduce operation using HalvingDoubling
double Graph::getDataParallelResyncCostTree (int d, double parameterSize, vector<int>& node_ids) const{
    auto result = create_partial_network(node_ids, ins.devices_per_level);
    int source_idx = result.first;
    int source_level = result.second;

    vector<CommLevel> levels;
    for (int i = 0; i < source_level; ++i) {
        CommLevel level;
        if(i == 0) {
            level.groups = 1;
        } else {
            level.groups = ins.devices_per_level[i];
        }
        level.procs_per_group = ins.devices_per_level[i];
        level.alpha = ins.latency_per_level[i];
        level.beta = 1/ins.bws_dp[i];
        levels.push_back(level);
    }
    double data_per_leaf_proc = parameterSize;
    double cost = hierarchical_allreduce(levels, data_per_leaf_proc, node_ids);
    return cost;
}

//returns the device index that contains a given graph node in the fatTree skeleton
int networkSkeleton::findDevice(unordered_map<int, vector<int>>& fatTree, int node) {
    for (const auto& [deviceId, nodes] : fatTree) {
        if(find(nodes.begin(), nodes.end(), node) != nodes.end()) {
            return deviceId;
        }
    }
    return -1;
}

//returns the minimum device index - required for Data Parallel degree determination
int networkSkeleton::findDevice_min(unordered_map<int, vector<int>>& fatTree, int node) {
    int minDeviceId = -1;
    for (const auto& [deviceId, nodes] : fatTree) {
        if (find(nodes.begin(), nodes.end(), node) != nodes.end()) {
            if (minDeviceId == -1 || deviceId < minDeviceId) {
                minDeviceId = deviceId;
            }
        }
    }
    return minDeviceId;
}

//returns a vector of a devices indices that represent a series of 'a' devices closest to prevDevice
vector<int> networkSkeleton::nearestDevices(unordered_map<int, vector<int>>& fatTree, int prevDevice, int a) {
    vector<int> result;

    //To check if a range of consecutive indices is valid
    auto areConsecutiveEmpty = [&](int start, int count) -> bool {
        for (int i = start; i < start + count; ++i) {
            if (fatTree.find(i) != fatTree.end() && !fatTree.at(i).empty()) {
                return false;
            }
        }
        return true;
    };

    //Search within the fatTree for consecutive empty indices before moving out of bounds
    for (int start = 0; start <= (int)fatTree.size() - a; ++start) {
        if (areConsecutiveEmpty(start, a)) {
            for (int i = 0; i < a; ++i) {
                result.push_back(start + i);
            }
            return result;
        }
    }

    //If no consecutive empty indices are found, use the next available indices
    int start = fatTree.size();
    for (int i = 0; i < a; ++i) {
        result.push_back(start + i);
    }

    return result;
}

//places the nodes into the fatTree at nearestDevices
int networkSkeleton::placeNodes(const Instance &ins, unordered_map<int, vector<int>>& fatTree, vector<int> nodes, int num_devices) {
    //find max of nodes
    int max_node = *max_element(nodes.begin(), nodes.end());
    int acc = findDevice(fatTree, max_node+1);  //acc holds the device id of the previous node
    vector<int> devices_per_level = ins.devices_per_level;

    //allocating required devices in the fatTree
    vector<int> devices = nearestDevices(fatTree, acc, num_devices);

    //Finding the level wise connection between the device of previous node vs this node
    int loadIdx = -1;
    int level_ids_prev = -1;
    int level_ids_next = -1;
    for (int i = 0; i < size(devices_per_level); i++) {
        if(i == 0) {
            level_ids_prev = int(acc/devices_per_level[i]);
            level_ids_next = int(devices[0]/devices_per_level[i]);
        } else {
            level_ids_prev = int(level_ids_prev/devices_per_level[i]);
            level_ids_next = int(level_ids_next/devices_per_level[i]);
        }
        if(level_ids_next == level_ids_prev) {
            loadIdx = i;
            break;
        }
    }

    for (int i = 0; i < devices.size(); i++) {
        fatTree[devices[i]] = nodes;
    }

    return loadIdx;
}

//returns the load index of the device that contains the previous graph node
int networkSkeleton::getLoadIdx(const Instance &ins, unordered_map<int, vector<int>>& fatTree, int prev_node, int next_node) {
    int acc_prev = findDevice(fatTree, prev_node);
    int acc_next = findDevice(fatTree, next_node);

    vector<int> devices_per_level = ins.devices_per_level;
    int level_ids_prev = -1;
    int level_ids_next = -1;
    int loadIdx = -1;
    for (int i = 0; i < size(devices_per_level); i++) {
        if(i == 0) {
            level_ids_prev = int(acc_prev/devices_per_level[i]);
            level_ids_next = int(acc_next/devices_per_level[i]);
        } else {
            level_ids_prev = int(level_ids_prev/devices_per_level[i]);
            level_ids_next = int(level_ids_next/devices_per_level[i]);
        }
        if(level_ids_next == level_ids_prev) {
            loadIdx = i;             
            break;
        }
    }
    return loadIdx;
}

//duplicates the fatTree based on data parallel degree
//the fatTree is duplicated (d - 1) times
unordered_map<int, vector<int>> networkSkeleton::duplicateFatTree(const unordered_map<int, vector<int>>& fatTree, int d) {
    unordered_map<int, vector<int>> duplicatedFatTree;
    int originalSize = fatTree.size();

    // Copy the original fatTree
    for (const auto& pair : fatTree) {
        duplicatedFatTree[pair.first] = pair.second;
    }

    // Duplicate the fatTree (d - 1) times
    for (int i = 1; i < d; ++i) {
        for (const auto& pair : fatTree) {
            int newDeviceId = pair.first + i * originalSize;
            duplicatedFatTree[newDeviceId] = pair.second;
        }
    }

    return duplicatedFatTree;
}

//creates a key for the memorization maps
//the key is a pair of data size and number of devices
struct InputKey {
    int dataSize;
    int num_devices;

    bool operator==(const InputKey& other) const {
        return dataSize == other.dataSize && num_devices == other.num_devices;
    }
};

// hash function for the InputKey struct
struct KeyHasher {
    std::size_t operator()(const InputKey& key) const {
        return std::hash<int>()(key.dataSize) ^ (std::hash<int>()(key.num_devices) << 1);
    }
};

//memorization for AllGather and ReduceScatter operations
std::unordered_map<InputKey, int, KeyHasher> gatherMemorization;
std::unordered_map<InputKey, int, KeyHasher> scatterMemorization;

//computes the cycles for AllGather and ReduceScatter operations
//using Python functions get_allgather_latency and get_reduce_scatter_latency
int compute_cycles(PyObject* pModule, int size, int zeroDeg, int a, int operation) {
    PyObject* pFunc = nullptr;

    if (operation == 1) {
        pFunc = PyObject_GetAttrString(pModule, "get_allgather_latency");
        if (!pFunc || !PyCallable_Check(pFunc)) {
            std::cerr << "Could not find function 'get_allgather_latency'!\n";
            Py_DECREF(pModule);
            Py_Finalize();
            return 1;
        }
    }
    else if (operation == 2) {
        pFunc = PyObject_GetAttrString(pModule, "get_reduce_scatter_latency");
        if (!pFunc || !PyCallable_Check(pFunc)) {
            std::cerr << "Could not find function 'get_reduce_scatter_latency'!\n";
            Py_DECREF(pModule);
            Py_Finalize();
            return 1;
        }
    }

    PyObject* pList = PyList_New(zeroDeg);
    int idx = 0;
    for (int i = 0; i < zeroDeg*a; i += a) {
        PyList_SetItem(pList, idx, PyLong_FromLong(i));        
        idx++;
    }
    PyObject* pInt = PyLong_FromLong(size);

    // Pack them into a tuple
    PyObject* pArgs = PyTuple_Pack(2, pList, pInt);
    PyObject* pValue = PyObject_CallObject(pFunc, pArgs);

    // Handle return value
    if (pValue) {
        std::cout << "Result: " << PyLong_AsLong(pValue) << std::endl;
        Py_DECREF(pValue);
    } else {
        std::cerr << "Function call failed!\n";
        PyErr_Print();
    }

    // Cleanup
    Py_DECREF(pArgs);
    Py_DECREF(pFunc);
    // return size*zeroDeg*a;
    return PyLong_AsLong(pValue);
}

//checks memorization maps for AllGather and ReduceScatter operations
//if the result is not found, it computes the cycles using compute_cycles
int getZeroCycles(PyObject* pModule, int dataSize, int zeroDeg, int a, int operation) {
    InputKey key{dataSize, zeroDeg*a};

    if(operation == 1) {
        if (gatherMemorization.find(key) != gatherMemorization.end()) {
            // std::cout << "Cache hit for size=" << dataSize << " and num_devices=" << zeroDeg*a << "\n";
            return gatherMemorization[key]; // Return cached value
        }

        // std::cout << "Cache miss for size=" << dataSize << " and num_devices=" << zeroDeg*a << "\n";
        int result = compute_cycles(pModule, dataSize, zeroDeg, a, operation);
        gatherMemorization[key] = result; // Store in the cache
        return result;
    }
    else if(operation == 2) {
        if (scatterMemorization.find(key) != scatterMemorization.end()) {
            // std::cout << "Cache hit for size=" << dataSize << " and num_devices=" << zeroDeg*a << "\n";
            return scatterMemorization[key]; // Return cached value
        }

        // std::cout << "Cache miss for size=" << dataSize << " and num_devices=" << zeroDeg*a << "\n";
        int result = compute_cycles(pModule, dataSize, zeroDeg, a, operation);
        scatterMemorization[key] = result; // Store in the cache
        return result;
    }
}

Result Graph::runDP (PyObject* pModule) {
    // initialize DP table: dp[level][downset][k][s]
    // (partition downset over AT MOST k devices/accelerators, with EXACTLY s stages)
    //store different possibilities at index 0 : level
    dp.assign(ins.num_levels, vector<vector<vector<pair<double, networkSkeleton>>>>(
        downsets.size(), vector<vector<pair<double, networkSkeleton>>>(
        ins.maxDevices+1, vector<pair<double, networkSkeleton>>(
        boundOnS+1, make_pair(INFTY, networkSkeleton())))));

    // case of the empty set (downset with ID 0)
    for (int i = 0; i < ins.num_levels; i++) {
        for (int k = 0; k <= ins.maxDevices; ++k) {
            dp[i][0][k][0].first = 0;
        }
    }

    // profiling stuff
    double timeSpentInGetLoadOfStage = 0.0;

    // here we go!
    dbg << "running DP..." << endl;
    const clock_t startTimeDP = clock();
    for (int id : downsetsSortedBySize) {
        if (id == 0) continue; // already filled above

        if (id == idOfFullSet) break; // will be handled separately below

        // we want to fill dp[*][id][*][*] (already initialized to INFTY).
        // we will loop over every subdownset subId (their list is already
        // precomputed in subDownsets[id] for convenience)

        //pre check condition to see if a single layer can be placed on a single device
        if(id == 1) {
            int subId = 0;
            int selectedA = -1;
            int max_s = 0;
            vector<map<int,LoadOfStage>> loadOfStages = getLoadOfStage(pModule, id, subId);
            for (const pair<int,LoadOfStage> &it : loadOfStages[0]) { // loop over a
                const int a = it.first;
                if(it.second.max_s_memory_feasible >= boundOnS) {
                    selectedA = a;
                }
                if(a == tmpcWidth) {
                    max_s = it.second.max_s_memory_feasible;
                }
            }
            if(selectedA == -1) {
                if(max_s > 0) {
                    isZero = false;
                    cout << "Pipeline parallelism" << endl;
                    cout << "max stages = " << max_s << endl;
                }
                else {
                    fail("Memory usage too high even for ZeRO");
                }
            }
            else if(selectedA == tmpcWidth) {
                isZero = false;
                cout << "Pipeline parallelism" << endl;
            }
            else {
                isZero = true;
                cout << "ZeRO Optimizer" << endl;
            }
        }

        cout << id << endl;
        for (int subId : subDownsets[id]) {

            // putting downsets[id] \ downsets[subId] (contiguous set) as the next stage

            const clock_t startTimeGetLoadOfStage = clock();
            vector<map<int,LoadOfStage>> loadOfStages = getLoadOfStage(pModule, id, subId);
            timeSpentInGetLoadOfStage += (clock() - startTimeGetLoadOfStage) * 1.0 / CLOCKS_PER_SEC;

            vector<int> nodes = vectorOfSetBits(getContiguousSet(id, subId));

            for (const pair<int,LoadOfStage> &it : loadOfStages[0]) { // loop over a
                const int a = it.first;
                const int max_s = min(it.second.max_s_memory_feasible, boundOnS);
                for (int s = 1; s <= max_s; ++s) {
                    const int recomp = (ins.activationRecomputation && s > 1) ? 1 : 0;
#pragma GCC unroll 16
                    for (int k = a; k <= ins.maxDevices; ++k) {
                        // subdownset could not be placed
                        if(dp[2][subId][k-a][s-1].first == INFINITY) {
                            continue;
                        }
                        if(recomp) {
                            //minimum of existing values is greater than the maximum of the updating values - then update
                            if(dp[0][id][k][s].first > max(dp[2][subId][k-a][s-1].first, loadOfStages[size(loadOfStages)-1][a].fw_bw_latency_with_recompute)) {
                                    int loadIdx = 0;
                                    dp[0][id][k][s].second.fatTree = dp[0][subId][k-a][s-1].second.fatTree;   //substituting the fatTree to be updated
                                    loadIdx = dp[0][id][k][s].second.placeNodes(ins, dp[0][id][k][s].second.fatTree, nodes, a);
                                    //based on loadIdx pick the required load values and place in dp table
                                    int level = 0;
                                    for (int i = loadIdx; i < loadOfStages.size(); i += ins.num_levels) {
                                        dp[level][id][k][s].second.fatTree = dp[0][id][k][s].second.fatTree;
                                        dp[level][id][k][s].first = min(dp[level][id][k][s].first, max(dp[loadIdx][subId][k-a][s-1].first, loadOfStages[i][a].fw_bw_latency_with_recompute));  
                                        level++;
                                    }
                            }
                            //based on placement of the load decide the update status and update
                            else {
                                int loadIdx = 0;
                                networkSkeleton temp_ns;
                                temp_ns.fatTree = dp[0][subId][k-a][s-1].second.fatTree;
                                loadIdx = temp_ns.placeNodes(ins, temp_ns.fatTree, nodes, a);

                                int level = 0;
                                double diff = 0;
                                for (int j = 0; j < ins.num_levels; j++) {
                                    diff += (dp[j][id][k][s].first - max(dp[loadIdx][subId][k-a][s-1].first, loadOfStages[loadIdx + (j)*ins.num_levels][a].fw_bw_latency_with_recompute));
                                }
                                if(diff > 0) {
                                    for (int i = loadIdx; i < loadOfStages.size(); i += ins.num_levels) {
                                        minify(dp[level][id][k][s].first, max(dp[loadIdx][subId][k-a][s-1].first, loadOfStages[i][a].fw_bw_latency_with_recompute));
                                        dp[level][id][k][s].second.fatTree = temp_ns.fatTree;
                                        level++;
                                    }
                                    
                                }
                            }
                        }
                        else {
                            //minimum of existing values is greater than the maximum of the updating values - then update
                            if(dp[0][id][k][s].first > max(dp[2][subId][k-a][s-1].first, loadOfStages[size(loadOfStages)-1][a].fw_bw_latency_wo_recompute)) {
                                int loadIdx = 0;
                                dp[0][id][k][s].second.fatTree = dp[0][subId][k-a][s-1].second.fatTree;   //substituting the fatTree to be updated
                                loadIdx = dp[0][id][k][s].second.placeNodes(ins, dp[0][id][k][s].second.fatTree, nodes, a);
                                //based on loadIdx pick the required load values and place in dp table
                                int level = 0;
                                for (int i = loadIdx; i < loadOfStages.size(); i += ins.num_levels) {
                                    dp[level][id][k][s].second.fatTree = dp[0][id][k][s].second.fatTree;
                                    dp[level][id][k][s].first = min(dp[level][id][k][s].first, max(dp[loadIdx][subId][k-a][s-1].first, loadOfStages[i][a].fw_bw_latency_wo_recompute));  
                                    level++;
                                }
                            }
                            //based on placement of the load decide the update status and update
                            else {
                                if(dp[0][id][k][s].first == dp[2][subId][k-a][s-1].first == INFINITY) {
                                    continue;
                                }
                                int loadIdx = 0;
                                networkSkeleton temp_ns;
                                temp_ns.fatTree = dp[0][subId][k-a][s-1].second.fatTree;
                                loadIdx = temp_ns.placeNodes(ins, temp_ns.fatTree, nodes, a);
                                int level = 0;
                                double diff = 0;
                                for (int j = 0; j < ins.num_levels; j++) {
                                    diff += (dp[j][id][k][s].first - max(dp[loadIdx][subId][k-a][s-1].first, loadOfStages[loadIdx + (j)*ins.num_levels][a].fw_bw_latency_wo_recompute));
                                }
                                if(diff > 0) {
                                    for (int i = loadIdx; i < loadOfStages.size(); i += ins.num_levels) {
                                        minify(dp[level][id][k][s].first, max(dp[loadIdx][subId][k-a][s-1].first, loadOfStages[i][a].fw_bw_latency_wo_recompute));
                                        dp[level][id][k][s].second.fatTree = temp_ns.fatTree;
                                        level++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    const double timeSpentInDPLoop = (clock() - startTimeDP) * 1.0 / CLOCKS_PER_SEC - timeSpentInGetLoadOfStage;
    // this is EXCLUDING the time spent in getLoadOfStage() calls
    const double timeSpentInGetLoadOfStage_mainDP = timeSpentInGetLoadOfStage;

    // final DP round (placing the first stage)
    const clock_t startTimeFinalDPRound = clock();
    double finalTimePerBatch = INFTY;
    int finalD = -1, finalS = -1, finalSubId = -1, finalA = -1;
    for (int d = 1; d <= ins.maxDevices && d <= ins.mbsInBatch; ++d) {
        // d = data-parallelism degree
        if (DATA_PARALLEL_DEGREE_MUST_DIVIDE_BATCH_SIZE && ins.mbsInBatch % d != 0) {
            continue;
        }
        if (DATA_PARALLEL_DEGREE_MUST_DIVIDE_NUM_DEVICES && ins.maxDevices % d != 0) {
            continue;
        }
        const int numDevicesPerPipeline = ins.maxDevices / d;
        const int mbsInBatchPerPipeline = ceildiv(ins.mbsInBatch, d);

        for (int subId : subDownsets[idOfFullSet]) {
            // putting (full set) \ downsets[subId] (contiguous set) as the first stage
            clock_t startTime = clock();
            vector<map<int,LoadOfStage>> loadOfStages = getLoadOfStage(pModule, idOfFullSet, subId);
            timeSpentInGetLoadOfStage += (clock() - startTime) * 1.0 / CLOCKS_PER_SEC;

            for (const pair<int,LoadOfStage> &it : loadOfStages[0]) { // loop over a
                const int a = it.first;
                if (numDevicesPerPipeline < a) continue; // not enough devices
                const int max_s = min(it.second.max_s_memory_feasible, boundOnS);
                for (int s = 1; s <= max_s; ++s) {
                    // since this is the first stage, s = pipeline depth (number of stages)
                    if (dp[0][subId][numDevicesPerPipeline - a][s-1].first > INFTY / 2) continue;
                    const double load = (ins.activationRecomputation && s > 1)
                            ? it.second.fw_bw_latency_with_recompute
                            : it.second.fw_bw_latency_wo_recompute;
                    vector<int> dp_nodes = {};
                    for (int i = 0; i < d*numDevicesPerPipeline; i += numDevicesPerPipeline) {
                        dp_nodes.push_back(i);
                    }
                    const double timePerBatch =
                            max(dp[0][subId][numDevicesPerPipeline - a][s-1].first, load)
                            *
                            (mbsInBatchPerPipeline + s - 1)
                            +
                            getDataParallelResyncCostTree(d, it.second.parameter_size, dp_nodes);
                    if (timePerBatch < finalTimePerBatch) {
                        finalTimePerBatch = timePerBatch;
                        finalD = d;
                        finalS = s;
                        finalA = a;
                        finalSubId = subId;
                    }
                }
            }
        }
    }
    if(finalD == -1 && finalS == -1 && finalA == -1 ) {
        fail("no feasible solution found - memory usage high for Pipeline parallelism");
    }
    // cout << "Final Values\n";
    // cout << "finalTimePerBatch : " << finalTimePerBatch << endl;
    // cout << "finalD : " << finalD << endl;
    // cout << "finalS : " << finalS << endl;
    // cout << "finalA : " << finalA << endl;
    int finalNumDevicesPerPipeline = ins.maxDevices / finalD;
    vector<int> finalNodes = vectorOfSetBits(getContiguousSet(idOfFullSet, finalSubId));
    networkSkeleton final_ns;
    networkSkeleton final_ns_dp;
    final_ns.fatTree = dp[0][finalSubId][finalNumDevicesPerPipeline - finalA][finalS-1].second.fatTree;
    final_ns.placeNodes(ins, final_ns.fatTree, finalNodes, finalA);
    // cout << "Final Placement\n";
    // map<int, vector<int>> sortedFatTree_fin(final_ns.fatTree.begin(), final_ns.fatTree.end());
    // for (const auto& pair : sortedFatTree_fin) {
    // cout << "device " << pair.first << " : ";
    //     for (const int& element : pair.second) {
    //         cout << element << " ";
    //     }
    // cout << endl;
    // }
    final_ns_dp.fatTree = final_ns.duplicateFatTree(final_ns.fatTree, finalD);

    const double timeSpentInFinalDPRound = (clock() - startTimeFinalDPRound) * 1.0 / CLOCKS_PER_SEC
                                    - (timeSpentInGetLoadOfStage - timeSpentInGetLoadOfStage_mainDP);
    // this is EXCLUDING the time spent in getLoadOfStage() calls

    if (finalTimePerBatch > INFTY/2) {
        // we didn't find a feasible solution
        dbg << "no feasible solution found" << endl;
        return Result();
    }
    dbg << "finalTimePerBatch = " << finalTimePerBatch << endl;

    // now we reconstruct the solution
    Result result;
    result.dataParallelDegree = finalD;
    result.tensorParallelDegree = tmpcWidth;
    result.netSkel = final_ns;
    result.netSkel_dp = final_ns_dp;
    // begin by placing the first stage
    ResultStage firstStage;
    firstStage.devicesForStage = finalA;
    firstStage.nodes = vectorOfSetBits(getContiguousSet(idOfFullSet, finalSubId));
    result.stages.push_back(firstStage);
    // now reconstruct the rest of the stages
    int curId = finalSubId, curS = finalS - 1, curK = ins.maxDevices / finalD - finalA;
    while (curId != 0) { // curId is not empty set
        assert(curK > 0);
        assert(curS > 0);
        bool found = false;
        for (int subId : subDownsets[curId]) {
            const clock_t startTimeGetLoadOfStage = clock();
            vector<map<int,LoadOfStage>> loadOfStages = getLoadOfStage(pModule, curId, subId);
            timeSpentInGetLoadOfStage += (clock() - startTimeGetLoadOfStage) * 1.0 / CLOCKS_PER_SEC;

            vector<int> nodes = vectorOfSetBits(getContiguousSet(curId, subId));
            int first_node = nodes[0];
            int last_node = nodes[nodes.size() - 1];

            int fwdIdx = final_ns.getLoadIdx(ins, final_ns.fatTree, first_node-1, first_node);
            int bwdIdx = final_ns.getLoadIdx(ins, final_ns.fatTree, last_node, last_node+1);
            int loadIdx = bwdIdx + ins.num_levels*fwdIdx;

            for (const pair<int,LoadOfStage> &it : loadOfStages[0]) { // loop over a
                int a;
                if(it.second.max_s_memory_feasible >= curS) {
                    a = it.first;
                }
                else {
                    continue;
                }
                if (curK < a) continue; // not enough devices
                const int recomp = (ins.activationRecomputation && curS > 1) ? 1 : 0;
                double load = 0;

                if(recomp) {
                    load = loadOfStages[loadIdx][a].fw_bw_latency_with_recompute;
                }
                else {
                    load = loadOfStages[loadIdx][a].fw_bw_latency_wo_recompute;
                }

                if (1e-9 > abs(dp[fwdIdx][curId][curK][curS].first - max(dp[bwdIdx][subId][curK-a][curS-1].first, load))) {
                    // found the next stage
                    found = true;

                    ResultStage rs;
                    rs.devicesForStage = a;
                    rs.nodes = vectorOfSetBits(getContiguousSet(curId, subId));
                    result.stages.push_back(rs);

                    dbg << "formed a stage with nodes [" << rs.nodes
                        << "] using a=" << a << " many devices, yielding load=" << load << endl;
                    
                    curS -= 1;
                    curK -= a;
                    curId = subId;
                    
                    break;
                }
            }
            if (found) break;
        }
        if (!found) fail("didn't find any reconstruction step to make?");
    }
    if (curS > 0) fail("s didn't fall to 0 by the end of reconstruction?");
    assert(result.stages.size() == finalS);
    // curK, however, might not be 0 by the end
    dbg << "solution used " << ins.maxDevices / finalD - curK
        << " out of available " << ins.maxDevices / finalD << " devices per pipeline" << endl;
    dbg << "data parallel degree: " << finalD << endl;
    dbg << "tensor parallel degree: " << tmpcWidth << endl;
    dbg << "number of stages: " << finalS << endl;
    dbg << endl;

    dbg << "time spent: " << endl;
    dbg << "  in getLoadOfStage: " << timeSpentInGetLoadOfStage << endl;
    dbg << "  in (rest of) DP loop: " << timeSpentInDPLoop << endl;
    dbg << "  in (rest of) final DP round: " << timeSpentInFinalDPRound << endl;

    const double finalTimePerBatchSanityCheck = getTimePerBatchForResult(pModule, result, final_ns);
    cout << finalTimePerBatchSanityCheck << endl;
    if (abs(finalTimePerBatch - finalTimePerBatchSanityCheck) > 1e-9) {
        dbg << "finalTimePerBatch = " << finalTimePerBatch << endl;
        dbg << "finalTimePerBatchSanityCheck = " << finalTimePerBatchSanityCheck << endl;
        fail("finalTimePerBatch != finalTimePerBatchSanityCheck");
    }

    return result;
}


//returns vector of loadOfStages - for all possibilities
vector<map<int,LoadOfStage>> Graph::getLoadOfStage (PyObject* pModule, int id, int subId) const {
    vector<bool> nodesVB = getContiguousSet(id, subId);
    vector<int> nodes = vectorOfSetBits(nodesVB);
    return getLoadOfStage(pModule, nodes, nodesVB);
}


//returns vector of loadOfStages - for all possibilities
vector<map<int,LoadOfStage>> Graph::getLoadOfStage (PyObject* pModule, const vector<int> &nodes) const {
    vector<bool> nodesVB = vector<bool>(ins.nodes.at(tmpcWidth).size(), false);
    for (int v : nodes) nodesVB[v] = true;
    return getLoadOfStage(pModule, nodes, nodesVB);
}

// Start running the greedy algorithm to place the nodes in the stage
// onto a accelerators. If at any point we find that having more accelerators might
// be useful, we split off another execution with a+1.
// At the end, save the result as resultMap[a].

// For now, just a version that does not handle any layer-level branching,
// but it does handle TMPCs.
// (It assumes it is run for `a` being enough to do tensor-parallel computation,
//  if there is a tensor-parallelized layer in `nodes`.)
void Graph::getLoadOfStageForA (PyObject* pModule, const vector<int> &nodes, const vector<bool> &nodesVB,
                                int a, vector<map<int,LoadOfStage>> &resultMap) const {
    // this better be deterministic, as it is run again during reconstruction

    // FW pass
    vector<unordered_map<int, double>> finishingTimes;
    finishingTimes.resize(ins.num_levels);
    vector<double> fwPassLatencys;
    fwPassLatencys.assign(ins.num_levels, 0.0);
    int prevNode = -1;
    for (int v : nodes) {
        // nodes in `nodes` come in topological order
        vector<double> startTimes;
        startTimes.assign(ins.num_levels, 0.0);
        bool seenEdgeFromPrevNode = false;
        for (const pair<int,double>& it : incomingEdges.at(v)) {
            if (prevNode == it.first) {
                seenEdgeFromPrevNode = true;
            }
            if (finishingTimes[0].count(it.first)) {
                // in our simple special case, the wanted tensor will already be on this device,
                // so just take the finishing time of it.first into account
                for (int i = 0; i < ins.num_levels; i++) {
                    startTimes[i] = max(startTimes[i], finishingTimes[i][it.first]);
                }
            } else {
                // the wanted tensor was available already at time 0, but on some other device,
                // so need to take transfer cost into account.
                for (int i = 0; i < ins.num_levels; i++) {
                    if(i == 0) {
                        startTimes[i] = max(startTimes[i], (it.second / ins.bws[i]) + (ins.latency_per_level[i]));
                    }
                    else {
                        startTimes[i] = max(startTimes[i], 2*((it.second / ins.bws[i]) + (ins.latency_per_level[i])));
                    }
                }
            }
        }
        if (prevNode != -1 && !seenEdgeFromPrevNode) fail("branching in the layer graph, but this is not supported yet");
        for (int i = 0; i < ins.num_levels; i++) {
            finishingTimes[i][v] = startTimes[i] + node[v]->optimalLatencyFw;
            fwPassLatencys[i] = finishingTimes[i][v];
        }
        prevNode = v;
    }

    // we ignore the outgoing edges - these are the following stages' problem, not ours
    // fwPassLatency computed
    // BW pass (standalone, without recomputing FW pass)
    finishingTimes.clear();
    finishingTimes.resize(ins.num_levels);
    vector<double> bwPassLatencys;
    bwPassLatencys.assign(ins.num_levels, 0.0);
    for (int j = nodes.size()-1; j >= 0; --j) {
        const int v = nodes[j];

        // when can we start running v?
        vector<double> startTimes;
        startTimes.assign(ins.num_levels, 0.0);
        for (const pair<int,double>& it : outgoingEdges.at(v)) {
            if (finishingTimes[0].count(it.first)) {
                for (int i = 0; i < ins.num_levels; i++) {
                    startTimes[i] = max(startTimes[i], finishingTimes[i][it.first]);
                }
            } else {
                for (int i = 0; i < ins.num_levels; i++) {
                    if(i == 0) {
                        startTimes[i] = max(startTimes[i], (it.second / ins.bws[i]) + (ins.latency_per_level[i]));
                    }
                    else {
                        startTimes[i] = max(startTimes[i], 2*((it.second / ins.bws[i]) + (ins.latency_per_level[i])));
                    }
                }
            }
        }
        for (int i = 0; i < ins.num_levels; i++) {
            finishingTimes[i][v] = startTimes[i] + node[v]->optimalLatencyBw;
            bwPassLatencys[i] = finishingTimes[i][v];
        }
    }
    // bwPassLatency computed

    // now FW+BW pass (when doing activation recomputation)
    finishingTimes.clear();
    finishingTimes.resize(ins.num_levels);
    vector<double> fwBwPassLatency_fw_parts;
    fwBwPassLatency_fw_parts.assign(ins.num_levels, 0.0);
    for (int v : nodes) {
        vector<double> startTimes;
        startTimes.assign(ins.num_levels, 0.0);
        for (const pair<int,double>& it : incomingEdges.at(v)) {
            if (finishingTimes[0].count(it.first)) {
                for (int i = 0; i < ins.num_levels; i++) {
                    startTimes[i] = max(startTimes[i], finishingTimes[i][it.first]);
                }
            }
        }
        for (int i = 0; i < ins.num_levels; i++) {
            finishingTimes[i][v] = startTimes[i] + node[v]->optimalLatencyBw;
            fwBwPassLatency_fw_parts[i] = finishingTimes[i][v];
        }
    }
    finishingTimes.clear();
    finishingTimes.resize(ins.num_levels);
    vector<double> fwBwPassLatency_entires;
    fwBwPassLatency_entires.assign(ins.num_levels, 0.0);
    for (int j = nodes.size()-1; j >= 0; --j) {
        const int v = nodes[j];
        vector<double> startTimes;
        startTimes.assign(fwBwPassLatency_fw_parts.begin(), fwBwPassLatency_fw_parts.end());
        for (const pair<int,double>& it : outgoingEdges.at(v)) {
            if (finishingTimes[0].count(it.first)) {
                for (int i = 0; i < ins.num_levels; i++) {
                    startTimes[i] = max(startTimes[i], finishingTimes[i][it.first]);
                }
            } else {
                // the wanted (backward) tensor is coming from another device,
                // but it could already begin the transfer at time 0
                // (during our forward pass)
                for (int i = 0; i < ins.num_levels; i++) {
                    if(i == 0) {
                        startTimes[i] = max(startTimes[i], (it.second / ins.bws[i]) + (ins.latency_per_level[i]));
                    }
                    else {
                        startTimes[i] = max(startTimes[i], 2*((it.second / ins.bws[i]) + (ins.latency_per_level[i])));
                    }
                }
            }
        }
        for (int i = 0; i < ins.num_levels; i++) {
            finishingTimes[i][v] = startTimes[i] + node[v]->optimalLatencyBw;
            fwBwPassLatency_entires[i] = finishingTimes[i][v];
        }
    }

    vector<LoadOfStage> results;
    results.resize(ins.num_levels*ins.num_levels);

    double param_size = 0.0;
    for (int v : nodes) {
        param_size += node[v]->parameterSize;
    }

    double momentum_size = 2*param_size;
    double variance_size = 2*param_size;
    double parameter_copy_size = 2*param_size;
    // left to compute the memory usage,
    // which is of the form const_memory_usage + (s-1) * stashed_data

    const double optimizerSize = (ins.optimizerAlgorithm == "Adam") ? (momentum_size + variance_size + parameter_copy_size) : 0;
    double activationsSize = 0.0;
    for (int v : nodes) {
        activationsSize += node[v]->activationSize;
    }
    const double constMemoryUsage = 2 * param_size + optimizerSize + activationsSize;
    double stashedData = 0.0;
    if (ins.activationRecomputation) {
        // stash all activations that come over incoming edges
        for (int v : nodes) {
            for (const pair<int,double>& it : incomingEdges.at(v)) {
                if (!nodesVB[it.first]) {
                    // edge is coming from a previous stage
                    stashedData += it.second;
                }
            }
        }
    } else {
        // stash all activations
        stashedData = activationsSize;
    }

    int max_s_mem_feas;
    // okay, now we want to return the largest s such that constMemoryUsage + (s-1) * stashedData <= ins.maxMemoryPerDevice
    // first check if s = ins.maxDevices would be fine
    if (constMemoryUsage + (ins.maxDevices-1) * stashedData <= ins.maxMemoryPerDevice) {
        max_s_mem_feas = ins.maxDevices;
    } else if (constMemoryUsage > ins.maxMemoryPerDevice) {
        // even s = 1 is impossible
        max_s_mem_feas = 0;
    } else {
        // something in between
        max_s_mem_feas = 1 + floor((ins.maxMemoryPerDevice - constMemoryUsage) / stashedData);
    }

    //updated the vector of values into results to return from this function
    for (int i = 0; i < ins.num_levels; i++) {
        for (int j = 0; j < ins.num_levels; j++) {
            results[i*ins.num_levels+j].fw_bw_latency_wo_recompute = fwPassLatencys[i] + bwPassLatencys[j];
            results[i*ins.num_levels+j].fw_bw_latency_with_recompute = fwPassLatencys[i] + fwBwPassLatency_entires[j];
            results[i*ins.num_levels+j].parameter_size = param_size;
            results[i*ins.num_levels+j].max_s_memory_feasible = max_s_mem_feas;
        }
    }

    // NOTE: the formula constMemoryUsage + (s-1) * stashedData is applicable to
    // pipelining schemes such as PipeDream-Flush (1F1B). For GPipe,
    // a more appropriate formula is constMemoryUsage + num_microbatches_per_pipeline * stashedData,
    // but num_microbatches_per_pipeline is mbsInBatch / d,
    // and so we would have to make the DP aware of d, which is not the case right now.
    // (Time complexity wise, it should actually be fine, but requires some changes.)
    // So for now we do not support GPipe

    // all good
    for (int i = 0; i < ins.num_levels*ins.num_levels; i++) {
    resultMap[i][a] = results[i];
    }

    //invoke getZeroLoadOfStageForA when there is a memory bottleneck - resultMap will automatically get updated

    if(isZero) {
        int memory_feasible = 0;
        bool break_outer_loop = false;
        int max_devices = 0;
        int comm_cycles = -1;
        if(max_s_mem_feas == 0) {
            for (int zD = 2; a*zD <= ins.maxDevices; zD *= 2) {
                for (int zT = 1; zT <= 3; zT++) {
                    memory_feasible = getMemoryFeasibilityZero(nodes, nodesVB, a, zD, zT, resultMap);
                    if(a*zD > max_devices) {
                        max_devices = a*zD;
                    }
                    if(memory_feasible >= boundOnS) {
                        comm_cycles = getZeroLoadOfStageForA(pModule, nodes, nodesVB, a, zD, zT, resultMap);
                        if(comm_cycles == 0) {
                            continue;
                        }
                        break_outer_loop = true;
                        break;
                    }
                }
                if(break_outer_loop) {
                    break;
                }
            }
        }
    }
}

int Graph::getMemoryFeasibilityZero (const vector<int> &nodes, const vector<bool> &nodesVB,
    int a, int zeroDeg, int zeroType, vector<map<int,LoadOfStage>> &resultMap) const {

    //compute the load (comp + comm) for ZeRO assuming a accelerators and a degree of zeroDeg
    //memory mapping will change and communication cost will change majorly

    vector<LoadOfStage> results;
    results.resize(resultMap.size());
    for (int i = 0; i < resultMap.size(); i++) {
        results[i] = resultMap[i][a];
    }

    LoadOfStage result;
    result = results[0];
    double momentum_size = 2*result.parameter_size;
    double variance_size = 2*result.parameter_size;
    double parameter_copy_size = 2*result.parameter_size;

    //num_parameters = w
    //parameter_size = 2w (fp16)
    //gradients_size = 2w (fp16)
    //optimizerSize:
    //momentum_size = 4w (fp32)
    //variance_size = 4w (fp32)
    //copy of parameters = 4w (fp32)
    double optimizerSize = (ins.optimizerAlgorithm == "Adam") ? (momentum_size + variance_size + parameter_copy_size) : 0;
    double activationsSize = 0.0;
    for (int v : nodes) {
        activationsSize += node[v]->activationSize;
    // these are ALL the intermediate activations
    }
    
    double constMemoryUsage = 0.0;
    if(zeroType == 1) {
        //optimizer states memory reduction
        optimizerSize /= zeroDeg;
        momentum_size /= zeroDeg;
        variance_size /= zeroDeg;
        parameter_copy_size /= zeroDeg;
        //reduction in memory usage
        constMemoryUsage = 2 * result.parameter_size + optimizerSize + activationsSize;
    }
    else if(zeroType == 2) {
        //optimizer states memory reduction
        optimizerSize /= zeroDeg;
        momentum_size /= zeroDeg;
        variance_size /= zeroDeg;
        parameter_copy_size /= zeroDeg;
        //reduction in memory usage
        constMemoryUsage = result.parameter_size + (result.parameter_size/zeroDeg) + optimizerSize + activationsSize;
    }
    else if(zeroType == 3) {
        //optimizer states memory reduction
        optimizerSize /= zeroDeg;
        momentum_size /= zeroDeg;
        variance_size /= zeroDeg;
        parameter_copy_size /= zeroDeg;
        //reduction in memory usage
        constMemoryUsage = 2 * (result.parameter_size/zeroDeg) + optimizerSize + activationsSize;
    }
    double stashedData = 0.0;
    if (ins.activationRecomputation) {
        // stash all activations that come over incoming edges
        for (int v : nodes) {
            for (const pair<int,double>& it : incomingEdges.at(v)) {
                if (!nodesVB[it.first]) {
                    // edge is coming from a previous stage
                    stashedData += it.second;
                }
            }
        }
    } else {
        // stash all activations
        stashedData = activationsSize;
    }
    // okay, now we want to return the largest s such that constMemoryUsage + (s-1) * stashedData <= ins.maxMemoryPerDevice
    // first check if s = ins.maxDevices would be fine
    assert(constMemoryUsage > 0);
    if (constMemoryUsage + (ins.maxDevices-1) * stashedData <= ins.maxMemoryPerDevice) {
        result.max_s_memory_feasible = ins.maxDevices; // we never need more than that
    } 
    else if (constMemoryUsage > ins.maxMemoryPerDevice) {
        // even s = 1 is impossible
        result.max_s_memory_feasible = 0;
    } 
    else {
        // something in between
        result.max_s_memory_feasible = 1 + floor((ins.maxMemoryPerDevice - constMemoryUsage) / stashedData);
    }
    
    return result.max_s_memory_feasible;
}

int Graph::getZeroLoadOfStageForA (PyObject* pModule, const vector<int> &nodes, const vector<bool> &nodesVB,
                             int a, int zeroDeg, int zeroType, vector<map<int,LoadOfStage>> &resultMap) const {
    vector<LoadOfStage> results;
    results.resize(resultMap.size());
    for (int i = 0; i < resultMap.size(); i++) {
        results[i] = resultMap[i][a];
    }

    LoadOfStage result;
    result = results[0];
    double momentum_size = 2*result.parameter_size;
    double variance_size = 2*result.parameter_size;
    double parameter_copy_size = 2*result.parameter_size;

    double optimizerSize = (ins.optimizerAlgorithm == "Adam") ? (momentum_size + variance_size + parameter_copy_size) : 0;
    double activationsSize = 0.0;
    for (int v : nodes) {
        activationsSize += node[v]->activationSize;
        // these are ALL the intermediate activations
    }

    //communication cost of model states
    //based on sizes compute the communication cost
    int param_comm_cycles = 0;
    int gradient_comm_cycles = 0;
    double param_comm_cost = 0.0;
    double gradient_comm_cost = 0.0;
    double opt_update_cost = 0.0;

    if(zeroType == 2 || zeroType == 3) {
        gradient_comm_cycles = getZeroCycles(pModule, result.parameter_size/zeroDeg, zeroDeg, a, 2);
        gradient_comm_cost = gradient_comm_cycles/ins.frequency;
        if(gradient_comm_cycles == 0) {
            return 0;
        }
    }
    
    if(zeroType == 3) {
        param_comm_cycles = getZeroCycles(pModule, result.parameter_size/zeroDeg, zeroDeg, a, 1);
        param_comm_cost = param_comm_cycles/ins.frequency;
    }

    double constMemoryUsage = 0.0;
    if(zeroType == 1) {
        for (int i = 0; i < results.size(); i++) {
            results[i].fw_bw_latency_wo_recompute += opt_update_cost;
            results[i].fw_bw_latency_with_recompute += opt_update_cost;
        }
        //optimizer states memory reduction
        optimizerSize /= zeroDeg;
        momentum_size /= zeroDeg;
        variance_size /= zeroDeg;
        parameter_copy_size /= zeroDeg;
        //reduction in memory usage
        constMemoryUsage = 2 * result.parameter_size + optimizerSize + activationsSize;
    }
    else if(zeroType == 2) {
        for (int i = 0; i < results.size(); i++) {
            results[i].fw_bw_latency_wo_recompute += opt_update_cost + gradient_comm_cost;
            results[i].fw_bw_latency_with_recompute += opt_update_cost + gradient_comm_cost;
        }
        //optimizer states memory reduction
        optimizerSize /= zeroDeg;
        momentum_size /= zeroDeg;
        variance_size /= zeroDeg;
        parameter_copy_size /= zeroDeg;
        //reduction in memory usage
        constMemoryUsage = result.parameter_size + (result.parameter_size/zeroDeg) + optimizerSize + activationsSize;
    }
    else if(zeroType == 3) {
        for (int i = 0; i < results.size(); i++) {
            results[i].fw_bw_latency_wo_recompute += opt_update_cost + gradient_comm_cost + param_comm_cost;
            results[i].fw_bw_latency_with_recompute += opt_update_cost + gradient_comm_cost + param_comm_cost;
        }
        //optimizer states memory reduction
        optimizerSize /= zeroDeg;
        momentum_size /= zeroDeg;
        variance_size /= zeroDeg;
        parameter_copy_size /= zeroDeg;
        //reduction in memory usage
        constMemoryUsage = 2 * (result.parameter_size/zeroDeg) + optimizerSize + activationsSize;
    }
    double stashedData = 0.0;
    if (ins.activationRecomputation) {
        // stash all activations that come over incoming edges
        for (int v : nodes) {
            for (const pair<int,double>& it : incomingEdges.at(v)) {
                if (!nodesVB[it.first]) {
                    // edge is coming from a previous stage
                    stashedData += it.second;
                }
            }
        }
    } else {
        // stash all activations
        stashedData = activationsSize;
    }
    // okay, now we want to return the largest s such that constMemoryUsage + (s-1) * stashedData <= ins.maxMemoryPerDevice
    // first check if s = ins.maxDevices would be fine
    assert(constMemoryUsage > 0);
    if (constMemoryUsage + (ins.maxDevices-1) * stashedData <= ins.maxMemoryPerDevice) {
        result.max_s_memory_feasible = ins.maxDevices; // we never need more than that
    } else if (constMemoryUsage > ins.maxMemoryPerDevice) {
        // even s = 1 is impossible
        result.max_s_memory_feasible = 0;
    } else {
        // something in between
        result.max_s_memory_feasible = 1 + floor((ins.maxMemoryPerDevice - constMemoryUsage) / stashedData);
    }

    //update max_s_memory_feasible
    for (int i = 0; i < results.size(); i++) {
        results[i].max_s_memory_feasible = result.max_s_memory_feasible;
        if(zeroType == 3) {
            results[i].parameter_size /= zeroDeg;
        }
    }

    // all good
    for (int i = 0; i < resultMap.size(); i++) {
        resultMap[i][a*zeroDeg] = results[i];
    }

    return gradient_comm_cycles;
}

//returns vector of loadOfStages - for all possibilities
vector<map<int,LoadOfStage>> Graph::getLoadOfStage (PyObject* pModule, const vector<int> &nodes, const vector<bool> &nodesVB) const {
    // this better be deterministic, as it is run again during reconstruction

    // by the consistent Megatron assumption, if there is a TMPC-able node
    // in this stage, we need to TMPC it, so then we need a >= t
    int startingA = 1;
    for (int v : nodes) {
        if (node[v]->isTensorParallelized) {
            startingA = tmpcWidth;
            break;
        }
    }
    map<int,LoadOfStage> result;
    vector<map<int, LoadOfStage>> results;
    results.resize(ins.num_levels*ins.num_levels);
    getLoadOfStageForA(pModule, nodes, nodesVB, startingA, results);
    return results;
}


void Graph::renumberResultBack (Result &r) const {
    for (ResultStage &rs : r.stages) {
        for (int &nodeId : rs.nodes) {
            nodeId = ins.oldNumber[nodeId];
        }
    }
}

double Graph::getTimePerBatchForResult (PyObject* pModule, const Result &r, networkSkeleton final_ns) const {
    // for sanity checks of returned solutions

    if (r.stages.empty()) {
        // infeasible/OOM/empty result
        return INFTY;
    }

    // check that the solution is contiguous
    // (and that there is some topological order in the contracted graph)
    // and that every node belongs to exactly one subgraph
    vector<int> stageOfNode(node.size(), -1);
    int devicesUsedPerPipeline = 0;
    for (int i = 0; i < r.stages.size(); ++i) {
        for (int v : r.stages[i].nodes) {
            if (stageOfNode[v] != -1) {
                fail("duplicate node");
            }
            stageOfNode[v] = i;
        }
        if (r.stages[i].devicesForStage < 1 || r.stages[i].devicesForStage > ins.maxDevices) {
            fail("wrong number of devices for stage");
        }
        devicesUsedPerPipeline += r.stages[i].devicesForStage;
    }
    for (int v = 0; v < ins.nodes.at(tmpcWidth).size(); ++v) {
        if (-1 == stageOfNode[v]) {
            fail("node does not appear in any subgraph");
        }
    }
    if (r.dataParallelDegree < 1 || r.dataParallelDegree > ins.maxDevices || r.dataParallelDegree > ins.mbsInBatch) {
        fail("wrong data-parallel degree");
    }
    if (DATA_PARALLEL_DEGREE_MUST_DIVIDE_NUM_DEVICES &&
        ins.maxDevices % r.dataParallelDegree != 0) {
        fail("data-parallel degree must divide the number of devices");
    }
    if (DATA_PARALLEL_DEGREE_MUST_DIVIDE_BATCH_SIZE &&
        ins.mbsInBatch % r.dataParallelDegree != 0) {
        fail("data-parallel degree must divide the number of microbatches in a batch");
    }
    if (r.tensorParallelDegree != tmpcWidth) {
        fail("wrong tensor-parallel degree (using wrong graph?)");
    }
    if (devicesUsedPerPipeline * r.dataParallelDegree > ins.maxDevices) {
        fail("too many devices used");
    }

    // compute the load, and check if memory usage is okay
    double maxLoad = 0.0;
    double parameterSizeInFirstLayer = -1.0;
    for (int stage = 0; stage < r.stages.size(); ++stage) {
        const int s = r.stages.size() - stage; // which stage this is, counting from the end
        const int a = r.stages[stage].devicesForStage;

        // compute the load of this stage
        vector<map<int,LoadOfStage>> loadOfStages = getLoadOfStage(pModule, r.stages[stage].nodes);
        
        int first_node = r.stages[stage].nodes[0];
        int last_node = r.stages[stage].nodes[r.stages[stage].nodes.size()-1];

        const int recomp = (ins.activationRecomputation && s > 1) ? 1 : 0;
        double load = 0;
        if(first_node == 0) {
            first_node = 1;
        }
        if(last_node == ins.nodes.at(tmpcWidth).size()-1) {
            last_node = ins.nodes.at(tmpcWidth).size()-2;
        }
        int fwdIdx = final_ns.getLoadIdx(ins, final_ns.fatTree, first_node-1, first_node);
        int bwdIdx = final_ns.getLoadIdx(ins, final_ns.fatTree, last_node, last_node+1);
        int loadIdx = bwdIdx + ins.num_levels*fwdIdx;

        // the consistent Megatron assumption is verified inside getLoadOfStage

        int bestA = -1;
        for (const pair<int,LoadOfStage> &it : loadOfStages[0]) {
            if (it.first <= a &&  it.second.max_s_memory_feasible >= s-1) {
                bestA = it.first;
            }
        }
        if (bestA == -1) {
            fail("no feasible solution using at most " + to_string(a) + " devices found for this stage");
        }
        if (bestA != a) {
            dbg << "WARNING: would be enough to use " << bestA << " devices instead of "
                << a << " for this stage" << endl;
        }
        if (stage == 0) {
            parameterSizeInFirstLayer = loadOfStages[0].at(bestA).parameter_size;
        }

        // check memory usage
        if (s > loadOfStages[0].at(bestA).max_s_memory_feasible) {
            fail("memory usage too high");
        }

        if(recomp) {
            load = loadOfStages[loadIdx][bestA].fw_bw_latency_with_recompute;
        }
        else {
            load = loadOfStages[loadIdx][bestA].fw_bw_latency_wo_recompute;
        }

        maxLoad = max(maxLoad, load);
    }

    // maxLoad is computed, now compute timePerBatch
    const int mbsInBatchPerPipeline = ceildiv(ins.mbsInBatch, r.dataParallelDegree);
    vector<int> dp_nodes = {};
    for (int i = 0; i < r.dataParallelDegree*devicesUsedPerPipeline; i += devicesUsedPerPipeline) {
        dp_nodes.push_back(i);
    }
    const double timePerBatch = maxLoad * (mbsInBatchPerPipeline + r.stages.size() - 1)
                + getDataParallelResyncCostTree(r.dataParallelDegree, parameterSizeInFirstLayer, dp_nodes);
    return timePerBatch;
}


// returns RENUMBERED-back result
pair<Result,double> run (PyObject* pModule, const Instance &ins) {
    double bestTPB = INFTY;
    Result bestResult;
    for (const auto &it : ins.nodes) {
        int tmpcWidth = it.first;
        dbg << "building Graph for tmpcWidth = " << tmpcWidth << endl;
        Graph g(ins, tmpcWidth);
        Result r = g.runDP(pModule);
        double tpb = g.getTimePerBatchForResult(pModule, r, r.netSkel);
        dbg << "TPB = " << tpb << " for tmpcWidth = " << tmpcWidth << endl;
        if (tpb < bestTPB) {
            bestTPB = tpb;
            g.renumberResultBack(r);
            bestResult = r;
        }
    }
    return make_pair(bestResult, bestTPB);
}


double fixedConfigTimePerBatch(PyObject* pModule, const Instance &ins) {
    dbg << endl << "now trying manual partitioning..." << endl;
    Result r;
    r.tensorParallelDegree = ins.fixedDPStrategy[1];
    r.dataParallelDegree = ins.fixedDPStrategy[2];
    int depthOfPipeline = ins.fixedDPStrategy[0];
    int numTransformers = ins.numTransformerLayers;

    int numTransformersPerStage = (numTransformers + depthOfPipeline - 1) / depthOfPipeline;

    for (int stage = depthOfPipeline-1; stage >= 0; --stage) {
        ResultStage rs;
        rs.devicesForStage = r.tensorParallelDegree;
        if (stage == 0) {
            for (const Node &v : ins.nodes.at(r.tensorParallelDegree)) {
                if (v.id < 0) rs.nodes.push_back(v.id); // junk goes to the first stage
            }
        }
        for (int i = 0; i < numTransformers; ++i) {
            if (i/numTransformersPerStage == stage) rs.nodes.push_back(i);
        }
        if (stage == depthOfPipeline-1) {
            for (const Node &v : ins.nodes.at(r.tensorParallelDegree)) {
                if (v.id >= numTransformers) rs.nodes.push_back(v.id); // junk goes to the last stage
            }
        }
        r.stages.push_back(rs);
        dbg << "stage: " << rs.nodes << endl;
    }
    Graph g(ins, r.tensorParallelDegree);
    networkSkeleton manual_ns;
    for (int i = 0; i < r.stages.size(); i++) {
        for (int j = 0; j < r.tensorParallelDegree; ++j) {
            manual_ns.fatTree[i*r.tensorParallelDegree+j] = r.stages[i].nodes;
        }
    }
    double tpb = g.getTimePerBatchForResult(pModule, r, manual_ns);
    dbg << "manual TPB = " << tpb << endl;
    return tpb;
}


int main (int argc, char **argv) {

    Py_Initialize();

    // Import the Python module (network from utils)
    PyObject* pModule = PyImport_ImportModule("utils.network");
    if (!pModule) {
        std::cerr << "Failed to load Python module!\n";
        PyErr_Print();
        Py_Finalize();
        return 1;
    }

    if (argc != 3) {
        fail("usage: device_placement <input file> <output file>");
    }
    const string inputFilename = argv[1];
    const string outputFilename = argv[2];
    json inputJson;
    ifstream inputFile(inputFilename);
    inputFile >> inputJson;
    Instance ins = inputJson.get<Instance>();
    dbg << "read instance" << endl;
    pair<Result,double> r = run(pModule, ins);
    dbg << "got result" << endl;
    ofstream outputFile(outputFilename);
    json outputJson = json(r.first);
    outputJson["finalTimePerBatch"] = r.second;
    outputJson["networkSkeleton"] = r.first.netSkel.fatTree;

    double fixedStrategyTimePerBatch = fixedConfigTimePerBatch(pModule, ins);

    outputJson["fixedStrategyTimePerBatch"] = fixedStrategyTimePerBatch;
    outputFile << outputJson.dump(4) << endl;

    Py_DECREF(pModule);

    Py_Finalize();
}
