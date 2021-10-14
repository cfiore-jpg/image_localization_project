//
// Created by Cameron Fiore on 7/2/21.
//

#ifndef IMAGEMATCHERPROJECT_ORB_KDTREE_H
#define IMAGEMATCHERPROJECT_ORB_KDTREE_H

#endif //IMAGEMATCHERPROJECT_ORB_KDTREE_H

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <utility>
#include <base/database.h>
#include <base/image.h>
#include "sevenScenes.h"

using namespace std;

struct Node
{
    string name;
    Eigen::Vector3d position;
    Node *left, *right;
};

namespace node {
    double getDistanceFrom(Node *n, Eigen::Vector3d p);
}

class KDTree
{
    Node* root;
public:
    explicit KDTree (vector<string>);
    Node* getRoot();
    Node* buildBalancedTree(vector<string>, int);
    Node* nearestNeighbor(Node*, int, Eigen::Vector3d, Node*);
};

