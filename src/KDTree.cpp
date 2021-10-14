//
// Created by Cameron Fiore on 7/2/21.
//

#include "../include/KDTree.h"
#include "../include/sevenScenes.h"
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <utility>
#include <base/database.h>
#include <base/image.h>

using namespace std;

//Node functions

Node* newNode(string n, Eigen::Vector3d p)
{
    Node* temp = new Node;
    temp->name = n;
    temp->position = p;
    temp->left = temp->right = nullptr;
    return temp;
}

namespace node {
    double getDistanceFrom(Node *n, Eigen::Vector3d p) {
        Eigen::Vector3d nC = n->position;
        return sqrt(pow((nC(0) - p(0)), 2.0) + pow((nC(1) - p(1)), 2.0) + pow((nC(2) - p(2)), 2.0));
    }
}

//KDTree functions
KDTree::KDTree(vector<string> listImage)
{
    root = buildBalancedTree(std::move(listImage), 0);
}

Node* KDTree::getRoot()
{
    return root;
}

Node* KDTree::buildBalancedTree(vector<string> listImage, int dim)
{
    sort(listImage.begin(), listImage.end(), [dim](const string& lhs, const string& rhs)
    {
        Eigen::Vector3d lC = sevenScenes::getT(lhs);
        Eigen::Vector3d rC = sevenScenes::getT(rhs);
        return lC(dim) < rC(dim);
    });
    if (dim == 2)
    {
        dim = 0;
    } else
    {
        ++dim;
    }
    if (listImage.empty())
    {
        return nullptr;
    } else if (listImage.size() == 1)
    {
        return newNode(listImage[0], sevenScenes::getT(listImage[0]));
    }
    int mid = (listImage.size() - 1) / 2;
    Node* r = newNode(listImage[mid], sevenScenes::getT(listImage[mid]));
    vector<string> leftList (listImage.begin(), listImage.begin() + mid);
    vector<string> rightList (listImage.begin() + mid + 1, listImage.begin() + listImage.size());
    r->left = buildBalancedTree(leftList, dim);
    r->right = buildBalancedTree(rightList, dim);
    return r;
}

Node* KDTree::nearestNeighbor(Node* cur, int dim, Eigen::Vector3d point, Node* closest)
{
    if (cur != nullptr)
    {
        if (node::getDistanceFrom(cur, point) < node::getDistanceFrom(closest, point))
        {
            closest = cur;
        }

        int nextDim;
        if (dim == 2)
        {
            nextDim = 0;
        } else
        {
            nextDim = dim + 1;
        }

        if (point(dim) <= cur->position(dim))
        {
            closest = nearestNeighbor(cur->left, nextDim, point, closest);
            if (node::getDistanceFrom(closest, point) >= abs(point(dim) - cur->position(dim)))
            {
                closest = nearestNeighbor(cur->right, nextDim, point, closest);
            }
        }

        if (point(dim) > cur->position(dim))
        {
            closest = nearestNeighbor(cur->right, nextDim, point, closest);
            if (node::getDistanceFrom(closest, point) >= abs(point(dim) - cur->position(dim)))
            {
                closest = nearestNeighbor(cur->left, nextDim, point, closest);
            }
        }
    }
    return closest;
}