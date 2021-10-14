//
// Created by Cameron Fiore on 6/30/21.
//

#ifndef IMAGEMATCHERPROJECT_ORB_SEVENSCENES_H
#define IMAGEMATCHERPROJECT_ORB_SEVENSCENES_H

#endif //IMAGEMATCHERPROJECT_ORB_SEVENSCENES_H

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <utility>
#include <base/database.h>
#include <base/image.h>

using namespace std;

namespace sevenScenes
{

    Eigen::Matrix3d getR(const string& image);

    Eigen::Vector3d getT(const string& image);

    double getDistBetween(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2);

    double getAngleBetween(const Eigen::Vector3d &d1, const Eigen::Vector3d &d2);

    vector<tuple<string, string, vector<string>, vector<string>>> createInfoVector();

    vector<tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d, int>> pairSelection(const vector<pair<int, float>> &result, const vector<string> &listImage, const string &query, double loThresh, int maxNum, int method);

    bool findEssentialMatrix(const string &db_image, const string &query_image, int method, Eigen::Matrix3d &R, Eigen::Vector3d &t, int& num_points);

    double triangulateRays(const Eigen::Vector3d& ci, const Eigen::Vector3d& di, const Eigen::Vector3d& cj, const Eigen::Vector3d& dj, Eigen::Vector3d &intersect);
}




