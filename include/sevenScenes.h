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
    vector<tuple<string, string, vector<string>, vector<string>>> createInfoVector();

    Eigen::Matrix3d getR(const string& image);

    Eigen::Vector3d getT(const string& image);

    void getAbsolutePose(const string& image, Eigen::Matrix3d &R_wi, Eigen::Vector3d &t_wi);
}




