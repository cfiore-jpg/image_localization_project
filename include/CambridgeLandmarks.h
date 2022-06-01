//
// Created by Cameron Fiore on 3/14/22.
//

#ifndef IMAGE_LOCALIZATION_PROJECT_CAMBRIDGELANDMARKS_H
#define IMAGE_LOCALIZATION_PROJECT_CAMBRIDGELANDMARKS_H

#endif //IMAGE_LOCALIZATION_PROJECT_CAMBRIDGELANDMARKS_H

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <utility>

using namespace std;

namespace cambridge
{
    void createPoseFiles(const string & folder);

    vector<string> getTestImages(const string & folder);

    Eigen::Matrix3d getR(const string& image);

    Eigen::Vector3d getT(const string& image);

    void getAbsolutePose(const string& image, Eigen::Matrix3d &R_i, Eigen::Vector3d &t_i);

}