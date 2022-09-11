//
// Created by Cameron Fiore on 6/30/21.
//

#ifndef IMAGEMATCHERPROJECT_ORB_SEVENSCENES_H
#define IMAGEMATCHERPROJECT_ORB_SEVENSCENES_H

#endif //IMAGEMATCHERPROJECT_ORB_SEVENSCENES_H

#include <iostream>
#include <string>
#include <map>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <utility>

using namespace std;

namespace sevenScenes
{
    vector<string> createQueryVector(const string & data_dir, const string & scene);

    void getCalibration(double & f, double & cx, double & cy);

    Eigen::Matrix3d getR(const string & image);

    Eigen::Vector3d getT(const string & image);

    Eigen::Matrix3d getR_BA(const string & image);

    Eigen::Vector3d getT_BA(const string & image);

    bool get3dfrom2d(const cv::Point2d & point_2d, const cv::Mat & depth, cv::Point3d & point_3d);

    void getAbsolutePose(const string& image, Eigen::Matrix3d & R_iw, Eigen::Vector3d & T_iw);

    void getAbsolutePose_BA(const string& image, Eigen::Matrix3d & R_iw, Eigen::Vector3d & T_iw);

    map<pair<string, string>, pair<Eigen::Matrix3d, Eigen::Vector3d>> getAnchorPoses(const string & dir);

}




