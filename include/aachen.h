//
// Created by Cameron Fiore on 7/14/22.
//

#ifndef IMAGE_LOCALIZATION_PROJECT_AACHEN_H
#define IMAGE_LOCALIZATION_PROJECT_AACHEN_H

#endif //IMAGE_LOCALIZATION_PROJECT_AACHEN_H

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <utility>

using namespace std;

namespace aachen {

    void createPoseFiles();

    void createCalibFiles();

    void createQueryVector(vector<string> & queries, vector<tuple<double, double, double>> & calibrations);

    bool getR(const string & image, Eigen::Matrix3d & R);

    bool getC(const string & image, Eigen::Vector3d & c);

    bool getK(const string & image, double & f, double & cx, double & cy);

    bool getAbsolutePose(const string& image, Eigen::Matrix3d & R_iw, Eigen::Vector3d & T_iw);

    vector<string> retrieveSimilar(const string & query_image, int max_num);

    bool getRelativePose(vector<cv::Point2d> & pts_q,
                         double q_f, double q_cx, double q_cy,
                         vector<cv::Point2d> & pts_db,
                         double db_f, double db_cx, double db_cy,
                         double match_thresh,
                         Eigen::Matrix3d & R_qk, Eigen::Vector3d & T_qk);

}
