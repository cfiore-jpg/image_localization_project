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

    bool findMatches(const string& db_image, const string& query_image, const string& method, vector<cv::Point2d>& pts_db, vector<cv::Point2d>& pts_query);

    vector<tuple<Eigen::Matrix3d, Eigen::Vector3d, vector<cv::Point2d>, vector<cv::Point2d>>> getTopXImageMatches(const vector<pair<int, float>> &result, const vector<string> &listImage, const string &queryImage, int num_images, const string& method);

    void handle(const vector<tuple<Eigen::Matrix3d, Eigen::Vector3d, vector<cv::Point2d>, vector<cv::Point2d>>>& lst, const cv::Mat& K, Eigen::Matrix3d& R, Eigen::Vector3d& t, vector<vector<double>>& cams, vector<vector<vector<double>>>& matches);

    double triangulateRays(const Eigen::Vector3d& ci, const Eigen::Vector3d& di, const Eigen::Vector3d& cj, const Eigen::Vector3d& dj, Eigen::Vector3d &intersect);
}




