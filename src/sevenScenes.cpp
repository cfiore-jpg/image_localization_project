//
// Created by Cameron Fiore on 7/2/21.
//

#include "../include/sevenScenes.h"

#include <iostream>
#include <map>
#include <string>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <fstream>
#include "../include/functions.h"

#define PI 3.14159265

using namespace std;

vector<string> sevenScenes::createQueryVector(const string & data_dir, const string & scene) {

    vector<string> queries;
    vector<string> seqs;
    string base = data_dir;
    int num_ims = 0;

    if (scene == "chess") {
        base += "chess/";
        seqs = {"03", "05"};
        num_ims = 1000;
    }

    if (scene == "fire") {
        base += "fire/";
        seqs = {"03", "04"};
        num_ims = 1000;
    }

    if (scene == "heads") {
        base += "heads/";
        seqs = {"01"};
        num_ims = 1000;
    }

    if (scene == "office") {
        base += "office/";
        seqs = {"02", "06", "07", "09"};
        num_ims = 1000;
    }

    if (scene == "pumpkin") {
        base += "pumpkin/";
        seqs = {"01", "07"};
        num_ims = 1000;
    }

    if (scene == "redkitchen") {
        base += "redkitchen/";
        seqs = {"03", "04", "06", "12", "14"};
        num_ims = 1000;
    }

    if (scene == "stairs") {
        base += "stairs/";
        seqs = {"01", "04"};
        num_ims = 500;
    }


    for (const auto &seq: seqs) {
        for (int i = 0; i < num_ims; i++) {
            string stem = "seq-" + seq + "/frame-";
            string base_num = to_string(i);
            int num_zeros = 6 - int(base_num.length());
            for (int j = 0; j < num_zeros; j++) {
                stem += "0";
            }
            stem += base_num;
            queries.push_back(base + stem);
        }
    }

    return queries;
}

void sevenScenes::getCalibration(double & f, double & cx, double & cy) {
    f = 585.;
    cx = 320.;
    cy = 240.;
}

Eigen::Matrix3d sevenScenes::getR(const string& image)
{
    Eigen::Matrix3d R;

    string poseFile = image + ".pose.txt";
    ifstream im_pose (poseFile);
    if (im_pose.is_open())
    {
        string line;
        int row = 0;
        while (getline(im_pose, line))
        {
            if (row < 3)
            {
                stringstream ss(line);
                string num;
                int col = 0;
                while (ss >> num) {
                    if (col < 3)
                    {
                        R(row, col) = stod(num);
                    }
                    ++col;
                }
                ++row;
            }
        }
        im_pose.close();
        double det = R.determinant();
        if(abs(det - 1.) > 0.00000000001) {
            Eigen::JacobiSVD<Eigen::Matrix3d> svd (R, Eigen::ComputeFullU | Eigen::ComputeFullV);
            const auto & u = svd.matrixU();
            auto v = svd.matrixV();
            R = u * v.transpose();
        }
        return R;
    } else
    {
        cout << "Pose file does not exist for: " << image << endl;
        Eigen::Matrix3d l;
        return l;
    }
}

Eigen::Vector3d sevenScenes::getT(const string& image)
{
    Eigen::Vector3d t;

    string poseFile = image + ".pose.txt";
    ifstream im_pose (poseFile);
    if (im_pose.is_open())
    {
        string line;
        int row = 0;
        while (getline(im_pose, line))
        {
            if (row < 3)
            {
                stringstream ss(line);
                string num;
                int col = 0;
                while (ss >> num) {
                    if (col == 3)
                    {
                        t(row) = stod(num);
                    }
                    ++col;
                }
                ++row;
            }
        }
        im_pose.close();
        return t;
    } else
    {
        cout << "Pose file does not exist for: " << image << endl;
        Eigen::Vector3d l;
        return l;
    }
}

Eigen::Matrix3d sevenScenes::getR_BA(const string & image) {

    Eigen::Matrix3d R;

    string poseFile = image + ".bundle_adjusted.pose.txt";
    ifstream im_pose(poseFile);
    if (im_pose.is_open()) {
        string line;
        int row = 0;
        while (row < 3) {
            getline(im_pose, line);
            stringstream ss(line);
            string num;
            int col = 0;
            while (ss >> num) {
                if (col < 3) {
                    R(row, col) = stod(num);
                }
                ++col;
            }
            ++row;
        }
        im_pose.close();

        double det = R.determinant();
        if(abs(det - 1.) > 0.00000000001) {
            Eigen::JacobiSVD<Eigen::Matrix3d> svd (R, Eigen::ComputeFullU | Eigen::ComputeFullV);
            const auto & u = svd.matrixU();
            auto v = svd.matrixV();
            R = u * v.transpose();
        }
        return R;
    } else {
        cout << "Pose file does not exist for: " << image << endl;
        Eigen::Matrix3d l;
        return l;
    }

}

Eigen::Vector3d sevenScenes::getT_BA(const string & image) {

    Eigen::Vector3d T;

    string poseFile = image + ".bundle_adjusted.pose.txt";
    ifstream im_pose(poseFile);
    if (im_pose.is_open()) {
        string line;
        int row = 0;
        while (row < 3) {
            getline(im_pose, line);
            stringstream ss(line);
            string num;
            int col = 0;
            while (ss >> num) {
                if (col == 3) {
                    T(row) = stod(num);
                }
                ++col;
            }
            ++row;
        }
        im_pose.close();
        return T;
    } else {
        cout << "Pose file does not exist for: " << image << endl;
        Eigen::Vector3d l;
        return l;
    }

}

bool sevenScenes::get3dfrom2d(const cv::Point2d & point_2d, const cv::Mat & depth, cv::Point3d & point_3d) {

    double d = depth.at<unsigned short>(point_2d);

    if (d != 65535 && d != 0) {

        double x_over_z = (point_2d.x - 320.) / 585.;
        double y_over_z = (point_2d.y - 240.) / 585.;

        double z = d / sqrt(1. + x_over_z*x_over_z + y_over_z*y_over_z);
        double x = x_over_z * z;
        double y = y_over_z * z;

        point_3d = cv::Point3d (x, y, z) / 1000.;

        return true;

    } else {
        return false;
    }
}

void sevenScenes::getAbsolutePose(const string& image, Eigen::Matrix3d &R_iw, Eigen::Vector3d & T_iw) {
    Eigen::Matrix3d R_wi = sevenScenes::getR(image);
    Eigen::Vector3d T_wi = sevenScenes::getT(image);
    R_iw = R_wi.transpose();
    T_iw = -R_iw * T_wi;
}

void sevenScenes::getAbsolutePose_BA(const string& image, Eigen::Matrix3d & R_iw, Eigen::Vector3d & T_iw) {
    R_iw = sevenScenes::getR_BA(image);
    T_iw = sevenScenes::getT_BA(image);
}

map<pair<string, string>, pair<Eigen::Matrix3d, Eigen::Vector3d>> sevenScenes::getAnchorPoses(const string & dir) {

    map<pair<string, string>, pair<Eigen::Matrix3d, Eigen::Vector3d>> map;
    string file = dir + "anchor_poses.txt";
    ifstream rel_poses(file);

    if (rel_poses.is_open()) {
        string line;
        while(getline(rel_poses, line)) {
            stringstream ss (line);
            int count = 0;
            string item;
            string query;
            string anchor;
            Eigen::Matrix3d R_qi;
            Eigen::Vector3d T_qi;
            while (ss >> item) {
                if (count == 0) query = item;
                if (count == 1) anchor = item;
                if (count == 2) R_qi(0, 0) = stod(item);
                if (count == 3) R_qi(0, 1) = stod(item);
                if (count == 4) R_qi(0, 2) = stod(item);
                if (count == 5) R_qi(1, 0) = stod(item);
                if (count == 6) R_qi(1, 1) = stod(item);
                if (count == 7) R_qi(1, 2) = stod(item);
                if (count == 8) R_qi(2, 0) = stod(item);
                if (count == 9) R_qi(2, 1) = stod(item);
                if (count == 10) R_qi(2, 2) = stod(item);
                if (count == 11) T_qi(0) = stod(item);
                if (count == 12) T_qi(1) = stod(item);
                if (count == 13) T_qi(2) = stod(item);
                count++;
            }
            if (!query.empty()) {
                pair<string, string> key = make_pair(dir + query, dir + anchor);
                pair<Eigen::Matrix3d, Eigen::Vector3d> value = make_pair(R_qi, T_qi);
                map.emplace(key, value);
            }
        }
        rel_poses.close();
    }

    return map;
}