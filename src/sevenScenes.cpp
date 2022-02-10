//
// Created by Cameron Fiore on 7/2/21.
//

#include "../include/sevenScenes.h"

#include <iostream>
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

vector<tuple<string, string, vector<string>, vector<string>>> sevenScenes::createInfoVector()
{
    vector<tuple<string, string, vector<string>, vector<string>>> info;

    tuple<string, string, vector<string>, vector<string>> chess;
    vector<string> train0 {"01", "02", "04", "06"};
    vector<string> test0 {"03", "05"};
    get<0>(chess) = "/Users/cameronfiore/C++/image_localization_project/data/chess/";
    get<1>(chess) = "/Users/cameronfiore/C++/image_localization_project/data/images_1000.txt";
    get<2>(chess) = train0;
    get<3>(chess) = test0;
    info.push_back(chess);

    tuple<string, string, vector<string>, vector<string>> fire;
    vector<string> train1 {"01", "02"};
    vector<string> test1 {"03", "04"};
    get<0>(fire) = "/Users/cameronfiore/C++/image_localization_project/data/fire/";
    get<1>(fire) = "/Users/cameronfiore/C++/image_localization_project/data/images_1000.txt";
    get<2>(fire) = train1;
    get<3>(fire) = test1;
    info.push_back(fire);

    tuple<string, string, vector<string>, vector<string>> heads;
    vector<string> train2 {"02"};
    vector<string> test2 {"01"};
    get<0>(heads) = "/Users/cameronfiore/C++/image_localization_project/data/heads/";
    get<1>(heads) = "/Users/cameronfiore/C++/image_localization_project/data/images_1000.txt";
    get<2>(heads) = train2;
    get<3>(heads) = test2;
    info.push_back(heads);

    tuple<string, string, vector<string>, vector<string>> office;
    vector<string> train3 {"01", "03", "04", "05", "08", "10"};
    vector<string> test3 {"02", "06", "07", "09"};
    get<0>(office) = "/Users/cameronfiore/C++/image_localization_project/data/office/";
    get<1>(office) = "/Users/cameronfiore/C++/image_localization_project/data/images_1000.txt";
    get<2>(office) = train3;
    get<3>(office) = test3;
    info.push_back(office);

    tuple<string, string, vector<string>, vector<string>> pumpkin;
    vector<string> train4 {"02", "03", "06", "08"};
    vector<string> test4 {"01", "07"};
    get<0>(pumpkin) = "/Users/cameronfiore/C++/image_localization_project/data/pumpkin/";
    get<1>(pumpkin) = "/Users/cameronfiore/C++/image_localization_project/data/images_1000.txt";
    get<2>(pumpkin) = train4;
    get<3>(pumpkin) = test4;
    info.push_back(pumpkin);

    tuple<string, string, vector<string>, vector<string>> redkitchen;
    vector<string> train5 {"01", "02", "05", "07", "08", "11", "13"};
    vector<string> test5 {"03", "04", "06", "12", "14"};
    get<0>(redkitchen) = "/Users/cameronfiore/C++/image_localization_project/data/redkitchen/";
    get<1>(redkitchen) = "/Users/cameronfiore/C++/image_localization_project/data/images_1000.txt";
    get<2>(redkitchen) = train5;
    get<3>(redkitchen) = test5;
    info.push_back(redkitchen);

    tuple<string, string, vector<string>, vector<string>> stairs;
    vector<string> train6 {"02", "03", "05", "06"};
    vector<string> test6 {"01", "04"};
    get<0>(stairs) = "/Users/cameronfiore/C++/image_localization_project/data/stairs/";
    get<1>(stairs) = "/Users/cameronfiore/C++/image_localization_project/data/images_500.txt";
    get<2>(stairs) = train6;
    get<3>(stairs) = test6;
    info.push_back(stairs);

    return info;
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
        Eigen::JacobiSVD<Eigen::Matrix3d> svd (R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d new_R = svd.matrixU() * svd.matrixV().transpose();
        return new_R;
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

void sevenScenes::getAbsolutePose(const string& image, Eigen::Matrix3d &R_wi, Eigen::Vector3d &t_wi) {
    Eigen::Matrix3d R_iw = sevenScenes::getR(image);
    Eigen::Vector3d t_iw = sevenScenes::getT(image);
    R_wi = R_iw.transpose();
    t_wi = -R_wi * t_iw;
}




