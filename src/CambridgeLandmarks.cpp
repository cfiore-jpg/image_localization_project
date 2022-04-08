//
// Created by Cameron Fiore on 3/14/22.
//

#include "../include/CambridgeLandmarks.h"

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

using namespace std;

void cambridge::createPoseFiles() {

    string folder = "/Users/cameronfiore/C++/image_localization_project/KingsCollege/";

    string test_file = "/Users/cameronfiore/C++/image_localization_project/KingsCollege/dataset_test.txt";
    ifstream tf(test_file);
    if (tf.is_open()) {
        string line;
        int pass = 0;
        while (getline(tf, line)) {
            if (pass < 3) {
                pass++;
                continue;
            }
            stringstream ss(line);
            string item;
            int count = 0;
            string fn;
            double nums[7];
            while (ss >> item) {
                if (count == 0) {
                    string f = item.substr(0, item.find(".png"));
                    fn = folder + f;
                } else {
                    nums[count - 1] = stod(item);
                }
                count++;
            }

            Eigen::Vector3d c{nums[0], nums[1], nums[2]};
            Eigen::Quaternion<double> q(nums[3], nums[4], nums[5], nums[6]);
            q.normalize();
            Eigen::Matrix3d R = q.toRotationMatrix();

            Eigen::Vector3d T = -R * c;

            ofstream pf;
            pf.open(fn + ".pose.txt");
            pf << R << endl;
            pf << T.transpose() << endl;
            pf.close();
        }
        tf.close();
    }

    string train_file = "/Users/cameronfiore/C++/image_localization_project/KingsCollege/dataset_train.txt";
    ifstream trf(train_file);
    if (trf.is_open()) {
        string line;
        int pass = 0;
        while (getline(trf, line)) {
            if (pass < 3) {
                pass++;
                continue;
            }
            stringstream ss(line);
            string item;
            int count = 0;
            string fn;
            double nums[7];
            while (ss >> item) {
                if (count == 0) {
                    string f = item.substr(0, item.find(".png"));
                    fn = folder + f;
                } else {
                    nums[count - 1] = stod(item);
                }
                count++;
            }

            Eigen::Vector3d c{nums[0], nums[1], nums[2]};
            Eigen::Quaternion<double> q(nums[3], nums[4], nums[5], nums[6]);
            q.normalize();
            Eigen::Matrix3d R = q.toRotationMatrix();

            Eigen::Vector3d T = -R * c;

            ofstream pf;
            pf.open(fn + ".pose.txt");
            pf << R << endl;
            pf << T.transpose() << endl;
            pf.close();
        }
        trf.close();
    }
}

vector<string> cambridge::getTestImages() {

    vector<string> ims;
    string folder = "/Users/cameronfiore/C++/image_localization_project/KingsCollege/";
    string test_file = "/Users/cameronfiore/C++/image_localization_project/KingsCollege/dataset_test.txt";

    ifstream tf(test_file);
    if (tf.is_open()) {
        string line;
        int pass = 0;
        while (getline(tf, line)) {
            if (pass < 3) {
                pass++;
                continue;
            }
            stringstream ss(line);
            string item;
            string fn;
            if (ss >> item) {
                string f = item.substr(0, item.find(".png"));
                fn = folder + f;
                ims.push_back(fn);
            }
        }
        tf.close();
    }
    return ims;
}

Eigen::Matrix3d cambridge::getR(const string& image) {

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

Eigen::Vector3d cambridge::getT(const string& image) {
    Eigen::Vector3d t;

    string poseFile = image + ".pose.txt";
    ifstream im_pose (poseFile);
    if (im_pose.is_open())
    {
        string line;
        int num = 0;
        while (getline(im_pose, line)) {
            if (num < 3) {
                num++;
                continue;
            }
            stringstream ss(line);
            string item;
            int row = 0;
            while (ss >> item) {
                t(row) = stod(item);
                row++;
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

void cambridge::getAbsolutePose(const string& image, Eigen::Matrix3d &R_i, Eigen::Vector3d &t_i) {

    R_i = cambridge::getR(image);
    t_i = cambridge::getT(image);
}





