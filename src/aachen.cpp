//
// Created by Cameron Fiore on 7/14/22.
//

#include "../include/aachen.h"

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

void aachen::createPoseFiles() {
    string folder = "/Users/cameronfiore/C++/image_localization_project/data/images_upright/db/";
    ifstream poses ("/Users/cameronfiore/C++/image_localization_project/data/images_upright/poses.txt");
    if (poses.is_open()) {
        string line;
        while (getline(poses, line)) {
            int count = 0;
            string item;
            stringstream ss (line);
            string file;
            double w = 0, x = 0, y = 0, z = 0, cx = 0, cy = 0, cz = 0;
            while(ss >> item) {
                if (count == 0) file = item;
                if (count == 2) w = stod(item);
                if (count == 3) x = stod(item);
                if (count == 4) y = stod(item);
                if (count == 5) z = stod(item);
                if (count == 6) cx = stod(item);
                if (count == 7) cy = stod(item);
                if (count == 8) {
                    cz = stod(item);
                    break;
                }
                count++;
            }

            Eigen::Quaterniond q (w, x, y, z);
            Eigen::Matrix3d R = q.toRotationMatrix();
            Eigen::Vector3d c {cx, cy, cz};

            ofstream poseFile;
            poseFile.open(folder + file + ".pose.txt");
            poseFile << setprecision(12) << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << endl;
            poseFile << setprecision(12) << R(1, 0) << " " << R(1, 1) << " " << R(1, 2) << endl;
            poseFile << setprecision(12) << R(2, 0) << " " << R(2, 1) << " " << R(2, 2) << endl;
            poseFile << endl;
            poseFile << setprecision(12) << c(0) << " " << c(1) << " " << c(2) << endl;
            poseFile.close();
        }
    }
}

void aachen::createCalibFiles() {
    string folder = "/Users/cameronfiore/C++/image_localization_project/data/images_upright/db/";
    ifstream calibs ("/Users/cameronfiore/C++/image_localization_project/data/images_upright/db_intrinsics.txt");
    if (calibs.is_open()) {
        string line;
        while (getline(calibs, line)) {
            int count = 0;
            string item;
            stringstream ss (line);
            string file, f, cx, cy, r;
            while(ss >> item) {
                if (count == 0) file = item;
                if (count == 3) f = item;
                if (count == 4) cx = item;
                if (count == 5) {
                    cy = item;
                    break;
                }
                count++;
            }

            ofstream poseFile;
            poseFile.open(folder + file + ".calib.txt");
            poseFile << setprecision(12) << f << " " << cx << " " << cy << endl;
            poseFile.close();
        }
    }
}

void aachen::createQueryVector(vector<string> & queries, vector<tuple<double, double, double>> & calibrations) {
    string folder = "/Users/cameronfiore/C++/image_localization_project/data/images_upright/";
    string day = "/Users/cameronfiore/C++/image_localization_project/data/images_upright/day_intrinsics.txt";
    string night = "/Users/cameronfiore/C++/image_localization_project/data/images_upright/night_intrinsics.txt";

    ifstream d (day);
    if (d.is_open()) {
        string line;
        while (getline(d, line)) {
            int count = 0;
            string item;
            stringstream ss (line);
            string file;
            double f, cx, cy;
            while(ss >> item) {
                if (count == 0) file = item;
                if (count == 3) f = stod(item);
                if (count == 4) cx = stod(item);
                if (count == 5) {
                    cy = stod(item);
                    break;
                }
                count++;
            }
            queries.push_back(folder+file);
            calibrations.emplace_back(f, cx, cy);
        }
        d.close();
    }

    ifstream n (night);
    if (n.is_open()) {
        string line;
        while (getline(n, line)) {
            int count = 0;
            string item;
            stringstream ss (line);
            string file;
            double f, cx, cy;
            while(ss >> item) {
                if (count == 0) file = item;
                if (count == 3) f = stod(item);
                if (count == 4) cx = stod(item);
                if (count == 5) {
                    cy = stod(item);
                    break;
                }
                count++;
            }
            queries.push_back(folder+file);
            calibrations.emplace_back(f, cx, cy);
        }
        n.close();
    }
}

bool aachen::getR(const string & image, Eigen::Matrix3d & R) {
    string im = image.substr(0, image.find(".jpg"));
    string poseFile = im + ".pose.txt";
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
        return true;
    } else {
        return false;
    }
}

bool aachen::getC(const string & image, Eigen::Vector3d & c) {
    string im = image.substr(0, image.find(".jpg"));
    string poseFile = im + ".pose.txt";
    ifstream im_pose (poseFile);
    if (im_pose.is_open())
    {
        string line;
        int row = 0;
        while (getline(im_pose, line))
        {
            if (row == 4)
            {
                stringstream ss(line);
                string num;
                int col = 0;
                while (ss >> num) {
                    c(col) = stod(num);
                    col++;
                }
            } else {
                row++;
            }
        }
        im_pose.close();
        return true;
    } else {
        return false;
    }
}

bool aachen::getK(const string & image, double & f, double & cx, double & cy) {
    string im = image.substr(0, image.find(".jpg"));
    string poseFile = im + ".calib.txt";
    ifstream im_calib (poseFile);
    if (im_calib.is_open()) {
        string line;
        getline(im_calib, line);

        stringstream ss(line);
        string num;
        int count = 0;
        while (ss >> num) {
            if (count == 0) f = stod(num);
            if (count == 1) cx = stod(num);
            if (count == 2) {
                cy = stod(num);
                break;
            }
            count++;
        }
        im_calib.close();
        return true;
    } else {
        return false;
    }
}

bool aachen::getAbsolutePose(const string& image, Eigen::Matrix3d & R_iw, Eigen::Vector3d & T_iw) {
    Eigen::Vector3d c;
    if (!aachen::getR(image, R_iw)) return false;
    if (!aachen::getC(image, c)) return false;
    T_iw = -R_iw * c;
    return true;
}

vector<string> aachen::retrieveSimilar(const string & query_image, int max_num) {
    vector<string> similar;
    ifstream file (query_image + ".1000nn.txt");
    if (file.is_open()) {
        string line;
        while (similar.size() < max_num && getline(file, line)) {
            stringstream ss (line);
            string fn;
            ss >> fn;
            Eigen::Vector3d c;
            if(!aachen::getC(fn, c)) {
                continue;
            }
            similar.push_back(fn);
        }
        file.close();
    }
    return similar;
}

bool aachen::getRelativePose(vector<cv::Point2d> & pts_q,
                     double q_f, double q_cx, double q_cy,
                     vector<cv::Point2d> & pts_db,
                     double db_f, double db_cx, double db_cy,
                     double match_thresh,
                     Eigen::Matrix3d & R_qk, Eigen::Vector3d & T_qk) {
    try {
        vector<cv::Point2d> undist_pts_q, undist_pts_db;
        cv::Mat q_K = (cv::Mat_<double>(3, 3) <<
                q_f, 0., q_cx,
                0., q_f, q_cy,
                0., 0., 1.);
        cv::Mat db_K = (cv::Mat_<double>(3, 3) <<
                db_f, 0., db_cx,
                0., db_f, db_cy,
                0., 0., 1.);
        vector<double> dc;
        cv::undistortPoints(pts_q, undist_pts_q, q_K, dc);
        cv::undistortPoints(pts_db, undist_pts_db, db_K, dc);

        double focal = 1.;
        cv::Point2d pp (0., 0.);

        cv::Mat mask;
        cv::Mat E_qk_sols = findEssentialMat(undist_pts_db, undist_pts_q, focal, pp, cv::RANSAC, 0.9999999999, match_thresh, mask);
        cv::Mat E_qk = E_qk_sols(cv::Range(0, 3), cv::Range(0, 3));

        vector<cv::Point2d> inlier_db_points, inlier_q_points, inlier_db_points_, inlier_q_points_;
        for (int i = 0; i < mask.rows; i++) {
            if (mask.at<unsigned char>(i)) {
                inlier_db_points.push_back(pts_db[i]);
                inlier_q_points.push_back(pts_q[i]);
                inlier_db_points_.push_back(undist_pts_db[i]);
                inlier_q_points_.push_back(undist_pts_q[i]);
            }
        }

        mask.release();
        cv::Mat R, T;
        cv::recoverPose(E_qk, inlier_db_points_, inlier_q_points_, R, T, focal, pp, mask);

        vector<cv::Point2d> recover_db_points, recover_q_points;
        for (int i = 0; i < mask.rows; i++) {
            if (mask.at<unsigned char>(i)) {
                recover_db_points.push_back(inlier_db_points_[i]);
                recover_q_points.push_back(inlier_q_points_[i]);
            }
        }

        cv2eigen(R, R_qk);
        cv2eigen(T, T_qk);

        pts_db = recover_db_points;
        pts_q = recover_q_points;
        return true;
    } catch (...) {
        return false;
    }
}