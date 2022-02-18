//
// Created by Cameron Fiore on 2/15/22.
//

#include "../include/synthetic.h"

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
#include <random>
#include "../include/functions.h"

#define PI 3.14159265

vector<string> synthetic::getAll() {

    vector<string> to_return;
    for(int i = 0; i < 100; i++) {
        if (i < 10) {
            string file_name = "/Users/cameronfiore/C++/image_localization_project/synthetic_data/Chicago_data-master-97f54aaa7a230f0ee2305a71659da7311c771fcd/synthcurves_dataset/spherical-ascii-100_views-perturb-radius_sigma10-normal_sigma0_01rad-minsep_15deg-no_two_cams_colinear_with_object/frame_000" + to_string(i);
            to_return.push_back(file_name);
        } else {
            string file_name = "/Users/cameronfiore/C++/image_localization_project/synthetic_data/Chicago_data-master-97f54aaa7a230f0ee2305a71659da7311c771fcd/synthcurves_dataset/spherical-ascii-100_views-perturb-radius_sigma10-normal_sigma0_01rad-minsep_15deg-no_two_cams_colinear_with_object/frame_00" + to_string(i);
            to_return.push_back(file_name);
        }
    }
    return to_return;
}

vector<Eigen::Vector3d> synthetic::get3DPoints() {

    vector<Eigen::Vector3d> to_return;

    string point_file = "/Users/cameronfiore/C++/image_localization_project/synthetic_data/Chicago_data-master-97f54aaa7a230f0ee2305a71659da7311c771fcd/synthcurves_dataset/spherical-ascii-100_views-perturb-radius_sigma10-normal_sigma0_01rad-minsep_15deg-no_two_cams_colinear_with_object/crv-3D-pts.txt";
    ifstream points (point_file);
    if (points.is_open()) {
        string line;
        while (getline(points, line)) {
            stringstream ss(line);
            string num;
            int col = 0;
            double x, y, z;
            while (ss >> num) {
                if (col == 0) {
                    x = stod(num);
                } else if (col == 1) {
                    y = stod(num);
                } else {
                    z = stod(num);
                }
                col++;
            }
            to_return.emplace_back(x, y, z);
        }
        points.close();
    }
    return to_return;
}


vector<string> synthetic::omitQuery(int i, const vector<string> & all_ims) {

    vector<string> to_return = all_ims;
    to_return.erase(to_return.begin() + i);
    return to_return;
}

Eigen::Matrix3d synthetic::getR(const string& image) {

    Eigen::Matrix3d R;

    string poseFile = image + ".extrinsic";
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
                    R(row, col) = stod(num);
                    ++col;
                }
                ++row;
            }
        }
        im_pose.close();
        return R;
    } else
    {
        cout << "Pose file does not exist for: " << image << endl;
        Eigen::Matrix3d l;
        return l;
    }

}

Eigen::Vector3d synthetic::getC(const string& image)
{
    Eigen::Vector3d c;

    string poseFile = image + ".extrinsic";
    ifstream im_pose (poseFile);
    if (im_pose.is_open()) {
        string line;
        int row = 0;
        while (getline(im_pose, line)) {
            if (row == 4) {
                stringstream ss(line);
                string num;
                int col = 0;
                while (ss >> num) {
                    c(col) = stod(num);
                    ++col;
                }
            }
            ++row;
        }
        im_pose.close();
        return c;
    } else
    {
        cout << "Pose file does not exist for: " << image << endl;
        Eigen::Vector3d l;
        return l;
    }
}

void synthetic::getAbsolutePose(const string& image, Eigen::Matrix3d &R_i, Eigen::Vector3d &t_i) {
    R_i = synthetic::getR(image);
    t_i = -R_i * synthetic::getC(image);
}

void synthetic::findSyntheticMatches(const string & db_im, const string & query, vector<cv::Point2d> & pts_db, vector<cv::Point2d> & pts_q) {

    string db_match_file = db_im + "-pts-2D.txt";
    string q_match_file = query + "-pts-2D.txt";

    ifstream db_pose (db_match_file);
    ifstream q_pose (q_match_file);
    if (db_pose.is_open() && q_pose.is_open()) {
        string db_line, q_line;
        while (getline(db_pose, db_line) && getline(q_pose, q_line)) {
            stringstream db_ss(db_line), q_ss(q_line);
            double db_x, db_y, q_x, q_y;
            int read_count = 0;
            while(read_count < 2) {
                if (read_count == 0) {
                    db_ss >> db_x;
                    q_ss >> q_x;
                } else {
                    db_ss >> db_y;
                    q_ss >> q_y;
                }
                read_count++;
            }
            pts_db.emplace_back(db_x, db_y);
            pts_q.emplace_back(q_x, q_y);
        }
        db_pose.close();
        q_pose.close();
    }
}

void synthetic::addGaussianNoise(vector<cv::Point2d> & pts_db, vector<cv::Point2d> & pts_q, double std_dev) {

    assert(pts_db.size() == pts_q.size());
    random_device generator;
    normal_distribution<double> gaussian(0., std_dev);
    uniform_real_distribution<double> uniform(0., 2.*PI);
    for (int i = 0; i < pts_db.size(); i++) {
        double mag_db = abs(gaussian(generator));
        double mag_q = abs(gaussian(generator));
        double theta_db = uniform(generator);
        double theta_q = uniform(generator);

        double x_db = mag_db * cos(theta_db);
        double y_db = mag_db * sin(theta_db);
        double x_q = mag_q * cos(theta_q);
        double y_q = mag_q * sin(theta_q);

        pts_db[i].x = pts_db[i].x + x_db;
        pts_db[i].y = pts_db[i].y + y_db;
        pts_q[i].x = pts_q[i].x + x_q;
        pts_q[i].y = pts_q[i].y + y_q;
    }
}

void synthetic::addOcclusion(vector<cv::Point2d> & pts_db, vector<cv::Point2d> & pts_q, int n) {
    random_device generator;
    while (int (pts_db.size()) > n) {
        uniform_int_distribution<int> distribution(0,int(pts_db.size())-1);
        int i = distribution(generator);
        pts_db.erase(pts_db.begin() + i);
        pts_q.erase(pts_q.begin() + i);
    }
}

void synthetic::addMismatches(vector<cv::Point2d> & pts_db, vector<cv::Point2d> & pts_q, double percentage) {
    random_device generator;
    int n = int( double (pts_db.size()) * percentage);
    uniform_real_distribution<double> x_random(0., 400.);
    uniform_real_distribution<double> y_random(0., 500.);
    uniform_int_distribution<int> distribution(0,int(pts_db.size())-1);
    vector<int> already_mofified;
    while (n > 0) {
        int i = distribution(generator);
        if (std::count(already_mofified.begin(), already_mofified.end(), i)) {
            continue;
        }
        already_mofified.push_back(i);
        pts_db[i].x = x_random(generator);
        pts_db[i].y = y_random(generator);
        n--;
    }
}




