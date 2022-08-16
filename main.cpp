#include "include/aachen.h"
#include "include/sevenScenes.h"
#include "include/synthetic.h"
#include "include/functions.h"
#include "include/poseEstimation.h"
#include "include/calibrate.h"
#include <iostream>
#include <fstream>
#include <Eigen/SVD>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <opencv2/xfeatures2d.hpp>
#include "include/OptimalRotationSolver.h"
#include "include/CambridgeLandmarks.h"
#include <thread>
#include <mutex>
#include <tuple>
#include <map>


#define PI 3.1415926536

#define FOLDER "/Users/cameronfiore/C++/image_localization_project/data/"
//#define SCENE 2
#define IMAGE_LIST "/Users/cameronfiore/C++/image_localization_project/data/images_1000.txt"
#define EXT ".color.png"
#define GENERATE_DATABASE false

using namespace std;
using namespace chrono;

double K[4] = {585., 585., 320., 240.};

int main() {

//// Testing Whole Dataset

    vector<string> listQuery;
    auto info = sevenScenes::createInfoVector();
    sevenScenes::createQueryVector(listQuery, info, 6);
    string scene = "stairs";

    ofstream R_calc_error_cpp;
    R_calc_error_cpp.open("/Users/cameronfiore/C++/image_localization_project/data/"+scene+"/R_calc_error_cpp2.txt");

    ofstream T_calc_error_cpp;
    T_calc_error_cpp.open("/Users/cameronfiore/C++/image_localization_project/data/"+scene+"/T_calc_error_cpp2.txt");

    for (int q = 0; q < listQuery.size(); q++) {

        cout << q << "/" << listQuery.size()-1 << endl;

        string query = listQuery[q];

        Eigen::Matrix3d R_q;
        Eigen::Vector3d T_q;
        sevenScenes::getAbsolutePose(query, R_q, T_q);
        Eigen::Vector3d c_q = sevenScenes::getT(query);

//        auto desc_kp_q = functions::getDescriptors(query, ".color.png", "SIFT");
//        cv::Mat desc_q = desc_kp_q.first;
//        vector<cv::KeyPoint> kp_q = desc_kp_q.second;

        vector<string> retrieved;
        vector<double> distances;
        functions::retrieveSimilar(query, "7-Scenes", ".color.png", 150, 3., retrieved, distances);

        string im = retrieved[0];

        Eigen::Matrix3d R_k, R_qk_calc, R_kq_calc;
        Eigen::Vector3d T_k, T_qk_calc, T_kq_calc;
        sevenScenes::getAbsolutePose(im, R_k, T_k);

//        auto desc_kp_i = functions::getDescriptors(im, ".color.png", "SIFT");
//        cv::Mat desc_i = desc_kp_i.first;
//        vector<cv::KeyPoint> kp_i = desc_kp_i.second;

        vector<cv::Point2d> pts_i, pts_q, pts_q_qk, pts_i_qk, pts_q_kq, pts_i_kq;

        string fn = "/Users/cameronfiore/C++/image_localization_project/data/"+scene+"/py_matches/"+ to_string(q)+".txt";
        ifstream py_matches (fn);
        if (py_matches.is_open()) {
            string line;
            int count = 0;
            while(getline(py_matches, line)) {
                double qx, qy, imx, imy;
                stringstream ss (line);
                string item;

                if (count == 0) {
                    ss >> item;
                    assert(item == query);
                } else if(count == 1) {
                    ss >> item;
                    assert(line == im);
                } else {
                    for (int i = 0; i < 4; i++) {
                        ss >> item;
                        if (i == 0) qx = stod(item);
                        if (i == 1) qy = stod(item);
                        if (i == 2) imx = stod(item);
                        if (i == 3) imy = stod(item);
                    }
                    pts_q.emplace_back(qx, qy);
                    pts_i.emplace_back(imx, imy);
                }
                count++;
            }
        }


//        functions::findMatches(0.8, desc_i, kp_i, desc_q, kp_q, pts_i, pts_q);
//
//        ofstream matches;
//        string fn = "/Users/cameronfiore/C++/image_localization_project/data/"+scene+"/matches/"+ to_string(q)+".txt";
//        matches.open(fn);
//        matches << query << endl;
//        matches << im << endl;
//        for (int i = 0; i < pts_q.size(); i++) {
//            matches << setprecision(10) << pts_q[i].x << " " << pts_q[i].y << " " << pts_i[i].x << " " << pts_i[i].y << endl;
//        }
//        matches.close();

        pts_q_qk = pts_q;
        pts_i_qk = pts_i;
        functions::getRelativePose(pts_i_qk, pts_q_qk, K, .5, R_qk_calc, T_qk_calc);

        Eigen::Matrix3d R_qk_real = R_q * R_k.transpose();
        Eigen::Vector3d T_qk_real = T_q - R_qk_real * T_k;
        T_qk_real.normalize();

        double r_dist = functions::rotationDifference(R_qk_real, R_qk_calc);
        double t_dist = functions::getAngleBetween(T_qk_real, T_qk_calc);

        R_calc_error_cpp << r_dist << endl;
        T_calc_error_cpp << t_dist << endl;
    }
    R_calc_error_cpp.close();
    T_calc_error_cpp.close();
    return 0;
}