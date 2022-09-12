#include "include/aachen.h"
#include "include/sevenScenes.h"
#include "include/synthetic.h"
#include "include/functions.h"
#include "include/poseEstimation.h"
#include <iostream>
#include <fstream>
#include <Eigen/SVD>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <opencv2/xfeatures2d.hpp>
#include "include/OptimalRotationSolver.h"
#include "include/CambridgeLandmarks.h"


#define PI 3.1415926536

using namespace std;
using namespace chrono;

int main() {

    string scene = "chess";
    string data_dir = "/Users/cameronfiore/C++/image_localization_project/data/";
    string scene_dir = data_dir + scene + "/";

    auto rel_poses = functions::getRelativePoses(scene_dir, scene_dir);


    for (const auto & query : rel_poses) {

        string query_name = query.first;

        cout << query_name << endl;

        Eigen::Matrix3d R_q; Eigen::Vector3d T_q;
        sevenScenes::getAbsolutePose(query_name, R_q, T_q);

        vector<string> anchor_names;
        vector<Eigen::Matrix3d> R_ks, R_qks;
        vector<Eigen::Vector3d> T_ks, T_qks;

        for (const auto & anchor : query.second) {

            string anchor_name = anchor.first;

            Eigen::Matrix3d R_k, R_qk; Eigen::Vector3d T_k, T_qk;
            sevenScenes::getAbsolutePose(anchor_name, R_k, T_k);
            R_qk = anchor.second.first;
            T_qk = anchor.second.second;

            Eigen::Matrix3d R_qk_real = R_q * R_k.transpose();
            Eigen::Vector3d T_qk_real = T_q - R_qk_real * T_k;
            T_qk.normalize();

            double r_error = functions::rotationDifference(R_qk, R_qk_real);
            double t_error = functions::getAngleBetween(T_qk, T_qk_real);

            cout << "      " << anchor_name << "  R Error: " << r_error << "  T Error: " << t_error << endl;

            R_ks.push_back(R_k);
            T_ks.push_back(T_k);
            R_qks.push_back(R_qk);
            T_qks.push_back(T_qk);
        }

        int stop = 0;

    }

    return 0;
}