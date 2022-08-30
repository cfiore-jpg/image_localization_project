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


#define PI 3.1415926536

#define FOLDER "/Users/cameronfiore/C++/image_localization_project/data/"
//#define SCENE 2
#define IMAGE_LIST "/Users/cameronfiore/C++/image_localization_project/data/images_1000.txt"
#define GENERATE_DATABASE false

using namespace std;
using namespace chrono;

int main() {

    string scene = "chess";
    string ext = ".color.png";

    string data_dir = "/Users/cameronfiore/C++/image_localization_project/data/";
    string base_dir = data_dir + scene + "/";

    vector<string> queries;
    auto info = sevenScenes::createInfoVector();
    sevenScenes::createQueryVector(queries, info, 0);

    ofstream superglue_r_error;
    superglue_r_error.open(base_dir+"superglue_r_error.txt");

    ofstream superglue_t_error;
    superglue_t_error.open(base_dir+"superglue_t_error.txt");

    for (int q = 0; q < queries.size(); q++) {

        cout << q << "/" << int(queries.size()) << ":" << endl;

        string query = queries[q];

        Eigen::Matrix3d R_q;
        Eigen::Vector3d T_q;
        sevenScenes::getAbsolutePose(query, R_q, T_q);
        Eigen::Vector3d c_q = -R_q.transpose() * T_q;

        vector<string> retrieved;
        vector<double> distances;
        functions::retrieveSimilar(query, "7-Scenes", ".color.png", 100, 3.0, retrieved, distances);

//        auto map = sevenScenes::getAnchorPoses(base_dir);
//
        vector<Eigen::Matrix3d> R_ks, R_qks;
        vector<Eigen::Vector3d> T_ks, T_qks;
        for (const auto & im : retrieved) {

            Eigen::Matrix3d R_k;
            Eigen::Vector3d T_k;
            sevenScenes::getAbsolutePose(im, R_k, T_k);


            Eigen::Matrix3d R_qk_real = R_q * R_k.transpose();
            Eigen::Vector3d T_qk_real = T_q - R_qk_real * T_k;


//            Eigen::Matrix3d R_qk;
//            Eigen::Vector3d T_qk;
//            try {
//                pair<string, string> key = make_pair(query+ext, im+ext);
//                pair<Eigen::Matrix3d, Eigen::Vector3d> value = map.at(key);
//                R_qk = value.first;
//                T_qk = value.second;
//            } catch(...) {
//                continue;
//            }
//
//            double r_dist = functions::rotationDifference(R_qk_real, R_qk);
//            double t_dist = functions::getAngleBetween(T_qk_real, T_qk);

            R_ks.push_back(R_k);
            T_ks.push_back(T_k);
            R_qks.push_back(R_qk_real);
            T_qks.push_back(T_qk_real);
        }

        auto result = pose::hypothesizeRANSAC(3., R_ks, T_ks, R_qks, T_qks);
        Eigen::Vector3d c_calc = get<0>(result);
        Eigen::Matrix3d R_calc = get<1>(result);
        int inliers = int(get<2>(result).size());

        double r_diff = functions::rotationDifference(R_calc, R_q);
        double c_diff = functions::getDistBetween(c_calc, c_q);

        superglue_r_error << r_diff << endl;
        superglue_t_error << c_diff << endl;
    }

    superglue_r_error.close();
    superglue_t_error.close();

    return 0;
}