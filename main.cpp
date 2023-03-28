
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

mutex mtx;

void findInliers (double thresh,
                  int i,
                  int j,
                  const vector<Eigen::Matrix3d> * R_ks,
                  const vector<Eigen::Vector3d> * T_ks,
                  const vector<Eigen::Matrix3d> * R_qks,
                  const vector<Eigen::Vector3d> * T_qks,
                  const vector<vector<cv::Point2d>> * match_num,
                  vector<tuple<int, int, double, vector<int>>> * results) {

    int K = int((*R_ks).size());
    vector<Eigen::Matrix3d> set_R_ks{(*R_ks)[i], (*R_ks)[j]};
    vector<Eigen::Vector3d> set_T_ks{(*T_ks)[i], (*T_ks)[j]};
    vector<Eigen::Matrix3d> set_R_qks{(*R_qks)[i], (*R_qks)[j]};
    vector<Eigen::Vector3d> set_T_qks{(*T_qks)[i], (*T_qks)[j]};
    vector<int> indices{i, j};
    int score = int((*match_num)[i].size()) + int((*match_num)[j].size());
    Eigen::Matrix3d R_q = pose::R_q_average(vector<Eigen::Matrix3d>{(*R_qks)[i] * (*R_ks)[i], (*R_qks)[j] * (*R_ks)[j]});
    Eigen::Vector3d c_q = pose::c_q_closed_form(set_R_ks, set_T_ks, set_R_qks, set_T_qks);
    for (int k = 0; k < K; k++) {
        if (k != i && k != j) {
            double rot_diff = functions::rotationDifference(R_q*(*R_ks)[k].transpose(), (*R_qks)[k]);
            double center_diff = functions::getAngleBetween(-(*R_qks)[k]*((*R_ks)[k]*c_q+(*T_ks)[k]), (*T_qks)[k]);
            if (rot_diff <= thresh && center_diff <= thresh) {
                indices.push_back(k);
                score += int((*match_num)[k].size());
            }
        }
    }
    auto t = make_tuple(i, j, score, indices);
    mtx.lock();
    results->push_back(t);
    mtx.unlock();
}



int main() {

    vector<string> scenes = {"chess/", "fire/", "heads/", "office/", "pumpkin/", "redkitchen/", "stairs/"};
    string dataset = "seven_scenes/";
    string error_file = "good_recovered";
    int cutoff = -1;
    double thresh = 10;
    string relpose_file = "relpose_SP_SFM";

//    vector<string> scenes = {"GreatCourt/", "KingsCollege/", "OldHospital/", "ShopFacade/", "StMarysChurch/"};
//     string dataset = "cambridge/";
//    string error_file = "good_recovered";
//     int cutoff = -1;
//     double thresh = 5;
//     string relpose_file = "relpose_SP";

//     vector<string> scenes = {"query/"};
//     string dataset = "aachen/";
//    string error_file = "good_recovered";
//     int cutoff = 3;
//     double thresh = 5;
//     string relpose_file = "relpose_SP";

//    vector<string> scenes = {"query/"};
//    string dataset = "robotcar/";
//    string error_file = "good_recovered";
//    int cutoff = 2;
//    double thresh = 5;
//    string relpose_file = "relpose_SP_v2";

    string ccv_dir = "/users/cfiore/data/cfiore/image_localization_project/data/" + dataset;
    string home_dir = "/Users/cameronfiore/C++/image_localization_project/data/" + dataset;
    string dir = ccv_dir;

    ofstream error;
    error.open(dir + error_file + ".txt");
    for (const auto &scene: scenes) {

        int start = 0;
        vector<string> queries = functions::getQueries(dir + "q.txt", scene);
        for (int q = start; q < queries.size(); q++) {

            cout << q + 1 << "/" << queries.size() << endl;
            string query = queries[q];

            auto info = functions::parseRelposeFile(dir, query, relpose_file);
            auto R_q = get<1>(info);
            auto T_q = get<2>(info);
            auto K_q = get<3>(info);
            Eigen::Vector3d c_q = -R_q.transpose() * T_q;

            auto anchors = get<4>(info);
            auto R_is = get<5>(info);
            auto T_is = get<6>(info);
            auto R_qis = get<7>(info);
            auto T_qis = get<8>(info);
            auto K_is = get<9>(info);
            auto inliers_q = get<10>(info);
            auto inliers_i = get<11>(info);
            int K = int(anchors.size());

            int num_good = 0;
            for(const auto & points : inliers_q) {
                if (points.size() >= 30) num_good++;
            }
            error << num_good << endl;
        }
    }
    error.close();
    return 0;
}