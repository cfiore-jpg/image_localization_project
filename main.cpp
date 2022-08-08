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
#define EXT ".color.png"
#define GENERATE_DATABASE false

using namespace std;
using namespace chrono;

int main() {

//// Testing Whole Dataset

    ofstream k_effect;
    k_effect.open("/Users/cameronfiore/C++/image_localization_project/data/chess/k_effect.csv");


    vector<string> listQuery;
    auto info = sevenScenes::createInfoVector();
    functions::createQueryVector(listQuery, info, 0);


    cout << "Running queries..." << endl;
    int startIdx = 0;
    vector<string> not_all_right;
    for (int q = startIdx; q < listQuery.size(); q++) {
        cout << q << "/" << listQuery.size() - 1 << ":";

        string query = listQuery[q];

        auto desc_kp_q = functions::getDescriptors(query, ".color.png", "SIFT");
        cv::Mat desc_q = desc_kp_q.first;
        vector<cv::KeyPoint> kp_q = desc_kp_q.second;

        Eigen::Matrix3d R_q;
        Eigen::Vector3d T_q;
        sevenScenes::getAbsolutePose(query, R_q, T_q);
        Eigen::Vector3d c_q = -R_q.transpose() * T_q;

        double K[4] = {585., 585., 320., 240.};

        int k_hat = 150; vector<string> retrieved; vector<double> distances;
        functions::retrieveSimilar(query, "7-Scenes", ".color.png", k_hat, 3., retrieved, distances);


        for (int s = 2; s <= 50; s++) {
            auto spaced = functions::optimizeSpacing(query, retrieved, distances, s, false, "7-Scenes");

            vector<Eigen::Matrix3d> R_ks, R_qks;
            vector<Eigen::Vector3d> T_ks, T_qks;
            for (const auto &im: spaced) {
                Eigen::Matrix3d R_k, R_qk_real, R_qk_calc, R_kq_calc;
                Eigen::Vector3d T_k, T_qk_real, T_qk_calc, T_kq_calc;
                sevenScenes::getAbsolutePose(im, R_k, T_k);

                auto desc_kp_i = functions::getDescriptors(im, ".color.png", "SIFT");
                cv::Mat desc_i = desc_kp_i.first;
                vector<cv::KeyPoint> kp_i = desc_kp_i.second;

                vector<cv::Point2d> pts_i, pts_q;
                functions::findMatches(0.8, desc_i, kp_i, desc_q, kp_q, pts_i, pts_q);

                vector<cv::Point2d> pts_q_qk = pts_q, pts_i_qk = pts_i;
                functions::getRelativePose(pts_i_qk, pts_q_qk, K, .5, R_qk_calc, T_qk_calc);

                vector<cv::Point2d> pts_q_kq = pts_q, pts_i_kq = pts_i;
                functions::getRelativePose(pts_q_kq, pts_i_kq, K, .5, R_kq_calc, T_kq_calc);

                Eigen::Matrix3d R_qk_from_kq = R_kq_calc.transpose();
                Eigen::Vector3d T_qk_from_kq = -R_kq_calc.transpose() * T_kq_calc;

                double r_consistency = functions::rotationDifference(R_qk_from_kq, R_qk_calc);
                double t_consistency = functions::getAngleBetween(T_qk_from_kq, T_qk_calc);
                if (r_consistency >= 0.0001 || t_consistency >= 0.0001) continue;

                R_ks.push_back(R_k);
                T_ks.push_back(T_k);
                R_qks.push_back(R_qk_calc);
                T_qks.push_back(T_qk_calc);
            }
            if (R_ks.size() < 2) {
                k_effect << s << "," << -1 << "," << -1 << "," << -1 << ",";
                continue;
            }

            auto results = pose::hypothesizeRANSAC(5., R_ks, T_ks, R_qks, T_qks);
            Eigen::Vector3d c_est = get<0>(results);
            Eigen::Matrix3d R_est = get<1>(results);

            int num_inliers = int(get<2>(results).size());
            double c_error = functions::getDistBetween(c_q, c_est);
            double r_error = functions::rotationDifference(R_q, R_est);

            k_effect << s << "," << num_inliers << "," << setprecision(5) << c_error << "," << r_error << ",";
            cout << s << " ";
        }
        k_effect << endl;
        cout << endl;
    }
    k_effect.close();
    return 0;
}