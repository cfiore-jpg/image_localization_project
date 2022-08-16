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


    vector<string> listQuery;
    auto info = sevenScenes::createInfoVector();
    sevenScenes::createQueryVector(listQuery, info, 6);

    cout << "Running queries..." << endl;
    int startIdx = 90;
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
        functions::retrieveSimilar(query, "7-Scenes", ".color.png", k_hat, 2.5, retrieved, distances);
        auto spaced = functions::optimizeSpacing(query, retrieved, 20, false, "7-Scenes");

//        functions::showTop150(query, retrieved, ".color.png");
//        functions::showSpacedInTop150(query, retrieved, spaced, ".color.png");
//        functions::showSpaced(query, spaced, ".color.png");
//
//        vector<Eigen::Matrix3d> R_ks, R_qks;
//        vector<Eigen::Vector3d> T_ks, T_qks;
//        vector<string> anchors;
//        vector<vector<pair<cv::Point2d, cv::Point2d>>> all_matches;
//        int num_good = 0;
//        double max_t = 0;
//        double min_t = 1000;
//        double max_r = 0;
//        double min_r = 1000;
//        double avg_t_dist = 0;
//        double avg_r_dist = 0;
//        int count = 0;
//        for (const auto &im: spaced) {
//            Eigen::Matrix3d R_k, R_qk_real, R_qk_calc, R_kq_calc;
//            Eigen::Vector3d T_k, T_qk_real, T_qk_calc, T_kq_calc;
//            sevenScenes::getAbsolutePose(im, R_k, T_k);
//
//            auto desc_kp_i = functions::getDescriptors(im, ".color.png", "SIFT");
//            cv::Mat desc_i = desc_kp_i.first;
//            vector<cv::KeyPoint> kp_i = desc_kp_i.second;
//
//            vector<cv::Point2d> pts_i, pts_q;
//            functions::findMatches(0.8, desc_i, kp_i, desc_q, kp_q, pts_i, pts_q);
//
//            cv::Mat src;
//            cv::Mat q_mat = cv::imread(query + ".color.png");
//            cv::Mat im_mat = cv::imread(im + ".color.png");
//            cv::hconcat(q_mat, im_mat, src);
//            random_device gen;
//            uniform_int_distribution<int> dis(0, 256);
//            for (int i = 0; i < pts_q.size(); i++) {
//                cv::Point2d pt_q = pts_q[i];
//                cv::Point2d pt_i(pts_i[i].x + q_mat.cols, pts_i[i].y);
//                cv::Scalar color = cv::Scalar(dis(gen), dis(gen), dis(gen));
//                circle(src, pt_q, 3, color, -1);
//                circle(src, pt_i, 3, color, -1);
//                cv::line(src, pt_q, pt_i, color, 2);
//            }
//            cv::imshow("ALL", src);
//            cv::waitKey(0);
//
//            vector<cv::Point2d> pts_q_qk = pts_q, pts_i_qk = pts_i;
//            functions::getRelativePose(pts_i_qk, pts_q_qk, K, .5, R_qk_calc, T_qk_calc);
//
//            vector<cv::Point2d> pts_q_kq = pts_q, pts_i_kq = pts_i;
//            functions::getRelativePose(pts_q_kq, pts_i_kq, K, .5, R_kq_calc, T_kq_calc);
//
//            Eigen::Matrix3d R_qk_from_kq = R_kq_calc.transpose();
//            Eigen::Vector3d T_qk_from_kq = -R_kq_calc.transpose() * T_kq_calc;
//
//            double r_consistency = functions::rotationDifference(R_qk_from_kq, R_qk_calc);
//            double t_consistency = functions::getAngleBetween(T_qk_from_kq, T_qk_calc);
//            if (r_consistency >= 0.0001 || t_consistency >= 0.0001) continue;
//
//            R_qk_real = R_q * R_k.transpose();
//            T_qk_real = T_q - R_qk_real * T_k;
//            T_qk_real.normalize();
//
//            double r_dist = functions::rotationDifference(R_qk_real, R_qk_calc);
//            avg_r_dist += r_dist;
//            if (r_dist > max_r) max_r = r_dist;
//            if (r_dist < min_r) min_r = r_dist;
//            double t_dist = functions::getAngleBetween(T_qk_real, T_qk_calc);
//            avg_t_dist += t_dist;
//            if (t_dist > max_t) max_t = t_dist;
//            if (t_dist < min_t) min_t = t_dist;
//            count++;
//            cout << r_dist << ", " << t_dist << " ";
//            if (r_dist <= 10 && t_dist <= 10) {
//                num_good++;
//                cout << "GOOD ";
//            } else {
//                cout << "BAD ";
//            }
//
//
//            cv::Mat src1;
//            cv::Mat q_mat1 = cv::imread(query + ".color.png");
//            cv::Mat im_mat1 = cv::imread(im + ".color.png");
//            cv::hconcat(q_mat1, im_mat1, src1);
//            random_device gen1;
//            uniform_int_distribution<int> dis1(0, 256);
//            for (int i = 0; i < pts_q_qk.size(); i++) {
//                cv::Point2d pt_q = pts_q_qk[i];
//                cv::Point2d pt_i(pts_i_qk[i].x + q_mat.cols, pts_i_qk[i].y);
//                cv::Scalar color = cv::Scalar(dis1(gen), dis1(gen), dis1(gen));
//                circle(src1, pt_q, 3, color, -1);
//                circle(src1, pt_i, 3, color, -1);
//                cv::line(src1, pt_q, pt_i, color, 2);
//            }
//            cv::imshow("INLIERS", src1);
//            cv::waitKey(0);
//
//        }

        cout << endl;
    }

    return 0;
}