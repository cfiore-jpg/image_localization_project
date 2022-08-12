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
mutex mtx;

void get_poses (const cv::Mat * desc_q, const vector<cv::KeyPoint> * kp_q, const string * im,
                map<string, tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d>> * poses) {

    Eigen::Matrix3d R_k, R_qk_calc, R_kq_calc;
    Eigen::Vector3d T_k, T_qk_calc, T_kq_calc;
    sevenScenes::getAbsolutePose(*im, R_k, T_k);

    auto * desc_kp_i =  new pair<cv::Mat, vector<cv::KeyPoint>> ();
    *desc_kp_i = functions::getDescriptors(*im, ".color.png", "SIFT");
    cv::Mat desc_i = desc_kp_i->first;
    vector<cv::KeyPoint> kp_i = desc_kp_i->second;

    auto * pts_i = new vector<cv::Point2d>();
    auto * pts_q = new vector<cv::Point2d>();
    auto * pts_q_qk = new vector<cv::Point2d>();
    auto * pts_i_qk = new vector<cv::Point2d>();
    auto * pts_q_kq = new vector<cv::Point2d>();
    auto * pts_i_kq = new vector<cv::Point2d>();

    functions::findMatches(0.8, desc_i, kp_i, *desc_q, *kp_q, *pts_i, *pts_q);

    *pts_q_qk = *pts_q;
    *pts_i_qk = *pts_i;
    functions::getRelativePose(*pts_i_qk, *pts_q_qk, K, .5, R_qk_calc, T_qk_calc);

    *pts_q_kq = *pts_q;
    *pts_i_kq = *pts_i;
    functions::getRelativePose(*pts_q_kq, *pts_i_kq, K, .5, R_kq_calc, T_kq_calc);

    Eigen::Matrix3d R_qk_from_kq = R_kq_calc.transpose();
    Eigen::Vector3d T_qk_from_kq = -R_kq_calc.transpose() * T_kq_calc;

    double r_consistency = functions::rotationDifference(R_qk_from_kq, R_qk_calc);
    double t_consistency = functions::getAngleBetween(T_qk_from_kq, T_qk_calc);

    if (r_consistency < 0.00001 || t_consistency < 0.00001) {
        mtx.lock();
        poses->insert(make_pair(*im, make_tuple(R_k, T_k, R_qk_calc, T_qk_calc)));
        mtx.unlock();
    }

    delete desc_kp_i;
    delete pts_i;
    delete pts_q;
    delete pts_q_qk;
    delete pts_i_qk;
    delete pts_q_kq;
    delete pts_i_kq;
}

void test_methods (const string * query,
                   const Eigen::Vector3d * c_q,
                   const Eigen::Matrix3d * R_q,
                   const vector<string> * K_hat,
                   int num_K,
                   const map<string, tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d>> * poses,
                   std::map <int, vector<tuple<double, double, double>>> * error) {

    vector<tuple<double, double, double>> v;

    vector<vector<string>> methods;
    methods.push_back(functions::optimizeSpacing(*query, *K_hat, num_K, false, "7-Scenes"));
    methods.push_back(functions::optimizeSpacingZhou(*K_hat, 0.05, 10, num_K, "7-Scenes"));
    methods.push_back(functions::kMeans(*K_hat, num_K, false));
    methods.push_back(functions::randomSelection(*K_hat, num_K));
    vector<string> first_k (num_K);
    for (int i = 0; i < num_K; i++) {
        first_k[i] = (*K_hat)[i];
    }
    methods.push_back(first_k);

    for(const auto & spaced : methods) {

        auto *R_ks = new vector<Eigen::Matrix3d>();
        auto *R_qks = new vector<Eigen::Matrix3d>();
        auto *T_ks = new vector<Eigen::Vector3d>();
        auto *T_qks = new vector<Eigen::Vector3d>();

        for (const string & im: spaced) {
            try {
                tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d> t = poses->at(im);
                Eigen::Matrix3d R_k = get<0>(t);
                Eigen::Vector3d T_k = get<1>(t);
                Eigen::Matrix3d R_qk_calc = get<2>(t);
                Eigen::Vector3d T_qk_calc = get<3>(t);

                R_ks->push_back(R_k);
                T_ks->push_back(T_k);
                R_qks->push_back(R_qk_calc);
                T_qks->push_back(T_qk_calc);
            } catch (...) {
                continue;
            }
        }

        if (R_ks->size() < 2) {
            v.emplace_back(-1, -1, -1);
        } else {
            auto results = pose::hypothesizeRANSAC(5., *R_ks, *T_ks, *R_qks, *T_qks);
            Eigen::Vector3d c_est = get<0>(results);
            Eigen::Matrix3d R_est = get<1>(results);

            int num_inliers = int(get<2>(results).size());
            double c_error = functions::getDistBetween(*c_q, c_est);
            double r_error = functions::rotationDifference(*R_q, R_est);

            v.emplace_back(num_inliers, c_error, r_error);
        }

        delete R_ks;
        delete T_ks;
        delete R_qks;
        delete T_qks;
    }
    mtx.lock();
    error->emplace(num_K, v);
    mtx.unlock();
}


int main() {

//// Testing Whole Dataset

    ofstream files [5];
    files[0].open("/Users/cameronfiore/C++/image_localization_project/data/chess/k_effect_ours.csv", ios::app);
    files[1].open("/Users/cameronfiore/C++/image_localization_project/data/chess/k_effect_zhou.csv", ios::app);
    files[2].open("/Users/cameronfiore/C++/image_localization_project/data/chess/k_effect_kmeans.csv", ios::app);
    files[3].open("/Users/cameronfiore/C++/image_localization_project/data/chess/k_effect_random.csv", ios::app);
    files[4].open("/Users/cameronfiore/C++/image_localization_project/data/chess/k_effect_firstk.csv", ios::app);

    vector<string> listQuery;
    auto info = sevenScenes::createInfoVector();
    sevenScenes::createQueryVector(listQuery, info, 0);

    int k_hat = 150;
    thread pose_threads[k_hat];
    thread error_threads[k_hat - 1];

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
        Eigen::Vector3d c_q = sevenScenes::getT(query);

        vector<string> retrieved; vector<double> distances;
        functions::retrieveSimilar(query, "7-Scenes", ".color.png", k_hat, 3., retrieved, distances);

        auto * poses = new map<string, tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d>> ();
        for (int k = 0; k < k_hat; k++)
            pose_threads[k] = std::thread(get_poses, &desc_q, &kp_q, &retrieved[k], poses);
        for (auto & th : pose_threads) th.join();

        auto * error = new std::map <int, vector<tuple<double, double, double>>> ();
        for (int k = 2; k <= k_hat; k++)
            error_threads[k-2] = thread(test_methods, &query, &c_q, &R_q, &retrieved, k, poses, error);
        for (auto & th : error_threads) th.join();

        for (int f = 0; f < 5; f++) {
            for (int k = 2; k <= k_hat; k++) {
                tuple<double, double, double> t = error->at(k)[f];
                files[f] << k << "," << get<0>(t) << "," << setprecision(5) << get<1>(t) << "," << get<2>(t) << ",";
            }
            files[f] << endl;
        }

        cout << endl;
        delete poses;
        delete error;
    }
    for (auto & file : files) {
        file.close();
    }
    return 0;
}