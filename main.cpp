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

void findInliers (double threshold,
                  int i,
                  int j,
                  const vector<Eigen::Matrix3d> * R_ks,
                  const vector<Eigen::Vector3d> * T_ks,
                  const vector<Eigen::Matrix3d> * R_qks,
                  const vector<Eigen::Vector3d> * T_qks,
                  vector<tuple<int, int, double, vector<int>>> * results) {

    int K = int((*R_ks).size());

    vector<Eigen::Matrix3d> set_R_ks{(*R_ks)[i], (*R_ks)[j]};
    vector<Eigen::Vector3d> set_T_ks{(*T_ks)[i], (*T_ks)[j]};
    vector<Eigen::Matrix3d> set_R_qks{(*R_qks)[i], (*R_qks)[j]};
    vector<Eigen::Vector3d> set_T_qks{(*T_qks)[i], (*T_qks)[j]};
    vector<int> indices{i, j};
    double score = 0;

    Eigen::Matrix3d R_h = pose::R_q_average(
            vector<Eigen::Matrix3d>{(*R_qks)[i] * (*R_ks)[i], (*R_qks)[j] * (*R_ks)[j]});
    Eigen::Vector3d c_h = pose::c_q_closed_form(set_R_ks, set_T_ks, set_R_qks, set_T_qks);

    for (int k = 0; k < K; k++) {
        if (k != i && k != j) {
            Eigen::Matrix3d R_qk_h = R_h * (*R_ks)[k].transpose();
            Eigen::Vector3d T_qk_h = -R_qk_h * ((*R_ks)[k] * c_h + (*T_ks)[k]);
            double T_angular_diff = functions::getAngleBetween(T_qk_h, (*T_qks)[k]);
            double R_angular_diff = functions::rotationDifference(R_qk_h, (*R_qks)[k]);
            if (T_angular_diff <= threshold && R_angular_diff <= threshold) {
                indices.push_back(k);
                score += R_angular_diff + T_angular_diff;
            }
        }
    }

    auto t = make_tuple(i, j, score, indices);

    mtx.lock();
    results->push_back(t);
//    cout << i << ", " << j << " finished." << endl;
    mtx.unlock();
}

void calculation(double angle_thresh,
                 double covis,
                 double pixel_thresh,
                 double post_ransac_percent,
                 double reproj_radius,
                 Eigen::Matrix3d * R_estimation,
                 Eigen::Vector3d * T_estimation,
                 Eigen::Matrix3d * R_q,
                 Eigen::Vector3d * T_q,
                 vector<double> * K_q,
                 vector<pair<pair<double, double>, vector<tuple<pair<double, double>, Eigen::Matrix3d, Eigen::Vector3d, vector<double>>>>> * r,
                 map<tuple<double, double, double, double, double>, vector<tuple<double, double, double, double, double>>> * m) {

    vector<double> v_GT, v_EST;
    for (const auto & p : *r) {
        if (p.second.size() < covis) break;
        auto p_and_s = pose::RANSAC3DPoint(pixel_thresh, p.second);
        if (p_and_s.second.size() / p.second.size() < post_ransac_percent) continue;
        cv::Point2d pt(p.first.first, p.first.second);
        Eigen::Vector3d point3d = p_and_s.first;
        cv::Point2d reprojEST = pose::reproject3Dto2D(point3d, *R_estimation, *T_estimation, *K_q);
        if (sqrt(pow(pt.x - reprojEST.x, 2.) + pow(pt.y - reprojEST.y, 2.)) > reproj_radius) continue;
        cv::Point2d reprojGT = pose::reproject3Dto2D(point3d, *R_q, *T_q, *K_q);
        v_GT.push_back(sqrt(pow(pt.x - reprojGT.x, 2.) + pow(pt.y - reprojGT.y, 2.)));
        v_EST.push_back(sqrt(pow(pt.x - reprojEST.x, 2.) + pow(pt.y - reprojEST.y, 2.)));
    }

    if (!v_GT.empty()) {
        auto gt = functions::mean_and_stdv(v_GT);
        auto est = functions::mean_and_stdv(v_EST);
        double m_GT = gt.first;
        double m_EST = est.first;
        double stdv_GT = gt.second;
        double stdv_EST = est.second;

        auto key = make_tuple(angle_thresh, covis, pixel_thresh, post_ransac_percent, reproj_radius);
        auto t = make_tuple(m_GT, stdv_GT, m_EST, stdv_EST, double(v_GT.size()));

        mtx.lock();
        if((*m).find(key) != (*m).end()) {
            (*m).at(key).push_back(t);
        } else {
            vector<tuple<double, double, double, double, double>> val = {t};
            (*m).insert({key, val});
        }
        mtx.unlock();
    }

}

int main() {

//    vector<string> scenes = {"chess/", "fire/", "heads/", "office/", "pumpkin/", "redkitchen/", "stairs/"};
//   vector<string> scenes = {"stairs/"};
//   string dataset = "seven_scenes/";

//    vector<string> scenes = {"KingsCollege/", "OldHospital/", "ShopFacade/", "StMarysChurch/"};
      vector<string> scenes = {"KingsCollege/"};
      string dataset = "cambridge/";


    string relpose_file = "relpose_SP";
    string error_file = "hp_sweep";

    string ccv_dir = "/users/cfiore/data/cfiore/image_localization_project/data/" + dataset;
    string home_dir = "/Users/cameronfiore/C++/image_localization_project/data/" + dataset;
    string dir = ccv_dir;

    for (const auto & scene: scenes) {
        ofstream error;
        error.open(dir + scene + error_file + ".txt");

        map<tuple<double, double, double, double, double>, vector<tuple<double, double, double, double, double>>> m;

        int start = 0;
        vector<string> queries = functions::getQueries(dir + "q.txt", scene);
        for (int q = start; q < queries.size(); q++) {
            cout << q + 1 << "/" << queries.size();


            string query = queries[q];
            string line = query;

            auto info = functions::parseRelposeFile(dir, query, relpose_file);
            auto R_q = get<1>(info);
            auto T_q = get<2>(info);
            auto K_q = get<3>(info);
            auto anchors = get<4>(info);
            auto R_is = get<5>(info);
            auto T_is = get<6>(info);
            auto R_qis = get<7>(info);
            auto T_qis = get<8>(info);
            auto K_is = get<9>(info);
            auto inliers_q = get<10>(info);
            auto inliers_i = get<11>(info);
            Eigen::Vector3d c_q = -R_q.transpose() * T_q;
            int K = int(anchors.size());


            for (double angle_thresh = 5.; angle_thresh < 6.; angle_thresh += 5.) {

                cout << " " << angle_thresh ;


                vector<thread> threads;
                vector<tuple<int, int, double, vector<int>>> results;
                for (int i = 0; i < K - 1; i++) {
                    for (int j = i + 1; j < K; j++) {
                        threads.emplace_back(thread(findInliers, angle_thresh, i, j, &R_is, &T_is, &R_qis, &T_qis, &results));
                        this_thread::sleep_for(std::chrono::microseconds(1));
                    }
                }
                for (auto & th: threads) th.join();

                // Sort by number of inliers
                sort(results.begin(), results.end(), [](const auto &a, const auto &b) {
                    return get<3>(a).size() > get<3>(b).size();
                });

                // Sort by lowest score
                auto best_set = results.at(0);
                int size = int(get<3>(best_set).size());
                double best_score = get<2>(best_set);
                int idx = 0;
                while (true) {
                    try {
                        auto set = results.at(idx);
                        if (get<3>(set).size() != size) break;
                        if (get<2>(set) < best_score) {
                            best_score = get<2>(set);
                            best_set = set;
                        }
                    } catch (...) {
                        break;
                    }
                    idx++;
                }

                vector<string> best_anchors;
                vector<Eigen::Matrix3d> best_R_is, best_R_qis;
                vector<Eigen::Vector3d> best_T_is, best_T_qis;
                vector<vector<double>> best_K_is;
                vector<vector<cv::Point2d>> best_inliers_q, best_inliers_i;
                for (const auto &i: get<3>(best_set)) {
                    best_anchors.push_back(anchors[i]);
                    best_R_is.push_back(R_is[i]);
                    best_R_qis.push_back(R_qis[i]);
                    best_T_is.push_back(T_is[i]);
                    best_T_qis.push_back(T_qis[i]);
                    best_inliers_q.push_back(inliers_q[i]);
                    best_inliers_i.push_back(inliers_i[i]);
                    best_K_is.push_back(K_is[i]);
                }
                vector<Eigen::Matrix3d> rotations(best_R_is.size());
                for (int i = 0; i < best_R_is.size(); i++) {
                    rotations[i] = best_R_qis[i] * best_R_is[i];
                }

                Eigen::Vector3d c_estimation = pose::c_q_closed_form(best_R_is, best_T_is, best_R_qis, best_T_qis);
                Eigen::Matrix3d R_estimation = pose::R_q_average(rotations);
                Eigen::Vector3d T_estimation = -R_estimation * c_estimation;

                auto r = functions::findSharedMatches(best_R_is, best_T_is, best_K_is, best_inliers_q, best_inliers_i);

                vector<thread> thrs;
                for (double covis = 15; covis < 16; covis += 5) {
                    for (double pixel_thresh = 5; pixel_thresh < 6; pixel_thresh += 3) {
                        for (double post_ransac_percent = .75; post_ransac_percent < 8.; post_ransac_percent += .25) {
                            for (double reproj_radius = 25; reproj_radius < 30; reproj_radius += 10) {
                                thrs.emplace_back(thread(calculation, angle_thresh, covis, pixel_thresh,
                                                         post_ransac_percent, reproj_radius, &R_estimation, &T_estimation,
                                                         &R_q, &T_q, &K_q, &r, &m));
                                this_thread::sleep_for(std::chrono::microseconds (1));
                            }
                        }
                    }
                }
                for (auto & th: thrs) th.join();

            }
            cout << endl;
        }
        for (const auto & p : m) {
            error << get<0>(p.first) << " " << get<1>(p.first) << " " << get<2>(p.first) << " " << get<3>(p.first) << " " << get<4>(p.first);
            for (const auto & t : p.second) {
                error << " " << get<0>(t) << " " << get<1>(t) << " " << get<2>(t) << " " << get<3>(t) << " " << get<4>(t);
            }
            error << endl;
        }
        error.close();
    }
    return 0;
}