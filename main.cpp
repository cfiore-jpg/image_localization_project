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
    functions::createQueryVector(listQuery, info, 6);

//    vector<tuple<double, double, double>> calibrations;
//    aachen::createQueryVector(listQuery, calibrations);

//    vector<string> listQuery = cambridge::getTestImages(scene);
//    vector<string> listQuery = synthetic::getAll();

//    ofstream num_good_anchors;
//    num_good_anchors.open("/Users/cameronfiore/C++/image_localization_project/data/stairs/num_good_anchors.csv");
//
//    ofstream anchor_dists;
//    anchor_dists.open("/Users/cameronfiore/C++/image_localization_project/data/stairs/anchor_dists.csv");
//
//    ofstream anchor_angles;
//    anchor_angles.open("/Users/cameronfiore/C++/image_localization_project/data/stairs/anchor_angles.csv");

//    ofstream distances;
//    distances.open("/Users/cameronfiore/C++/image_localization_project/data/stairs/distances.csv");

//    ofstream distances_spaced;
//    distances_spaced.open("/Users/cameronfiore/C++/image_localization_project/data/stairs/distances_spaced.csv");

//    ofstream angles;
//    angles.open("/Users/cameronfiore/C++/image_localization_project/data/stairs/angles.csv");

//    ofstream angles_spaced;
//    angles_spaced.open("/Users/cameronfiore/C++/image_localization_project/data/stairs/angles_spaced.csv");

//    ofstream poses;
//    poses.open("/Users/cameronfiore/C++/image_localization_project/data/images_upright/poses.txt");
//
//    ofstream poses_adj;
//    poses_adj.open("/Users/cameronfiore/C++/image_localization_project/data/images_upright/poses_adj.txt");






        for (const auto & query : listQuery) {

            auto desc_kp_q = functions::getDescriptors(query, ".color.png", "SIFT");
            cv::Mat desc_q = desc_kp_q.first;
            vector<cv::KeyPoint> kp_q = desc_kp_q.second;

            Eigen::Matrix3d R_q;
            Eigen::Vector3d T_q;
            sevenScenes::getAbsolutePose(query, R_q, T_q);
            Eigen::Vector3d c_q = -R_q.transpose() * T_q;

            double K[4] = {585., 585., 320., 240.};

            int k_hat = 150;
            auto retrieved = functions::retrieveSimilar(query, "7-Scenes", ".color.png", k_hat, 2.5);
            for (const auto & quer)



        }

        cout << "Running queries..." << endl;
        int startIdx = 0;
        vector<string> not_all_right;
        for (int q = startIdx; q < listQuery.size(); q++) {

            cout << q << "/" << listQuery.size() - 1 << ":";

            string query = listQuery[q];
//
//            double q_f = get<0>(calibrations[q]);
//            double q_cx = get<1>(calibrations[q]);
//            double q_cy = get<2>(calibrations[q]);

            auto desc_kp_q = functions::getDescriptors(query, ".color.png", "SIFT");
            cv::Mat desc_q = desc_kp_q.first;
            vector<cv::KeyPoint> kp_q = desc_kp_q.second;

            Eigen::Matrix3d R_q;
            Eigen::Vector3d T_q;
            sevenScenes::getAbsolutePose(query, R_q, T_q);
            Eigen::Vector3d c_q = -R_q.transpose() * T_q;


//            double K[4] = {1., 1., 0., 0.};
            double K[4] = {585., 585., 320., 240.};
//        double K[4] = {1680., 1680., 960., 540.};
//        double K[4] = {744.375, 744.375, 960., 540.};
//        double K[4] = {2584.9325098195013197, 2584.7918606057692159, 249.77137587221417903, 278.31267937919352562};
//            cv::Mat K_mat = (cv::Mat_<double>(3, 3) << K[0], 0., K[2], 0., K[1], K[3], 0., 0., 1.);
//            Eigen::Matrix3d K_eig{{K[0], 0.,   K[2]},
//                                  {0.,   K[1], K[3]},
//                                  {0.,   0.,   1.}};
            int k_hat = 150;
            auto retrieved = functions::retrieveSimilar(query, "7-Scenes", ".color.png", k_hat, 2.5);
            auto spaced = functions::optimizeSpacing(retrieved, 20, false, "7-Scenes");


//            Eigen::Vector3d c_q = sevenScenes::getT(query);
//
//            double max_dist = -1;
//            double min_dist = 100000;
//            double total_dist = 0;
//            for (const auto & im : retrieved) {
//                Eigen::Vector3d c_i = sevenScenes::getT(im);
//                double d = functions::getDistBetween(c_q, c_i);
//                if (d > 2.2) {
//                    cout << q << endl;
//                    break;
//                }
//                total_dist += d;
//                if (d > max_dist) {
//                    max_dist = d;
//                }
//                if (d < min_dist) {
//                    min_dist = d;
//                }
//            }

//            distances << setprecision(6) << min_dist << "," << max_dist << "," << total_dist/k_hat << endl;
//
//            Eigen::Vector3d z = {0, 0, 1};
//            Eigen::Matrix3d R_wq = sevenScenes::getR(query);
//            Eigen::Vector3d op_ax_q = R_wq * z;

//            double max_ang = -1;
//            double min_ang = 100000;
//            double total_ang = 0;
//            for (const auto & im : retrieved) {
//                Eigen::Matrix3d R_wi = sevenScenes::getR(im);
//                Eigen::Vector3d op_ax_i = R_wi * z;
//                double ang = functions::getAngleBetween(op_ax_q, op_ax_i);
//                if (ang >= 85) {
//                    cout << q << endl;
//                    break;
//                }
//                total_ang += ang;
//                if (ang > max_ang) {
//                    max_ang = ang;
//                }
//                if (ang < min_ang) {
//                    min_ang = ang;
//                }
//            }
//
//            angles << setprecision(6) << min_ang << "," << max_ang << "," << total_ang/k_hat << endl;
//            angles_spaced << setprecision(6) << min_ang << "," << max_ang << "," << total_ang/k_hat << endl;


//        string scene = functions::getScene(query, "");
//        for (const auto & im: retrieved) {
//            string this_scene = functions::getScene(im, "");
//            if (this_scene != scene) {
//                cout << q << endl;
//                break;
//            }
//        }
//
        functions::showTop150(query, retrieved, ".color.png");
        functions::showSpacedInTop150(query, retrieved, spaced, ".color.png");
        functions::showSpaced(query, spaced, ".color.png");
//
//        int stop = 0;


//        auto retrieved = synthetic::omitQuery(q, listQuery);

//        if (scene == "/Users/cameronfiore/C++/image_localization_project/data/Street/") {
//            if(functions::getScene(retrieved[0], "Street") != "Street") {
//                cout << query << endl;
//                continue;
//            }
//        } else {
//            string s = functions::getScene(query, "");
//            if(s != functions::getScene(retrieved[0], "")) {
//                cout << query << endl;
//                continue;
//            }
//        }

//        auto spaced = functions::optimizeSpacing(retrieved, 10, false, "aachen");
//        auto spaced = functions::optimizeSpacingZhou(retrieved, 0.05, 10., 20, "7-Scenes");
//        functions::record_spaced(query, spaced, "/Users/cameronfiore/Python/Localization/SuperGluePretrainedNetwork/7-Scenes/stairs/seq-01/frame-000200");
//

            vector<Eigen::Matrix3d> R_ks, R_qks;
            vector<Eigen::Vector3d> T_ks, T_qks;
            vector<string> anchors;
            vector<vector<pair<cv::Point2d, cv::Point2d>>> all_matches;
            int num_good = 0;
            double max_t = 0; double min_t = 1000;
            double max_r = 0; double min_r = 1000;
            double avg_t_dist = 0;
            double avg_r_dist = 0;
            int count = 0;
            for (const auto & im : spaced) {
                Eigen::Matrix3d R_k, R_qk_real, R_qk_calc, R_kq_calc;
                Eigen::Vector3d T_k, T_qk_real, T_qk_calc, T_kq_calc;
                sevenScenes::getAbsolutePose(im, R_k, T_k);
//                double i_f, i_cx, i_cy = 0;
//                aachen::getK(im, i_f, i_cx, i_cy);
//
//                double thresh = 1./((i_f + q_f)/2.);
//                avg_thresh += thresh;
//                count ++;

                auto desc_kp_i = functions::getDescriptors(im, ".color.png", "SIFT");
                cv::Mat desc_i = desc_kp_i.first;
                vector<cv::KeyPoint> kp_i = desc_kp_i.second;

                vector<cv::Point2d> pts_i, pts_q;
                functions::findMatches(0.8, desc_i, kp_i, desc_q, kp_q, pts_i, pts_q);

                vector<cv::Point2d> pts_q_qk = pts_q, pts_i_qk = pts_i;
                functions::getRelativePose(pts_i_qk, pts_q_qk, K, .5, R_qk_calc, T_qk_calc);
//                aachen::getRelativePose(pts_q_qk, q_f, q_cx, q_cy, pts_i_qk, i_f, i_cx, i_cy, 2.*thresh, R_qk_calc, T_qk_calc);

                vector<cv::Point2d> pts_q_kq = pts_q, pts_i_kq = pts_i;
                functions::getRelativePose(pts_q_kq, pts_i_kq, K, .5, R_kq_calc, T_kq_calc);
//                aachen::getRelativePose(pts_i_kq, i_f, i_cx, i_cy, pts_q_kq, q_f, q_cx, q_cy, 2.*thresh, R_kq_calc, T_kq_calc);

                Eigen::Matrix3d R_qk_from_kq = R_kq_calc.transpose();
                Eigen::Vector3d T_qk_from_kq = -R_kq_calc.transpose() * T_kq_calc;

                double r_consistency = functions::rotationDifference(R_qk_from_kq, R_qk_calc);
                double t_consistency = functions::getAngleBetween(T_qk_from_kq, T_qk_calc);
                if (r_consistency >= 0.0001 || t_consistency >= 0.0001) continue;

//                for(int i = 0; i < pts_q_qk.size(); i++) {
//                    cv::Mat q_mat = cv::imread(query);
//                    cv::Mat im_mat = cv::imread(im);
//                    cv::Point2d pt_q = pts_q_qk[i];
//                    cv::Point2d pt_i = pts_i_qk[i];
//                    cv::Scalar color = cv::Scalar(0, 255, 0);
//                    circle(q_mat, pt_q, 10, color, -1);
//                    circle(im_mat, pt_i, 10, color, -1);
//                    cv::imshow("Query", q_mat);
//                    cv::imshow(im, im_mat);
//                    cv::waitKey(0);
//                }

//                cout << setprecision(2) << double(pts_q_qk.size()) / double(pts_q.size()) << "|";

                R_qk_real = R_q * R_k.transpose();
                T_qk_real = T_q - R_qk_real * T_k;
                T_qk_real.normalize();
//
//                Eigen::Matrix3d R_kq_real = R_k * R_q.transpose();
//                Eigen::Vector3d T_kq_real = T_k - R_kq_real * T_q;
//                T_kq_real.normalize();

            random_device gen;
            uniform_int_distribution<int> dis (0, 256) ;
            for(int i = 0; i < pts_q_qk.size(); i++) {
                cv::Mat src;
                cv::Mat q_mat = cv::imread(query + ".color.png");
                cv::Mat im_mat = cv::imread(im + ".color.png");
                cv::hconcat(q_mat, im_mat, src);
                cv::Point2d pt_q = pts_q_qk[i];
                cv::Point2d pt_i(pts_i_qk[i].x + q_mat.cols, pts_i_qk[i].y);
                cv::Scalar color = cv::Scalar(dis(gen), dis(gen), dis(gen));
                circle(src, pt_q, 3, color, -1);
                circle(src, pt_i, 3, color, -1);
                cv::line(src, pt_q, pt_i, color, 2);
                cv::imshow("NEW MATCHING", src);
                cv::waitKey(0);
            }
                double r_dist = functions::rotationDifference(R_qk_real, R_qk_calc);
                avg_r_dist += r_dist; if (r_dist > max_r) max_r = r_dist; if (r_dist < min_r) min_r = r_dist;
                double t_dist = functions::getAngleBetween(T_qk_real, T_qk_calc);
                avg_t_dist += t_dist; if (t_dist > max_t) max_t = t_dist; if (t_dist < min_t) min_t = t_dist;
                count++;
                if (r_dist <= 10 && t_dist <= 10) {
                    num_good++;
                }

                //            cout << r_dist << ", " << t_dist << "       SIFT RANSAC Inliers: " << pts_q_qk.size() << "/" << pts_q.size() << endl;





//            vector<cv::Point2d> SG_points_q, SG_points_i;
//            functions::get_SG_points(query, im, SG_points_q, SG_points_i);
//
//            vector<cv::Point2d> inliers_q = SG_points_q, inliers_i = SG_points_i;
//            functions::findInliers(R_q, T_q, R_k, T_k, K_eig, 1., inliers_q, inliers_i);
//
//
//            pts_q_qk = SG_points_q;
//            pts_i_qk = SG_points_i;
//            functions::getRelativePose(pts_i_qk, pts_q_qk, K, 1., R_qk_calc, T_qk_calc);
//
//            pts_q_kq = SG_points_q;
//            pts_i_kq = SG_points_i;
//            functions::getRelativePose(pts_q_kq, pts_i_kq, K, 1., R_kq_calc, T_kq_calc);
//
//            R_qk_from_kq = R_kq_calc.transpose();
//            T_qk_from_kq = -R_kq_calc.transpose() * T_kq_calc;
//
//            r_consistency = functions::rotationDifference(R_qk_from_kq, R_qk_calc);
//            t_consistency = functions::getAngleBetween(T_qk_from_kq, T_qk_calc);
//            if (r_consistency >= 0.0001 || t_consistency >= 0.0001) {
//                continue;
//            }

//            R_qk_real = R_q * R_k.transpose();
//            T_qk_real = T_q - R_qk_real * T_k;
//            T_qk_real.normalize();
//
//            double r_dist_SG = functions::rotationDifference(R_qk_real, R_qk_calc);
//            double t_dist_SG = functions::getAngleBetween(T_qk_real, T_qk_calc);
//
//            int cross = 0;
//            for (auto & i : inliers_q) {
//                double x = i.x;
//                double y = i.y;
//                for (auto & j : pts_q_qk) {
//                    if (x == j.x && y == j.y) {
//                        cross++;
//                    }
//                }
//            }
//
//            cout << r_dist_SG << ", " << t_dist_SG << "      SuperGlue RANSAC Inliers: " << pts_q_qk.size() << "/" << SG_points_q.size() << endl;
//            cout << "True Inlier Rate: " << inliers_q.size() << "/" << SG_points_q.size() << endl;
//            cout << cross << " RANSAC inliers are true inliers" << endl;
//            cout << endl << endl;


//                vector<pair<cv::Point2d, cv::Point2d>> matches(pts_i_qk.size());
//                for (int i = 0; i < pts_i_qk.size(); i++) {
//                    matches[i] = pair<cv::Point2d, cv::Point2d>{pts_q_qk[i], pts_i_qk[i]};
//                }
//                all_matches.push_back(matches);
//                anchors.push_back(im);
//                R_ks.push_back(R_k);
//                T_ks.push_back(T_k);
//                R_qks.push_back(R_qk_calc);
//                T_qks.push_back(T_qk_calc);
            }
//            avg_t_dist /= count;
//            avg_r_dist /= count;
//            num_good_anchors << num_good << endl;
//            anchor_dists << min_t << "," << max_t << "," << avg_t_dist << endl;
//            anchor_angles << min_r << "," << max_r << "," << avg_r_dist << endl;
//
//            if (anchors.size() < 3) {
//                cout << "Bad query..." << endl;
//                continue;
//            }
//
//            auto results = pose::hypothesizeRANSAC(15., R_ks, T_ks, R_qks, T_qks);
//
//            Eigen::Vector3d c_q_est = get<0>(results);
//            Eigen::Matrix3d R_q_est = get<1>(results);
//            Eigen::Vector3d T_q_est = -R_q_est * c_q_est;
//            vector<int> inlier_indices = get<2>(results);
//            cout << "Inliers:" << inlier_indices.size() << "/" << R_ks.size();

//            vector<Eigen::Matrix3d> R_ks_adj;
//            vector<Eigen::Vector3d> T_ks_adj;
//            vector<vector<pair<cv::Point2d, cv::Point2d>>> matches_adj;
//            for (const auto & idx : inlier_indices) {
//                R_ks_adj.push_back(R_ks[idx]);
//                T_ks_adj.push_back(T_ks[idx]);
//                matches_adj.push_back(all_matches[idx]);
//            }


//            Eigen::Matrix3d R_q_adj = R_q_est;
//            Eigen::Vector3d T_q_adj = T_q_est;
//            avg_thresh = avg_thresh / count;
//            pose::adjustHypothesis(R_ks, T_ks, all_matches, 10.*avg_thresh, K, R_q_adj, T_q_adj);
//            cout << endl;
//
//            string name = query;
//            name = name.substr(name.find("query"), name.length());
//            name = name.substr(name.find('/')+1, name.length());
//            name = name.substr(name.find('/')+1, name.length());
//            name = name.substr(name.find('/')+1, name.length());
//
//            Eigen::Quaterniond R_q (R_q_est);
//            R_q.normalize();
//
//            poses << name << " " << setprecision(12) << R_q.w() << " " <<  R_q.x() << " " <<  R_q.y() << " " <<  R_q.z() << " ";
//            poses << setprecision(12) << T_q_est[0] << " " << T_q_est[1] << " " << T_q_est[2] << endl;
//
//            Eigen::Quaterniond R_q_adj_est (R_q_adj);
//            R_q_adj_est.normalize();
//
//            poses_adj << name << " " << setprecision(12) << R_q_adj_est.w() << " " <<  R_q_adj_est.x() << " " <<  R_q_adj_est.y() << " " <<  R_q_adj_est.z() << " ";
//            poses_adj << setprecision(12) << T_q_adj[0] << " " << T_q_adj[1] << " " << T_q_adj[2] << endl;

            cout << endl;
        }
//        num_good_anchors.close();
//        anchor_dists.close();
//        anchor_angles.close();
//        distances.close();
//        angles.close();
//        distances_spaced.close();
//        angles_spaced.close();
//        poses.close();
//        poses_adj.close();

    return 0;
}