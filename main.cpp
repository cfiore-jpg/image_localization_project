
#include "include/sevenScenes.h"
#include "include/synthetic.h"
#include "include/functions.h"
#include "include/poseEstimation.h"
#include "include/calibrate.h"
#include <iostream>
#include <Eigen/SVD>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <opencv2/xfeatures2d.hpp>
#include "include/OptimalRotationSolver.h"
#include "include/CambridgeLandmarks.h"


#define PI 3.1415926536

#define FOLDER "/Users/cameronfiore/C++/image_localization_project/data/"
#define SCENE 0
#define IMAGE_LIST "/Users/cameronfiore/C++/image_localization_project/data/images_1000.txt"
#define EXT ".color.png"
#define GENERATE_DATABASE false

using namespace std;
using namespace chrono;

int main() {

//    double K[4] = {525., 525., 320., 240.};
//    pose::sceneBundleAdjust(1000, K, "/Users/cameronfiore/C++/image_localization_project/data",
//                            "chess", "seq-01", "frame-", ".pose.txt", ".color.png");
//    pose::sceneBundleAdjust(1000, K, "/Users/cameronfiore/C++/image_localization_project/data",
//                            "chess", "seq-02", "frame-", ".pose.txt", ".color.png");
//    pose::sceneBundleAdjust(1000, K, "/Users/cameronfiore/C++/image_localization_project/data",
//                            "chess", "seq-03", "frame-", ".pose.txt", ".color.png");
//    pose::sceneBundleAdjust(1000, K, "/Users/cameronfiore/C++/image_localization_project/data",
//                            "chess", "seq-04", "frame-", ".pose.txt", ".color.png");
//    pose::sceneBundleAdjust(1000, K, "/Users/cameronfiore/C++/image_localization_project/data",
//                            "chess", "seq-05", "frame-", ".pose.txt", ".color.png");
//    pose::sceneBundleAdjust(1000, K, "/Users/cameronfiore/C++/image_localization_project/data",
//                            "chess", "seq-06", "frame-", ".pose.txt", ".color.png");
//
//    exit(0);


//// Testing Whole Dataset
//    cambridge::createPoseFiles("/Users/cameronfiore/C++/image_localization_project/data/KingsCollege/");
    vector<string> listQuery;// = cambridge::getTestImages("/Users/cameronfiore/C++/image_localization_project/data/KingsCollege/");
//    vector<string> listQuery = synthetic::getAll();
    vector<tuple<string, string, vector<string>, vector<string>>> info = sevenScenes::createInfoVector();
    functions::createQueryVector(listQuery, info, SCENE);


    //    ofstream angles;
//    angles.open ("/Users/cameronfiore/C++/image_localization_project/data/spacing_analysis/angles.txt");

//    ofstream R_changes;
//    R_changes.open ("/Users/cameronfiore/C++/image_localization_project/data/analysis/BA_analysis/R_changes.txt");
//    ofstream T_angular_changes;
//    T_angular_changes.open ("/Users/cameronfiore/C++/image_localization_project/data/analysis/BA_analysis/T_angular_changes.txt");
//    ofstream T_mag_changes;
//    T_mag_changes.open ("/Users/cameronfiore/C++/image_localization_project/data/analysis/BA_analysis/T_mag_changes.txt");
//    ofstream rep_before_BA;
//    rep_before_BA.open ("/Users/cameronfiore/C++/image_localization_project/data/analysis/BA_analysis/rep_before_BA.txt");
//    ofstream rep_after_BA;
//    rep_after_BA.open ("/Users/cameronfiore/C++/image_localization_project/data/analysis/BA_analysis/rep_after_BA.txt");



//    ofstream triplet_error;
//    triplet_error.open ("/Users/cameronfiore/C++/image_localization_project/data/analysis/rel_pose_analysis/triplet_error.txt");
//
//
//
//    ofstream R_dist_before;
//    R_dist_before.open ("/Users/cameronfiore/C++/image_localization_project/data/analysis/BA_and_rel_pose_analysis/R_dist_before.txt");
//    ofstream T_dist_before;
//    T_dist_before.open ("/Users/cameronfiore/C++/image_localization_project/data/analysis/BA_and_rel_pose_analysis/T_dist_before.txt");
//    ofstream R_dist_after;
//    R_dist_after.open ("/Users/cameronfiore/C++/image_localization_project/data/analysis/BA_and_rel_pose_analysis/R_dist_after.txt");
//    ofstream T_dist_after;
//    T_dist_after.open ("/Users/cameronfiore/C++/image_localization_project/data/analysis/BA_and_rel_pose_analysis/T_dist_after.txt");

    ofstream c_error;
    c_error.open("/Users/cameronfiore/C++/image_localization_project/data/analysis/results/c_error.txt");

//    ofstream T_gov_error;
//    T_gov_error.open("/Users/cameronfiore/C++/image_localization_project/data/analysis/results/T_gov_error.txt");
//    ofstream T_cf_error;
//    T_cf_error.open("/Users/cameronfiore/C++/image_localization_project/data/analysis/results/T_cf_error.txt");

    ofstream R_avg_error;
    R_avg_error.open("/Users/cameronfiore/C++/image_localization_project/data/analysis/results/R_avg_error.txt");

//    ofstream R_cf_R_error;
//    R_cf_R_error.open("/Users/cameronfiore/C++/image_localization_project/data/analysis/results/R_cf_R_error.txt");
//    ofstream R_cf_R_NORM_error;
//    R_cf_R_NORM_error.open("/Users/cameronfiore/C++/image_localization_project/data/analysis/results/R_cf_R_NORM_error.txt");
//    ofstream R_cf_T_error;
//    R_cf_T_error.open("/Users/cameronfiore/C++/image_localization_project/data/analysis/results/R_cf_T_error.txt");
//    ofstream R_cf_T_NORM_error;
//    R_cf_T_NORM_error.open("/Users/cameronfiore/C++/image_localization_project/data/analysis/results/R_cf_T_NORM_error.txt");
//
//    ofstream R_cf_R_New_error;
//    R_cf_R_New_error.open("/Users/cameronfiore/C++/image_localization_project/data/analysis/results/R_cf_R_New_error.txt");
//    ofstream R_cf_T_New_error;
//    R_cf_T_New_error.open("/Users/cameronfiore/C++/image_localization_project/data/analysis/results/R_cf_T_New_error.txt");
//
//    ofstream R_blend_R_error;
//    R_blend_R_error.open("/Users/cameronfiore/C++/image_localization_project/data/analysis/results/R_blend_R_error.txt");
//    ofstream R_blend_T_error;
//    R_blend_T_error.open("/Users/cameronfiore/C++/image_localization_project/data/analysis/results/R_blend_T_error.txt");

    cout << "Running queries..." << endl;
    int startIdx = 0;
    for (int q = startIdx; q < listQuery.size(); q++) {

        cout << q + 1 << "/" << listQuery.size() << ": ";

        string query = listQuery[q];

        auto desc_kp_q = functions::getDescriptors(query, ".color.png", "SIFT");
//        auto desc_kp_q = functions::getDescriptors(query, ".png", "SIFT");
        cv::Mat desc_q = desc_kp_q.first;
        vector<cv::KeyPoint> kp_q = desc_kp_q.second;

        Eigen::Matrix3d R_q;
        Eigen::Vector3d T_q;
        sevenScenes::getAbsolutePose(query, R_q, T_q);
//        Eigen::Vector3d c_q = synthetic::getC(query);
//        synthetic::getAbsolutePose(query, R_q, t_q);
//        cambridge::getAbsolutePose(query, R_q, T_q);
        Eigen::Vector3d c_q = -R_q.transpose() * T_q;


        double K[4] = {525., 525., 320., 240.};
//        double K[4] = {1670., 1670., 960., 540.};

//        double K[4] = {744.375, 744.375, 960., 540.};
//        double K[4] = {2584.9325098195013197, 2584.7918606057692159, 249.77137587221417903, 278.31267937919352562};

        auto retrieved = functions::retrieveSimilar(query, ".color.png", 100, 1.6);
//        auto retrieved = synthetic::omitQuery(q, listQuery);
//        auto retrieved = functions::retrieveSimilar(query, ".png", 100, 1.);

        auto spaced = functions::optimizeSpacing(retrieved, 20, false, "7-Scenes");
//        auto spaced = functions::optimizeSpacing(retrieved, 20, false, "Cambridge");
//        auto spaced = functions::optimizeSpacing(retrieved, 10, false, "synthetic");

        vector<Eigen::Matrix3d> R_ks, R_ks_BA, R_qks, R_qk_reals;
        vector<Eigen::Vector3d> T_ks, T_ks_BA, T_qks, T_qk_reals;
        vector<pair<cv::Mat, vector<cv::KeyPoint>>> desc_kp_anchors;
        vector<string> anchors;
        vector<vector<pair<cv::Point2d, cv::Point2d>>> all_matches;
        for (const auto & im : spaced) {
            Eigen::Matrix3d R_k, R_qk_calc, R_kq_calc, R_qk_real;
            Eigen::Vector3d T_k, T_qk_calc, T_kq_calc, T_qk_real;
            sevenScenes::getAbsolutePose(im, R_k, T_k);
//            synthetic::getAbsolutePose(im, R_k, t_k);
//            cambridge::getAbsolutePose(im, R_k, T_k);

            cv::Mat K_mat = (cv::Mat_<double>(3, 3) << K[0], 0., K[2], 0., K[1], K[3], 0., 0., 1.);

            auto desc_kp_i = functions::getDescriptors(im, ".color.png", "SIFT");
//            auto desc_kp_i = functions::getDescriptors(im, ".png", "SIFT");
            cv::Mat desc_i = desc_kp_i.first;
            vector<cv::KeyPoint> kp_i = desc_kp_i.second;

            vector<cv::Point2d> pts_i, pts_q;
            functions::findMatches(0.8, desc_i, kp_i, desc_q, kp_q, pts_i, pts_q);
            if (pts_i.size() < 5) continue;

//            synthetic::findSyntheticMatches(im, query, pts_db, pts_q);
//            synthetic::addGaussianNoise(pts_db, pts_q, .1);
//            synthetic::addOcclusion(pts_db, pts_q, 1000);
//            synthetic::addMismatches(pts_db, pts_q, .10);

            vector<cv::Point2d> pts_q_qk = pts_q, pts_i_qk = pts_i;
            functions::getRelativePose(pts_i_qk, pts_q_qk, K, 1., R_qk_calc, T_qk_calc);

            vector<cv::Point2d> pts_q_kq = pts_q, pts_i_kq = pts_i;
            functions::getRelativePose(pts_q_kq, pts_i_kq, K, 1., R_kq_calc, T_kq_calc);

            Eigen::Matrix3d R_qk_from_kq = R_kq_calc.transpose();
            Eigen::Vector3d T_qk_from_kq = - R_kq_calc.transpose() * T_kq_calc;

            double r_consistency = functions::rotationDifference(R_qk_from_kq, R_qk_calc);
            double t_consistency = functions::getAngleBetween(T_qk_from_kq, T_qk_calc);
            if (r_consistency >= 0.0001 || t_consistency >= 0.0001) {
                continue;
            }

            cout << " " << pts_i_qk.size() << ",";

            R_qk_real = R_q * R_k.transpose();
            T_qk_real = T_q - R_qk_real * T_k;
            T_qk_real.normalize();

            Eigen::Matrix3d R_kq_real = R_k * R_q.transpose();
            Eigen::Vector3d T_kq_real = T_k - R_kq_real * T_q;
            T_kq_real.normalize();

            double r_dist = functions::rotationDifference(R_qk_real, R_qk_calc);
            double t_dist = functions::getAngleBetween(T_qk_real, T_qk_calc);


//            random_device generator;
//            uniform_int_distribution<int> distribution(1,255);
//            for(int x = 0; x < pts_i_qk.size(); x+=10) {
////                cv::Mat src;
//                cv::Mat q_mat = cv::imread(query+".png");
//                cv::Mat im_mat = cv::imread(im+".png");
//                Eigen::Vector3d pt_i {pts_i_qk[x].x, pts_i_qk[x].y, 1.};
//                Eigen::Vector3d pt_q {pts_q_qk[x].x, pts_q_qk[x].y, 1.};
////                cv::hconcat(q_mat, im_mat, src);
////                for(int i = x; i < MIN(x + 10, pts_i_qk.size()) ; i++) {
////                    cv::Point2d pt_q = pts_q_qk[i];
////                    cv::Point2d pt_i (pts_i_qk[i].x + q_mat.cols, pts_i_qk[i].y);
////                    cv::Scalar color = cv::Scalar(distribution(generator), distribution(generator),
////                                                  distribution(generator));
////                    circle(src, pt_q, 3, color, -1);
////                    circle(src, pt_i, 3, color, -1);
////                    cv::line(src, pt_q, pt_i, color, 2);
////                }
////                cv::imshow("Feature Matching", src);
////                cv::waitKey(0);
//
//                Eigen::Matrix3d K_eig{{K[0], 0.,   K[2]},
//                                      {0.,  K[1], K[3]},
//                                      {0.,   0.,   1.}};
//
//                T_qk_real.normalize();
//                Eigen::Matrix3d t_qk_cross {
//                        {0,               -T_qk_real(2), T_qk_real(1)},
//                        {T_qk_real(2),  0,              -T_qk_real(0)},
//                        {-T_qk_real(1), T_qk_real(0),               0}};
//                Eigen::Matrix3d E_qk = t_qk_cross * R_qk_real;
//                Eigen::Vector3d epiline_qk = K_eig.inverse().transpose() * E_qk * K_eig.inverse() * pt_i;
//
//                vector<cv::Point3d> lines_q {cv::Point3d (epiline_qk[0], epiline_qk[1], epiline_qk[2])};
//                vector<cv::Point2d> points_q {cv::Point2d (pt_q[0], pt_q[1])};
//                cv::Scalar color_q = cv::Scalar(distribution(generator), distribution(generator), distribution(generator));
//                vector<cv::Scalar> colors {color_q};
//
//
//                Eigen::Matrix3d t_kq_cross {
//                        {0,               -T_kq_real(2), T_kq_real(1)},
//                        {T_kq_real(2),  0,              -T_kq_real(0)},
//                        {-T_kq_real(1), T_kq_real(0),               0}};
//                Eigen::Matrix3d E_kq = t_kq_cross * R_kq_real;
//                Eigen::Vector3d epiline_kq = K_eig.inverse().transpose() * E_kq * K_eig.inverse() * pt_q;
//
//                vector<cv::Point3d> lines_k {cv::Point3d (epiline_kq[0], epiline_kq[1], epiline_kq[2])};
//                vector<cv::Point2d> points_k {cv::Point2d (pt_i[0], pt_i[1])};
//                cv::Scalar color_k = cv::Scalar(distribution(generator), distribution(generator), distribution(generator));
//                colors.push_back(color_k);
//
//                functions::drawLines(q_mat, im_mat, lines_q, lines_k, points_q, points_k, colors);
//                cv::Mat src;
//                cv::hconcat(q_mat, im_mat, src);
//                cv::imshow("epiline", src);
//                cv::waitKey(0);
//            }

            vector<pair<cv::Point2d, cv::Point2d>> matches (pts_i_qk.size());
            for(int i = 0; i < pts_i_qk.size(); i++) {
                matches[i] = pair<cv::Point2d, cv::Point2d> {pts_q_qk[i], pts_i_qk[i]};
            }

            all_matches.push_back(matches);
            anchors.push_back(im);
            R_ks.push_back(R_k);
            T_ks.push_back(T_k);
            R_qks.push_back(R_qk_calc);
            T_qks.push_back(T_qk_calc);
            desc_kp_anchors.push_back(desc_kp_i);
            R_qk_reals.push_back(R_qk_real);
            T_qk_reals.push_back(T_qk_real);
        }

        if (anchors.size() < 3) {
            cout << "Bad query..." << endl;
            continue;
        }

        auto results = pose::hypothesizeRANSAC(10., R_ks, T_ks, R_qks, T_qks);

        vector<int> inlier_indices = get<2>(results);
        cout << " Inliers: " << inlier_indices.size() << endl;

        Eigen::Vector3d c_q_est = get<0>(results);
        Eigen::Matrix3d R_q_est = get<1>(results);

        Eigen::Vector3d T_q_est = - R_q_est * c_q_est;
        pose::adjustHypothesis(R_ks, T_ks, all_matches, 1., K, R_q_est, T_q_est);
        c_q_est = - R_q_est.transpose() * T_q_est;

        double c_ = functions::getDistBetween(c_q_est, c_q);

//        double T_gov_ = functions::getDistBetween(get<1>(results), T_q);
//        double T_cf_ = functions::getDistBetween(get<2>(results), T_q);

        double R_avg_ = functions::rotationDifference(R_q_est, R_q);




//        double R_cf_R_ = functions::rotationDifference(get<4>(results), R_q);
//        double R_cf_R_NORM_ = functions::rotationDifference(get<5>(results), R_q);
//        double R_cf_T_ = functions::rotationDifference(get<6>(results), R_q);
//        double R_cf_T_NORM_ = functions::rotationDifference(get<7>(results), R_q);
//        double R_cf_R_New_ = functions::rotationDifference(get<8>(results), R_q);
//        double R_cf_T_New_ = functions::rotationDifference(get<9>(results), R_q);

//        vector<Eigen::Matrix3d> rotations_R {get<8>(results), get<3>(results), get<3>(results)};
//        vector<Eigen::Matrix3d> rotations_T {get<9>(results), get<3>(results), get<3>(results)};
//        Eigen::Matrix3d blend_R = pose::R_q_average(rotations_R);
//        Eigen::Matrix3d blend_T = pose::R_q_average(rotations_T);
//        double R_blend_R_ = functions::rotationDifference(blend_R, R_q);
//        double R_blend_T_ = functions::rotationDifference(blend_T, R_q);


        int stop = 0;

//        double total_d = 0, total_c = 0;
//        for(const auto & n : inlier_indices) {
//            Eigen::Vector3d T_qk_direct = T_qks[n];
//            Eigen::Vector3d T_qk_composed = - R_qks[n] * (R_ks[n] * get<0>(results) + T_ks[n]);
//            T_qk_composed.normalize();
//
//            Eigen::Vector3d T_qk_real = T_qk_reals[n];
//
//            total_d += functions::getAngleBetween(T_qk_direct, T_qk_real);
//            total_c += functions::getAngleBetween(T_qk_composed, T_qk_real);
//        }
//        double avg_direct = total_d / inlier_indices.size();
//        double avg_composed = total_c / inlier_indices.size();
//        int x = 0;
//
//
//        int stop = 0;

////Write to error files
        c_error << c_ << endl;

//        T_gov_error << T_gov_ << endl;
//        T_cf_error << T_cf_ << endl;

        R_avg_error << R_avg_ << endl;
//        R_cf_R_error << R_cf_R_ << endl;
//        R_cf_R_NORM_error << R_cf_R_NORM_ << endl;
//        R_cf_T_error << R_cf_T_ << endl;
//        R_cf_T_NORM_error << R_cf_T_NORM_ << endl;
//
//        R_cf_R_New_error << R_cf_R_New_ << endl;
//        R_cf_T_New_error << R_cf_T_New_ << endl;
//
//        R_blend_R_error << R_blend_R_ << endl;
//        R_blend_T_error << R_blend_T_ << endl;


        // BUNDLE ADJUSTMENT TESTING
//        double avg_rep_error_before_BA = pose::averageReprojectionError(K, query, R_q, T_q, anchors, R_ks, T_ks);
//        rep_before_BA << avg_rep_error_before_BA << endl;

//        vector<Eigen::Matrix3d> new_R_ks;
//        vector<Eigen::Vector3d> new_T_ks;
//        for (const auto & anchor : anchors) {
//            Eigen::Matrix3d new_R_k;
//            Eigen::Vector3d new_T_k;
//            sevenScenes::getAbsolutePose_BA(anchor, new_R_k, new_T_k);
//            new_R_ks.push_back(new_R_k);
//            new_T_ks.push_back(new_T_k);
//        }
//        Eigen::Matrix3d new_R_q = sevenScenes::getR_BA(query);
//        Eigen::Vector3d new_T_q = sevenScenes::getT_BA(query);
//        Eigen::Vector3d new_c_q = -new_R_q.transpose() * new_T_q;
//
//        for(int i = 0; i < anchors.size(); i++) {
//            double angle_change = functions::rotationDifference(R_ks[i], new_R_ks[i]);
//            R_changes << angle_change << endl;
//
//            double t_angular_change = functions::getAngleBetween(T_ks[i], new_T_ks[i]);
//            T_angular_changes << t_angular_change << endl;
//
//
//            double dist_change = functions::getDistBetween(T_ks[i], new_T_ks[i]);
//            double t_mag_change = dist_change / T_ks[i].norm();
//            T_mag_changes << t_mag_change * 100 << endl;
//        }
//        double query_angle_change = functions::rotationDifference(R_q, new_R_q);
//        R_changes << query_angle_change << endl;
//
//        double q_angular_change = functions::getAngleBetween(T_q, new_T_q);
//        T_angular_changes << q_angular_change << endl;
//
//        double q_dist_change = functions::getDistBetween(T_q, new_T_q);
//        double q_mag_change = q_dist_change / T_q.norm();
//        T_mag_changes << q_mag_change * 100 << endl;
//
//        double avg_rep_error_after_BA = pose::averageReprojectionError(K, query, new_R_q, new_T_q, anchors, new_R_ks, new_T_ks);
//        rep_after_BA << avg_rep_error_after_BA << endl;
//
//        int stop = 0;



//        // TRIPLET TESTING
//        double avg_triplet_error = pose::tripletCircles(K, desc_kp_anchors);
//        triplet_error << avg_triplet_error << endl;

    }

//    rep_before_BA.close();
//    R_changes.close();
//    T_angular_changes.close();
//    T_mag_changes.close();
//    rep_after_BA.close();

//    triplet_error.close();
//    R_qk_to_gt.close();
//    T_qk_to_gt.close();
//
//    R_dist_before.close();
//    T_dist_before.close();
//    R_dist_after.close();
//    T_dist_after.close();

//    angles.close();

    c_error.close();
//    T_gov_error.close();
//    T_cf_error.close();
    R_avg_error.close();
//    R_cf_R_error.close();
//    R_cf_R_NORM_error.close();
//    R_cf_T_error.close();
//    R_cf_T_NORM_error.close();
//    R_cf_R_New_error.close();
//    R_cf_T_New_error.close();
//    R_blend_R_error.close();
//    R_blend_T_error.close();

    return 0;
}