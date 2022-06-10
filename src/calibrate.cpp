//
// Created by Cameron Fiore on 12/14/21.
//

#include "ceres/ceres.h"
#include "glog/logging.h"
#include "../include/sevenScenes.h"
#include "../include/functions.h"
#include "../include/calibrate.h"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <fstream>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
using namespace std;

#define PI 3.14159265
#define EXT ".color.png"
#define FOLDER "/Users/cameronfiore/C++/image_localization_project/data/"

struct ReprojectionError2D {
    ReprojectionError2D(Eigen::Matrix3d &E_q1,
                        Eigen::Matrix3d &E_q2,
                        cv::Point2d &point_1,
                        cv::Point2d &point_2,
                        cv::Point2d &point_q)
            : E_q1(E_q1), E_q2(E_q2), point_1(point_1), point_2(point_2), point_q(point_q) {}

    template<typename T>
    bool operator()(const T *const K, T *residuals) const {

        T px2point_1[3];
        px2point_1[0] = (point_1.x - K[2]) / K[0];
        px2point_1[1] = (point_1.y - K[3]) / K[1];
        px2point_1[2] = T(1.);

        T point2point_1[3];
        point2point_1[0] = E_q1(0, 0) * px2point_1[0] + E_q1(0, 1) * px2point_1[1] + E_q1(0, 2) * px2point_1[2];
        point2point_1[1] = E_q1(1, 0) * px2point_1[0] + E_q1(1, 1) * px2point_1[1] + E_q1(1, 2) * px2point_1[2];
        point2point_1[2] = E_q1(2, 0) * px2point_1[0] + E_q1(2, 1) * px2point_1[1] + E_q1(2, 2) * px2point_1[2];

        T epiline_1[3];
        epiline_1[0] = (1.0 / K[0]) * point2point_1[0];
        epiline_1[1] = (1.0 / K[1]) * point2point_1[1];
        epiline_1[2] = -(K[2] / K[0]) * point2point_1[0] - (K[3] / K[1]) * point2point_1[1] + point2point_1[2];


        // Get second epiline

        T px2point_2[3];
        px2point_2[0] = (point_2.x - K[2]) / K[0];
        px2point_2[1] = (point_2.y - K[3]) / K[1];
        px2point_2[2] = T(1.);

        T point2point_2[3];
        point2point_2[0] = E_q2(0, 0) * px2point_2[0] + E_q2(0, 1) * px2point_2[1] + E_q2(0, 2) * px2point_2[2];
        point2point_2[1] = E_q2(1, 0) * px2point_2[0] + E_q2(1, 1) * px2point_2[1] + E_q2(1, 2) * px2point_2[2];
        point2point_2[2] = E_q2(2, 0) * px2point_2[0] + E_q2(2, 1) * px2point_2[1] + E_q2(2, 2) * px2point_2[2];

        T epiline_2[3];
        epiline_2[0] = (1.0 / K[0]) * point2point_2[0];
        epiline_2[1] = (1.0 / K[1]) * point2point_2[1];
        epiline_2[2] = -(K[2] / K[0]) * point2point_2[0] - (K[3] / K[1]) * point2point_2[1] + point2point_2[2];


        // Find where epilines intersect
        T x_pred = ((epiline_1[1] / epiline_2[1]) * epiline_2[2] - epiline_1[2])
                   / (epiline_1[0] - (epiline_1[1] / epiline_2[1]) * epiline_2[0]);
        T y_pred = -(epiline_1[2] + epiline_1[0] * x_pred) / epiline_1[1];

        // compute residuals
        T x_error = x_pred - point_q.x;
        T y_error = y_pred - point_q.y;


        residuals[0] = x_error;
        residuals[1] = y_error;

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *Create(Eigen::Matrix3d &E_q1,
                                       Eigen::Matrix3d &E_q2,
                                       cv::Point2d &point_1,
                                       cv::Point2d &point_2,
                                       cv::Point2d &point_q) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError2D, 2, 4>(
                new ReprojectionError2D(E_q1, E_q2, point_1, point_2, point_q)));
    }

    Eigen::Matrix3d E_q1;
    Eigen::Matrix3d E_q2;
    cv::Point2d point_1;
    cv::Point2d point_2;
    cv::Point2d point_q;
};

void calibrate::calibrate (double K [4], int scene, bool display_triplets) {

    double total_reprojection_error = 0.;
    int total_triplets = 0;

// convert K to eigen vector
    Eigen::Matrix3d K_eig{
            {K[0], 0.,   K[2]},
            {0.,   K[1], K[3]},
            {0.,   0.,   1.}};

// get queries
    vector<string> listQuery;
    auto info = sevenScenes::createInfoVector();
    functions::createQueryVector(listQuery, info, scene);

    int old_percent;
    int new_percent = 0;
    vector<Eigen::Matrix3d> all_E_q1, all_E_q2;
    vector<vector<cv::Point2d>> all_points_1, all_points_2, all_points_q;
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    for (int q = 0; q < listQuery.size(); q++) {

        string query = listQuery[q];
        vector<string> topN = functions::retrieveSimilar(query, ".pose.png", "", 20, 1.6);
        vector<string> top2 = functions::optimizeSpacing(topN, 2, false, "7-Scenes");
        if (top2.size() < 2) continue;
        string im_1 = top2[0];
        string im_2 = top2[1];

        // Get point correspondences for this triplet
        vector<cv::Point2d> pts_1, pts_2, pts_q1, pts_q2;
        functions::findMatches(im_1, ".pose.png",query, "ORB", 0.8, pts_1, pts_q1);
        functions::findMatches(im_2, ".pose.png",query, "ORB", 0.8, pts_2, pts_q2);

        if (pts_1.size() >= 24 && pts_2.size() >= 24) {

//             Filter outliers
            cv::Mat mask1, mask2;
            vector<cv::Point2d> inliers_1, inliers_2, inliers_q1, inliers_q2;
            cv::findFundamentalMat(pts_1, pts_q1, cv::FM_RANSAC, 3.0, 0.999999, mask1);
            cv::findFundamentalMat(pts_2, pts_q2, cv::FM_RANSAC, 3.0, 0.999999, mask2);
            for (int i = 0; i < pts_1.size(); i++) {
                if (mask1.at<unsigned char>(i)) {
                    inliers_1.push_back(pts_1[i]);
                    inliers_q1.push_back(pts_q1[i]);
                }
            }
            for (int i = 0; i < pts_2.size(); i++) {
                if (mask2.at<unsigned char>(i)) {
                    inliers_2.push_back(pts_2[i]);
                    inliers_q2.push_back(pts_q2[i]);
                }
            }


            // find triplet matches
            vector<cv::Point2d> triplets_1, triplets_2, triplets_q;
            for (int i = 0; i < inliers_q1.size(); i++) {
                int x1 = int(round(inliers_q1[i].x));
                int y1 = int(round(inliers_q1[i].y));
                for (int j = 0; j < inliers_q2.size(); j++) {
                    int x2 = int(round(inliers_q2[j].x));
                    int y2 = int(round(inliers_q2[j].y));
                    if (x1 == x2 && y1 == y2) {
                        triplets_1.push_back(inliers_1[i]);
                        triplets_2.push_back(inliers_2[j]);
                        triplets_q.push_back((inliers_q1[i] + inliers_q2[j]) / 2.);

                        inliers_2.erase(inliers_2.begin() + j);
                        inliers_q2.erase(inliers_q2.begin() + j);
                        break;
                    }
                }
            }

            if (!triplets_q.empty()) {

                // Get absolute poses for each image
                Eigen::Matrix3d R_1;
                Eigen::Vector3d t_1;
                sevenScenes::getAbsolutePose(im_1, R_1, t_1);

                Eigen::Matrix3d R_2;
                Eigen::Vector3d t_2;
                sevenScenes::getAbsolutePose(im_2, R_2, t_2);

                Eigen::Matrix3d R_q;
                Eigen::Vector3d t_q;
                sevenScenes::getAbsolutePose(query, R_q, t_q);

                // Calculate relative poses
                Eigen::Matrix3d R_q1 = R_q * R_1.transpose();
                Eigen::Vector3d t_q1 = t_q - R_q1 * t_1;

                Eigen::Matrix3d R_q2 = R_q * R_2.transpose();
                Eigen::Vector3d t_q2 = t_q - R_q2 * t_2;


                // Compute Essential matrices
                Eigen::Matrix3d t_q1_skew{{0,        -t_q1(2), t_q1(1)},
                                          {t_q1(2),  0,        -t_q1(0)},
                                          {-t_q1(1), t_q1(0),  0}};
                Eigen::Matrix3d E_q1 = t_q1_skew * R_q1;


                Eigen::Matrix3d t_q2_skew{{0,        -t_q2(2), t_q2(1)},
                                          {t_q2(2),  0,        -t_q2(0)},
                                          {-t_q2(1), t_q2(0),  0}};
                Eigen::Matrix3d E_q2 = t_q2_skew * R_q2;

                Eigen::Matrix3d F_q1 = K_eig.inverse().transpose() * E_q1 * K_eig.inverse();
                Eigen::Matrix3d F_q2 = K_eig.inverse().transpose() * E_q2 * K_eig.inverse();
                cv::Mat F_q1_mat, F_q2_mat;
                cv::eigen2cv(F_q1, F_q1_mat);
                cv::eigen2cv(F_q2, F_q2_mat);

                vector<cv::Point3d> lines_on_q_from_1, lines_on_q_from_2;
                cv::computeCorrespondEpilines(triplets_1, 1, F_q1_mat, lines_on_q_from_1);
                cv::computeCorrespondEpilines(triplets_2, 1, F_q2_mat, lines_on_q_from_2);

                for (int c = 0; c < triplets_q.size(); c++) {
                    cv::Point3d line_1 = lines_on_q_from_1[c];
                    cv::Point3d line_2 = lines_on_q_from_2[c];

                    double x_pred = ((line_1.y / line_2.y) * line_2.z - line_1.z) /
                                    (line_1.x - (line_1.y / line_2.y) * line_2.x);
                    double y_pred = -(line_1.z + line_1.x * x_pred) / line_1.y;
                    double error = sqrt(pow(triplets_q[c].x - x_pred, 2) + pow(triplets_q[c].y - y_pred, 2));


                    if (error > 15.) {
//                            cv::Mat image_1 = cv::imread(im_1 + ".color.png");
//                            cv::Mat image_2 = cv::imread(im_2 + ".color.png");
//                            cv::Mat image_q = cv::imread(query + ".color.png");
//
//                            vector<cv::Scalar> colors = {cv::Scalar((double) std::rand() / RAND_MAX * 255,
//                                                                    (double) std::rand() / RAND_MAX * 255,
//                                                                    (double) std::rand() / RAND_MAX * 255)};
//
//                            vector<cv::Point2d> triplet_q = {triplets_q[c]};
//                            vector<cv::Point2d> triplet_1 = {triplets_1[c]};
//                            vector<cv::Point2d> triplet_2 = {triplets_2[c]};
//                            vector<cv::Point3d> line_on_q_from_1 = {line_1};
//                            vector<cv::Point3d> line_on_q_from_2 = {line_2};
//                            vector<cv::Point3d> line_on_1_from_q, line_on_2_from_q;
//                            cv::computeCorrespondEpilines(triplet_q, 2, F_q1_mat, line_on_1_from_q);
//                            cv::computeCorrespondEpilines(triplet_q, 2, F_q2_mat, line_on_2_from_q);
//                            functions::drawLines(image_1, image_q, line_on_1_from_q, line_on_q_from_1, triplet_1,
//                                                 triplet_q,
//                                                 colors);
//                            functions::drawLines(image_2, image_q, line_on_2_from_q, line_on_q_from_2, triplet_2,
//                                                 triplet_q,
//                                                 colors);
//                            cv::imshow("Query", image_q);
//                            cv::waitKey();
//                            cv::imshow("Image 1", image_1);
//                            cv::waitKey();
//                            cv::imshow("Image 2", image_2);
//                            cv::waitKey();
                    } else {
                        CostFunction *cost_function = ReprojectionError2D::Create(E_q1, E_q2,
                                                                                  triplets_1[c],
                                                                                  triplets_2[c],
                                                                                  triplets_q[c]);
                        problem.AddResidualBlock(cost_function, loss_function, K);
                        total_reprojection_error += error;
                        total_triplets++;
                    }
                }
                if (display_triplets) {

                    cv::Mat image_1 = cv::imread(im_1 + ".color.png");
                    cv::Mat image_2 = cv::imread(im_2 + ".color.png");
                    cv::Mat image_q = cv::imread(query + ".color.png");

                    vector<cv::Scalar> colors;
                    colors.reserve(triplets_q.size());
                    for (int i = 0; i < triplets_q.size(); i++) {
                        colors.emplace_back(
                                (double) std::rand() / RAND_MAX * 255,
                                (double) std::rand() / RAND_MAX * 255,
                                (double) std::rand() / RAND_MAX * 255
                        );
                    }

                    vector<cv::Point3d> lines_on_1_from_q, lines_on_2_from_q;
                    cv::computeCorrespondEpilines(triplets_q, 2, F_q1_mat, lines_on_1_from_q);
                    cv::computeCorrespondEpilines(triplets_q, 2, F_q2_mat, lines_on_2_from_q);

                    functions::drawLines(image_1, image_q, lines_on_1_from_q, lines_on_q_from_1, triplets_1, triplets_q,
                                         colors);
                    functions::drawLines(image_2, image_q, lines_on_2_from_q, lines_on_q_from_2, triplets_2, triplets_q,
                                         colors);
                    cv::imshow("Query", image_q);
                    cv::waitKey();
                    cv::imshow("Image 1", image_1);
                    cv::waitKey();
                    cv::imshow("Image 2", image_2);
                    cv::waitKey();
                }
            }
        }

        int temp = new_percent;
        new_percent = int(100. * (double(q + 1) / double(listQuery.size())));
        old_percent = temp;
        if (new_percent != old_percent) {
            cout << "\r" << "[";
            for (int line = 0; line < new_percent; line++) { cout << "|"; }
            for (int space = 0; space < 100 - new_percent; space++) { cout << " "; }
            cout << "] ";
            if (total_triplets > 0) {
                cout << "Average reprojection error: " << total_reprojection_error / total_triplets;
            }
        }
    }

    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    cout << endl << K[0] << ", " << K[1] << ", " << K[2] << ", " << K[3] << endl;
    Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    cout << K[0] << ", " << K[1] << ", " << K[2] << ", " << K[3] << endl;
}

void calibrate::run() {


    int scene = 0;
    string first = "chess/seq-03/frame-000000";
    string second = "chess/seq-04/frame-000629";
    double K[4] = {523.538, 529.669, 314.245, 237.595};

//    int scene = 1;
//    string first = "fire/seq-02/frame-000011";
//    string second = "fire/seq-01/frame-000309";
//    double K[4] = {520.208, 525.499, 314.99, 240.479};

//    int scene = 2;
//    string first = "heads/seq-02/frame-000369";
//    string second = "heads/seq-02/frame-000866";
//    double K[4] = {535.661, 530.986, 310.016, 237.958};

//    int scene = 3;
//    string first = "office/seq-04/frame-000005";
//    string second = "office/seq-08/frame-000956";
//    double K[4] = {524.264, 520.141, 314.283, 241.606};

//    int scene = 4;
//    string first = "pumpkin/seq-08/frame-000469";
//    string second = "pumpkin/seq-06/frame-000454";
//    double K[4] = {528.865, 525.468, 318.446, 237.14};

//    int scene = 5;
//    string first = "redkitchen/seq-07/frame-000069";
//    string second = "redkitchen/seq-02/frame-000201";
//    double K[4] = {516.217, 523.613, 314.216, 238.271};

//    int scene = 6;
//    string first = "stairs/seq-03/frame-000072";
//    string second = "stairs/seq-03/frame-000049";
//    double K[4] = {510.482, 542.358, 311.003, 235.719};

//    calibrate::calibrate(K, scene, false);

    string im1 = FOLDER + first;
    string im2 = FOLDER + second;
    cout << "Testing reprojection for " << first << " to " << second << endl;
    Eigen::Vector3d c1 = sevenScenes::getT(im1);
    Eigen::Vector3d c2 = sevenScenes::getT(im2);
    double separation = functions::getDistBetween(c1, c2);
    cout << "Separation: " << separation << " meters." << endl;


    //// GT GEOMETRY -------------------------------------------------------------------------------
    Eigen::Matrix3d R_1, R_2, R_real;
    Eigen::Vector3d t_1, t_2, t_real;
    sevenScenes::getAbsolutePose(im1, R_1, t_1);
    sevenScenes::getAbsolutePose(im2, R_2, t_2);

    R_real = R_2 * R_1.transpose();
    t_real = t_2 - R_real * t_1;

    Eigen::Matrix3d t_cross {
            {0,               -t_real(2), t_real(1)},
            {t_real(2),  0,              -t_real(0)},
            {-t_real(1), t_real(0),               0}};
    Eigen::Matrix3d E_12 = t_cross * R_real;

    Eigen::Matrix3d K_rgb_eig{{K[0], 0.,   K[2]},
                              {0.,   K[1], K[3]},
                              {0.,   0.,   1.}};
    cv::Mat K_rgb; cv::eigen2cv(K_rgb_eig, K_rgb);

    Eigen::Matrix3d F_gt_eig = K_rgb_eig.inverse().transpose() * E_12 * K_rgb_eig.inverse();
    cv::Mat F_gt; cv::eigen2cv(F_gt_eig, F_gt);
    //// GT GEOMETRY -------------------------------------------------------------------------------




    //// 5-point GEOMETRY -------------------------------------------------------------------------------
    vector<cv::Point2d> pts1, pts2;
    functions::findMatches(im1, ".pose.png",im2, "SIFT", 0.8, pts1, pts2);

    cv::Mat mask1;
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K_rgb,
                                     cv::RANSAC,0.999999, 3.0, mask1);

    vector<cv::Point2d> essential_inliers1, essential_inliers2;
    for (int i = 0; i < pts1.size(); i++) {
        if (mask1.at<unsigned char>(i)) {
            essential_inliers1.push_back(pts1[i]);
            essential_inliers2.push_back(pts2[i]);
        }
    }

    cv::Mat mask2;
    cv::Mat R_5pt, t_5pt;
    cv::recoverPose(E, essential_inliers1, essential_inliers2, K_rgb,
                    R_5pt, t_5pt, mask2);
    Eigen::Matrix3d R_5pt_eig; cv::cv2eigen(R_5pt, R_5pt_eig);
    Eigen::Vector3d t_5pt_eig; cv::cv2eigen(t_5pt, t_5pt_eig);

    vector<cv::Point2d> recover_inliers1, recover_inliers2;
    for (int i = 0; i < essential_inliers1.size(); i++) {
        if (mask2.at<unsigned char>(i)) {
            recover_inliers1.push_back(essential_inliers1[i]);
            recover_inliers2.push_back(essential_inliers2[i]);
        }
    }

    cv::Mat t_5pt_skew = (cv::Mat_<double>(3, 3) <<
                                                 0., -t_5pt.at<double>(2), t_5pt.at<double>(1),
            t_5pt.at<double>(2), 0, -t_5pt.at<double>(0),
            -t_5pt.at<double>(1), t_5pt.at<double>(0), 0);
    cv::Mat F_5pt = K_rgb.inv().t() * t_5pt_skew * R_5pt * K_rgb.inv();
    //// 5-point GEOMETRY DONE-------------------------------------------------------------------------------


    //// DISPLAY ITEMS---------------------------------------------------------------------------------------
    cv::Mat mask3;
    vector<cv::Point2d> points_display_1;
    vector<cv::Point2d> points_display_2;
    cv::findFundamentalMat(pts1, pts2,
                           cv::FM_RANSAC,3.0,0.999999, mask3);
    for (int i = 0; i < pts1.size(); i++) {
        if (mask3.at<unsigned char>(i)){
            points_display_1.push_back(pts1[i]);
            points_display_2.push_back(pts2[i]);
        }
    }

    cout << "Using " << points_display_1.size() << " test points." << endl;

    vector<cv::Scalar> colors;
    colors.reserve(points_display_1.size());
    for (int i = 0; i < points_display_1.size(); i++) {
        colors.emplace_back((double) std::rand() / RAND_MAX * 255,
                            (double) std::rand() / RAND_MAX * 255,
                            (double) std::rand() / RAND_MAX * 255);
    }
    //// DISPLAY ITEMS END-----------------------------------------------------------------------------------




    //// GR DRAW -------------------------------------------------------------------------------
    vector<cv::Point3d> lines_gt1, lines_gt2;
    cv::computeCorrespondEpilines(points_display_1, 1, F_gt, lines_gt1);
    cv::computeCorrespondEpilines(points_display_2, 2, F_gt, lines_gt2);

    cv::Mat image1 = cv::imread(im1 + ".color.png");
    cv::Mat image2 = cv::imread(im2 + ".color.png");
    double total_error = functions::drawLines(image1, image2, lines_gt2, lines_gt1,
                                          points_display_1, points_display_2, colors);
    cv::imshow(first + " Ground Truth", image1);
    cv::waitKey();
    cv::imshow(second + " Ground Truth", image2);
    cv::waitKey();
    cout << "Average reprojection error for ground truth pose: " << total_error << endl << endl;
    //// GR DRAW DONE-------------------------------------------------------------------------------



    //// 5-point DRAW -------------------------------------------------------------------------------
    vector<cv::Point3d> lines_5pt1, lines_5pt2;
    cv::computeCorrespondEpilines(points_display_1, 1, F_5pt, lines_5pt1);
    cv::computeCorrespondEpilines(points_display_2, 2, F_5pt, lines_5pt2);

    image1 = cv::imread(im1 + ".color.png");
    image2 = cv::imread(im2 + ".color.png");
    total_error = functions::drawLines(image1, image2, lines_5pt2, lines_5pt1, points_display_1,
                                   points_display_2, colors);
    cv::imshow(first + " 5-point", image1);
    cv::waitKey();
    cv::imshow(second + " 5-point", image2);
    cv::waitKey();
    cout << "Number of matched points: " << pts1.size() << endl;
    cout << "Number of 5-point inliers: " << recover_inliers1.size() << endl;
    cout << "Average reprojection error for 5-point pose: " << total_error << endl;
    double r_error = functions::rotationDifference(R_5pt_eig, R_real);
    double t_error = functions::getAngleBetween(t_5pt_eig, t_real);
    cout << "5-point R error: " << r_error << endl;
    cout << "5-point T error: " << t_error << endl << endl;
    //// 5-point DRAW DONE-------------------------------------------------------------------------------



    //// P3P GEOMETRY -------------------------------------------------------------------------------
    vector<cv::Point3d> pts1_3d;
    vector<cv::Point2d> pts2_2d;
    cv::Mat depth = cv::imread(im1 + ".depth.png");
    for (int k = 0; k < pts1.size(); k++) {
        cv::Point3d pt1_3d;
        if (sevenScenes::get3dfrom2d(pts1[k], depth, pt1_3d)) {
            pts1_3d.push_back(pt1_3d);
            pts2_2d.push_back(pts2[k]);
        }
    }

    cv::Mat K_ir = (cv::Mat_<double>(3, 3) <<
                                           585., 0., 320.,
            0., 585., 240.,
            0., 0., 1.);

    cv::Mat mask4, distCoeffs, R_ir, Rvec, t_ir;
    cv::solvePnPRansac(pts1_3d, pts2_2d, K_ir,
                       distCoeffs, Rvec, t_ir,mask4);
    cv::Rodrigues(Rvec, R_ir);

    int inlier_points = 0;
    for (int i = 0; i < mask4.rows; i++) {
        if (mask4.at<unsigned char>(i)) {
            inlier_points++;
        }
    }
    cv::Mat t_ir_skew = (cv::Mat_<double>(3, 3) <<
                                                0., -t_ir.at<double>(2), t_ir.at<double>(1),
            t_ir.at<double>(2), 0, -t_ir.at<double>(0),
            -t_ir.at<double>(1), t_ir.at<double>(0), 0);

    cv::Mat F_ir = K_ir.inv().t() * t_ir_skew * R_ir * K_ir.inv();
    //// P3P GEOMETRY DONE-------------------------------------------------------------------------------



    //// P3P DRAW -------------------------------------------------------------------------------
    vector<cv::Point3d> lines_ir1, lines_ir2;
    cv::computeCorrespondEpilines(points_display_1, 1, F_ir, lines_ir1);
    cv::computeCorrespondEpilines(points_display_2, 2, F_ir, lines_ir2);

    image1 = cv::imread(im1 + ".color.png");
    image2 = cv::imread(im2 + ".color.png");
    total_error = functions::drawLines(image1, image2, lines_ir2, lines_ir1, points_display_1, points_display_2,
                                   colors);
    cv::imshow(first + " P3P", image1);
    cv::waitKey();
    cv::imshow(second + " P3P", image2);
    cv::waitKey();
    cout << "Number of P3PRansac inliers: " << inlier_points << endl;
    cout << "Average reprojection error for P3P pose: " << total_error << endl;

    Eigen::Matrix3d R_p3p; cv::cv2eigen(R_ir, R_p3p);
    Eigen::Vector3d t_p3p; cv::cv2eigen(t_ir, t_p3p);
    double r_error_p3p = functions::rotationDifference(R_p3p, R_real);
    double t_error_p3p = functions::getAngleBetween(t_p3p, t_real);

    cout << "P3P R error: " << r_error_p3p << endl;
    cout << "P3P T error: " << t_error_p3p << endl;
    //// P3P DRAW DONE-------------------------------------------------------------------------------
}

