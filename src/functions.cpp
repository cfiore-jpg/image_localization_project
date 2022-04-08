//
// Created by Cameron Fiore on 12/15/21.
//

#include "../include/imageMatcher_Orb.h"
#include "../include/sevenScenes.h"
#include "../include/functions.h"
#include "../include/Space.h"
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <algorithm>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "5point.h"
#include "defines.h"
#include "fmatrix.h"
#include "poly1.h"
#include "poly3.h"
#include "matrix.h"
#include "qsort.h"
#include "svd.h"
#include "triangulate.h"
#include "vector.h"

#define PI 3.1415926536
#define FOLDER "/Users/cameronfiore/C++/image_localization_project/data/"

using namespace std;
using namespace cv;

Eigen::Matrix3d functions::smallRandomRotationMatrix() {

        random_device generator;
        uniform_real_distribution<double> uniform (0., 2*PI);
        normal_distribution<double> gaussian (0., PI/4.);

        double alpha = uniform(generator);
        double beta = uniform(generator);
        double gamma = gaussian(generator);

        double z = sin(alpha);
        double h = abs(cos(alpha));

        double x = h * cos(beta);
        double y = h * sin(beta);

        Eigen::Vector3d vec {x, y, z};
        vec.normalize();

        double n = vec.norm();

        Eigen::AngleAxis<double> aa (gamma, vec);

        Eigen::Matrix3d r = aa.toRotationMatrix();

        double d = r.determinant();

        return r;
}

void functions::compute_pose_from_points(const vector<cv::Point2d> & pts_l, const vector<cv::Point2d> & pts_r,
                                         const Eigen::Matrix3d & K, Eigen::Matrix3d & R, Eigen::Vector3d & T) {

    v2_t pts1[pts_l.size()];
    v2_t pts2[pts_r.size()];

    //Transfer types for bundler
    for (int i = 0; i < pts_l.size(); i++)
    {
        v2_t p1;
        p1.p[0] = pts_r[i].x;
        p1.p[1] = pts_r[i].y;
        pts1[i] = p1;

        v2_t p2;
        p2.p[0] = pts_l[i].x;
        p2.p[1] = pts_l[i].y;
        pts2[i] = p2;
    }

    double Rout_qk[9];
    double Rout_kq[9];
    double Tout_qk[3];
    double Tout_kq[3];

    double bundler_K[9] = {K(0,0), 0., K(0,2),
                           0., K(1,1), K(1,2),
                           0., 0., 1.};

    int numInliers_qk = compute_pose_ransac(int (pts_l.size()),pts1,pts2,bundler_K,bundler_K,3.,5000,Rout_qk,Tout_qk);
    int numInliers_kq = compute_pose_ransac(int (pts_l.size()),pts2,pts1,bundler_K,bundler_K,3.,5000,Rout_kq,Tout_kq);

    Eigen::Matrix3d R_qk_calc_bun, R_kq_calc_bun;
    Eigen::Vector3d t_qk_calc_bun, t_kq_calc_bun;

    R_qk_calc_bun(0,0) = Rout_qk[0];
    R_qk_calc_bun(0,1) = Rout_qk[1];
    R_qk_calc_bun(0,2) = -Rout_qk[2];
    R_qk_calc_bun(1,0) = Rout_qk[3];
    R_qk_calc_bun(1,1) = Rout_qk[4];
    R_qk_calc_bun(1,2) = -Rout_qk[5];
    R_qk_calc_bun(2,0) = -Rout_qk[6];
    R_qk_calc_bun(2,1) = -Rout_qk[7];
    R_qk_calc_bun(2,2) = Rout_qk[8];

    t_qk_calc_bun(0) = Tout_qk[0];
    t_qk_calc_bun(1) = Tout_qk[1];
    t_qk_calc_bun(2) = -Tout_qk[2];

    R_kq_calc_bun(0,0) = Rout_kq[0];
    R_kq_calc_bun(0,1) = Rout_kq[1];
    R_kq_calc_bun(0,2) = -Rout_kq[2];
    R_kq_calc_bun(1,0) = Rout_kq[3];
    R_kq_calc_bun(1,1) = Rout_kq[4];
    R_kq_calc_bun(1,2) = -Rout_kq[5];
    R_kq_calc_bun(2,0) = -Rout_kq[6];
    R_kq_calc_bun(2,1) = -Rout_kq[7];
    R_kq_calc_bun(2,2) = Rout_kq[8];

    t_kq_calc_bun(0) = Tout_kq[0];
    t_kq_calc_bun(1) = Tout_kq[1];
    t_kq_calc_bun(2) = -Tout_kq[2];

    auto inlier_pts = functions::same_inliers(pts_l, pts_r, R_qk_calc_bun, t_qk_calc_bun, R_kq_calc_bun, t_kq_calc_bun, K);



    int n = int (inlier_pts.first.size());
    v2_t inlier_pts_r[inlier_pts.first.size()], inlier_pts_l[inlier_pts.first.size()];

    //Transfer types for bundler
    for (int i = 0; i < n; i++)
    {
        v2_t pr;
        pr.p[0] = inlier_pts.second[i].x;
        pr.p[1] = inlier_pts.second[i].y;
        inlier_pts_r[i] = pr;

        v2_t pl;
        pl.p[0] = inlier_pts.first[i].x;
        pl.p[1] = inlier_pts.first[i].y;
        inlier_pts_l[i] = pl;
    }

    v2_t r_pts_norm[inlier_pts.first.size()], l_pts_norm[inlier_pts.first.size()];

    double K_inv[9];
    matrix_invert(3, bundler_K, K_inv);

    for (int i = 0; i < n; i++) {
        double r[3] = { Vx(inlier_pts_r[i]), Vy(inlier_pts_r[i]), 1.0 };
        double l[3] = { Vx(inlier_pts_l[i]), Vy(inlier_pts_l[i]), 1.0 };

        double r_norm[3], l_norm[3];

        matrix_product331(K_inv, r, r_norm);
        matrix_product331(K_inv, l, l_norm);

        r_pts_norm[i] = v2_new(-r_norm[0], -r_norm[1]);
        l_pts_norm[i] = v2_new(-l_norm[0], -l_norm[1]);

        cout << "(" << r_pts_norm[i].p[0] << ", " << r_pts_norm[i].p[1] << "), (" << l_pts_norm[i].p[0] << ", " << l_pts_norm[i].p[1] << ")" << endl;
    }

    int num_hyp;
    double E[90];
    generate_Ematrix_hypotheses(n, r_pts_norm, l_pts_norm, &num_hyp, E);

//    for (int i = 0; i < num_hyp; i++) {
//        for (int j = 0; j < 3; j++) {
//                cout << E[9*i+3*j] << ", " << E[9*i+3*j+1] << ", " << E[9*i+3*j+2] << endl;
//        }
//        cout << endl;
//    }

    int max_inliers = 0;
    double min_score = DBL_MAX;
    double E_best[9];
    for (int i = 0; i < num_hyp; i++) {
        int best_inlier;
        double score = 0.0;

        double E2[9], tmp[9], F[9];
        memcpy(E2, E + 9 * i, 9 * sizeof(double));
        E2[0] = -E2[0];
        E2[1] = -E2[1];
        E2[3] = -E2[3];
        E2[4] = -E2[4];
        E2[8] = -E2[8];

        matrix_transpose_product(3, 3, 3, 3, K_inv, E2, tmp);
        matrix_product(3, 3, 3, 3, tmp, K_inv, F);

        int inliers = evaluate_Ematrix(n, inlier_pts_r, inlier_pts_l, // r_pts_norm, l_pts_norm,
                                   9.0, F, // E + 9 * i,
                                   &best_inlier, &score);

        if (inliers > max_inliers ||
            (inliers == max_inliers && score < min_score)) {
            max_inliers = inliers;
            min_score = score;
            memcpy(E_best, E + 9 * i, sizeof(double) * 9);
        }
    }

    if (max_inliers > 0) {
        int best_inlier;
        double score;
        // find_extrinsics_essential(E_best, r_best, l_best, R_out, t_out);
        double Rout[9];
        double Tout[3];
        int success = find_extrinsics_essential_multipt(E_best, n,
                                                  r_pts_norm, l_pts_norm,
                                                  Rout, Tout);
        R(0,0) = Rout[0];
        R(0,1) = Rout[1];
        R(0,2) = -Rout[2];
        R(1,0) = Rout[3];
        R(1,1) = Rout[4];
        R(1,2) = -Rout[5];
        R(2,0) = -Rout[6];
        R(2,1) = -Rout[7];
        R(2,2) = Rout[8];

        T(0) = Tout[0];
        T(1) = Tout[1];
        T(2) = -Tout[2];

    }


    int x = 0;


}

pair<vector<cv::Point2d>, vector<cv::Point2d>> functions::same_inliers(
        const vector<cv::Point2d> & pts_q, const vector<cv::Point2d> & pts_db,
        const Eigen::Matrix3d & R_qk, const Eigen::Vector3d & T_qk,
        const Eigen::Matrix3d & R_kq, const Eigen::Vector3d & T_kq,
        const Eigen::Matrix3d & K) {

    vector<cv::Point2d> inliers_q, inliers_db;

    vector<cv::Point2d> pts_q_inliers_qk;
    vector<cv::Point2d> pts_db_inliers_qk;

    Eigen::Matrix3d t_qk_cross{
            {0,        -T_qk(2), T_qk(1)},
            {T_qk(2),  0,        -T_qk(0)},
            {-T_qk(1), T_qk(0),  0}};

    Eigen::Matrix3d E_qk = t_qk_cross * R_qk;

    for (int i = 0; i < pts_q.size(); i++) {

        Eigen::Vector3d pt_q{pts_q[i].x, pts_q[i].y, 1.};
        Eigen::Vector3d pt_db{pts_db[i].x, pts_db[i].y, 1.};

        Eigen::Vector3d epiline = K.inverse().transpose() * E_qk * K.inverse() * pt_db;

        double dist = abs(epiline(0) * pt_q(0) + epiline(1) * pt_q(1) + epiline(2)) /
                      sqrt(epiline(0) * epiline(0) + epiline(1) * epiline(1));

        if (dist <= 3.0) {
            pts_q_inliers_qk.push_back(pts_q[i]);
            pts_db_inliers_qk.push_back(pts_db[i]);
        }
    }

    vector<cv::Point2d> pts_q_inliers_kq;
    vector<cv::Point2d> pts_db_inliers_kq;

    Eigen::Matrix3d t_kq_cross{
            {0,        -T_kq(2), T_kq(1)},
            {T_kq(2),  0,        -T_kq(0)},
            {-T_kq(1), T_kq(0),  0}};

    Eigen::Matrix3d E_kq = t_kq_cross * R_kq;

    for (int i = 0; i < pts_q.size(); i++) {

        Eigen::Vector3d pt_q{pts_q[i].x, pts_q[i].y, 1.};
        Eigen::Vector3d pt_db{pts_db[i].x, pts_db[i].y, 1.};

        Eigen::Vector3d epiline = K.inverse().transpose() * E_kq * K.inverse() * pt_q;

        double dist = abs(epiline(0) * pt_db(0) + epiline(1) * pt_db(1) + epiline(2)) /
                      sqrt(epiline(0) * epiline(0) + epiline(1) * epiline(1));

        if (dist <= 3.0) {
            pts_q_inliers_kq.push_back(pts_q[i]);
            pts_db_inliers_kq.push_back(pts_db[i]);
        }
    }

    vector<cv::Point2d> pts_q_inliers_kq_diff = pts_q_inliers_kq;
    vector<cv::Point2d> pts_q_inliers_qk_diff;

    for (int i = 0; i < pts_q_inliers_qk.size(); i++) {
        bool matched = false;
        for (int j = 0; j < pts_q_inliers_kq_diff.size(); j++) {
            double dist = sqrt(pow((pts_q_inliers_qk[i].x - pts_q_inliers_kq_diff[j].x), 2.)
                               + pow((pts_q_inliers_qk[i].y - pts_q_inliers_kq_diff[j].y), 2.));
            if (dist <= 0.000001) {
                pts_q_inliers_kq_diff.erase(pts_q_inliers_kq_diff.begin() + j);
                inliers_q.push_back(pts_q_inliers_qk[i]);
                matched = true;
                break;
            }
        }
        if (!matched) {
            pts_q_inliers_qk_diff.push_back(pts_q_inliers_qk[i]);
        }
    }

    vector<cv::Point2d> pts_db_inliers_kq_diff = pts_db_inliers_kq;
    vector<cv::Point2d> pts_db_inliers_qk_diff;

    for (int i = 0; i < pts_db_inliers_qk.size(); i++) {
        bool matched = false;
        for (int j = 0; j < pts_db_inliers_kq_diff.size(); j++) {
            double dist = sqrt(pow((pts_db_inliers_qk[i].x - pts_db_inliers_kq_diff[j].x), 2.)
                               + pow((pts_db_inliers_qk[i].y - pts_db_inliers_kq_diff[j].y), 2.));
            if (dist <= 0.000001) {
                pts_db_inliers_kq_diff.erase(pts_db_inliers_kq_diff.begin() + j);
                inliers_db.push_back(pts_db_inliers_qk[i]);
                matched = true;
                break;
            }
        }
        if (!matched) {
            pts_db_inliers_qk_diff.push_back(pts_db_inliers_qk[i]);
        }
    }
    int x = 0;
    return {inliers_q, inliers_db};
}

double functions::triangulateRays(const Eigen::Vector3d &ci, const Eigen::Vector3d &di, const Eigen::Vector3d &cj,
                                   const Eigen::Vector3d &dj, Eigen::Vector3d &intersect) {
    Eigen::Vector3d dq = di.cross(dj);
    Eigen::Matrix3d D;
    D.col(0) = di;
    D.col(1) = -dj;
    D.col(2) = dq;
    Eigen::Vector3d b = cj - ci;
    Eigen::Vector3d sol = D.colPivHouseholderQr().solve(b);
    intersect = ci + sol(0) * di + (sol(2) / 2) * dq;
    Eigen::Vector3d toReturn = sol(2) * dq;
    return toReturn.norm();
}

double functions::getDistBetween(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2) {
    double x = p1(0) - p2(0);
    double y = p1(1) - p2(1);
    double z = p1(2) - p2(2);
    return sqrt(pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0));
}

double functions::getPoint3DLineDist(const Eigen::Vector3d & h, const Eigen::Vector3d & c_k, const Eigen::Vector3d & v_k) {
    double numerator = (h - c_k).cross(h - (c_k + v_k)).norm();
    double denominator = v_k.norm();
    return numerator/denominator;
}

double functions::getAngleBetween(const Eigen::Vector3d &d1, const Eigen::Vector3d &d2) {
    double dot = d1.dot(d2);
    double d1_norm = d1.norm();
    double d2_norm = d2.norm();
    double frac = dot / (d1_norm * d2_norm);
    if ((frac - 1.) > 0. && (frac - 1.) < 0.0000000000000010) frac = 1.;
    double rad = acos(frac);
    return rad * 180.0 / PI;
}

double functions::rotationDifference(const Eigen::Matrix3d & r1, const Eigen::Matrix3d & r2) {

    Eigen::Matrix3d R12 = r1 * r2.transpose();
    double trace = R12.trace();
    if (trace > 3.) trace = 3.;
    double theta = acos((trace-1.)/2.);
    theta = theta * 180. / PI;
    return theta;
}

vector<double> functions::getReprojectionErrors(const vector<cv::Point2d> & pts_to, const vector<cv::Point2d> & pts_from,
                                     const Eigen::Matrix3d & K, const Eigen::Matrix3d & R, const Eigen::Vector3d & T) {
    vector<double> to_return(301);

    Eigen::Vector3d T_normalized = T.normalized();
    for (int i = 0; i < pts_from.size(); i++) {

        Eigen::Vector3d pt_from {pts_from[i].x, pts_from[i].y, 1.};
        Eigen::Vector3d pt_to {pts_to[i].x, pts_to[i].y, 1.};

        Eigen::Matrix3d T_cross {
                {0,               -T_normalized(2), T_normalized(1)},
                {T_normalized(2),  0,              -T_normalized(0)},
                {-T_normalized(1), T_normalized(0),               0}};
        Eigen::Matrix3d E = T_cross * R;

        Eigen::Vector3d ep_line = K.inverse().transpose() * E * K.inverse() * pt_from;

        double error = abs(ep_line(0) * pt_to(0) + ep_line(1) * pt_to(1) + ep_line(2)) /
                sqrt(ep_line(0) * ep_line(0) + ep_line(1) * ep_line(1));

        int idx = int (round(error*100));

        if (idx < 301) to_return[idx]++;
    }
    return to_return;
}

bool functions::findMatches(const string &db_image, const string & ext, const string &query_image, const string &method, double ratio,
                             vector<Point2d> &pts_db, vector<Point2d> &pts_query) {


    vector<KeyPoint> kp_vec1, kp_vec2;
    Mat desc1, desc2;
    if (method == "ORB") {
        Ptr<ORB> orb = ORB::create();
        cv::UMat image1, image2, gray1, gray2;
        imread(query_image + ext, IMREAD_COLOR).copyTo(image1);
        cvtColor(image1, gray1, COLOR_RGB2GRAY);
        imread(db_image + ext, IMREAD_COLOR).copyTo(image2);
        cvtColor(image2, gray2, COLOR_RGB2GRAY);
        orb->detectAndCompute(gray1, Mat(), kp_vec1, desc1);
        orb->detectAndCompute(gray2, Mat(), kp_vec2, desc2);
    } else if (method == "SURF") {
        Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(400, 4, 2, false);
        cv::UMat image1, image2, gray1, gray2;
        imread(query_image + ext, IMREAD_COLOR).copyTo(image1);
        cvtColor(image1, gray1, COLOR_RGB2GRAY);
        imread(db_image + ext, IMREAD_COLOR).copyTo(image2);
        cvtColor(image2, gray2, COLOR_RGB2GRAY);
        surf->detectAndCompute(image1, Mat(), kp_vec1, desc1);
        surf->detectAndCompute(image2, Mat(), kp_vec2, desc2);
    } else if (method == "SIFT") {
        Ptr<SIFT> sift = SIFT::create();

        Mat image1 = imread(query_image + ext);
        Mat image2 = imread(db_image + ext);
        sift->detectAndCompute(image1, Mat(), kp_vec1, desc1);
        sift->detectAndCompute(image2, Mat(), kp_vec2, desc2);

//        cv::Mat kp_im1;
//        cv::drawKeypoints(image1, kp_vec1, kp_im1);
//        cv::imshow("Query Keypoints", kp_im1);
//        cv::waitKey(0);
//
//        cv::Mat kp_im2;
//        cv::drawKeypoints(image2, kp_vec2, kp_im2);
//        cv::imshow("Database Keypoints", kp_im2);
//        cv::waitKey(0);
    } else {
        cout << "Not a valid method for feature matching..." << endl;
        exit(1);
    }

    auto matcher = BFMatcher::create(NORM_L2, false);
    vector< vector<DMatch> > matches_2nn_12, matches_2nn_21;
    matcher->knnMatch( desc1, desc2, matches_2nn_12, 2 );
    matcher->knnMatch( desc2, desc1, matches_2nn_21, 2 );
    vector<Point2d> selected_points1, selected_points2;

    for (const auto & m : matches_2nn_12) { // i is queryIdx
        if( m[0].distance/m[1].distance < ratio
            and
            matches_2nn_21[m[0].trainIdx][0].distance
            / matches_2nn_21[m[0].trainIdx][1].distance < ratio )
        {
            if(matches_2nn_21[m[0].trainIdx][0].trainIdx
               == m[0].queryIdx)
            {
                selected_points1.push_back(kp_vec1[m[0].queryIdx].pt);
                selected_points2.push_back(kp_vec2[matches_2nn_21[m[0].trainIdx][0].queryIdx].pt);
            }
        }
    }

//    cv::Mat src;
//    Mat image1 = imread(query_image + ext);
//    Mat image2 = imread(db_image + ext);
//    cv::hconcat(image1, image2, src);
//    for(int i = 0; i < selected_points1.size(); i++) {
//        cv::line( src, selected_points1[i],
//                  cv::Point2d(selected_points2[i].x + image1.cols, selected_points2[i].y),
//                  1, 1, 0 );
//    }
//    cv::imshow("Match Results", src);
//    cv::waitKey(0);

    pts_query = selected_points1;
    pts_db = selected_points2;

    if (pts_db.empty()) {
        return false;
    } else {
        return true;
    }
}

bool functions::findMatchesSorted(const string &db_image, const string & ext, const string &query_image, const string &method, double ratio,
                                  vector<tuple<Point2d, Point2d, double>> & points) {


    vector<KeyPoint> kp_vec1, kp_vec2;
    Mat desc1, desc2;
    if (method == "ORB") {
        Ptr<ORB> orb = ORB::create();
        cv::UMat image1, image2, gray1, gray2;
        imread(query_image + ext, IMREAD_COLOR).copyTo(image1);
        cvtColor(image1, gray1, COLOR_RGB2GRAY);
        imread(db_image + ext, IMREAD_COLOR).copyTo(image2);
        cvtColor(image2, gray2, COLOR_RGB2GRAY);
        orb->detectAndCompute(gray1, Mat(), kp_vec1, desc1);
        orb->detectAndCompute(gray2, Mat(), kp_vec2, desc2);
    } else if (method == "SURF") {
        Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(400, 4, 2, false);
        cv::UMat image1, image2, gray1, gray2;
        imread(query_image + ext, IMREAD_COLOR).copyTo(image1);
        cvtColor(image1, gray1, COLOR_RGB2GRAY);
        imread(db_image + ext, IMREAD_COLOR).copyTo(image2);
        cvtColor(image2, gray2, COLOR_RGB2GRAY);
        surf->detectAndCompute(image1, Mat(), kp_vec1, desc1);
        surf->detectAndCompute(image2, Mat(), kp_vec2, desc2);
    } else if (method == "SIFT") {
        Ptr<SIFT> sift = SIFT::create();
        Mat image1 = imread(query_image + ext);
        Mat image2 = imread(db_image + ext);
        sift->detectAndCompute(image1, Mat(), kp_vec1, desc1);
        sift->detectAndCompute(image2, Mat(), kp_vec2, desc2);
    } else {
        cout << "Not a valid method for feature matching..." << endl;
        exit(1);
    }

    auto matcher = BFMatcher::create(NORM_L2, false);
    vector< vector<DMatch> > matches_2nn_12, matches_2nn_21;
    matcher->knnMatch( desc1, desc2, matches_2nn_12, 2 );
    matcher->knnMatch( desc2, desc1, matches_2nn_21, 2 );
    vector<tuple<Point2d, Point2d, double>> selected_points; // query, db, match_ratio

    for (const auto & m : matches_2nn_12) {
        double r = m[0].distance/m[1].distance;
        if (r < ratio and matches_2nn_21[m[0].trainIdx][0].distance/matches_2nn_21[m[0].trainIdx][1].distance < ratio )
        {
            if(matches_2nn_21[m[0].trainIdx][0].trainIdx == m[0].queryIdx)
            {
                selected_points.emplace_back(kp_vec1[m[0].queryIdx].pt,
                                             kp_vec2[matches_2nn_21[m[0].trainIdx][0].queryIdx].pt,
                                             r);
            }
        }
    }

    if (selected_points.empty()) return false;

    sort(selected_points.begin(), selected_points.end(), [](const auto & lhs, const auto & rhs){
        return get<2>(lhs) < get<2>(rhs);
    });
    points = selected_points;
    return true;
}

vector<pair<cv::Point2d, cv::Point2d>> functions::findInliersForFundamental(const Eigen::Matrix3d & F, double threshold,
                                                                 const vector<tuple<cv::Point2d, cv::Point2d, double>> & points) {
    vector<pair<cv::Point2d, cv::Point2d>> inliers;
    for (const auto & tup : points) {
        Eigen::Vector3d pt_q {get<0>(tup).x, get<0>(tup).y, 1.};
        Eigen::Vector3d pt_db {get<1>(tup).x, get<1>(tup).y, 1.};

        Eigen::Vector3d epiline = F * pt_db;

        double error = abs(epiline[0] * pt_q[0] + epiline[1] * pt_q[1] + epiline[2]) /
                       sqrt(epiline[0] * epiline[0] + epiline[1] * epiline[1]);

        if (error <= threshold) {
            inliers.emplace_back(cv::Point2d {get<0>(tup).x, get<0>(tup).y},
                                 cv::Point2d {get<1>(tup).x, get<1>(tup).y});
        }
    }
    return inliers;
}

int functions::getRelativePose(const string & db_image, const string & ext, const string & query_image, const double * K, const string & method,
                               Eigen::Matrix3d &R_kq, Eigen::Vector3d &t_kq) {
    try {
        vector<Point2d> pts_db, pts_q;
        if (!findMatches(db_image, ext, query_image, method, 0.8, pts_db, pts_q)) return 0;

        Mat im = imread(db_image + ext);
        Mat mask;
        Mat K_mat = (Mat_<double>(3, 3) <<
                K[0], 0., K[2],
                0., K[1], K[3],
                0., 0.,   1.);
        Mat E_kq = findEssentialMat(pts_db,pts_q,K_mat, RANSAC, 0.999999, 3.0, mask);

        vector<Point2d> inlier_db_points, inlier_q_points;
        for (int i = 0; i < mask.rows; i++) {
            if (mask.at<unsigned char>(i)) {
                inlier_db_points.push_back(pts_db[i]);
                inlier_q_points.push_back(pts_q[i]);
            }
        }

        mask.release();
        Mat R, t;
        recoverPose(E_kq,inlier_db_points,inlier_q_points,K_mat, R, t, mask);

        vector<Point2d> recover_db_points, recover_q_points;
        for (int i = 0; i < mask.rows; i++) {
            if (mask.at<unsigned char>(i)) {
                recover_db_points.push_back(inlier_db_points[i]);
                recover_q_points.push_back(inlier_q_points[i]);
            }
        }

        cv2eigen(R, R_kq);
        cv2eigen(t, t_kq);

        return int (recover_db_points.size());
    } catch (...) {
        return 0;
    }
}

bool functions::getRelativePose(vector<cv::Point2d> & pts_db, vector<cv::Point2d> & pts_q, const double * K,
                               Eigen::Matrix3d &R_kq, Eigen::Vector3d &t_kq) {
    Mat mask;
    Mat K_mat = (Mat_<double>(3, 3) <<
                                    K[0], 0., K[2],
            0., K[1], K[3],
            0., 0., 1.);

    Mat E_qk_sols = findEssentialMat(pts_db, pts_q, K_mat, RANSAC, 0.9999999999999, 3.0, mask);

    Mat E_qk = E_qk_sols(cv::Range(0, 3), cv::Range(0, 3));

    vector<Point2d> inlier_db_points, inlier_q_points;
    for (int i = 0; i < mask.rows; i++) {
        if (mask.at<unsigned char>(i)) {
            inlier_db_points.push_back(pts_db[i]);
            inlier_q_points.push_back(pts_q[i]);
        }
    }


    Eigen::Matrix3d K_eig{{K[0], 0.,   K[2]},
                          {0.,   K[1], K[3]},
                          {0.,   0.,   1.}};

//    Eigen::Matrix3d E_eig;
//    cv::cv2eigen(E_qk, E_eig);
//
//    int num_points = 0;
//
//    for (int i = 0; i < inlier_db_points.size(); i++) {
//
//        Eigen::Vector3d pt_db{inlier_db_points[i].x, inlier_db_points[i].y, 1.};
//        Eigen::Vector3d pt_q{inlier_q_points[i].x, inlier_q_points[i].y, 1.};
//
//        double d = pt_q.transpose() * K_eig.inverse().transpose() * E_eig * K_eig.inverse() * pt_db;
//
//        if (abs(d) <= 0.00000000001) {
//            num_points++;
//        }
//    }

    mask.release();
    Mat R, t;
    recoverPose(E_qk, inlier_db_points, inlier_q_points, K_mat, R, t, mask);

    vector<Point2d> recover_db_points, recover_q_points;
    for (int i = 0; i < mask.rows; i++) {
        if (mask.at<unsigned char>(i)) {
            recover_db_points.push_back(inlier_db_points[i]);
            recover_q_points.push_back(inlier_q_points[i]);
        }
    }

    cv2eigen(R, R_kq);
    cv2eigen(t, t_kq);

    pts_db = recover_db_points;
    pts_q = recover_q_points;
    return true;
//    } catch (...) {
//        return false;
//    }
}

int functions::getRelativePose3D(const string &db_image, const string & ext, const string &query_image, const string &method,
                                Eigen::Matrix3d &R_kq, Eigen::Vector3d &t_kq) {
//    try {
    vector<Point2d> pts_db, pts_q;
    if (!findMatches(db_image, ext, query_image, method, 0.8, pts_db, pts_q)) { return 0; }

    vector<Point3d> pts_db_3d;
    vector<Point2d> pts_q_2d;
    Mat depth = imread(db_image + ".depth.png");
    for (int i = 0; i < pts_db.size(); i++) {
        Point3d pt_3d;
        if (sevenScenes::get3dfrom2d(pts_db[i], depth, pt_3d)) {
            pts_db_3d.push_back(pt_3d);
            pts_q_2d.push_back(pts_q[i]);
        }
    }

    Mat im = imread(db_image + ext);
    Mat mask;
    Mat K = (Mat_<double>(3, 3) <<
                                        585.6, 0., 316.,
                                        0., 585.6, 247.6,
                                        0., 0., 1.);
    Mat R, Rvec, t;

    Mat distCoeffs = Mat();
    solvePnPRansac(pts_db_3d, pts_q_2d, K,
                       distCoeffs, Rvec, t, true,
                       1000, 8.0, 0.999, mask);
    Rodrigues(Rvec, R);


    int inlier_points = 0;
    for (int i = 0; i < mask.rows; i++) {
        if (mask.at<unsigned char>(i)) {
            inlier_points++;
        }
    }

    cv2eigen(R, R_kq);
    cv2eigen(t, t_kq);

    return inlier_points;
//    } catch (...) {
//        return 0;
//    }
}

double functions::drawLines(Mat & im1, Mat & im2,
               const vector<Point3d> & lines_draw_on_1,
               const vector<Point3d> & lines_draw_on_2,
               const vector<Point2d> & pts1,
               const vector<Point2d> & pts2,
               const vector<Scalar> & colors) {
    int c = im1.cols;
    double total_rep_error = 0.;
    for (int i = 0; i < pts1.size(); i++) {
        Point3d l1 = lines_draw_on_1[i];
        Point3d l2 = lines_draw_on_2[i];

        Point2d pt1 = pts1[i];
        Point2d pt2 = pts2[i];
        Scalar color = colors[i];

        Point2d i1 (0., -l1.z/l1.y);
        Point2d j1 (c, -(l1.z+l1.x*c)/l1.y);
        double error = abs(l1.x*pt1.x + l1.y*pt1.y + l1.z) / sqrt(l1.x*l1.x + l1.y*l1.y);
        total_rep_error += error;

        Point2d i2 (0., -l2.z/l2.y);
        Point2d j2 (c, -(l2.z+l2.x*c)/l2.y);

        line(im1, i1, j1, color);
        line(im2, i2, j2, color);

        circle(im1, pt1, 5, color, -1);
        circle(im2, pt2, 5, color, -1);
    }
    return total_rep_error / double (pts1.size());
}

vector<string> functions::optimizeSpacing(const vector<string> & images, int N, bool show_process, const string & dataset) {
    Space space (images, dataset);
    space.getOptimalSpacing(N, show_process);
    vector<string> names = space.getPointNames();
    return names;
}

// DATABASE PREPARATION FUNCTIONS
void functions::createImageVector(vector<string> &listImage, vector<tuple<string, string, vector<string>, vector<string>>> &info, int scene)
{
    cout << "Creating image vector..." << endl;
    int begin;
    int end;
    if (scene == -1)
    {
        begin = 0;
        end = int (info.size());
    } else
    {
        begin = scene;
        end = scene + 1;
    }
    for (int i = begin; i < end; ++i)
    {
        string folder = get<0>(info[i]);
        string imageList = get<1>(info[i]);
        vector<string> train = get<2>(info[i]);
        for (auto & seq : train)
        {
            ifstream im_list(imageList);
            if (im_list.is_open())
            {
                string image;
                while (getline(im_list, image))
                {
                    string file = "seq-"; file.append(seq).append("/").append(image);
                    listImage.push_back(folder + file);
                }
                im_list.close();
            }
        }
    }
    cout << "done" << endl;
}

void functions::createQueryVector(vector<string> &listQuery, vector<tuple<string, string, vector<string>, vector<string>>> &info, int scene)
{
    cout << "Creating query vector..." << endl;
    int begin;
    int end;
    if (scene == -1)
    {
        begin = 0;
        end = int (info.size());
    } else
    {
        begin = scene;
        end = scene + 1;
    }
    for (int i = begin; i < end; ++i)
    {
        string folder = get<0>(info[i]);
        string imageList = get<1>(info[i]);
        vector<string> test = get<3>(info[i]);
        for (auto & seq : test) {
            ifstream im_list(imageList);
            if (im_list.is_open()) {
                string image;
                while (getline(im_list, image)) {
                    string file = "seq-"; file.append(seq).append("/").append(image);
                    listQuery.push_back(folder + file);
                }
                im_list.close();
            }
        }
    }
    cout << "done" << endl;
}



////SURF functions
//void functions::loadFeaturesSURF(const vector<string> &listImage, vector<vector<vector<float>>> &features)
//{
//    features.clear();
//    Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(400, 4, 2, false);
//
//    for (const string& im : listImage)
//    {
//        Mat image = imread(im + ext, IMREAD_GRAYSCALE);
//        if (image.empty())
//        {
//            cout << "The image " << im << "is empty" << endl;
//            exit(1);
//        }
//        Mat mask;
//        vector<KeyPoint> keyPoints;
//        vector<float> descriptors;
//
//        surf->detectAndCompute(image, mask, keyPoints, descriptors);
//        features.push_back(vector<vector<float>>());
//        changeStructureSURF(descriptors, features.back(), surf->descriptorSize());
//
//        cout << im << endl;
//    }
//    cout << "Done!" << endl;
//}
//
//void functions::changeStructureSURF(const vector<float> &plain, vector<vector<float>> &out, int L)
//{
//    out.resize(plain.size() / L);
//    unsigned int j = 0;
//
//    for(unsigned int i = 0; i < plain.size(); i += L, ++j)
//    {
//        out[j].resize(L);
//        std::copy(plain.begin() + i, plain.begin() + i + L, out[j].begin());
//    }
//}
//
//void functions::DatabaseSaveSURF(const vector<vector<vector<float>>> &features)
//{
//    Surf64Vocabulary voc;
//
//    cout << "Creating vocabulary..." << endl;
//    voc.create(features);
//    cout << "... done!" << endl;
//
//    cout << "Vocabulary information: " << endl << voc << endl;
//
//    cout << "Saving vocabulary..." << endl;
//    string vocFile = "voc.yml.gz";
//    voc.save(FOLDER + vocFile);
//    cout << "Done" << endl;
//    cout << "Creating database..." << endl;
//
//    Surf64Database db(voc, false, 0);
//
//    for(const auto & feature : features)
//    {
//        db.add(feature);
//    }
//
//    cout << "... done!" << endl;
//
//    cout << "Database information: " << endl << db << endl;
//
//    cout << "Saving database..." << endl;
//    string dbFile = "db.yml.gz";
//    db.save(FOLDER + dbFile);
//    cout << "... done!" << endl;
//
//}


//// ORB functions
//void functions::loadFeaturesORB(const vector<string> &listImage, vector<vector<Mat>> &features)
//{
//    features.clear();
//    Ptr<ORB> orb = ORB::create();
//
//    for (const string& im : listImage)
//    {
//        Mat image = imread(im + ext, IMREAD_GRAYSCALE);
//        if (image.empty())
//        {
//            cout << "The image " << im << "is empty" << endl;
//            exit(1);
//        }
//        Mat mask;
//        vector<KeyPoint> keyPoints;
//        Mat descriptors;
//
//        orb->detectAndCompute(image, mask, keyPoints, descriptors);
//        features.emplace_back();
//        changeStructureORB(descriptors, features.back());
//
//        cout << im << endl;
//    }
//    cout << "Done!" << endl;
//}
//
//void functions::changeStructureORB(const Mat &plain, vector<Mat> &out)
//{
//    out.resize(plain.rows);
//
//    for(int i = 0; i < plain.rows; ++i)
//    {
//        out[i] = plain.row(i);
//    }
//}
//
//void functions::DatabaseSaveORB(const vector<vector<Mat>> &features)
//{
//    OrbVocabulary voc;
//
//    cout << "Creating vocabulary..." << endl;
//    voc.create(features);
//    cout << "... done!" << endl;
//
//    cout << "Vocabulary information: " << endl << voc << endl;
//
//    cout << "Saving vocabulary..." << endl;
//    string vocFile = "voc_orb.yml.gz";
//    voc.save(FOLDER + vocFile);
//    cout << "Done" << endl;
//    cout << "Creating database..." << endl;
//
//    OrbDatabase db(voc, false, 0);
//
//    for(const auto & feature : features)
//    {
//        db.add(feature);
//    }
//
//    cout << "... done!" << endl;
//
//    cout << "Database information: " << endl << db << endl;
//
//    cout << "Saving database..." << endl;
//    string dbFile = "db_orb.yml.gz";
//    db.save(FOLDER + dbFile);
//    cout << "... done!" << endl;
//
//}

// surf
// orb
// optional _jts meaning "just this scene"
//void functions::saveResultVector(const vector<pair<int, float>>& result, const string& query_image, const string& method, const string & num)
//{
//    ofstream file (query_image + "." + method + ".top" + num + ".txt");
//    if(file.is_open())
//    {
//        for (const pair<int, float>& r : result)
//        {
//            file << to_string(r.first) + "  " + to_string(r.second) + "\n";
//        }
//        file.close();
//    }
//}
//
//vector<pair<int, float>> functions::getResultVector(const string& query_image, const string& method, const string & num)
//{
//    vector<pair<int, float>> result;
//    ifstream file (query_image + "." + method + ".top" + num + ".txt");
//    if(file.is_open())
//    {
//        string line;
//        while (getline(file, line))
//        {
//            stringstream ss (line);
//            string n;
//            vector<string> temp;
//            while(ss >> n)
//            {
//                temp.push_back(n);
//            }
//            result.emplace_back(stoi(temp[0]), stof(temp[1]));
//        }
//        file.close();
//    }
//    return result;
//}

string functions::getScene(const string & image) {
    string scene = image;
    scene.erase(0, scene.find("data/") + 5);
    scene = scene.substr(0, scene.find("/seq"));
    return scene;
}

string functions::getSequence(const string & image) {
    string seq = image;
    string scene = functions::getScene(image);
    seq.erase(0, seq.find(scene) + scene.size() + 5);
    seq = seq.substr(0, seq.find("/frame"));
    return seq;
}

vector<string> functions::getTopN(const string& query_image, const string & ext, int N) {
    vector<string> result;
    ifstream file (query_image + ext + ".1000nn.txt");
    if(file.is_open())
    {
        string line;
        int n = 0;
        while (n < N)
        {
            getline(file, line);
            stringstream ss (line);
            string im;
            string fn;
            if (ss >> im)
            {
                fn = im.substr(0, im.find(ext));
            }
            result.push_back(fn);
            n++;
        }
    }
    return result;
}

vector<string> functions::retrieveSimilar(const string& query_image, const string & ext, int max_num, double max_descriptor_dist) {
    int N = (max_num > 20) ? 20 : max_num;
    vector<string> similar;
    ifstream file(query_image + ext + ".1000nn.txt");
    if (file.is_open()) {
        string line;
        double descriptor_dist;
        string scene;
        unordered_map<string, int> mc;
        string most_common;
        int num_most_common = 0;
        while (similar.size() < max_num && getline(file, line)) {
            stringstream ss(line);
            string buf, fn;
            bool name = true;
            while (ss >> buf) {
                if (name) {
                    fn = buf.substr(0, buf.find(ext));
                    name = false;
                } else {
                    descriptor_dist = stod(buf);
                }
            }

            if (descriptor_dist > max_descriptor_dist) break;

            if (scene.empty()) {
                similar.push_back(fn);
                if (similar.size() <= N) {
                    string this_scene = getScene(fn);
                    if (mc.find(this_scene) == mc.end()) {
                        mc[this_scene] = 1;
                    } else {
                        mc[this_scene]++;
                    }
                    if (mc[this_scene] > num_most_common) most_common = this_scene;
                } else {
                    scene = most_common;
                    vector<string> filtered;
                    for (const auto & s: similar) {
                        if (getScene(s) == scene) filtered.push_back(s);
                    }
                    similar = filtered;
                }
            } else {
                if (getScene(fn) == scene) similar.push_back(fn);
            }
        }
    }
    return similar;
}

vector<string> functions::spaceWithMostMatches (const string & query_image, const string & ext,
                                                const double * cam_matrix, int K,
                                                int N_thresh, double max_descriptor_dist, double separation, int min_matches,
                                                vector<Eigen::Matrix3d> & R_k,
                                                vector<Eigen::Vector3d> & t_k,
                                                vector<Eigen::Matrix3d> & R_qk,
                                                vector<Eigen::Vector3d> & t_qk) {

    auto images = retrieveSimilar(query_image, ext, N_thresh, max_descriptor_dist);

    vector<string> to_return;
    to_return.push_back(images[0]);

    Eigen::Matrix3d R_1; Eigen::Vector3d t_1;
    sevenScenes::getAbsolutePose(images[0], R_1, t_1);
    R_k.push_back(R_1); t_k.push_back(t_1);

    Eigen::Matrix3d R_q1; Eigen::Vector3d t_q1;
    getRelativePose(images[0], ext, query_image, cam_matrix, "SIFT", R_q1, t_q1);
    R_qk.push_back(R_q1); t_qk.push_back(t_q1);

    int i = 1;
    while (R_k.size() < K && i < images.size()) {

        Eigen::Matrix3d R_i; Eigen::Vector3d t_i;
        sevenScenes::getAbsolutePose(images[i], R_i, t_i);
        Eigen::Vector3d c_i = sevenScenes::getT(images[i]);

        bool too_close = false;
        for (const auto & im : to_return) {
            Eigen::Vector3d c_j = sevenScenes::getT(im);
            double dist = getDistBetween(c_i, c_j);
            if (dist < separation) {
                too_close = true;
                break;
            }
        }

        if (!too_close) {
            Eigen::Matrix3d R_qi; Eigen::Vector3d t_qi;
            int num = getRelativePose(images[i], ext, query_image, cam_matrix, "SIFT", R_qi, t_qi);

            if (num >= min_matches) {
                to_return.push_back(images[i]);
                R_k.push_back(R_i); t_k.push_back(t_i);
                R_qk.push_back(R_qi); t_qk.push_back(t_qi);
            }
        }
        i++;
    }
    return to_return;
}





// Visualize
void functions::showTop1000(const string & query_image, const string & ext, int max_num, double max_descriptor_dist, int inliers) {

    // Show Query
    Mat q = imread(query_image + ext);
    imshow(query_image, q);
    waitKey();

    // Get Query Scene
    string scene = getScene(query_image);

    // Get Top 1000 NetVLAD images
    vector<string> topN = getTopN(query_image, ext, 1000);
    vector<pair<Mat, string>> vecMat;
    vecMat.reserve(topN.size());
    for(const auto & p : topN) {
        vecMat.emplace_back(imread(p + ext), p);
    }

    // Threshold, filter, and space the top 1000
    vector<string> retrieved = functions::retrieveSimilar(query_image, ext, max_num, max_descriptor_dist);
    vector<string> spaced = functions::optimizeSpacing(retrieved, inliers, false, "7-Scenes");

    // Do the following for images/100 windows
    int N = 100;
    int num = int (vecMat.size()) / 100;
    for (int n = 0; n < num; n++) {

        // Get window parameters
        int nRows = 10;
        int windowHeight = 5000;
        nRows = nRows > N ? N : nRows;
        int edgeThickness = 20;
        int imagesPerRow = ceil(double(N) / nRows);
        int resizeHeight = int (floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0))) - edgeThickness;
        int maxRowLength = 0;
        std::vector<int> resizeWidth;
        for (int i = 0; i < N;) {
            int thisRowLen = 0;
            for (int k = 0; k < imagesPerRow; k++) {
                double aspectRatio = double(vecMat[100 * n + i].first.cols) / vecMat[100 * n + i].first.rows;
                int temp = int(ceil(resizeHeight * aspectRatio));
                resizeWidth.push_back(temp);
                thisRowLen += temp;
                if (++i == N) break;
            }
            if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength) {
                maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
            }
        }

        // Create canvas
        int windowWidth = maxRowLength;
        Mat canvasImage(windowHeight, windowWidth, CV_8UC3, Scalar(0, 0, 0));

        // Draw Images
        for (int k = 0, i = 0; i < nRows; i++) {
            int y = i * resizeHeight + (i + 1) * edgeThickness;
            int x_end = edgeThickness;
            for (int j = 0; j < imagesPerRow && k < N; k++, j++) {
                int x = x_end;
                int y_color = y - edgeThickness / 2;
                int x_color = x - edgeThickness / 2;

                // Determine Highlight
                Rect roi_color(x_color, y_color, resizeWidth[k] + edgeThickness, resizeHeight + edgeThickness);
                Size s_color = canvasImage(roi_color).size();
                Mat target_ROI_color;
                string this_scene = getScene(vecMat[100 * n + k].second);
                if (this_scene != scene) {
                    target_ROI_color = Mat (s_color, CV_8UC3, Scalar(0., 0., 255.));
                } else {
                    if (count(retrieved.begin(), retrieved.end(), vecMat[100 * n + k].second)) {
                        target_ROI_color = Mat(s_color, CV_8UC3, Scalar(0., 0., 0.));
                        if (count(spaced.begin(), spaced.end(), vecMat[100 * n + k].second)) {
                            target_ROI_color = Mat(s_color, CV_8UC3, Scalar(0., 255., 0.));
                        }
                    } else {
                        target_ROI_color = Mat(s_color, CV_8UC3, Scalar(255., 255., 255.));
                    }
                }
                target_ROI_color.copyTo(canvasImage(roi_color));

                // Draw Image
                Rect roi(x, y, resizeWidth[k], resizeHeight);
                Size s = canvasImage(roi).size();
                // change the number of channels to three
                Mat target_ROI(s, CV_8UC3);
                if (vecMat[100 * n + k].first.channels() != canvasImage.channels()) {
                    if (vecMat[100 * n + k].first.channels() == 1) {
                        cvtColor(vecMat[100 * n + k].first, target_ROI, COLOR_GRAY2BGR);
                    }
                } else {
                    vecMat[100 * n + k].first.copyTo(target_ROI);
                }
                resize(target_ROI, target_ROI, s);
                if (target_ROI.type() != canvasImage.type()) {
                    target_ROI.convertTo(target_ROI, canvasImage.type());
                }
                target_ROI.copyTo(canvasImage(roi));
                x_end += resizeWidth[k] + edgeThickness;
            }
        }
        string title = query_image + "NN from " + to_string(100 * n + 1) + " to " + to_string(100 * n + 100);
        imshow(title, canvasImage);
        waitKey();
    }
}

Mat functions::projectCentersTo2D(const string & query, const vector<string> & images,
                                  unordered_map<string, cv::Scalar> & seq_colors,
                                  const string & Title) {

    Eigen::Vector3d total {0, 0, 0};
    Eigen::Vector3d z {0, 0, 1};
    vector<tuple<Eigen::Vector3d, Eigen::Vector3d, Scalar>> centers;

    // get query center
    auto c_q = sevenScenes::getT(query);
    total += c_q;

    // get view direction
    auto r_q = sevenScenes::getR(query);
    Eigen::Vector3d dir_q = r_q * z;

    centers.emplace_back(c_q, dir_q, Scalar(0., 0., 255.));

    // do same for all images
    for (const auto & image : images) {
        // get camera center
        auto c = sevenScenes::getT(image);
        total += c;

        // get view direction
        auto r = sevenScenes::getR(image);
        Eigen::Vector3d dir = r * z;

        // get color
        string seq = getSequence(image);
        Scalar color = seq_colors[seq];

        centers.emplace_back(c, dir, color);
    }
    Eigen::Vector3d average = total / centers.size();

    double farthest_x = 0.;
    double farthest_y = 0.;
    double max_z = 0;
    double min_z = 0;
    for (const auto & c : centers) {
        double x_dist = abs(average[0] - get<0>(c)[0]);
        double y_dist = abs(average[2] - get<0>(c)[2]);
        double h = get<0>(c)[1];
        if (x_dist > farthest_x) farthest_x = x_dist;
        if (y_dist > farthest_y) farthest_y = y_dist;
        if (h > max_z) {
            max_z = h;
        } else if (h < min_z) {
            min_z = h;
        }
    }
    double med_z = (max_z + min_z) / 2.;


    double height = 1900.;
    double width = 3000.;
    double avg_radius = 15.;
    double radial_variance = avg_radius * .5;
    double border = 4 * avg_radius;
    double m_over_px_x = (farthest_x / (width/2. - border));
    double m_over_px_y = (farthest_y / (height/2. - border));
    double m_over_px_z = ((max_z - med_z) / radial_variance);
    Mat canvasImage(int (height), int (width), CV_8UC3, Scalar(255., 255., 255.));
    cout << "Window is " << m_over_px_x * width << " meters wide and " << m_over_px_y * height << " meters tall." << endl;


    Point2d im_center {width / 2, height / 2};
    for (const auto & c : centers) {

        // get center point
        Point2d c_pt {
                im_center.x + ((get<0>(c)[0] - average[0]) / m_over_px_x),
                im_center.y + ((get<0>(c)[2] - average[2]) / m_over_px_y)
        };

        // get radius
        double radius = avg_radius + (get<0>(c)[1] - med_z) / m_over_px_z;

        // get view point
        double x_dir = get<1>(c)[0] / m_over_px_x;
        double y_dir = get<1>(c)[2] / m_over_px_y;
        double length_dir = sqrt(pow(x_dir, 2) + pow(y_dir, 2));
        double scale =  2. * radius / length_dir;
        x_dir = scale * x_dir;
        y_dir = scale * y_dir;
        Point2d dir_pt {c_pt.x + x_dir, c_pt.y + y_dir};

        // get color
        Scalar color = get<2>(c);

        // draw center and direction
        circle(canvasImage, c_pt, int (radius), color, -1);
        line(canvasImage, c_pt, dir_pt, color, int (radius/2.));
    }
    imshow(Title, canvasImage);
    waitKey();
    return canvasImage;
}

cv::Mat functions::showAllImages(const string & query, const vector<string> & images,
                                 unordered_map<string, cv::Scalar> & seq_colors,
                                 const unordered_set<string> & unincluded_all,
                                 const unordered_set<string> & unincluded_top1000,
                                 const string & Title) {

    Eigen::Vector3d total {0, 0, 0};
    Eigen::Vector3d z {0, 0, 1};
    vector<tuple<Eigen::Vector3d, Eigen::Vector3d, Scalar>> centers, first_layer, second_layer;

    // get query center
    auto c_q = sevenScenes::getT(query);
    total += c_q;

    // get view direction
    auto r_q = sevenScenes::getR(query);
    Eigen::Vector3d dir_q = r_q * z;

    centers.emplace_back(c_q, dir_q, Scalar(0., 0., 255.));
    second_layer.emplace_back(c_q, dir_q, Scalar(0., 0., 255.));
    // do same for all images
    for (const auto & image : images) {
        // get camera center
        auto c = sevenScenes::getT(image);
        total += c;

        // get view direction
        auto r = sevenScenes::getR(image);
        Eigen::Vector3d dir = r * z;

        // get color
        Scalar color;
        string seq = getSequence(image);
        bool found = false;
        if (unincluded_all.find(image) != unincluded_all.end()) {
            color = cv::Scalar (200., 200., 200.);
            first_layer.emplace_back(c, dir, color);
        } else if (unincluded_top1000.find(image) != unincluded_top1000.end()) {
            color = cv::Scalar (0., 0., 0.);
            first_layer.emplace_back(c, dir, color);
        } else {
            color = seq_colors[seq];
            second_layer.emplace_back(c, dir, color);
        }
        centers.emplace_back(c, dir, color);
    }
    Eigen::Vector3d average = total / centers.size();

    double farthest_x = 0.;
    double farthest_y = 0.;
    double max_z = 0;
    double min_z = 0;
    for (const auto & c : centers) {
        double x_dist = abs(average[0] - get<0>(c)[0]);
        double y_dist = abs(average[2] - get<0>(c)[2]);
        double h = get<0>(c)[1];
        if (x_dist > farthest_x) farthest_x = x_dist;
        if (y_dist > farthest_y) farthest_y = y_dist;
        if (h > max_z) {
            max_z = h;
        } else if (h < min_z) {
            min_z = h;
        }
    }
    double med_z = (max_z + min_z) / 2.;


    double height = 1900.;
    double width = 3000.;
    double avg_radius = 15.;
    double radial_variance = avg_radius * .5;
    double border = 4 * avg_radius;
    double m_over_px_x = (farthest_x / (width/2. - border));
    double m_over_px_y = (farthest_y / (height/2. - border));
    double m_over_px_z = ((max_z - med_z) / radial_variance);
    Mat canvasImage(int (height), int (width), CV_8UC3, Scalar(255., 255., 255.));
    cout << "Window is " << m_over_px_x * width << " meters wide and " << m_over_px_y * height << " meters tall." << endl;


    Point2d im_center {width / 2, height / 2};
    for (const auto & f : first_layer) {

        // get center point
        Point2d c_pt {
                im_center.x + ((get<0>(f)[0] - average[0]) / m_over_px_x),
                im_center.y + ((get<0>(f)[2] - average[2]) / m_over_px_y)
        };

        // get radius
        double radius = avg_radius + (get<0>(f)[1] - med_z) / m_over_px_z;

        // get view point
        double x_dir = get<1>(f)[0] / m_over_px_x;
        double y_dir = get<1>(f)[2] / m_over_px_y;
        double length_dir = sqrt(pow(x_dir, 2) + pow(y_dir, 2));
        double scale =  2. * radius / length_dir;
        x_dir = scale * x_dir;
        y_dir = scale * y_dir;
        Point2d dir_pt {c_pt.x + x_dir, c_pt.y + y_dir};

        // get color
        Scalar color = get<2>(f);

        // draw center and direction
        circle(canvasImage, c_pt, int (radius), color, -1);
        line(canvasImage, c_pt, dir_pt, color, int (radius/2.));
    }
    for (const auto & s : second_layer) {

        // get center point
        Point2d c_pt {
                im_center.x + ((get<0>(s)[0] - average[0]) / m_over_px_x),
                im_center.y + ((get<0>(s)[2] - average[2]) / m_over_px_y)
        };

        // get radius
        double radius = avg_radius + (get<0>(s)[1] - med_z) / m_over_px_z;

        // get view point
        double x_dir = get<1>(s)[0] / m_over_px_x;
        double y_dir = get<1>(s)[2] / m_over_px_y;
        double length_dir = sqrt(pow(x_dir, 2) + pow(y_dir, 2));
        double scale =  2. * radius / length_dir;
        x_dir = scale * x_dir;
        y_dir = scale * y_dir;
        Point2d dir_pt {c_pt.x + x_dir, c_pt.y + y_dir};

        // get color
        Scalar color = get<2>(s);

        // draw center and direction
        circle(canvasImage, c_pt, int (radius), color, -1);
        line(canvasImage, c_pt, dir_pt, color, int (radius/2.));
    }
    imshow(Title, canvasImage);
    waitKey();
    return canvasImage;
}






