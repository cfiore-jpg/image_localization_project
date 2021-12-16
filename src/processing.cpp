//
// Created by Cameron Fiore on 12/15/21.
//

#include "../include/processing.h"
#include "../include/sevenScenes.h"
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <base/database.h>
#include <base/image.h>
#include <fstream>

#define PI 3.14159265

using namespace std;

double processing::triangulateRays(const Eigen::Vector3d &ci, const Eigen::Vector3d &di, const Eigen::Vector3d &cj,
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

double processing::getDistBetween(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2) {
    double x = p1(0) - p2(0);
    double y = p1(1) - p2(1);
    double z = p1(2) - p2(2);
    return sqrt(pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0));
}

double processing::getAngleBetween(const Eigen::Vector3d &d1, const Eigen::Vector3d &d2) {
    return acos(d1.dot(d2) / (d1.norm() * d2.norm())) * 180.0 / PI;
}

bool processing::findMatches(const string &db_image, const string &query_image, const string &method,
                             vector<cv::Point2d> &pts_db, vector<cv::Point2d> &pts_query) {

    cv::Mat image1 = cv::imread(query_image + ".color.png");
    cv::Mat image2 = cv::imread(db_image + ".color.png");

    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(400, 4, 2, false);
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    cv::Mat mask1, mask2;
    vector<cv::KeyPoint> kp_vec1, kp_vec2;
    cv::Mat desc1, desc2;

    if (method == "ORB") {
        orb->detectAndCompute(image1, mask1, kp_vec1, desc1);
        orb->detectAndCompute(image2, mask2, kp_vec2, desc2);
    } else if (method == "SURF") {
        surf->detectAndCompute(image1, mask1, kp_vec1, desc1);
        surf->detectAndCompute(image2, mask2, kp_vec2, desc2);
    } else if (method == "SIFT") {
        sift->detectAndCompute(image1, mask1, kp_vec1, desc1);
        sift->detectAndCompute(image2, mask2, kp_vec2, desc2);
    } else {
        cout << "Not a valid method for feature matching..." << endl;
        exit(1);
    }

    cv::BFMatcher matcher(cv::NORM_L2, true);
    vector<cv::DMatch> matches;
    vector<vector<cv::DMatch>> vmatches;
    matcher.knnMatch(desc1, desc2, vmatches, 1);
    for (auto & vmatch : vmatches) {
        if (vmatch.empty()) {
            continue;
        }
        matches.push_back(vmatch[0]);
    }

    std::sort(matches.begin(), matches.end());

    if (matches.size() < 5) {
        return false;
    }

    vector<cv::Point2d> pts1, pts2;
    for (int i = 0; i < matches.size(); i++) {
        pts1.push_back(kp_vec1[matches[i].queryIdx].pt);
        pts2.push_back(kp_vec2[matches[i].trainIdx].pt);
    }

    pts_query = pts1;
    pts_db = pts2;
    return true;
}

void processing::getRelativePose(const string &db_image, const string &query_image, const string &method,
                                 Eigen::Matrix3d &R_kq, Eigen::Vector3d &t_kq) {

    vector<cv::Point2d> pts_db, pts_q;
    findMatches(db_image, query_image, method, pts_db, pts_q);

    cv::Mat im = cv::imread(db_image + ".color.png");
    cv::Mat mask;
    cv::Point2d pp(im.cols / 2., im.rows / 2.);
    double focal = 500;
    cv::Mat E_kq = cv::findEssentialMat(pts_db, pts_q,
                                        focal,
                                        pp,
                                        cv::RANSAC, 0.999, 1.0, mask);

    vector<cv::Point2f> inlier_match_points1, inlier_match_points2;
    for (int i = 0; i < mask.rows; i++) {
        if (mask.at<unsigned char>(i)) {
            inlier_match_points1.push_back(pts_db[i]);
            inlier_match_points2.push_back(pts_q[i]);
        }
    }

    mask.release();
    cv::Mat R, t;
    cv::recoverPose(E_kq,
                    inlier_match_points1,
                    inlier_match_points2,
                    R, t,
                    focal, pp, mask);

    cv::cv2eigen(R, R_kq);
    cv::cv2eigen(t, t_kq);
}

Eigen::Vector3d hypothesizeQueryCenter(const string &query_image, const vector<string> &ensemble) {

    double K = ensemble.size();

    double tl, tc, tr, ml, mc, mr, bl, bc, br = 0;
    Eigen::Vector3d sum_c_k; sum_c_k << 0, 0, 0;
    Eigen::Vector3d sum_ckt_vk_vk; sum_ckt_vk_vk << 0, 0, 0;

    for (const auto& im_k : ensemble) {
        Eigen::Matrix3d R_wk;
        Eigen::Vector3d t_wk;
        sevenScenes::getAbsolutePose(im_k, R_wk, t_wk);

        Eigen::Matrix3d R_kq;
        Eigen::Vector3d t_kq;
        processing::getRelativePose(im_k, query_image, "SIFT", R_kq, t_kq);

        Eigen::Vector3d c_k = -R_wk.transpose() * t_wk;
        Eigen::Vector3d v_k = -R_wk.transpose() * R_kq.transpose() * t_kq;

        tl += v_k[0]*v_k[0]; tc += v_k[0]*v_k[1]; tr += v_k[0]*v_k[2];
        ml += v_k[1]*v_k[0]; mc += v_k[1]*v_k[1]; mr += v_k[1]*v_k[2];
        bl += v_k[2]*v_k[0]; bc += v_k[2]*v_k[1]; br += v_k[2]*v_k[2];

        sum_c_k += c_k;
        sum_ckt_vk_vk += c_k.dot(v_k) * v_k;
    }

    Eigen::Matrix3d A;
    A << K-tl, -tc, -tr,
         -ml, K-mc, -mr,
         -bl, -bc, K-br;
    Eigen::Vector3d b = sum_c_k - sum_ckt_vk_vk;

    Eigen::Vector3d c_q = A.colPivHouseholderQr().solve(b);

    return c_q;
}



