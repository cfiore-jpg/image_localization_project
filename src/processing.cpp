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

double processing::rotationDifference(const Eigen::Matrix3d & r1, const Eigen::Matrix3d & r2) {

    Eigen::Matrix3d R = r1 * r2.transpose();
    double trace = R.trace();
    double theta = acos((trace-1)/2);
    theta = theta * 180. / PI;
    return theta;
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
    cv::Point2d pp(320, 240);
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

    cout << "Number of points: " << to_string(inlier_match_points1.size()) << endl;

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

Eigen::Vector3d processing::hypothesizeQueryCenter(const string &query_image, const vector<string> &ensemble, const string & dataset) {

    double K = ensemble.size();

    double tl, tc, tr, ml, mc, mr, bl, bc, br = 0;
    Eigen::Vector3d sum_c_k; sum_c_k << 0, 0, 0;
    Eigen::Vector3d sum_ckt_vk_vk; sum_ckt_vk_vk << 0, 0, 0;

    for (const auto& im_k : ensemble) {
        Eigen::Matrix3d R_wk;
        Eigen::Vector3d t_wk;
        if (dataset == "7-Scenes") {
            sevenScenes::getAbsolutePose(im_k, R_wk, t_wk);
        } else {
            cout << "No support for this dataset." << endl;
            exit(1);
        }
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

Eigen::Vector3d processing::hypothesizeQueryCenter(const vector<Eigen::Matrix3d> &R_k,
                                                   const vector<Eigen::Vector3d> &t_k,
                                                   const vector<Eigen::Matrix3d> &R_qk,
                                                   const vector<Eigen::Vector3d> &t_qk) {

    double K = R_k.size();

    double tl, tc, tr, ml, mc, mr, bl, bc, br = 0;
    Eigen::Vector3d sum_c_k; sum_c_k << 0, 0, 0;
    Eigen::Vector3d sum_ckt_vk_vk; sum_ckt_vk_vk << 0, 0, 0;

    for (int i = 0; i < K; i++) {
        Eigen::Matrix3d R_wk = R_k[i];
        Eigen::Vector3d t_wk = t_k[i];
        Eigen::Matrix3d R_kq = R_qk[i];
        Eigen::Vector3d t_kq = t_qk[i];

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

template<typename DataType, typename ForwardIterator>
Eigen::Quaternion<DataType> processing::averageQuaternions(ForwardIterator const & begin, ForwardIterator const & end) {

    if (begin == end) {
            throw std::logic_error("Cannot average orientations over an empty range.");
    }

    Eigen::Matrix<DataType, 4, 4> A = Eigen::Matrix<DataType, 4, 4>::Zero();
    uint sum(0);
    for (ForwardIterator it = begin; it != end; ++it) {
        Eigen::Matrix<DataType, 1, 4> q(1,4);
        q(0) = it->w();
        q(1) = it->x();
        q(2) = it->y();
        q(3) = it->z();
        A += q.transpose()*q;
        sum++;
    }
    A /= sum;

    Eigen::EigenSolver<Eigen::Matrix<DataType, 4, 4>> es(A);

    Eigen::Matrix<std::complex<DataType>, 4, 1> mat(es.eigenvalues());
    int index;
    mat.real().maxCoeff(&index);
    Eigen::Matrix<DataType, 4, 1> largest_ev(es.eigenvectors().real().block(0, index, 4, 1));

    return Eigen::Quaternion<DataType>(largest_ev(0), largest_ev(1), largest_ev(2), largest_ev(3));
}

Eigen::Matrix3d processing::rotationAverage(const vector<Eigen::Matrix3d> & rotations) {

    vector<Eigen::Quaterniond> quaternions;
    for (const auto & rot: rotations) {
        Eigen::Quaterniond q(rot);
        quaternions.push_back(q);
    }

    Eigen::Quaterniond avg_Q = averageQuaternions<double>(quaternions.begin(), quaternions.end());
    Eigen::Matrix3d avg_M = avg_Q.toRotationMatrix();
    return avg_M;

}

void processing::getEnsemble(const int & max_size,
                             const string & dataset,
                             const string & query,
                             const vector<string> & images,
                             vector<Eigen::Matrix3d> & R_k,
                             vector<Eigen::Vector3d> & t_k,
                             vector<Eigen::Matrix3d> & R_qk,
                             vector<Eigen::Vector3d> & t_qk) {

    Eigen::Vector3d c;

    if (dataset == "7-Scenes") {
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        sevenScenes::getAbsolutePose(query, R, t);
        c = - R.transpose() * t;
    } else {
        cout << "No support for this dataset." << endl;
        exit(1);
    }

    for (const auto & image : images) {
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        Eigen::Matrix3d R_;
        Eigen::Vector3d t_;

        if (dataset == "7-Scenes") {
            sevenScenes::getAbsolutePose(image, R, t);
        } else {
            cout << "No support for this dataset." << endl;
            exit(1);
        }

        Eigen::Vector3d c_k = - R.transpose() * t;
        double d = getDistBetween(c, c_k);

        if (d <= 0.05) {
            continue;
        } else {
            bool b = true;
            for (int j = 0; j < R_k.size(); j++) {
                Eigen::Vector3d c_l = - R_k[j].transpose() * t_k[j];
                d = getDistBetween(c_k, c_l);
                if (d <= 0.05) {
                    b = false;
                    break;
                }
            }
            if (b) {
                getRelativePose(image, query, "ORB", R_, t_);
                R_k.push_back(R);
                t_k.push_back(t);
                R_qk.push_back(R_);
                t_qk.push_back(t_);
            }
        }
        if (R_k.size() == max_size) {
            break;
        }
    }
}

void processing::useRANSAC(const vector<Eigen::Matrix3d> &R_k,
                           const vector<Eigen::Vector3d> &t_k,
                           const vector<Eigen::Matrix3d> &R_qk,
                           const vector<Eigen::Vector3d> &t_qk,
                           Eigen::Matrix3d & R_q,
                           Eigen::Vector3d & c_q) {

    auto K = R_k.size();
    vector<Eigen::Matrix3d> R;
    vector<Eigen::Vector3d> t;
    vector<Eigen::Matrix3d> R_;
    vector<Eigen::Vector3d> t_;

    for (int i = 0; i < K - 1; i++) {
        for (int j = i + 1; j < K; j++) {
            vector<Eigen::Matrix3d>  R_1; R_1.push_back(R_k[i]); R_1.push_back(R_k[j]);
            vector<Eigen::Vector3d> t_1; t_1.push_back(t_k[i]); t_1.push_back(t_k[j]);
            vector<Eigen::Matrix3d>  R_q1; R_q1.push_back(R_qk[i]); R_q1.push_back(R_qk[j]);
            vector<Eigen::Vector3d> t_q1; t_q1.push_back(t_qk[i]); t_q1.push_back(t_qk[j]);

            Eigen::Vector3d h = hypothesizeQueryCenter(R_1, t_1, R_q1, t_q1);

            for (int k = 0; k < K; k++) {
                if (k != i && k != j) {
                    Eigen::Vector3d t_pred = -R_qk[k] * (R_k[k]*h + t_k[k]);
                    Eigen::Vector3d t_real = t_qk[k];
                    double angle = getAngleBetween(t_pred, t_real);

                    if (angle <= 15) {
                        R_1.push_back(R_k[k]);
                        t_1.push_back(t_k[k]);
                        R_q1.push_back(R_qk[k]);
                        t_q1.push_back(t_qk[k]);
                    }
                }
            }
            if (R_1.size() > R.size()) {
                R = R_1;
                t = t_1;
                R_ = R_q1;
                t_ = t_q1;
            }
        }
    }
    cout << "Number of inliers: " << to_string(R.size()) << endl;
    c_q = hypothesizeQueryCenter(R, t, R_, t_);
    vector<Eigen::Matrix3d> rotations;
    for (int x = 0; x < R.size(); x++) {
        rotations.push_back(R[x] * R_[x]);
    }
    R_q = rotationAverage(rotations);
}







