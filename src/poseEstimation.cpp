//
// Created by Cameron Fiore on 1/23/22.
//
#include "../include/sevenScenes.h"
#include "../include/functions.h"
#include "../include/poseEstimation.h"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <algorithm>
#include <opencv2/imgcodecs.hpp>
#include <utility>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "glog/logging.h"
#include "../include/sevenScenes.h"
#include "../include/functions.h"
#include "../include/OptimalRotationSolver.h"
#include <iomanip>
#include <chrono>


#define EXT ".color.png"


/// CERES COST FUNCTIONS

struct Top2 {
    Top2 (vector<double> K,
          cv::Point2d pt2D,
          Eigen::Vector3d pt3D)
            : K(std::move(K)), pt2D(std::move(pt2D)), pt3D(std::move(pt3D)) {}

    template<typename T>
    bool operator()(
            const T * const lm,
            const T * const A_q,
            const T * const T_q,
            T * residuals) const {

        T R_q[9];
        ceres::AngleAxisToRotationMatrix(A_q, R_q);

        T lambda = lm[0];
        T mu = lm[1];

        T A[3];
        A[0] = R_q[0] * T(pt3D(0)) + R_q[3] * T(pt3D(1)) + R_q[6] * T(pt3D(2)) + T_q[0];
        A[1] = R_q[1] * T(pt3D(0)) + R_q[4] * T(pt3D(1)) + R_q[7] * T(pt3D(2)) + T_q[1];
        A[2] = R_q[2] * T(pt3D(0)) + R_q[5] * T(pt3D(1)) + R_q[8] * T(pt3D(2)) + T_q[2];
        T B[3];
        B[0] = T(K[2]) * A[0] + T(K[0]) * A[2];
        B[1] = T(K[3]) * A[1] + T(K[1]) * A[2];
        B[2] = A[2];

        T mx = T(pt2D.x) - T(K[0]);
        T my = T(pt2D.y) - T(K[1]);
        T f = (T(K[2]) + T(K[3])) / T(2);
        T r = T(K[4]) / (f * f);
        T r2 = r * (mx * mx + my * my);
        T xi = (T(1)+r2)*mx + T(K[0]);
        T eta = (T(1)+r2)*my + T(K[1]);

        T left1 = 0.5 * B[2] * B[2] * lambda;
        T right1 = xi * B[2] - B[0];
        T res0 = left1 - right1;
        residuals[0] = res0;

        T left2 = 0.5 * B[2] * B[2] * mu;
        T right2 = eta * B[2] - B[1];
        T res1 = left2 - right2;
        residuals[1] = res1;

        return true;
    }

    static ceres::CostFunction * Create(const vector<double> & K,
                                        const cv::Point2d & pt2D,
                                        const Eigen::Vector3d & pt3D) {
        return (new ceres::AutoDiffCostFunction<Top2, 2, 2, 3, 3> (new Top2(K, pt2D, pt3D)));
    }

    vector<double> K;
    cv::Point2d pt2D;
    Eigen::Vector3d pt3D;
};


struct Bottom2 {
    Bottom2 (vector<double> K,
             vector<cv::Point2d> pts2D,
             vector<Eigen::Vector3d> pts3D)
            : K(std::move(K)), pts2D(std::move(pts2D)), pts3D(std::move(pts3D)) {}

    template<typename T>
    bool operator()(T const * const * parameters, T * residuals) const {

        T dR_d1[9];
        R_deriv1(parameters[0], dR_d1);

        T dR_d2[9];
        R_deriv2(parameters[0], dR_d2);

        T dR_d3[9];
        R_deriv3(parameters[0], dR_d3);

        T A_q[3];
        A_q[0] = parameters[0][0];
        A_q[1] = parameters[0][1];
        A_q[2] = parameters[0][2];

        T R_q[9];
        ceres::AngleAxisToRotationMatrix(A_q, R_q);

        T T_q[3];
        T_q[0] = parameters[1][0];
        T_q[1] = parameters[1][1];
        T_q[2] = parameters[1][2];

        T res0 = T(0);
        T res1 = T(0);
        T res2 = T(0);
        T res3 = T(0);
        T res4 = T(0);
        T res5 = T(0);
        for (int i = 0; i < pts2D.size(); i++) {

            cv::Point2d pt2D = pts2D[i];
            Eigen::Vector3d pt3D = pts3D[i];

            T A0[3];
            A0[0] = R_q[0] * T(pt3D(0)) + R_q[3] * T(pt3D(1)) + R_q[6] * T(pt3D(2)) + T_q[0];
            A0[1] = R_q[1] * T(pt3D(0)) + R_q[4] * T(pt3D(1)) + R_q[7] * T(pt3D(2)) + T_q[1];
            A0[2] = R_q[2] * T(pt3D(0)) + R_q[5] * T(pt3D(1)) + R_q[8] * T(pt3D(2)) + T_q[2];
            T B0[3];
            B0[0] = T(K[2]) * A0[0] + T(K[0]) * A0[2];
            B0[1] = T(K[3]) * A0[1] + T(K[1]) * A0[2];
            B0[2] = A0[2];

            T lambda = parameters[i + 2][0];
            T mu = parameters[i + 2][1];

            T mx = T(pt2D.x) - T(K[0]);
            T my = T(pt2D.y) - T(K[1]);
            T f = (T(K[2]) + T(K[3])) / T(2);
            T r = K[4] / (f * f);
            T r2 = r * (mx * mx + my * my);
            T xi = (T(1) + r2) * mx + T(K[0]);
            T eta = (T(1) + r2) * my + T(K[1]);

            T fx = T(K[2]);
            T fy = T(K[3]);
            T cx = T(K[0]);
            T cy = T(K[1]);

            T v[3];
            v[0] = lambda;
            v[1] = mu;
            v[2] = lambda * (xi - 0.5 * lambda * B0[2]) + mu * (eta - 0.5 * mu * B0[2]);

            res0 += v[0];
            res1 += v[1];
            res2 += v[2];

            T A1[3];
            A1[0] = dR_d1[0] * T(pt3D[0]) + dR_d1[3] * T(pt3D[1]) + dR_d1[6] * T(pt3D[2]);
            A1[1] = dR_d1[1] * T(pt3D[0]) + dR_d1[4] * T(pt3D[1]) + dR_d1[7] * T(pt3D[2]);
            A1[2] = dR_d1[2] * T(pt3D[0]) + dR_d1[5] * T(pt3D[1]) + dR_d1[8] * T(pt3D[2]);
            T B1[3];
            B1[0] = fx * A1[0] + cx * A1[2];
            B1[1] = fy * A1[1] + cy * A1[2];
            B1[2] = A1[2];
            res3 += v[0] * B1[0] + v[1] * B1[1] + v[2] * B1[2];

            T A2[3];
            A2[0] = dR_d2[0] * T(pt3D[0]) + dR_d2[3] * T(pt3D[1]) + dR_d2[6] * T(pt3D[2]);
            A2[1] = dR_d2[1] * T(pt3D[0]) + dR_d2[4] * T(pt3D[1]) + dR_d2[7] * T(pt3D[2]);
            A2[2] = dR_d2[2] * T(pt3D[0]) + dR_d2[5] * T(pt3D[1]) + dR_d2[8] * T(pt3D[2]);
            T B2[3];
            B2[0] = fx * A2[0] + cx * A2[2];
            B2[1] = fy * A2[1] + cy * A2[2];
            B2[2] = A2[2];
            res4 += v[0] * B2[0] + v[1] * B2[1] + v[2] * B2[2];

            T A3[3];
            A3[0] = dR_d3[0] * T(pt3D[0]) + dR_d3[3] * T(pt3D[1]) + dR_d3[6] * T(pt3D[2]);
            A3[1] = dR_d3[1] * T(pt3D[0]) + dR_d3[4] * T(pt3D[1]) + dR_d3[7] * T(pt3D[2]);
            A3[2] = dR_d3[2] * T(pt3D[0]) + dR_d3[5] * T(pt3D[1]) + dR_d3[8] * T(pt3D[2]);
            T B3[3];
            B3[0] = fx * A3[0] + cx * A3[2];
            B3[1] = fy * A3[1] + cy * A3[2];
            B3[2] = A3[2];
            res5 += v[0] * B3[0] + v[1] * B3[1] + v[2] * B3[2];
        }

        residuals[0] = res0;
        residuals[1] = res1;
        residuals[2] = res2;
        residuals[3] = res3;
        residuals[4] = res4;
        residuals[5] = res5;

        return true;
    }

    template<typename T>
    void R_deriv1(const T* aa, T* dR_d1) const {
        T a1 = aa[0];
        T a2 = aa[1];
        T a3 = aa[2];

        T sigma16 = a1 * a1 + a2 * a2 + a3 * a3;
        T sigma15 = cos(sqrt(sigma16)) - T(1);
        T sigma14 = sin(sqrt(sigma16)) / sqrt(sigma16);
        T sigma13 = (a1 * a2 * cos(sqrt(sigma16))) / sigma16;
        T sigma12 = (a1 * a3 * cos(sqrt(sigma16))) / sigma16;
        T sigma11 = (a1 * a1 * cos(sqrt(sigma16))) / sigma16;
        T sigma10 = (T(2) * a1 * a2 * a3 * sigma15) / (sigma16 * sigma16);
        T sigma9 = (T(2) * a1 * a1 * a2 * sigma15) / (sigma16 * sigma16);
        T sigma8 = (T(2) * a1 * a1 * a3 * sigma15) / (sigma16 * sigma16);
        T sigma7 = (a1 * a2 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        T sigma6 = (a1 * a3 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        T sigma5 = (a1 * a1 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        T sigma4 = (a1 * a2 * a3 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        T sigma3 = (a1 * a1 * a2 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        T sigma2 = (a1 * a1 * a3 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        T sigma1 = (a1 * sin(sqrt(sigma16))) / sqrt(sigma16);

        dR_d1[0] = (a1 * a1 * a1 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16) - (T(2) * a1 * sigma15) / (sigma16) - (sigma1) + (T(2) * a1 * a1 * a1 * sigma15) / (sigma16 * sigma16);
        dR_d1[1] = sigma9 - (a2 * sigma15) / sigma16 + sigma12 - sigma6 + sigma3;
        dR_d1[2] = sigma8 - (a3 * sigma15) / sigma16 - sigma13 + sigma7 + sigma2;
        dR_d1[3] = sigma9 - (a2 * sigma15) / sigma16 - sigma12 + sigma6 + sigma3;
        dR_d1[4] = (T(2) * a1 * a2 * a2 * sigma15) / (sigma16 * sigma16) - sigma1 + (a1 * a2 * a2 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        dR_d1[5] = sigma14 + sigma11 - sigma5 + sigma10 + sigma4;
        dR_d1[6] = sigma8 - (a3 * sigma15) / sigma16 + sigma13 - sigma7 + sigma2;
        dR_d1[7] = sigma5 - sigma11 - sigma14 + sigma10 + sigma4;
        dR_d1[8] = (T(2) * a1 * a3 * a3 * sigma15) / (sigma16 * sigma16) - sigma1 + (a1 * a3 * a3 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
    }

    template<typename T>
    void R_deriv2(const T* aa, T* dR_d2) const {
        T a1 = aa[0];
        T a2 = aa[1];
        T a3 = aa[2];

        T sigma16 = a1 * a1 + a2 * a2 + a3 * a3;
        T sigma15 = cos(sqrt(sigma16)) - T(1);
        T sigma14 = sin(sqrt(sigma16)) / sqrt(sigma16);
        T sigma13 = (a1 * a2 * cos(sqrt(sigma16))) / sigma16;
        T sigma12 = (a2 * a3 * cos(sqrt(sigma16))) / sigma16;
        T sigma11 = (a2 * a2 * cos(sqrt(sigma16))) / sigma16;
        T sigma10 = (T(2) * a1 * a2 * a3 * sigma15) / (sigma16 * sigma16);
        T sigma9 = (T(2) * a1 * a2 * a2 * sigma15) / (sigma16 * sigma16);
        T sigma8 = (T(2) * a2 * a2 * a3 * sigma15) / (sigma16 * sigma16);
        T sigma7 = (a1 * a2 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        T sigma6 = (a2 * a3 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        T sigma5 = (a2 * a2 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        T sigma4 = (a1 * a2 * a3 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        T sigma3 = (a1 * a2 * a2 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        T sigma2 = (a2 * a2 * a3 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        T sigma1 = (a2 * sin(sqrt(sigma16))) / sqrt(sigma16);

        dR_d2[0] = (T(2) * a1 * a1 * a2 * sigma15) / (sigma16 * sigma16) - sigma1 + (a1 * a1 * a2 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        dR_d2[1] = sigma9 - (a1 * sigma15) / sigma16 + sigma12 - sigma6 + sigma3;
        dR_d2[2] = sigma5 - sigma11 - sigma14 + sigma10 + sigma4;
        dR_d2[3] = sigma9 - (a1 * sigma15) / sigma16 - sigma12 + sigma6 + sigma3;
        dR_d2[4] = (a2 * a2 * a2 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16) - (T(2) * a2 * sigma15) / (sigma16) - (sigma1) + (T(2) * a2 * a2 * a2 * sigma15) / (sigma16 * sigma16);
        dR_d2[5] = sigma8 - (a3 * sigma15) / sigma16 + sigma13 - sigma7 + sigma2;
        dR_d2[6] = sigma14 + sigma11 - sigma5 + sigma10 + sigma4;
        dR_d2[7] = sigma8 - (a3 * sigma15) / sigma16 - sigma13 + sigma7 + sigma2;
        dR_d2[8] = (T(2) * a2 * a3 * a3 * sigma15) / (sigma16 * sigma16) - sigma1 + (a2 * a3 * a3 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
    }

    template<typename T>
    void R_deriv3(const T* aa, T* dR_d3) const {
        T a1 = aa[0];
        T a2 = aa[1];
        T a3 = aa[2];

        T sigma16 = a1 * a1 + a2 * a2 + a3 * a3;
        T sigma15 = cos(sqrt(sigma16)) - T(1);
        T sigma14 = sin(sqrt(sigma16)) / sqrt(sigma16);
        T sigma13 = (a1 * a3 * cos(sqrt(sigma16))) / sigma16;
        T sigma12 = (a2 * a3 * cos(sqrt(sigma16))) / sigma16;
        T sigma11 = (a3 * a3 * cos(sqrt(sigma16))) / sigma16;
        T sigma10 = (T(2) * a1 * a2 * a3 * sigma15) / (sigma16 * sigma16);
        T sigma9 = (T(2) * a1 * a3 * a3 * sigma15) / (sigma16 * sigma16);
        T sigma8 = (T(2) * a2 * a3 * a3 * sigma15) / (sigma16 * sigma16);
        T sigma7 = (a1 * a3 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        T sigma6 = (a2 * a3 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        T sigma5 = (a3 * a3 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        T sigma4 = (a1 * a2 * a3 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        T sigma3 = (a1 * a3 * a3 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        T sigma2 = (a2 * a3 * a3 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        T sigma1 = (a3 * sin(sqrt(sigma16))) / sqrt(sigma16);

        dR_d3[0] = (T(2) * a1 * a1 * a3 * sigma15) / (sigma16 * sigma16) - sigma1 + (a1 * a1 * a3 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        dR_d3[1] = sigma14 + sigma11 - sigma5 + sigma10 + sigma4;
        dR_d3[2] = sigma9 - (a1 * sigma15) / sigma16 - sigma12 + sigma6 + sigma3;
        dR_d3[3] = sigma5 - sigma11 - sigma14 + sigma10 + sigma4;
        dR_d3[4] = (T(2) * a2 * a2 * a3 * sigma15) / (sigma16 * sigma16) - sigma1 + (a2 * a2 * a3 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16);
        dR_d3[5] = sigma8 - (a2 * sigma15) / sigma16 + sigma13 - sigma7 + sigma2;
        dR_d3[6] = sigma9 - (a1 * sigma15) / sigma16 + sigma12 - sigma6 + sigma3;
        dR_d3[7] = sigma8 - (a2 * sigma15) / sigma16 - sigma13 + sigma7 + sigma2;
        dR_d3[8] = (a3 * a3 * a3 * sin(sqrt(sigma16))) / sqrt(sigma16 * sigma16 * sigma16) - (T(2) * a3 * sigma15) / (sigma16) - (sigma1) + (T(2) * a3 * a3 * a3 * sigma15) / (sigma16 * sigma16);
    }

    static ceres::CostFunction * Create(const vector<double> & K,
                                        const vector<cv::Point2d> & pts2D,
                                        const vector<Eigen::Vector3d> & pts3D) {

        auto cf = new ceres::DynamicAutoDiffCostFunction<Bottom2, 3> (new Bottom2(K, pts2D, pts3D));
        cf->AddParameterBlock(3);
        cf->AddParameterBlock(3);
        for (int i = 0; i < pts2D.size(); i++) {
            cf->AddParameterBlock(2);
        }
        cf->SetNumResiduals(6);
        return (cf);
    }

    vector<double> K;
    vector<cv::Point2d> pts2D;
    vector<Eigen::Vector3d> pts3D;
};

//pair<vector<cv::Point2d>, vector<Eigen::Vector3d>>
//pose::num_sys_solution(const vector<Eigen::Matrix3d> & R_is,
//                       const vector<Eigen::Vector3d> & T_is,
//                       const vector<vector<double>> & K_is,
//                       const vector<double> & K_q,
//                       const vector<vector<cv::Point2d>> & all_pts_q,
//                       const vector<vector<cv::Point2d>> & all_pts_i,
//                       Eigen::Matrix3d & R_q,
//                       Eigen::Vector3d & T_q) {
//
//    vector<cv::Point2d> points2d {
//            {480, 270},
//            {1440, 810},
//            {1440, 270},
//            {480, 810}
//    }
//    vector<Eigen::Vector3d> points3d;
//
//
//
//    int K = int(R_is.size());
//
//    double A[3];
//    double R[9]{R_q(0, 0), R_q(1, 0), R_q(2, 0), R_q(0, 1), R_q(1, 1), R_q(2, 1), R_q(0, 2), R_q(1, 2), R_q(2, 2)};
//    ceres::RotationMatrixToAngleAxis(R, A);
//    double T[3]{T_q[0], T_q[1], T_q[2]};
//    double lms[10000][2];
//
////    vector<pair<pair<double, double>, vector<pair<int, int>>>> all_matches = functions::findSharedMatches(2, R_is, T_is,
////                                                                                                          K_is,
////                                                                                                          all_pts_q,
////                                                                                                          all_pts_i);
//
//    ceres::Problem problem;
//    ceres::LossFunction *loss = new ceres::CauchyLoss(1);
//    vector<double *> parameters{A, T};
////    vector<cv::Point2d> points2d;
////    vector<Eigen::Vector3d> points3d;
////    for (int i = 0; i < min(int(all_matches.size()), 1500); i++) {
//
//        auto p = all_matches[i];
//        cv::Point2d pt2D(p.first.first, p.first.second);
//        Eigen::Vector3d pt3D = pose::nview(p.second, R_is, T_is, K_is, all_pts_i);
//        points2d.push_back(pt2D);
//        points3d.push_back(pt3D);
//
//        Eigen::Matrix3d K_{{K_q[2], 0,      K_q[0]},
//                           {0,      K_q[3], K_q[1]},
//                           {0,      0,      1}};
//        Eigen::Vector3d B = K_ * (R_q * pt3D + T_q);
//
//        cv::Point2d undist = pose::undistort_point(pt2D, (K_q[2] + K_q[3]) / 2., K_q[0], K_q[1], K_q[4]);
//
//        double xi = undist.x;
//        double eta = undist.y;
//
//        lms[i][0] = (xi * B[2] - B[0]) / (0.5 * B[2] * B[2]);
//        lms[i][1] = (eta * B[2] - B[1]) / (0.5 * B[2] * B[2]);
//
//        auto top2 = Top2::Create(K_q, pt2D, pt3D);
//        problem.AddResidualBlock(top2, loss, lms[i], A, T);
//
//        parameters.push_back(lms[i]);
////    }
//    auto bottom2 = Bottom2::Create(K_q, points2d, points3d);
//    problem.AddResidualBlock(bottom2, loss, parameters);
//
//    ceres::Solver::Options options;
//    options.minimizer_progress_to_stdout = true;
//    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
//    options.preconditioner_type = ceres::SCHUR_JACOBI;
//    ceres::Solver::Summary summary;
//    if (problem.NumResidualBlocks() > 0) {
//        ceres::Solve(options, &problem, &summary);
//    } else {
//        cout << " Can't Adjust ";
//    }
//    cout << summary.FullReport() << endl;
//
//    double R_adj[9];
//    ceres::AngleAxisToRotationMatrix(A, R_adj);
//    R_q = Eigen::Matrix3d{{R_adj[0], R_adj[3], R_adj[6]},
//                          {R_adj[1], R_adj[4], R_adj[7]},
//                          {R_adj[2], R_adj[5], R_adj[8]}};
//    T_q = Eigen::Vector3d{T[0], T[1], T[2]};
//
//    return {points2d, points3d};
//}






struct NView {
    NView(Eigen::Matrix3d R_,
          Eigen::Vector3d T_,
          vector<double> K_,
          cv::Point2d pt)
            : pt(std::move(pt)), R_(std::move(R_)), T_(std::move(T_)), K_(std::move(K_)) {}

    template<typename T>
    bool operator()(const T * const point3D, T * residuals) const {

        T point_C[3];
        point_C[0] = T(R_(0, 0)) * point3D[0] + T(R_(0, 1)) * point3D[1] + T(R_(0, 2)) * point3D[2] + T(T_[0]);
        point_C[1] = T(R_(1, 0)) * point3D[0] + T(R_(1, 1)) * point3D[1] + T(R_(1, 2)) * point3D[2] + T(T_[1]);
        point_C[2] = T(R_(2, 0)) * point3D[0] + T(R_(2, 1)) * point3D[1] + T(R_(2, 2)) * point3D[2] + T(T_[2]);

        T reprojected_x = T(K_[2]) * (point_C[0] / point_C[2]);
        T reprojected_y = T(K_[3]) * (point_C[1] / point_C[2]);

        T mx = T(pt.x) - T(K_[0]);
        T my = T(pt.y) - T(K_[1]);

        T f = (T(K_[2]) + T(K_[3])) / T(2);
        T r = K_[4] / (f * f);

        T r2 = r * (mx * mx + my * my);
        T u = (T(1) + r2) * mx;
        T v = (T(1) + r2) * my;

        T error_x = reprojected_x - u;
        T error_y = reprojected_y - v;

        residuals[0] = error_x;
        residuals[1] = error_y;

        return true;
    }

    static ceres::CostFunction * Create(const Eigen::Matrix3d &R_,
                                        const Eigen::Vector3d &T_,
                                        const vector<double> &K_,
                                        const cv::Point2d &pt) {
        return (new ceres::AutoDiffCostFunction<NView, 2, 3>(new NView(R_, T_, K_, pt)));
    }

    Eigen::Matrix3d R_;
    Eigen::Vector3d T_;
    vector<double> K_;
    cv::Point2d pt;
};


struct ReprojectionError {
    ReprojectionError(cv::Point2d pt2D,
                      Eigen::Vector3d pt3D,
                      vector<double> K_)
            : pt2D(std::move(pt2D)), pt3D(std::move(pt3D)), K_(std::move(K_)) {}

    template<typename T>
    bool operator()(const T *const camera, T *residuals) const {

        T AA[3];
        AA[0] = camera[0];
        AA[1] = camera[1];
        AA[2] = camera[2];

        T R_q[9];
        ceres::AngleAxisToRotationMatrix(AA, R_q);

        T T_q[3];
        T_q[0] = camera[3];
        T_q[1] = camera[4];
        T_q[2] = camera[5];

        T point_C[3];
        point_C[0] = R_q[0] * T(pt3D[0]) + R_q[3] * T(pt3D[1]) + R_q[6] * T(pt3D[2]) + T_q[0];
        point_C[1] = R_q[1] * T(pt3D[0]) + R_q[4] * T(pt3D[1]) + R_q[7] * T(pt3D[2]) + T_q[1];
        point_C[2] = R_q[2] * T(pt3D[0]) + R_q[5] * T(pt3D[1]) + R_q[8] * T(pt3D[2]) + T_q[2];

        T reprojected_x = T(K_[2]) * (point_C[0] / point_C[2]);
        T reprojected_y = T(K_[3]) * (point_C[1] / point_C[2]);

        T mx = T(pt2D.x) - T(K_[0]);
        T my = T(pt2D.y) - T(K_[1]);

        T f = (T(K_[2]) + T(K_[3])) / T(2);
        T r = K_[4] / (f * f);

        T r2 = r * (mx * mx + my * my);
        T u = (T(1) + r2) * mx;
        T v = (T(1) + r2) * my;

        T error_x = reprojected_x - u;
        T error_y = reprojected_y - v;

        residuals[0] = error_x;
        residuals[1] = error_y;

        return true;
    }

    static ceres::CostFunction *Create(const cv::Point2d &pt2D,
                                       const Eigen::Vector3d &pt3D,
                                       const vector<double> &K_) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6>(
                new ReprojectionError(pt2D, pt3D, K_)));
    }

    cv::Point2d pt2D;
    Eigen::Vector3d pt3D;
    vector<double> K_;
};


struct ReprojectionErrorWithPoint {
    ReprojectionErrorWithPoint(cv::Point2d pt,
                               vector<double> K_)
            : pt(std::move(pt)), K_(std::move(K_)) {}

    template<typename T>
    bool operator()(const T *const camera, const T *const point3D, T *residuals) const {

        T AA[3];
        AA[0] = camera[0];
        AA[1] = camera[1];
        AA[2] = camera[2];

        T R_q[9];
        ceres::AngleAxisToRotationMatrix(AA, R_q);

        T T_q[3];
        T_q[0] = camera[3];
        T_q[1] = camera[4];
        T_q[2] = camera[5];

        T point_C[3];
        point_C[0] = R_q[0] * point3D[0] + R_q[3] * point3D[1] + R_q[6] * point3D[2] + T_q[0];
        point_C[1] = R_q[1] * point3D[0] + R_q[4] * point3D[1] + R_q[7] * point3D[2] + T_q[1];
        point_C[2] = R_q[2] * point3D[0] + R_q[5] * point3D[1] + R_q[8] * point3D[2] + T_q[2];

        T reprojected_x = T(K_[2]) * (point_C[0] / point_C[2]);
        T reprojected_y = T(K_[3]) * (point_C[1] / point_C[2]);

        T mx = T(pt.x) - T(K_[0]);
        T my = T(pt.y) - T(K_[1]);

        T f = (T(K_[2]) + T(K_[3])) / T(2);
        T r = K_[4] / (f * f);

        T r2 = r * (mx * mx + my * my);
        T u = (T(1) + r2) * mx;
        T v = (T(1) + r2) * my;

        T error_x = reprojected_x - u;
        T error_y = reprojected_y - v;

        residuals[0] = error_x;
        residuals[1] = error_y;

        return true;
    }

    static ceres::CostFunction *Create(const cv::Point2d &pt,
                                       const vector<double> &K_) {
        return (new ceres::AutoDiffCostFunction<ReprojectionErrorWithPoint, 2, 6, 3>(
                new ReprojectionErrorWithPoint(pt, K_)));
    }

    cv::Point2d pt;
    vector<double> K_;
};



//// FINAL POSE ADJUSTMENT
pair<vector<cv::Point2d>, vector<Eigen::Vector3d>>
pose::adjustHypothesis (const vector<Eigen::Matrix3d> & R_is,
                        const vector<Eigen::Vector3d> & T_is,
                        const vector<vector<double>> & K_is,
                        const vector<double> & K_q,
                        const vector<vector<cv::Point2d>> & all_pts_q,
                        const vector<vector<cv::Point2d>> & all_pts_i,
                        Eigen::Matrix3d & R_q,
                        Eigen::Vector3d & T_q) {

    int K = int(R_is.size());

    double AA[3];
    double R_arr[9]{R_q(0, 0), R_q(1, 0), R_q(2, 0), R_q(0, 1), R_q(1, 1), R_q(2, 1), R_q(0, 2), R_q(1, 2), R_q(2, 2)};
    ceres::RotationMatrixToAngleAxis(R_arr, AA);
    double camera[6]{AA[0], AA[1], AA[2], T_q[0], T_q[1], T_q[2]};

    vector<pair<pair<double,double>,vector<pair<int,int>>>> all_matches = functions::findSharedMatches(2, R_is, T_is, K_is, all_pts_q, all_pts_i);

    ceres::Problem problem;
    ceres::LossFunction *loss = new ceres::CauchyLoss(1);

    vector<cv::Point2d> points2d;
//    points2d.reserve(all_matches.size());
    vector<Eigen::Vector3d> points3d;
//    points3d.reserve(all_matches.size());

//    double points3d_adj[10000][3];
//    auto startTime = std::chrono::high_resolution_clock::now();
//    double total = 0;

    for (int i = 0; i < all_matches.size(); i++) {
        auto p = all_matches[i];

        cv::Point2d pt2D(p.first.first, p.first.second);
        Eigen::Vector3d pt3D = pose::nview(p.second, R_is, T_is, K_is, all_pts_i);

    //    if (pose::reprojError(pt3D, R_q, T_q, K_q, pt2D.x, pt2D.y) > 15) continue;

    //    points3d_adj[i][0] = pt3D[0];
    //    points3d_adj[i][1] = pt3D[1];
    //    points3d_adj[i][2] = pt3D[2];

    //    for(const auto & m : p.second) {
    //        auto cost = NView::Create(R_is[m.first], T_is[m.first], K_is[m.first], pt2D);
    //        problem.AddResidualBlock(cost, loss, points3d_adj[i]);
    //    }

        auto cost = ReprojectionError::Create(pt2D, pt3D, K_q);
        problem.AddResidualBlock(cost, loss, camera);
//        points2d.push_back(pt2D);
//        points3d.push_back(pt3D);

    }

//    auto endTime = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

//    cout << duration << endl;

    ceres::Solver::Options options;
//    options.minimizer_progress_to_stdout = true;
   options.function_tolerance = 1e-12;
   options.gradient_tolerance = 1e-12;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    ceres::Solver::Summary summary;
    if (problem.NumResidualBlocks() > 0) {
        ceres::Solve(options, &problem, &summary);
    } else {
        cout << " Can't Adjust ";
    }

//    cout << summary.FullReport() << endl;

    double AA_adj[3]{camera[0], camera[1], camera[2]};
    double R_adj[9];
    ceres::AngleAxisToRotationMatrix(AA_adj, R_adj);
    R_q = Eigen::Matrix3d{{R_adj[0], R_adj[3], R_adj[6]},
                          {R_adj[1], R_adj[4], R_adj[7]},
                          {R_adj[2], R_adj[5], R_adj[8]}};
    T_q = Eigen::Vector3d{camera[3], camera[4], camera[5]};

    return {points2d, points3d};
}




Eigen::Vector3d pose::estimate3Dpoint(const vector<pair<int, int>> & matches,
                                      const vector<Eigen::Matrix3d> & R_is,
                                      const vector<Eigen::Vector3d> & T_is,
                                      const vector<vector<double>> & K_is,
                                      const vector<vector<cv::Point2d>> & all_pts_i) {
    size_t K = matches.size();
    Eigen::Matrix3d I{{1, 0, 0},
                      {0, 1, 0},
                      {0, 0, 1}};
    Eigen::MatrixXd A(K * 3, 3);
    Eigen::MatrixXd b(K * 3, 1);
    for (int i = 0; i < K; i++) {
        auto R_i = R_is[matches[i].first];
        auto T_i = T_is[matches[i].first];
        auto K_i = K_is[matches[i].first];
        auto obv = all_pts_i[matches[i].first][matches[i].second];

        double f = (K_i[2] + K_i[3]) / 2;
        double r = K_i[4] / (f * f);
        double mx = obv.x - K_i[0];
        double my = obv.y - K_i[1];
        double r2 = r * (mx * mx + my * my);
        double u = (1 + r2) * mx;
        double v = (1 + r2) * my;

        Eigen::Vector3d pt{u, v, 1.};
        Eigen::Matrix3d K_{{K_i[2], 0.,     0.},
                           {0.,     K_i[3], 0.},
                           {0.,     0.,     1.}};
        Eigen::Vector3d ray = K_.inverse() * pt;
        ray.normalize();
        Eigen::Vector3d c_i = -R_i.transpose() * T_i;
        Eigen::Matrix3d M = I - R_i.transpose() * ray * ray.transpose() * R_i;
        Eigen::Vector3d Mc = M * c_i;

        A.block(i * 3, 0, 3, 3) = M;
        b.block(i * 3, 0, 3, 1) = Mc;
    }
    Eigen::Vector3d est = A.colPivHouseholderQr().solve(b);
    return est;
}



Eigen::Vector3d pose::nview(const vector<pair<int, int>> & matches,
                            const vector<Eigen::Matrix3d> & R_is,
                            const vector<Eigen::Vector3d> & T_is,
                            const vector<vector<double>> & K_is,
                            const vector<vector<cv::Point2d>> & all_pts_i) {

    Eigen::Vector3d pt3D = pose::estimate3Dpoint(matches, R_is, T_is, K_is, all_pts_i);
    double pt3D_arr [3] {pt3D[0], pt3D[1], pt3D[2]};

    ceres::Problem problem;
    ceres::LossFunction * loss = new ceres::CauchyLoss(1);
    for(const auto & m : matches) {
        cv::Point2d pt2D (all_pts_i[m.first][m.second].x, all_pts_i[m.first][m.second].y);
        auto cost = NView::Create(R_is[m.first], T_is[m.first], K_is[m.first], pt2D);
        problem.AddResidualBlock(cost, loss, pt3D_arr);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::Vector3d pt3D_new {pt3D_arr[0], pt3D_arr[1], pt3D_arr[2]};

    return pt3D_new;
}


cv::Point2d pose::reproject3Dto2D(const Eigen::Vector3d & point3d,
                                  const Eigen::Matrix3d & R,
                                  const Eigen::Vector3d & T,
                                  const vector<double> & K) {
    Eigen::Matrix3d K_q_eig{{K[2], 0.,   K[0]},
                            {0.,   K[3], K[1]},
                            {0.,   0.,   1.}};
    Eigen::Vector3d p = K_q_eig * (R * point3d + T);
    cv::Point2d point2d(p[0] / p[2], p[1] / p[2]);
    return point2d;
}

cv::Point2d pose::undistort_point(const cv::Point2d & pt, double f, double cx, double cy, double rn) {
    double r = rn / (f * f);
    double mx_ = pt.x - cx;
    double my_ = pt.y - cy;
    double r2 = r * (mx_ * mx_ + my_ * my_);
    double u = (1 + r2) * mx_;
    double v = (1 + r2) * my_;
    cv::Point2d r_new (u+cx, v+cy);
    return r_new;
}



double pose::reprojError(const Eigen::Vector3d & point3d,
                         const Eigen::Matrix3d & R,
                         const Eigen::Vector3d & T,
                         const vector<double> & K,
                         double mx, double my) {

    vector<double> K_wo_pp = {0, 0, K[2], K[3]};
    cv::Point2d reproj = pose::reproject3Dto2D(point3d, R, T, K_wo_pp);
    double f = (K[2] + K[3]) / 2;
    double r = K[4] / (f * f);
    double mx_ = mx - K[0];
    double my_ = my - K[1];
    double r2 = r * (mx_ * mx_ + my_ * my_);
    double u = (1 + r2) * mx_;
    double v = (1 + r2) * my_;
    double d = sqrt(pow(u - reproj.x, 2.) + pow(v - reproj.y, 2.));
    return d;
}

cv::Point2d pose::delta_g(const Eigen::Vector3d & point3d,
                          const cv::Point2d & point2d,
                          const Eigen::Matrix3d & R,
                          const Eigen::Vector3d & T,
                          const vector<double> & K) {

    cv::Point2d reproj = pose::reproject3Dto2D(point3d, R, T, K);

    double f = (K[2] + K[3]) / 2;
    double r = K[4] / (f * f);
    double x = point2d.x - K[0];
    double y = point2d.y - K[1];
    double r2 = r * (x * x + y * y);

    x = (1 + r2)*x + K[0];
    y = (1 + r2)*y + K[1];

    double u = x - reproj.x;
    double v = y - reproj.y;

    cv::Point2d dg (u, v);

    return dg;
}















void pose::visualizeRelpose(const string & query,
                            const vector<string> & anchors,
                            const vector<Eigen::Matrix3d> & R_is,
                            const vector<Eigen::Vector3d> & T_is,
                            const vector<Eigen::Matrix3d> & R_qis,
                            const vector<Eigen::Vector3d> & T_qis,
                            const vector<vector<double>> & K_is,
                            const vector<double> & K_q,
                            const vector<vector<cv::Point2d>> & all_pts_q,
                            const vector<vector<cv::Point2d>> & all_pts_i,
                            const Eigen::Matrix3d & R_q_before,
                            const Eigen::Vector3d & T_q_before,
                            const Eigen::Matrix3d & R_q_after,
                            const Eigen::Vector3d & T_q_after) {

    cv::Mat im_q = cv::imread(query);

    Eigen::Matrix3d K_q_eig {{K_q[2], 0., K_q[0]},
                             {0., K_q[3], K_q[1]},
                             {0.,     0.,     1.}};

    int K = int (anchors.size());

    double total_total = 0;
    for (int i = 0; i < K; i++) {

        string anchor_name = "/Users/cameronfiore/C++/image_localization_project/data/" + anchors[i].substr(anchors[i].find("chess"), anchors[i].size());
        cv::Mat im_i = cv::imread(anchor_name);
        Eigen::Matrix3d K_i_eig {{K_is[i][2], 0., K_is[i][0]},
                                 {0., K_is[i][3], K_is[i][1]},
                                 {0.,         0.,         1.}};


        Eigen::Matrix3d R_qi_est = R_qis[i];
        Eigen::Vector3d T_qi_est = T_qis[i];
        T_qi_est.normalize();
        Eigen::Matrix3d T_qi_est_x {
                {       0, -T_qi_est(2),  T_qi_est(1)},
                { T_qi_est(2),        0, -T_qi_est(0)},
                {-T_qi_est(1),  T_qi_est(0),        0}};
        Eigen::Matrix3d E_qi_est = T_qi_est_x * R_qi_est;
        Eigen::Matrix3d F_est = K_q_eig.inverse().transpose() * E_qi_est * K_i_eig.inverse();


        Eigen::Matrix3d R_qi_before = R_q_before * R_is[i].transpose();
        Eigen::Vector3d T_qi_before = T_q_before - R_qi_before * T_is[i];
        T_qi_before.normalize();
        Eigen::Matrix3d T_qi_before_x {
                {       0, -T_qi_before(2),  T_qi_before(1)},
                { T_qi_before(2),        0, -T_qi_before(0)},
                {-T_qi_before(1),  T_qi_before(0),        0}};
        Eigen::Matrix3d E_qi_before = T_qi_before_x * R_qi_before;
        Eigen::Matrix3d F_before = K_q_eig.inverse().transpose() * E_qi_before * K_i_eig.inverse();


        Eigen::Matrix3d R_qi_after = R_q_after * R_is[i].transpose();
        Eigen::Vector3d T_qi_after = T_q_after - R_qi_after * T_is[i];
        T_qi_after.normalize();
        Eigen::Matrix3d T_qi_after_x {
                {       0, -T_qi_after(2),  T_qi_after(1)},
                { T_qi_after(2),        0, -T_qi_after(0)},
                {-T_qi_after(1),  T_qi_after(0),        0}};
        Eigen::Matrix3d E_qi_after = T_qi_after_x * R_qi_after;
        Eigen::Matrix3d F_after = K_q_eig.inverse().transpose() * E_qi_after * K_i_eig.inverse();

        vector<int> indices (K);
        for(int n = 0; n < K; n++) {
            indices[n] = n;
        }
        random_device gen;
//        shuffle(indices.begin(), indices.end(), default_random_engine(123));
        shuffle(indices.begin(), indices.end(), gen);


        vector<cv::Point3d> est_lines_q, est_lines_i, before_lines_q, before_lines_i, after_lines_q, after_lines_i;
        vector<cv::Point2d> pts_q, pts_i;
        vector<cv::Scalar> colors;
        uniform_int_distribution<int> d (0, 255);
        assert(all_pts_i[i].size() == all_pts_q[i].size());
        for (int x = 0; x < 20; x++) {
            int idx = indices[x];

            colors.emplace_back(d(gen), d(gen), d(gen));

            Eigen::Vector3d pq {all_pts_q[i][idx].x, all_pts_q[i][idx].y, 1.};
            Eigen::Vector3d pi {all_pts_i[i][idx].x, all_pts_i[i][idx].y, 1.};

            Eigen::Vector3d elq = F_est * pi;
            Eigen::Vector3d eli = pq.transpose() * F_est;
            cv::Point3d est_line_q (elq[0], elq[1], elq[2]);
            cv::Point3d est_line_i (eli[0], eli[1], eli[2]);
            est_lines_q.push_back(est_line_q);
            est_lines_i.push_back(est_line_i);

            Eigen::Vector3d blq = F_before * pi;
            Eigen::Vector3d bli = pq.transpose() * F_before;
            cv::Point3d before_line_q (blq[0], blq[1], blq[2]);
            cv::Point3d before_line_i (bli[0], bli[1], bli[2]);
            before_lines_q.push_back(before_line_q);
            before_lines_i.push_back(before_line_i);

            Eigen::Vector3d alq = F_after * pi;
            Eigen::Vector3d ali = pq.transpose() * F_after;
            cv::Point3d after_line_q (alq[0], alq[1], alq[2]);
            cv::Point3d after_line_i (ali[0], ali[1], ali[2]);
            after_lines_q.push_back(after_line_q);
            after_lines_i.push_back(after_line_i);

            cv::Point2d pt_q (all_pts_q[i][idx].x, all_pts_q[i][idx].y);
            cv::Point2d pt_i (all_pts_i[i][idx].x, all_pts_i[i][idx].y);
            pts_q.push_back(pt_q);
            pts_i.push_back(pt_i);
        }

        cv::Mat q_est = im_q.clone(), i_est = im_i.clone();
        functions::drawLines(q_est, i_est, est_lines_q, est_lines_i, pts_q, pts_i, colors);
        cv::Mat h_est;
        cv::hconcat(q_est, i_est, h_est);
        cv::imshow("<Query>                                 Local Reprojection                                 <Anchor>", h_est);


        cv::Mat q_before = im_q.clone(), i_before = im_i.clone();
        functions::drawLines(q_before, i_before, before_lines_q, before_lines_i, pts_q, pts_i, colors);
        cv::Mat h_before;
        cv::hconcat(q_before, i_before, h_before);
        cv::imshow("<Query>                                 Reprojection Before Adjustment                                 <Anchor>", h_before);


        cv::Mat q_after = im_q.clone(), i_after = im_i.clone();
        functions::drawLines(q_after, i_after, after_lines_q, after_lines_i, pts_q, pts_i, colors);
        cv::Mat h_after;
        cv::hconcat(q_after, i_after, h_after);
        cv::imshow("<Query>                                 Reprojection After Adjustment                                 <Anchor>", h_after);

        cv::waitKey(0);
    }
}









struct ReprojectionErrorBA{
    ReprojectionErrorBA(cv::Point2d pt_1, cv::Point2d pt_2)
            : pt_1(pt_1), pt_2(pt_2) {}

    template<typename T>
    bool operator()(const T * const camera_1, const T * const camera_2, T * residuals) const {

        T R_1[3], R_2[3], T_1[3], T_2[3];
        R_1[0] = camera_1[0]; R_1[1] = camera_1[1]; R_1[2] = camera_1[2];
        T_1[0] = camera_1[3]; T_1[1] = camera_1[4]; T_1[2] = camera_1[5];

        R_2[0] = camera_2[0]; R_2[1] = camera_2[1]; R_2[2] = camera_2[2];
        T_2[0] = camera_2[3]; T_2[1] = camera_2[4]; T_2[2] = camera_2[5];

        T K_1[4], K_2[4];
        K_1[0] = camera_1[6];
        K_1[1] = camera_1[7];
        K_1[2] = camera_1[8];
        K_1[3] = camera_1[9];

        K_2[0] = camera_2[6];
        K_2[1] = camera_2[7];
        K_2[2] = camera_2[8];
        K_2[3] = camera_2[9];

        T px2point[3];
        px2point[0] = (T(pt_1.x) - K_1[2]) / K_1[0];
        px2point[1] = (T(pt_1.y) - K_1[3]) / K_1[1];
        px2point[2] = T(1.);

        T R_1_m [9], R_2_m [9];
        ceres::AngleAxisToRotationMatrix(R_1, R_1_m);
        ceres::AngleAxisToRotationMatrix(R_2, R_2_m);

        T R_21 [9];
        R_21[0] = R_2_m[0] * R_1_m[0] + R_2_m[3] * R_1_m[3] + R_2_m[6] * R_1_m[6];
        R_21[1] = R_2_m[1] * R_1_m[0] + R_2_m[4] * R_1_m[3] + R_2_m[7] * R_1_m[6];
        R_21[2] = R_2_m[2] * R_1_m[0] + R_2_m[5] * R_1_m[3] + R_2_m[8] * R_1_m[6];
        R_21[3] = R_2_m[0] * R_1_m[1] + R_2_m[3] * R_1_m[4] + R_2_m[6] * R_1_m[7];
        R_21[4] = R_2_m[1] * R_1_m[1] + R_2_m[4] * R_1_m[4] + R_2_m[7] * R_1_m[7];
        R_21[5] = R_2_m[2] * R_1_m[1] + R_2_m[5] * R_1_m[4] + R_2_m[8] * R_1_m[7];
        R_21[6] = R_2_m[0] * R_1_m[2] + R_2_m[3] * R_1_m[5] + R_2_m[6] * R_1_m[8];
        R_21[7] = R_2_m[1] * R_1_m[2] + R_2_m[4] * R_1_m[5] + R_2_m[7] * R_1_m[8];
        R_21[8] = R_2_m[2] * R_1_m[2] + R_2_m[5] * R_1_m[5] + R_2_m[8] * R_1_m[8];

        T T_21 [3];
        T_21[0] = T_2[0] - (R_21[0] * T_1[0] + R_21[3] * T_1[1] + R_21[6] * T_1[2]);
        T_21[1] = T_2[1] - (R_21[1] * T_1[0] + R_21[4] * T_1[1] + R_21[7] * T_1[2]);
        T_21[2] = T_2[2] - (R_21[2] * T_1[0] + R_21[5] * T_1[1] + R_21[8] * T_1[2]);

        T E_21 [9];
        E_21[0] = -T_21[2] * R_21[1] + T_21[1] * R_21[2];
        E_21[1] = T_21[2] * R_21[0] - T_21[0] * R_21[2];
        E_21[2] = -T_21[1] * R_21[0] + T_21[0] * R_21[1];

        E_21[3] = -T_21[2] * R_21[4] + T_21[1] * R_21[5];
        E_21[4] = T_21[2] * R_21[3] - T_21[0] * R_21[5];
        E_21[5] = -T_21[1] * R_21[3] + T_21[0] * R_21[4];

        E_21[6] = -T_21[2] * R_21[7] + T_21[1] * R_21[8];
        E_21[7] = T_21[2] * R_21[6] - T_21[0] * R_21[8];
        E_21[8] = -T_21[1] * R_21[6] + T_21[0] * R_21[7];

        T point2point[3];
        point2point[0] = E_21[0] * px2point[0] + E_21[3] * px2point[1] + E_21[6] * px2point[2];
        point2point[1] = E_21[1] * px2point[0] + E_21[4] * px2point[1] + E_21[7] * px2point[2];
        point2point[2] = E_21[2] * px2point[0] + E_21[5] * px2point[1] + E_21[8] * px2point[2];

        T epiline[3];
        epiline[0] = (T(1.) /  K_2[0]) * point2point[0];
        epiline[1] = (T(1.) /  K_2[1]) * point2point[1];
        epiline[2] = -(K_2[2] /  K_2[0]) * point2point[0] - (K_2[3] /  K_2[1]) * point2point[1] + point2point[2];

        T error = ceres::abs(epiline[0] * T(pt_2.x) + epiline[1] * T(pt_2.y) + epiline[2]) /
                  ceres::sqrt(epiline[0] * epiline[0] + epiline[1] * epiline[1]);

        residuals[0] = error;

        return true;
    }

    static ceres::CostFunction *Create(const cv::Point2d & pt_1, const cv::Point2d & pt_2) {
        return (new ceres::AutoDiffCostFunction<ReprojectionErrorBA, 1, 10, 10> (new ReprojectionErrorBA(pt_1, pt_2)));
    }

    cv::Point2d pt_1;
    cv::Point2d pt_2;
};

struct TripletErrorBA{
    TripletErrorBA(cv::Point2d pt_c, cv::Point2d pt_l, cv::Point2d pt_r, Eigen::Matrix3d K_eig)
            : pt_c(pt_c), pt_l(pt_l), pt_r(pt_r), K_eig(std::move(K_eig)) {}

    template<typename T>
    bool operator()(const T * const camera_c, const T * const camera_l, const T * const camera_r, T * residuals) const {

        T K[4];
        K[0] = T (K_eig(0, 0));
        K[1] = T (K_eig(1, 1));
        K[2] = T (K_eig(0, 2));
        K[3] = T (K_eig(1, 2));


        T AA_c[3], T_c[3], AA_l[3], T_l[3], AA_r[3], T_r[3];
        AA_c[0] = camera_c[0]; AA_c[1] = camera_c[1]; AA_c[2] = camera_c[2];
        T_c[0] = camera_c[3]; T_c[1] = camera_c[4]; T_c[2] = camera_c[5];

        AA_l[0] = camera_l[0]; AA_l[1] = camera_l[1]; AA_l[2] = camera_l[2];
        T_l[0] = camera_l[3]; T_l[1] = camera_l[4]; T_l[2] = camera_l[5];

        AA_r[0] = camera_r[0]; AA_r[1] = camera_r[1]; AA_r[2] = camera_r[2];
        T_r[0] = camera_r[3]; T_r[1] = camera_r[4]; T_r[2] = camera_r[5];


        // LEFT ------> CENTER
        T px2point[3];
        px2point[0] = (T(pt_l.x) - K[2]) / K[0];
        px2point[1] = (T(pt_l.y) - K[3]) / K[1];
        px2point[2] = T(1.);

        T R_c [9], R_l [9];
        ceres::AngleAxisToRotationMatrix(AA_c, R_c);
        ceres::AngleAxisToRotationMatrix(AA_l, R_l);

        T R_cl [9];
        R_cl[0] = R_c[0] * R_l[0] + R_c[3] * R_l[3] + R_c[6] * R_l[6];
        R_cl[1] = R_c[1] * R_l[0] + R_c[4] * R_l[3] + R_c[7] * R_l[6];
        R_cl[2] = R_c[2] * R_l[0] + R_c[5] * R_l[3] + R_c[8] * R_l[6];
        R_cl[3] = R_c[0] * R_l[1] + R_c[3] * R_l[4] + R_c[6] * R_l[7];
        R_cl[4] = R_c[1] * R_l[1] + R_c[4] * R_l[4] + R_c[7] * R_l[7];
        R_cl[5] = R_c[2] * R_l[1] + R_c[5] * R_l[4] + R_c[8] * R_l[7];
        R_cl[6] = R_c[0] * R_l[2] + R_c[3] * R_l[5] + R_c[6] * R_l[8];
        R_cl[7] = R_c[1] * R_l[2] + R_c[4] * R_l[5] + R_c[7] * R_l[8];
        R_cl[8] = R_c[2] * R_l[2] + R_c[5] * R_l[5] + R_c[8] * R_l[8];

        T T_cl [3];
        T_cl[0] = T_c[0] - (R_cl[0] * T_l[0] + R_cl[3] * T_l[1] + R_cl[6] * T_l[2]);
        T_cl[1] = T_c[1] - (R_cl[1] * T_l[0] + R_cl[4] * T_l[1] + R_cl[7] * T_l[2]);
        T_cl[2] = T_c[2] - (R_cl[2] * T_l[0] + R_cl[5] * T_l[1] + R_cl[8] * T_l[2]);

        T E_cl [9];
        E_cl[0] = -T_cl[2] * R_cl[1] + T_cl[1] * R_cl[2];
        E_cl[1] = T_cl[2] * R_cl[0] - T_cl[0] * R_cl[2];
        E_cl[2] = -T_cl[1] * R_cl[0] + T_cl[0] * R_cl[1];
        E_cl[3] = -T_cl[2] * R_cl[4] + T_cl[1] * R_cl[5];
        E_cl[4] = T_cl[2] * R_cl[3] - T_cl[0] * R_cl[5];
        E_cl[5] = -T_cl[1] * R_cl[3] + T_cl[0] * R_cl[4];
        E_cl[6] = -T_cl[2] * R_cl[7] + T_cl[1] * R_cl[8];
        E_cl[7] = T_cl[2] * R_cl[6] - T_cl[0] * R_cl[8];
        E_cl[8] = -T_cl[1] * R_cl[6] + T_cl[0] * R_cl[7];

        T point2point[3];
        point2point[0] = E_cl[0] * px2point[0] + E_cl[3] * px2point[1] + E_cl[6] * px2point[2];
        point2point[1] = E_cl[1] * px2point[0] + E_cl[4] * px2point[1] + E_cl[7] * px2point[2];
        point2point[2] = E_cl[2] * px2point[0] + E_cl[5] * px2point[1] + E_cl[8] * px2point[2];

        T epiline_cl[3];
        epiline_cl[0] = (T(1.) /  K[0]) * point2point[0];
        epiline_cl[1] = (T(1.) /  K[1]) * point2point[1];
        epiline_cl[2] = -(K[2] /  K[0]) * point2point[0] - (K[3] /  K[1]) * point2point[1] + point2point[2];




        // RIGHT ------> CENTER
        px2point[0] = (T(pt_r.x) - K[2]) / K[0];
        px2point[1] = (T(pt_r.y) - K[3]) / K[1];
        px2point[2] = T(1.);

        T R_r [9];
        ceres::AngleAxisToRotationMatrix(AA_r, R_r);

        T R_cr [9];
        R_cr[0] = R_c[0] * R_r[0] + R_c[3] * R_r[3] + R_c[6] * R_r[6];
        R_cr[1] = R_c[1] * R_r[0] + R_c[4] * R_r[3] + R_c[7] * R_r[6];
        R_cr[2] = R_c[2] * R_r[0] + R_c[5] * R_r[3] + R_c[8] * R_r[6];
        R_cr[3] = R_c[0] * R_r[1] + R_c[3] * R_r[4] + R_c[6] * R_r[7];
        R_cr[4] = R_c[1] * R_r[1] + R_c[4] * R_r[4] + R_c[7] * R_r[7];
        R_cr[5] = R_c[2] * R_r[1] + R_c[5] * R_r[4] + R_c[8] * R_r[7];
        R_cr[6] = R_c[0] * R_r[2] + R_c[3] * R_r[5] + R_c[6] * R_r[8];
        R_cr[7] = R_c[1] * R_r[2] + R_c[4] * R_r[5] + R_c[7] * R_r[8];
        R_cr[8] = R_c[2] * R_r[2] + R_c[5] * R_r[5] + R_c[8] * R_r[8];

        T T_cr [3];
        T_cr[0] = T_c[0] - (R_cr[0] * T_r[0] + R_cr[3] * T_r[1] + R_cr[6] * T_r[2]);
        T_cr[1] = T_c[1] - (R_cr[1] * T_r[0] + R_cr[4] * T_r[1] + R_cr[7] * T_r[2]);
        T_cr[2] = T_c[2] - (R_cr[2] * T_r[0] + R_cr[5] * T_r[1] + R_cr[8] * T_r[2]);

        T E_cr [9];
        E_cr[0] = -T_cr[2] * R_cr[1] + T_cr[1] * R_cr[2];
        E_cr[1] = T_cr[2] * R_cr[0] - T_cr[0] * R_cr[2];
        E_cr[2] = -T_cr[1] * R_cr[0] + T_cr[0] * R_cr[1];
        E_cr[3] = -T_cr[2] * R_cr[4] + T_cr[1] * R_cr[5];
        E_cr[4] = T_cr[2] * R_cr[3] - T_cr[0] * R_cr[5];
        E_cr[5] = -T_cr[1] * R_cr[3] + T_cr[0] * R_cr[4];
        E_cr[6] = -T_cr[2] * R_cr[7] + T_cr[1] * R_cr[8];
        E_cr[7] = T_cr[2] * R_cr[6] - T_cr[0] * R_cr[8];
        E_cr[8] = -T_cr[1] * R_cr[6] + T_cr[0] * R_cr[7];

        point2point[0] = E_cr[0] * px2point[0] + E_cr[3] * px2point[1] + E_cr[6] * px2point[2];
        point2point[1] = E_cr[1] * px2point[0] + E_cr[4] * px2point[1] + E_cr[7] * px2point[2];
        point2point[2] = E_cr[2] * px2point[0] + E_cr[5] * px2point[1] + E_cr[8] * px2point[2];

        T epiline_cr[3];
        epiline_cr[0] = (T(1.) /  K[0]) * point2point[0];
        epiline_cr[1] = (T(1.) /  K[1]) * point2point[1];
        epiline_cr[2] = -(K[2] /  K[0]) * point2point[0] - (K[3] /  K[1]) * point2point[1] + point2point[2];

        T x_pred = (epiline_cl[1] * epiline_cr[2] - epiline_cr[1] * epiline_cl[2]) /
                   (epiline_cl[0] * epiline_cr[1] - epiline_cr[0] * epiline_cl[1]);
        T y_pred = (epiline_cl[2] * epiline_cr[0] - epiline_cr[2] * epiline_cl[0]) /
                   (epiline_cl[0] * epiline_cr[1] - epiline_cr[0] * epiline_cl[1]);

        residuals[0] = x_pred - pt_c.x;
        residuals[1] = y_pred - pt_c.y;

        return true;
    }

    static ceres::CostFunction *Create(const cv::Point2d & pt_c,
                                       const cv::Point2d & pt_l,
                                       const cv::Point2d & pt_r,
                                       const Eigen::Matrix3d & K_eig) {
        return (new ceres::AutoDiffCostFunction<TripletErrorBA, 2, 6, 6, 6>(
                new TripletErrorBA(pt_c, pt_l, pt_r, K_eig)));
    }

    cv::Point2d pt_c, pt_l, pt_r;
    Eigen::Matrix3d K_eig;
};

struct RotationError {
    RotationError(Eigen::Vector3d c_k,
                  Eigen::Vector3d c_q,
                  Eigen::Vector3d t_qk)
            : c_k(std::move(c_k)), c_q(std::move(c_q)), t_qk(std::move(t_qk)) {}

    template<typename T>
    bool operator()(const T * const R, const T * const lambda, T * residuals) const {

        T x = R[0];
        T y = R[1];
        T z = R[2];

        T Z = T(1) + x*x + y*y + z*z;

        T R_q [9];
        R_q[0] = (T(1) + x*x - y*y - z*z) / Z;
        R_q[1] = (T(2)*x*y - T(2)*z) / Z;
        R_q[2] = (T(2)*x*z + T(2)*y) / Z;
        R_q[3] = (T(2)*x*y + T(2)*z) / Z;
        R_q[4] = (T(1) + y*y - x*x - z*z) / Z;
        R_q[5] = (T(2)*y*z - T(2)*x) / Z;
        R_q[6] = (T(2)*x*z - T(2)*y) / Z;
        R_q[7] = (T(2)*y*z + T(2)*x) / Z;
        R_q[8] = (T(1) + z*z - x*x - y*y) / Z;

        T ray[3];
        ray[0] = lambda[0] * (R_q[0]*T(t_qk[0]) + R_q[3]*T(t_qk[1]) + R_q[6]*T(t_qk[2]));
        ray[1] = lambda[0] * (R_q[1]*T(t_qk[0]) + R_q[4]*T(t_qk[1]) + R_q[7]*T(t_qk[2]));
        ray[2] = lambda[0] * (R_q[2]*T(t_qk[0]) + R_q[5]*T(t_qk[1]) + R_q[8]*T(t_qk[2]));

        T x_error = T(c_k[0]) - (T(c_q[0]) + ray[0]);
        T y_error = T(c_k[1]) - (T(c_q[1]) + ray[1]);
        T z_error = T(c_k[2]) - (T(c_q[2]) + ray[2]);

        residuals[0] = x_error;
        residuals[1] = y_error;
        residuals[2] = z_error;

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *Create(const Eigen::Vector3d & c_k,
                                       const Eigen::Vector3d & c_q,
                                       const Eigen::Vector3d & t_qk) {
        return (new ceres::AutoDiffCostFunction<RotationError, 3, 3, 1>(
                new RotationError(c_k, c_q, t_qk)));
    }

    Eigen::Vector3d c_k;
    Eigen::Vector3d c_q;
    Eigen::Vector3d t_qk;
};


struct SimultaneousEstimation {
    SimultaneousEstimation(Eigen::Vector3d c_k,
                           Eigen::Vector3d t_qk)
            : c_k(std::move(c_k)), t_qk(std::move(t_qk)) {}

    template<typename T>
    bool operator()(const T * const c_q, const T * const r_q, const T * const lambda, T * residuals) const {

        T R_q [9];
        ceres::AngleAxisToRotationMatrix(r_q, R_q);

        T ray[3];
        ray[0] = lambda[0] * (R_q[0]*T(t_qk[0]) + R_q[1]*T(t_qk[1]) + R_q[2]*T(t_qk[2]));
        ray[1] = lambda[0] * (R_q[3]*T(t_qk[0]) + R_q[4]*T(t_qk[1]) + R_q[5]*T(t_qk[2]));
        ray[2] = lambda[0] * (R_q[6]*T(t_qk[0]) + R_q[7]*T(t_qk[1]) + R_q[8]*T(t_qk[2]));

        T x_error = T(c_k[0]) - (c_q[0] + ray[0]);
        T y_error = T(c_k[1]) - (c_q[1] + ray[1]);
        T z_error = T(c_k[2]) - (c_q[2] + ray[2]);

        residuals[0] = x_error;
        residuals[1] = y_error;
        residuals[2] = z_error;

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *Create(const Eigen::Vector3d & c_k,
                                       const Eigen::Vector3d & t_qk) {
        return (new ceres::AutoDiffCostFunction<SimultaneousEstimation, 3, 3, 3, 1>(
                new SimultaneousEstimation(c_k, t_qk)));
    }

    Eigen::Vector3d c_k;
    Eigen::Vector3d t_qk;
};


void pose::estimatePose(const vector<Eigen::Matrix3d> & R_ks,
                        const vector<Eigen::Vector3d> & T_ks,
                        const vector<Eigen::Vector3d> & T_qks,
                        Eigen::Matrix3d & R,
                        Eigen::Vector3d & c) {

    int k = int(R_ks.size());

    double c_q_arr [3] {c[0], c[1], c[2]};
    double AA_arr [3];
    double R_arr[9]{R(0, 0), R(1, 0), R(2, 0), R(0, 1), R(1, 1), R(2, 1), R(0, 2), R(1, 2), R(2, 2)};
    ceres::RotationMatrixToAngleAxis(R_arr, AA_arr);

    ceres::Problem problem;
    ceres::LossFunction * loss_function = new ceres::HuberLoss(1.0);
    for (int i = 0; i < k; i++) {
        Eigen::Vector3d c_k = - R_ks[i].transpose() * T_ks[i];
        double lambda [1] {(c_k - c).transpose() * R.transpose() * T_qks[i]};
        ceres::CostFunction *cost_function = SimultaneousEstimation::Create(c_k, T_qks[i]);
        problem.AddResidualBlock(cost_function, loss_function, c_q_arr, AA_arr, lambda);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

//    cout << summary.FullReport() << "\n";

    ceres::AngleAxisToRotationMatrix(AA_arr, R_arr);
    R = Eigen::Matrix3d {{R_arr[0], R_arr[3], R_arr[6]},
                         {R_arr[1], R_arr[4], R_arr[7]},
                         {R_arr[2], R_arr[5], R_arr[8]}};
    c = Eigen::Vector3d {c_q_arr[0], c_q_arr[1], c_q_arr[2]};
}









/// BUNDLE ADJUSTMENT
string formatNum(int n, int num_digits) {
    string s = to_string(n);
    int num_zeros = num_digits - int (s.size());
    for(int i = 0; i < num_zeros; i++) {
        string zero = "0";
        s = zero.append(s);
    }
    return s;
}

void percentageBar(double fraction_done) {

    int bar_stops = int (100 * fraction_done);

    cout << "\r" << "[";
    for (int i = 0; i < bar_stops; i++) {
        cout << "|";
    }
    for (int i = 0; i < 100 - bar_stops; i++) {
        cout << " ";
    }
    cout << "] ";

}

void pose::sceneBundleAdjust(const int num_ims, const double K[4],
                             const string & folder, const string & scene, const string & sequence, const string & frame,
                             const string & pose_ext, const string & rgb_ext) {

    //Make sure that array sizes and pose acquisition functions are appropriate for dataset
    cout << "Bundle adjustment for " << scene << " scene, sequence " << sequence << "." << endl;

    string base = folder + "/" + scene + "/" + sequence + "/" + frame;
    Eigen::Matrix3d K_eig {{K[0], 0.,   K[2]},
                           {0.,   K[1], K[3]},
                           {0.,   0.,     1.}};
    double cameras [1000][10];
    int bit_arr [1000];
    vector<Eigen::Matrix3d> R_ks;
    vector<Eigen::Vector3d> T_ks;
    vector<pair<cv::Mat, vector<cv::KeyPoint>>> d_kps;

    for (int i = 0; i < num_ims; i++) {

        string im = base + formatNum(i, 6);

        bit_arr[i] = 0;

        d_kps.push_back(functions::getDescriptors(im, rgb_ext, "SIFT"));

        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        sevenScenes::getAbsolutePose(im, R, T);
        R_ks.push_back(R);
        T_ks.push_back(T);

        double AA_arr[3];
        double T_arr[3] {T[0], T[1], T[2]};
        double R_arr[9]{R(0, 0), R(1, 0), R(2, 0),
                        R(0, 1), R(1, 1), R(2, 1),
                        R(0, 2), R(1, 2), R(2, 2)};
        ceres::RotationMatrixToAngleAxis(R_arr, AA_arr);

        cameras[i][0] = AA_arr[0];
        cameras[i][1] = AA_arr[1];
        cameras[i][2] = AA_arr[2];

        cameras[i][3] = T_arr[0];
        cameras[i][4] = T_arr[1];
        cameras[i][5] = T_arr[2];

        cameras[i][6] = K[0];
        cameras[i][7] = K[1];
        cameras[i][8] = K[2];
        cameras[i][9] = K[3];
        cout << "Descriptor " << i << endl;
    }

    double match_theshold = 2.;
    int min_good_matches = 50;
    ceres::Problem problem;
    ceres::LossFunction * loss_function = new ceres::HuberLoss(1.0);
    cout << "Adding residuals..." << endl;
    for(int i = 0; i < num_ims - 1; i++) {
        for(int j = i + 1; j < num_ims; j++) {

            Eigen::Matrix3d R_ji = R_ks[j] * R_ks[i].transpose();
            Eigen::Vector3d T_ji = T_ks[j] - R_ji * T_ks[i];
            if (T_ji.norm() == 0) continue;
            T_ji.normalize();
            Eigen::Matrix3d T_ji_cross{{0,        -T_ji(2), T_ji(1)},
                                       {T_ji(2),  0,        -T_ji(0)},
                                       {-T_ji(1), T_ji(0),  0}};
            Eigen::Matrix3d E_ji = T_ji_cross * R_ji;
            Eigen::Matrix3d F_ji = K_eig.inverse().transpose() * E_ji * K_eig.inverse();

            vector<cv::Point2d> pts_i, pts_j;
            functions::findMatches(0.8, d_kps[i].first, d_kps[i].second, d_kps[j].first, d_kps[j].second, pts_i, pts_j);
            cout << "Matching " << i << " with " << j << endl;

            vector<int> indices;
            for(int k = 0; k < pts_i.size(); k++) {
                Eigen::Vector3d pt_i {pts_i[k].x, pts_i[k].y, 1.};
                Eigen::Vector3d pt_j {pts_j[k].x, pts_j[k].y, 1.};

                Eigen::Vector3d epiline = F_ji * pt_i;

                double error = abs(epiline[0] * pt_j[0] + epiline[1] * pt_j[1] + epiline[2]) /
                               sqrt(epiline[0] * epiline[0] + epiline[1] * epiline[1]);

                if(error <= match_theshold) {
                    indices.push_back(k);
                }
            }

            if(indices.size() >= min_good_matches) {
                for(const auto & idx : indices) {
                    ceres::CostFunction *cost_function = ReprojectionErrorBA::Create(pts_i[idx], pts_j[idx]);
                    problem.AddResidualBlock(cost_function, loss_function, cameras[i], cameras[j]);
                }
            }
        }
    }

    cout << endl << "Bundle Adjusting..." << endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.max_num_iterations = 1000;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << "\n";

    cout << endl << "Writing Bundle Adjusted Pose Files..." << endl;
    for (int i = 0; i < num_ims; i++) {
        double AA_arr [3] = {cameras[i][0], cameras[i][1], cameras[i][2]};
        double T_arr [3] = {cameras[i][3], cameras[i][4], cameras[i][5]};
        double R_arr[9];
        ceres::AngleAxisToRotationMatrix(AA_arr, R_arr);
        Eigen::Matrix3d R {{R_arr[0], R_arr[3], R_arr[6]},
                           {R_arr[1], R_arr[4], R_arr[7]},
                           {R_arr[2], R_arr[5], R_arr[8]}};
        Eigen::Vector3d T {T_arr[0], T_arr[1], T_arr[2]};

        string fn = base; fn += formatNum(i, 6); fn += ".bundle_adjusted"; fn += pose_ext;
        ofstream BA_pose;
        BA_pose.open(fn);
        for(int row = 0; row < 4; row++) {
            if (row < 3) {
                for (int col = 0; col < 4; col++) {
                    if (col < 3) {
                        double num = R(row, col);
                        if (num < 0) BA_pose << setprecision(16) << num << " ";
                        else BA_pose << " " << setprecision(16) << num << " ";
                    } else {
                        double num = T(row);
                        if (num < 0) BA_pose << setprecision(16) << num << " ";
                        else BA_pose << " " << setprecision(16) << num << " ";
                    }
                }
            } else {
                for (int col = 0; col < 4; col++) {
                    if (col < 3) {
                        BA_pose << " " << 0 << "                  ";
                    } else {
                        BA_pose << " " << 1 << "                  ";
                    }
                }
            }
            BA_pose << endl;
        }
        BA_pose << "Transformation matrix from world to camera coordinates.";
        BA_pose.close();
    }
    cout << "Done." << endl;
}



/// PERFORMANCE TESTING
double pose::averageReprojectionError (const double K[4],
                                       const string & query,
                                       const Eigen::Matrix3d & R_q,
                                       const Eigen::Vector3d & T_q,
                                       const vector<string> & anchors,
                                       const vector<Eigen::Matrix3d> & R_ks,
                                       const vector<Eigen::Vector3d> & T_ks) {

    double total_rep_error = 0;
    int count = 0;

    if (anchors.size() != R_ks.size() && anchors.size() != T_ks.size()) {
        cout << "problem" << endl;
        exit(0);
    }

    Eigen::Matrix3d K_eig{{K[0], 0.,   K[2]},
                          {0.,   K[1], K[3]},
                          {0.,   0.,   1.}};

    for(int i = 0; i < anchors.size(); i++) {
        vector<cv::Point2d> pts_im, pts_q;
        functions::findMatches(anchors[i], query, ".color.png", "SIFT", 0.6, pts_im, pts_q);

        Eigen::Matrix3d R_qk = R_q * R_ks[i].transpose();
        Eigen::Vector3d T_qk = T_q - R_qk * T_ks[i];
        T_qk.normalize();
        Eigen::Matrix3d t_cross {
                {0,               -T_qk(2), T_qk(1)},
                {T_qk(2),  0,              -T_qk(0)},
                {-T_qk(1), T_qk(0),               0}};
        Eigen::Matrix3d E_real = t_cross * R_qk;
        Eigen::Matrix3d F = K_eig.inverse().transpose() * E_real * K_eig.inverse();

        for(int m = 0; m < pts_im.size(); m++) {
            Eigen::Vector3d pt_q {pts_q[m].x, pts_q[m].y, 1.};
            Eigen::Vector3d pt_im {pts_im[m].x, pts_im[m].y, 1.};
            Eigen::Vector3d epiline_real = F * pt_im;
            double error = abs(epiline_real[0] * pt_q[0] + epiline_real[1] * pt_q[1] + epiline_real[2]) /
                           sqrt(epiline_real[0] * epiline_real[0] + epiline_real[1] * epiline_real[1]);
            total_rep_error += error;
            count++;
        }
    }
    return total_rep_error / count;
}

double pose::tripletCircles (const double K[4], const vector<pair<cv::Mat, vector<cv::KeyPoint>>> & desc_kp_anchors) {

    double total_error = 0;
    int count = 0;

    for(int i = 0; i < desc_kp_anchors.size() - 2; i++) {

        int j = i + 1;
        if (j >= desc_kp_anchors.size() - 1) break;

        int k = j + 1;
        if (k >= desc_kp_anchors.size()) break;

        auto desc_kp_i = desc_kp_anchors[i];
        auto desc_kp_j = desc_kp_anchors[j];
        auto desc_kp_k = desc_kp_anchors[k];


        vector<cv::Point2d> pts_i_j, pts_j_i;
        functions::findMatches(0.8, desc_kp_i.first, desc_kp_i.second, desc_kp_j.first, desc_kp_j.second, pts_i_j, pts_j_i);
        if (pts_i_j.size() < 5) continue;
        Eigen::Matrix3d R_ji;
        Eigen::Vector3d T_ji;
        if(!functions::getRelativePose(pts_i_j, pts_j_i, K, 1.0, R_ji, T_ji)) continue;


        vector<cv::Point2d> pts_j_k, pts_k_j;
        functions::findMatches(0.8, desc_kp_j.first, desc_kp_j.second, desc_kp_k.first, desc_kp_k.second, pts_j_k, pts_k_j);
        if (pts_j_k.size() < 5) continue;
        Eigen::Matrix3d R_kj;
        Eigen::Vector3d T_kj;
        if(!functions::getRelativePose(pts_j_k, pts_k_j, K, 1.0, R_kj, T_kj)) continue;


        vector<cv::Point2d> pts_i_k, pts_k_i;
        functions::findMatches(0.8, desc_kp_i.first, desc_kp_i.second, desc_kp_k.first, desc_kp_k.second, pts_i_k, pts_k_i);
        if (pts_i_k.size() < 5) continue;
        Eigen::Matrix3d R_ki;
        Eigen::Vector3d T_ki;
        if(!functions::getRelativePose(pts_i_k, pts_k_i, K, 1.0, R_ki, T_ki)) continue;

        Eigen::Matrix3d R_ki_composed = R_kj * R_ji;

        double error = functions::rotationDifference(R_ki, R_ki_composed);

        total_error += error;
        count++;

    }
    return total_error / count;
}



/// CENTER HYPOTHESIS
Eigen::Vector3d pose::c_q_closed_form (const vector<Eigen::Matrix3d> & R_ks,
                                       const vector<Eigen::Vector3d> & T_ks,
                                       const vector<Eigen::Matrix3d> & R_qks,
                                       const vector<Eigen::Vector3d> & T_qks) {

    size_t K = R_ks.size();
    Eigen::Matrix3d I {{1, 0, 0},
                       {0, 1, 0},
                       {0, 0, 1}};

    Eigen::MatrixXd A (K*3, 3);
    Eigen::MatrixXd b (K*3, 1);

    for (int i = 0; i < K; i++) {
        Eigen::Matrix3d R_k = R_ks[i];
        Eigen::Vector3d T_k = T_ks[i];
        Eigen::Matrix3d R_qk = R_qks[i];
        Eigen::Vector3d T_qk = T_qks[i];

        Eigen::Vector3d c_k = - R_k.transpose() * T_k;
        Eigen::Vector3d T_kq = -R_qk.transpose() * T_qk;

        Eigen::Matrix3d M_kq = I - R_k.transpose() * T_kq * T_kq.transpose() * R_k;

        A.block(i*3,0,3,3) = M_kq;
        b.block(i*3,0,3,1) = M_kq * c_k;
    }

    Eigen::Vector3d c_q = A.colPivHouseholderQr().solve(b);

    return c_q;
}



//// TRANSLATION HYPOTHESIS
Eigen::Vector3d pose::T_q_govindu(const vector<Eigen::Vector3d> & T_ks,
                                  const vector<Eigen::Matrix3d> & R_qks,
                                  const vector<Eigen::Vector3d> & T_qks)
{

    int nImgs = int(T_ks.size());

    //Intermediate arrays
    Eigen::MatrixXd A(nImgs*3,3);
    Eigen::MatrixXd b(nImgs*3,1);

    //Info of our query

    //Weights:
    double lambda[nImgs];
    for (int i = 0; i < nImgs; i++)
        lambda[i] = 1.0;

    //Result container
    Eigen::MatrixXd results_prev(3,1);
    results_prev << 100,100,100;
    Eigen::MatrixXd results_curr(3,1);
    results_curr << 0,0,0;
    double dist_prev = 1e8;

    //Iterative method shown in the paper
    while((results_prev - results_curr).norm() > 1e-8 && dist_prev > (results_prev - results_curr).norm())
    {
        dist_prev = (results_prev - results_curr).norm();
        results_prev = results_curr;
        //Build Linear systems
        for (int i = 0; i < nImgs; i++)
        {

            Eigen::Matrix<double,3,3> R12 = R_qks[i];
            Eigen::Matrix<double,3,1> T2 = T_ks[i];
            Eigen::Matrix<double,3,1> T12 = T_qks[i];

            Eigen::MatrixXd t_ij_x(3,3);
            t_ij_x.setZero();
            t_ij_x(0,1) = -T12(2,0);
            t_ij_x(1,0) = T12(2,0);
            t_ij_x(0,2) = T12(1,0);
            t_ij_x(2,0) = -T12(1,0);
            t_ij_x(1,2) = -T12(0,0);
            t_ij_x(2,1) = T12(0,0);

            A.block(i*3,0,3,3) = lambda[i] * t_ij_x * R12;
            b.block(i*3,0,3,1) = lambda[i] * t_ij_x * T2;
        }
        //Solve linear system
        results_curr = A.colPivHouseholderQr().solve(b);

        //Update weights
        for (int i = 0; i < nImgs; i++)
        {
            Eigen::Matrix<double,3,3> R12 = R_qks[i];
            Eigen::Matrix<double,3,1> T2 = T_ks[i];

            lambda[i] = 1 / ((T2 - R12 * results_curr).norm());
        }
    }

    return results_curr;
}

Eigen::Vector3d pose::T_q_closed_form (const vector<Eigen::Vector3d> & T_ks,
                                       const vector<Eigen::Matrix3d> & R_qks,
                                       const vector<Eigen::Vector3d> & T_qks) {

    size_t K = T_ks.size();

    Eigen::Matrix3d I {{1, 0, 0},
                       {0, 1, 0},
                       {0, 0, 1}};

    Eigen::MatrixXd A(K*3,3);
    Eigen::MatrixXd b(K*3,1);

    for (int i = 0; i < K; i++) {

        const Eigen::Vector3d& T_k = T_ks[i];
        const Eigen::Matrix3d& R_qk = R_qks[i];
        const Eigen::Vector3d& T_qk = T_qks[i];

        Eigen::Matrix3d M = I - T_qk * T_qk.transpose();

        A.block(i*3,0,3,3) = M;
        b.block(i*3,0,3,1) = M * R_qk * T_k;

    }

    Eigen::Vector3d T_q = A.colPivHouseholderQr().solve(b);
    return T_q;
}



//// ROTATION HYPOTHESIS
template<typename DataType, typename ForwardIterator>
Eigen::Quaternion<DataType> pose::averageQuaternions(ForwardIterator const & begin, ForwardIterator const & end) {

    if (begin == end) {
        throw logic_error("Cannot average orientations over an empty range.");
    }

    Eigen::Matrix<DataType, 4, 4> A = Eigen::Matrix<DataType, 4, 4>::Zero();
    uint sum(0);
    for (ForwardIterator it = begin; it != end; ++it) {
        Eigen::Matrix<DataType, 1, 4> q(1,4);
        q(0) = it->w();
        q(1) = it->x();
        q(2) = it->y();
        q(3) = it->z();
        auto a = q.transpose()*q;
        A += a;
        sum++;
    }
    A /= sum;

    Eigen::EigenSolver<Eigen::Matrix<DataType, 4, 4>> es(A);

    Eigen::Matrix<complex<DataType>, 4, 1> mat(es.eigenvalues());
    int index;
    mat.real().maxCoeff(&index);
    Eigen::Matrix<DataType, 4, 1> largest_ev(es.eigenvectors().real().block(0, index, 4, 1));

    return Eigen::Quaternion<DataType>(largest_ev(0), largest_ev(1), largest_ev(2), largest_ev(3));
}

Eigen::Matrix3d pose::R_q_average (const vector<Eigen::Matrix3d> & rotations) {

    vector<Eigen::Quaterniond> quaternions;
    for (const auto & rot: rotations) {
        Eigen::Quaterniond q(rot);
        quaternions.push_back(q);
    }

    Eigen::Quaterniond avg_Q = averageQuaternions<double>(quaternions.begin(), quaternions.end());
    Eigen::Matrix3d avg_M = avg_Q.toRotationMatrix();
    return avg_M;

}

Eigen::Matrix3d pose::R_q_average_govindu(const vector<Eigen::Matrix3d> & R_qis, const vector<Eigen::Matrix3d> & R_is) {

    size_t K = R_is.size();
    Eigen::MatrixXd A (K * 4, 4);
    Eigen::MatrixXd b (K * 4, 1);

    for (int i = 0; i < K; i++) {
        Eigen::Matrix3d R_qi = R_qis[i];
        Eigen::Matrix3d R_i = R_is[i];

        Eigen::Quaterniond q_ri (R_is[i]);
        Eigen::Vector4d a {q_ri.w(), q_ri.x(), q_ri.y(), q_ri.z()};

        Eigen::Matrix3d R_qi_T = R_qis[i].transpose();
        Eigen::Quaterniond q (R_qi_T);

        Eigen::Matrix4d Q {{q.w(), -q.x(), -q.y(), -q.z()},
                           {q.x(), q.w(), -q.z(), q.y()},
                           {q.y(), q.z(), q.w(), -q.x()},
                           {q.z(), -q.y(), q.x(), q.w()}};

        A.block(i * 4, 0, 4, 4) = Q;
        b.block(i * 4, 0, 4, 1) = a;
    }

    Eigen::Vector4d sol = A.colPivHouseholderQr().solve(b);

    sol.normalize();

    Eigen::Quaterniond q_q (sol[0], sol[1], sol[2], sol[3]);
    Eigen::Matrix3d R_q = q_q.toRotationMatrix();

    return R_q;
}

Eigen::Matrix3d pose::R_q_closed_form (bool use_Rqk, bool normalize, bool version,
                                       const Eigen::Vector3d & c_q,
                                       const vector<Eigen::Matrix3d> & R_ks,
                                       const vector<Eigen::Vector3d> & T_ks,
                                       const vector<Eigen::Matrix3d> & R_qks,
                                       const vector<Eigen::Vector3d> & T_qks) {

    Eigen::Matrix3d R_q;

    if (version) {
        if (use_Rqk) {
            R_q = rotation::optimal_rotation_using_R_qks_NEW(c_q, R_ks, T_ks, R_qks, T_qks);
        } else {
            R_q = rotation::optimal_rotation_using_T_qks_NEW(c_q, R_ks, T_ks, R_qks, T_qks);
        }
        return R_q;
    }


    if (use_Rqk) {
        if (normalize) {
            R_q = rotation::NORM_optimal_rotation_using_R_qks(c_q, R_ks, T_ks, R_qks, T_qks);
        } else {
            R_q = rotation::optimal_rotation_using_R_qks_NEW_2(c_q, R_ks, T_ks, R_qks, T_qks);
        }
    } else {
        if (normalize) {
            R_q = rotation::NORM_optimal_rotation_using_T_qks(c_q, R_ks, T_ks, R_qks, T_qks);
        } else {
            R_q = rotation::optimal_rotation_using_T_qks_NEW_2(c_q, R_ks, T_ks, R_qks, T_qks);
        }
    }

    return R_q;
}



//// RANSAC
tuple<Eigen::Vector3d,
//Eigen::Vector3d,
//Eigen::Vector3d,
//Eigen::Matrix3d,
        Eigen::Matrix3d,
//Eigen::Matrix3d,
//Eigen::Matrix3d,
//Eigen::Matrix3d,
//Eigen::Matrix3d,
//Eigen::Matrix3d,
        vector<int>>
pose::hypothesizeRANSAC(double threshold,
                        const vector<Eigen::Matrix3d> & R_ks,
                        const vector<Eigen::Vector3d> & T_ks,
                        const vector<Eigen::Matrix3d> & R_qks,
                        const vector<Eigen::Vector3d> & T_qks) {

    int K = int(R_ks.size());

    vector<Eigen::Matrix3d> best_R_ks;
    vector<Eigen::Vector3d> best_T_ks;
    vector<Eigen::Matrix3d> best_R_qks;
    vector<Eigen::Vector3d> best_T_qks;
    vector<int> best_indices;
    double best_score = 0;

    for (int i = 0; i < K - 1; i++) {
        for (int j = i + 1; j < K; j++) {
            vector<Eigen::Matrix3d> set_R_ks {R_ks[i], R_ks[j]};
            vector<Eigen::Vector3d> set_T_ks {T_ks[i], T_ks[j]};
            vector<Eigen::Matrix3d> set_R_qks {R_qks[i], R_qks[j]};
            vector<Eigen::Vector3d> set_T_qks {T_qks[i], T_qks[j]};
            vector<int> indices{i, j};
            double score = 0;

            Eigen::Matrix3d R_h = pose::R_q_average(vector<Eigen::Matrix3d>{R_qks[i] * R_ks[i], R_qks[j] * R_ks[j]});
            Eigen::Vector3d c_h = pose::c_q_closed_form(set_R_ks, set_T_ks, set_R_qks, set_T_qks);

            for (int k = 0; k < K; k++) {
                if (k != i && k != j) {

                    Eigen::Matrix3d R_qk_h = R_h * R_ks[k].transpose();
                    Eigen::Vector3d T_qk_h = -R_qk_h * (R_ks[k] * c_h + T_ks[k]);

                    double T_angular_diff = functions::getAngleBetween(T_qk_h, T_qks[k]);
                    double R_angular_diff = functions::rotationDifference(R_qk_h, R_qks[k]);

                    if (T_angular_diff <= threshold && R_angular_diff <= threshold) {
                        set_R_ks.push_back(R_ks[k]);
                        set_T_ks.push_back(T_ks[k]);
                        set_R_qks.push_back(R_qks[k]);
                        set_T_qks.push_back(T_qks[k]);
                        indices.push_back(k);
                        score += R_angular_diff + T_angular_diff;
                    }
                }
            }

            if (set_R_ks.size() > best_R_ks.size() || (set_R_ks.size() == best_R_ks.size() && score < best_score)) {
                best_R_ks = set_R_ks;
                best_T_ks = set_T_ks;
                best_R_qks = set_R_qks;
                best_T_qks = set_T_qks;
                best_indices = indices;
                best_score = score;
            }
        }
    }

    vector<Eigen::Matrix3d> rotations(best_R_ks.size());
    for (int i = 0; i < best_R_ks.size(); i++) {
        rotations[i] = best_R_qks[i] * best_R_ks[i];
    }

//    auto t_q_s = pose::T_q_closed_form (best_R_ks, best_T_ks, best_R_qks, best_T_qks);

    // Center Methods
    Eigen::Vector3d c = pose::c_q_closed_form(best_R_ks, best_T_ks, best_R_qks, best_T_qks);


    // Translation Methods
//    Eigen::Vector3d T_gov = pose::T_q_govindu(best_T_ks, best_R_qks, best_T_qks);
//    Eigen::Vector3d T_cf = pose::T_q_closed_form(best_R_ks, best_T_ks, best_R_qks, best_T_qks);


    // Rotation Methods
    Eigen::Matrix3d R_avg = pose::R_q_average(rotations);
//    Eigen::Matrix3d R_cf_R = pose::R_q_closed_form(true, false, false, c, best_R_ks, best_T_ks, best_R_qks, best_T_qks);
//    Eigen::Matrix3d R_cf_R_NORM = pose::R_q_closed_form(true, true, false, c, best_R_ks, best_T_ks, best_R_qks, best_T_qks);
//    Eigen::Matrix3d R_cf_R_NEW = pose::R_q_closed_form(true, false, true, c, best_R_ks, best_T_ks, best_R_qks, best_T_qks);
//
//    Eigen::Matrix3d R_cf_T = pose::R_q_closed_form(false, false, false, c, best_R_ks, best_T_ks, best_R_qks, best_T_qks);
//    Eigen::Matrix3d R_cf_T_NORM = pose::R_q_closed_form(false, true, false, c, best_R_ks, best_T_ks, best_R_qks, best_T_qks);
//    Eigen::Matrix3d R_cf_T_NEW = pose::R_q_closed_form(false, false, true, c, best_R_ks, best_T_ks, best_R_qks, best_T_qks);

    return {
            c,
//            T_gov,
//            T_cf,
            R_avg,
//            R_cf_R,
//            R_cf_R_NORM,
//            R_cf_T,
//            R_cf_T_NORM,
//            R_cf_R_NEW,
//            R_cf_T_NEW,
            best_indices
    };
}