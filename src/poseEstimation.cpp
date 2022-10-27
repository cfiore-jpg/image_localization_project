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
struct ReprojectionError{
    ReprojectionError(Eigen::Matrix3d R_i,
                      Eigen::Vector3d T_i,
                      Eigen::Matrix3d K_i,
                      Eigen::Matrix3d K_q,
                      cv::Point2d pt_q,
                      cv::Point2d pt_i)
                      : R_i(move(R_i)), T_i(move(T_i)), K_i(move(K_i)), K_q(move(K_q)), pt_q(move(pt_q)), pt_i(move(pt_i)) {}

    template<typename T>
    bool operator()(const T * const R, const T * const T_q, T * residuals) const {

        T R_q [9];
        ceres::AngleAxisToRotationMatrix(R, R_q);

        T px2point[3];
        px2point[0] = (T(pt_i.x) - K_i(0,2)) / K_i(0,0);
        px2point[1] = (T(pt_i.y) - K_i(1,2)) / K_i(1,1);
        px2point[2] = T(1.);

        T R_qk [9];
        R_qk[0] = R_q[0] * T(R_i(0,0)) + R_q[3] * T(R_i(0,1)) + R_q[6] * T(R_i(0,2));
        R_qk[1] = R_q[1] * T(R_i(0,0)) + R_q[4] * T(R_i(0,1)) + R_q[7] * T(R_i(0,2));
        R_qk[2] = R_q[2] * T(R_i(0,0)) + R_q[5] * T(R_i(0,1)) + R_q[8] * T(R_i(0,2));
        R_qk[3] = R_q[0] * T(R_i(1,0)) + R_q[3] * T(R_i(1,1)) + R_q[6] * T(R_i(1,2));
        R_qk[4] = R_q[1] * T(R_i(1,0)) + R_q[4] * T(R_i(1,1)) + R_q[7] * T(R_i(1,2));
        R_qk[5] = R_q[2] * T(R_i(1,0)) + R_q[5] * T(R_i(1,1)) + R_q[8] * T(R_i(1,2));
        R_qk[6] = R_q[0] * T(R_i(2,0)) + R_q[3] * T(R_i(2,1)) + R_q[6] * T(R_i(2,2));
        R_qk[7] = R_q[1] * T(R_i(2,0)) + R_q[4] * T(R_i(2,1)) + R_q[7] * T(R_i(2,2));
        R_qk[8] = R_q[2] * T(R_i(2,0)) + R_q[5] * T(R_i(2,1)) + R_q[8] * T(R_i(2,2));

        T T_qk [3];
        T_qk[0] = T_q[0] - (R_qk[0] * T_i[0] + R_qk[3] * T_i[1] + R_qk[6] * T_i[2]);
        T_qk[1] = T_q[1] - (R_qk[1] * T_i[0] + R_qk[4] * T_i[1] + R_qk[7] * T_i[2]);
        T_qk[2] = T_q[2] - (R_qk[2] * T_i[0] + R_qk[5] * T_i[1] + R_qk[8] * T_i[2]);

        T E_qk [9];
        E_qk[0] =                     - T_qk[2] * R_qk[1] + T_qk[1] * R_qk[2];
        E_qk[1] =   T_qk[2] * R_qk[0]                     - T_qk[0] * R_qk[2];
        E_qk[2] = - T_qk[1] * R_qk[0] + T_qk[0] * R_qk[1];
        E_qk[3] =                     - T_qk[2] * R_qk[4] + T_qk[1] * R_qk[5];
        E_qk[4] =   T_qk[2] * R_qk[3]                     - T_qk[0] * R_qk[5];
        E_qk[5] = - T_qk[1] * R_qk[3] + T_qk[0] * R_qk[4];
        E_qk[6] =                     - T_qk[2] * R_qk[7] + T_qk[1] * R_qk[8];
        E_qk[7] =   T_qk[2] * R_qk[6]                     - T_qk[0] * R_qk[8];
        E_qk[8] = - T_qk[1] * R_qk[6] + T_qk[0] * R_qk[7];

        T point2point[3];
        point2point[0] = E_qk[0] * px2point[0] + E_qk[3] * px2point[1] + E_qk[6] * px2point[2];
        point2point[1] = E_qk[1] * px2point[0] + E_qk[4] * px2point[1] + E_qk[7] * px2point[2];
        point2point[2] = E_qk[2] * px2point[0] + E_qk[5] * px2point[1] + E_qk[8] * px2point[2];

        T epiline[3];
        epiline[0] = (T(1.) /  K_q(0,0)) * point2point[0];
        epiline[1] = (T(1.) /  K_q(1,1)) * point2point[1];
        epiline[2] = (-K_q(0,2)/K_q(0,0)) * point2point[0] - (K_q(1,2)/K_q(1,1)) * point2point[1] + point2point[2];

        T error = ceres::abs(epiline[0] * T(pt_q.x) + epiline[1] * T(pt_q.y) + epiline[2]) /
                  ceres::sqrt(epiline[0] * epiline[0] + epiline[1] * epiline[1]);

        residuals[0] = error;

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Matrix3d & R_i,
                                       const Eigen::Vector3d & T_i,
                                       const Eigen::Matrix3d & K_i,
                                       const Eigen::Matrix3d & K_q,
                                       const cv::Point2d & pt_q,
                                       const cv::Point2d & pt_i) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 1, 3, 3>(
                new ReprojectionError(R_i, T_i, K_i, K_q, pt_q, pt_i)));
    }

    Eigen::Matrix3d R_i;
    Eigen::Vector3d T_i;
    Eigen::Matrix3d K_i;
    Eigen::Matrix3d K_q;
    cv::Point2d pt_q;
    cv::Point2d pt_i;
};

//// FINAL POSE ADJUSTMENT
void pose::adjustHypothesis (const vector<Eigen::Matrix3d> & R_is,
                             const vector<Eigen::Vector3d> & T_is,
                             const vector<vector<double>> & K_is,
                             const vector<double> & K_q,
                             const vector<vector<cv::Point2d>> & all_pts_q,
                             const vector<vector<cv::Point2d>> & all_pts_i,
                             const double & error_thresh,
                             Eigen::Matrix3d & R_q,
                             Eigen::Vector3d & T_q) {

    double R[3];
    double R_arr[9] {R_q(0, 0), R_q(1, 0), R_q(2, 0), R_q(0, 1), R_q(1, 1), R_q(2, 1), R_q(0, 2), R_q(1, 2), R_q(2, 2)};
    ceres::RotationMatrixToAngleAxis(R_arr, R);

    double T [3] = {T_q[0], T_q[1], T_q[2]};

    Eigen::Matrix3d K_q_eig {{K_q[2], 0., K_q[0]},
                             {0., K_q[3], K_q[1]},
                             {0.,     0.,     1.}};

    int K = int (R_is.size());

    ceres::Problem problem;
    ceres::LossFunction * loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < K; i++) {

        Eigen::Matrix3d K_i_eig {{K_is[i][2], 0., K_is[i][0]},
                                 {0., K_is[i][3], K_is[i][1]},
                                 {0.,         0.,         1.}};

        Eigen::Matrix3d R_qi = R_q * R_is[i].transpose();
        Eigen::Vector3d T_qi = T_q - R_qi * T_is[i];
        T_qi.normalize();

        Eigen::Matrix3d T_qi_cross {
                {       0, -T_qi(2),  T_qi(1)},
                { T_qi(2),        0, -T_qi(0)},
                {-T_qi(1),  T_qi(0),        0}};
        Eigen::Matrix3d E_qi = T_qi_cross * R_qi;

        Eigen::Matrix3d F = K_q_eig.inverse().transpose() * E_qi * K_i_eig.inverse();

        assert(all_pts_i[i].size() == all_pts_q[i].size());

        for (int j = 0; j < all_pts_i[i].size(); j++) {

            Eigen::Vector3d pt_q {all_pts_q[i][j].x, all_pts_q[i][j].y, 1.};
            Eigen::Vector3d pt_i {all_pts_i[i][j].x, all_pts_i[i][j].y, 1.};

            Eigen::Vector3d epiline = F * pt_i;

            double error = abs(epiline[0] * pt_q[0] + epiline[1] * pt_q[1] + epiline[2]) /
                           sqrt(epiline[0] * epiline[0] + epiline[1] * epiline[1]);

            if (error <= error_thresh) {
                auto cost_function = ReprojectionError::Create(R_is[i], T_is[i], K_i_eig, K_q_eig, all_pts_q[i][j], all_pts_i[i][j]);
                problem.AddResidualBlock(cost_function, loss_function, R, T);
            }
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;

    if (problem.NumResidualBlocks() > 0) {
        ceres::Solve(options, &problem, &summary);
    } else {
        cout << " Can't Adjust ";
    }
//    std::cout << summary.FullReport() << "\n";

    double R_adj[9];
    ceres::AngleAxisToRotationMatrix(R, R_adj);
    R_q = Eigen::Matrix3d {{R_adj[0], R_adj[3], R_adj[6]},
                           {R_adj[1], R_adj[4], R_adj[7]},
                           {R_adj[2], R_adj[5], R_adj[8]}};
    T_q = Eigen::Vector3d {T[0], T[1], T[2]};

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
            : pt_c(pt_c), pt_l(pt_l), pt_r(pt_r), K_eig(move(K_eig)) {}

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
                  : c_k(move(c_k)), c_q(move(c_q)), t_qk(move(t_qk)) {}

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
                           : c_k(move(c_k)), t_qk(move(t_qk)) {}

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

//    std::cout << summary.FullReport() << "\n";

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
    std::cout << summary.FullReport() << "\n";

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
            t_ij_x(0,1) = -T12(2,0); t_ij_x(1,0) = T12(2,0);
            t_ij_x(0,2) = T12(1,0); t_ij_x(2,0) = -T12(1,0);
            t_ij_x(1,2) = -T12(0,0); t_ij_x(2,1) = T12(0,0);

            A.block(i*3,0,3,3) = lambda[i] * t_ij_x * R12;
            b.block(i*3,0,3,1) = lambda[i] * t_ij_x * T2;
        }
        //Solve linear system
        results_curr = A.colPivHouseholderQr().solve(b);

        //Update weights
//        for (int i = 0; i < nImgs; i++)
//        {
//            Eigen::Matrix<double,3,3> R12 = R_qks[i];
//            Eigen::Matrix<double,3,1> T2 = T_ks[i];
//
//            lambda[i] = 1 / ((T2 - R12 * results_curr).norm());
//        }
    }

    return results_curr;
}

pair<Eigen::Vector3d, Eigen::Vector3d> pose::T_q_closed_form (const vector<Eigen::Matrix3d> & R_ks,
                                                   const vector<Eigen::Vector3d> & T_ks,
                                                   const vector<Eigen::Matrix3d> & R_qks,
                                                   const vector<Eigen::Vector3d> & T_qks) {

    size_t K = R_ks.size();

    Eigen::Matrix3d I {{1, 0, 0},
                       {0, 1, 0},
                       {0, 0, 1}};

    Eigen::MatrixXd A_10 {{0, 0, 0},
                          {0, 0, 0},
                          {0, 0, 0}};
    Eigen::Vector3d b_10 {0, 0, 0};
    Eigen::MatrixXd A_14 {{0, 0, 0},
                          {0, 0, 0},
                          {0, 0, 0}};
    Eigen::Vector3d b_14 {0, 0, 0};

    for (int i = 0; i < K; i++) {

        Eigen::Matrix3d R_k = R_ks[i];
        Eigen::Vector3d T_k = T_ks[i];
        Eigen::Matrix3d R_qk = R_qks[i];
        Eigen::Vector3d T_qk = T_qks[i];

        Eigen::Matrix3d R_kq = R_qk.transpose();
        Eigen::Vector3d T_kq = -R_qk.transpose() * T_qk;

        Eigen::Matrix3d N = I - T_qk * T_qk.transpose();
        Eigen::Matrix3d M = I - T_kq * T_kq.transpose();

        A_10 += N;
        b_10 += R_kq.transpose() * M * T_k;

        A_14 += N;
        b_14 += N * R_qk * T_k;

    }

    Eigen::Vector3d T_q_10 = A_10.colPivHouseholderQr().solve(b_10);
    Eigen::Vector3d T_q_14 = A_14.colPivHouseholderQr().solve(b_14);

    return {T_q_10, T_q_14};
}



//// ROTATION HYPOTHESIS
template<typename DataType, typename ForwardIterator>
Eigen::Quaternion<DataType> pose::averageQuaternions(ForwardIterator const & begin, ForwardIterator const & end) {

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
        auto a = q.transpose()*q;
        A += a;
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