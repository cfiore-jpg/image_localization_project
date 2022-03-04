//
// Created by Cameron Fiore on 1/23/22.
//
#include "../include/sevenScenes.h"
#include "../include/functions.h"
#include "../include/poseEstimation.h"
#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include <opencv2/imgcodecs.hpp>
#include <utility>
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "../include/sevenScenes.h"
#include "../include/functions.h"
#include "../include/calibrate.h"

struct ReprojectionError{
    ReprojectionError(Eigen::Matrix3d R_k,
                      Eigen::Vector3d T_k,
                      Eigen::Matrix3d K,
                      cv::Point2d pt_db,
                      cv::Point2d pt_q)
                      : R_k(move(R_k)), T_k(move(T_k)), K(move(K)), pt_db(pt_db), pt_q(pt_q) {}

    template<typename T>
    bool operator()(const T * const R, const T * const T_q, T * residuals) const {

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

        T px2point_1[3];
        px2point_1[0] = (T(pt_db.x) - K(0,2)) / K(0,0);
        px2point_1[1] = (T(pt_db.y) - K(1,2)) / K(1,1);
        px2point_1[2] = T(1.);

        T R_qk [9];
        R_qk[0] = R_q[0] * T(R_k(0,0)) + R_q[1] * T(R_k(0,1)) + R_q[2] * T(R_k(0,2));
        R_qk[1] = R_q[0] * T(R_k(1,0)) + R_q[1] * T(R_k(1,1)) + R_q[2] * T(R_k(1,2));
        R_qk[2] = R_q[0] * T(R_k(2,0)) + R_q[1] * T(R_k(2,1)) + R_q[2] * T(R_k(2,2));
        R_qk[3] = R_q[3] * T(R_k(0,0)) + R_q[4] * T(R_k(0,1)) + R_q[5] * T(R_k(0,2));
        R_qk[4] = R_q[3] * T(R_k(1,0)) + R_q[4] * T(R_k(1,1)) + R_q[5] * T(R_k(1,2));
        R_qk[5] = R_q[3] * T(R_k(2,0)) + R_q[4] * T(R_k(2,1)) + R_q[5] * T(R_k(2,2));
        R_qk[6] = R_q[6] * T(R_k(0,0)) + R_q[7] * T(R_k(0,1)) + R_q[8] * T(R_k(0,2));
        R_qk[7] = R_q[6] * T(R_k(1,0)) + R_q[7] * T(R_k(1,1)) + R_q[8] * T(R_k(1,2));
        R_qk[8] = R_q[6] * T(R_k(2,0)) + R_q[7] * T(R_k(2,1)) + R_q[8] * T(R_k(2,2));

        T T_qk [3];
        T_qk[0] = T_q[0] - (R_qk[0] * T_k[0] + R_qk[1] * T_k[1] + R_qk[2] * T_k[2]);
        T_qk[1] = T_q[1] - (R_qk[3] * T_k[0] + R_qk[4] * T_k[1] + R_qk[5] * T_k[2]);
        T_qk[2] = T_q[2] - (R_qk[6] * T_k[0] + R_qk[7] * T_k[1] + R_qk[8] * T_k[2]);

        T E_qk [9];
        E_qk[0] = -T_qk[2] * R_qk[3] + T_qk[1] * R_qk[6];
        E_qk[1] = -T_qk[2] * R_qk[4] + T_qk[1] * R_qk[7];
        E_qk[2] = -T_qk[2] * R_qk[5] + T_qk[1] * R_qk[8];
        E_qk[3] = T_qk[2] * R_qk[0] - T_qk[0] * R_qk[6];
        E_qk[4] = T_qk[2] * R_qk[1] - T_qk[0] * R_qk[7];
        E_qk[5] = T_qk[2] * R_qk[2] - T_qk[0] * R_qk[8];
        E_qk[6] = -T_qk[1] * R_qk[0] + T_qk[0] * R_qk[3];
        E_qk[7] = -T_qk[1] * R_qk[1] + T_qk[0] * R_qk[4];
        E_qk[8] = -T_qk[1] * R_qk[2] + T_qk[0] * R_qk[5];


        T point2point_1[3];
        point2point_1[0] = E_qk[0] * px2point_1[0] + E_qk[1] * px2point_1[1] + E_qk[2] * px2point_1[2];
        point2point_1[1] = E_qk[3] * px2point_1[0] + E_qk[4] * px2point_1[1] + E_qk[5] * px2point_1[2];
        point2point_1[2] = E_qk[6] * px2point_1[0] + E_qk[7] * px2point_1[1] + E_qk[8] * px2point_1[2];

        T epiline[3];
        epiline[0] = (T(1.) /  K(0,0)) * point2point_1[0];
        epiline[1] = (T(1.) /  K(1,1)) * point2point_1[1];
        epiline[2] = -( K(0,2) /  K(0,0)) * point2point_1[0]
                - ( K(1,2) /  K(1,1)) * point2point_1[1]
                + point2point_1[2];

        T error = ceres::abs(epiline[0] * T(pt_q.x) + epiline[1] * T(pt_q.y) + epiline[2]) /
                ceres::sqrt(epiline[0] * epiline[0] + epiline[1] * epiline[1]);

        residuals[0] = error;

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Matrix3d & R_k,
                                       const Eigen::Vector3d & T_k,
                                       const Eigen::Matrix3d & K,
                                       const cv::Point2d & pt_db,
                                       const cv::Point2d & pt_q) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 1, 3, 3>(
                new ReprojectionError(R_k, T_k, K, pt_db, pt_q)));
    }


    Eigen::Matrix3d R_k;
    Eigen::Vector3d T_k;
    Eigen::Matrix3d K;
    cv::Point2d pt_db;
    cv::Point2d pt_q;
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

Eigen::Vector3d pose::hypothesizeQueryCenter (const vector<Eigen::Matrix3d> & R_k,
                                              const vector<Eigen::Vector3d> & t_k,
                                              const vector<Eigen::Matrix3d> & R_qk,
                                              const vector<Eigen::Vector3d> & t_qk) {

    size_t K = R_k.size();

    Eigen::Matrix3d I {{1, 0, 0},
                       {0, 1, 0},
                       {0, 0, 1}};
    Eigen::Matrix3d A {{0, 0, 0},
                       {0, 0, 0},
                       {0, 0, 0}};
    Eigen::Vector3d b {0, 0, 0};


    for (int i = 0; i < K; i++) {

        Eigen::Matrix3d R = R_k[i];
        Eigen::Vector3d t = t_k[i];
        Eigen::Matrix3d R_ = R_qk[i];
        Eigen::Vector3d t_ = t_qk[i];

        Eigen::Vector3d c_k = -R.transpose() * t;
        Eigen::Vector3d t_kq = -R_.transpose() * t_;

        Eigen::Matrix3d M_kq = I - R.transpose() * t_kq * t_kq.transpose() * R;

        A += M_kq;
        b += M_kq * c_k;
    }

    Eigen::Vector3d c_q = A.colPivHouseholderQr().solve(b);
    return c_q;
}

Eigen::Vector3d pose::hypothesizeQueryCenterRANSAC (const double & inlier_thresh,
                                                    vector<Eigen::Matrix3d> & R_k,
                                                    vector<Eigen::Vector3d> & t_k,
                                                    vector<Eigen::Matrix3d> & R_qk,
                                                    vector<Eigen::Vector3d> & t_qk,
                                                    vector<vector<tuple<cv::Point2d, cv::Point2d, double>>> & all_points) {
    int K = int (R_k.size());
    vector<Eigen::Matrix3d> best_R;
    vector<Eigen::Vector3d> best_t;
    vector<Eigen::Matrix3d> best_R_;
    vector<Eigen::Vector3d> best_t_;
    vector<vector<tuple<cv::Point2d, cv::Point2d, double>>> best_p;
    for (int i = 0; i < K - 1; i++) {
        for (int j = i + 1; j < K; j++) {
            vector<Eigen::Matrix3d> R {R_k[i], R_k[j]};
            vector<Eigen::Vector3d> t {t_k[i], t_k[j]};
            vector<Eigen::Matrix3d> R_ {R_qk[i], R_qk[j]};
            vector<Eigen::Vector3d> t_ {t_qk[i], t_qk[j]};
            vector<vector<tuple<cv::Point2d, cv::Point2d, double>>> p {all_points[i], all_points[j]};
//            Eigen::Vector3d c_i = -R_k[i].transpose() * t_k[i], c_j = -R_k[j].transpose() * t_k[j];
//            Eigen::Vector3d v_i = -R_k[i].transpose() * R_qk[i].transpose() * t_qk[i],
//            v_j = -R_k[j].transpose() * R_qk[j].transpose() * t_qk[j];
            Eigen::Vector3d h = hypothesizeQueryCenter(R, t, R_, t_);
            for (int k = 0; k < K; k++) {
                if (k != i && k != j) {
                    Eigen::Vector3d t_qk_real = -R_qk[k] * (R_k[k] * h + t_k[k]);
                    double angle = functions::getAngleBetween(t_qk_real, t_qk[k]);

                    if (angle <= inlier_thresh) {
                        R.push_back(R_k[k]);
                        t.push_back(t_k[k]);
                        R_.push_back(R_qk[k]);
                        t_.push_back(t_qk[k]);
                        p.push_back(all_points[k]);
                    }
                }
            }
            if (R.size() > best_R.size()) {
                best_R = R;
                best_t = t;
                best_R_ = R_;
                best_t_ = t_;
                best_p = p;
            }
        }
    }
    Eigen::Vector3d c_q = hypothesizeQueryCenter(best_R, best_t, best_R_, best_t_);
    R_k = best_R;
    t_k = best_t;
    R_qk = best_R_;
    t_qk = best_t_;
    all_points = best_p;
    return c_q;
}

Eigen::Matrix3d pose::hypothesizeQueryRotation(const Eigen::Vector3d & c_q,
                                               const Eigen::Matrix3d & R_q,
                                               const vector<Eigen::Matrix3d> & R_k,
                                               const vector<Eigen::Vector3d> & t_k,
                                               const vector<Eigen::Matrix3d> & R_qk,
                                               const vector<Eigen::Vector3d> & t_qk) {

    Eigen::Quaterniond q (R_q);

    double theta = 2. * acos(q.w());
    Eigen::Vector3d w_hat {q.x(), q.y(), q.z()};
    w_hat = w_hat * (1./ sin(theta/2.));
    Eigen::Vector3d r = tan(theta/2.) * w_hat;

    double R [3] = {r(0), r(1), r(2)};

    double x_test = R[0];
    double y_test = R[1];
    double z_test = R[2];
    double Z_test = 1. + x_test*x_test + y_test*y_test + z_test*z_test;
    Eigen::Matrix3d R_q_new_test {{(1. + x_test*x_test - y_test*y_test - z_test*z_test) / Z_test, (2.*x_test*y_test - 2.*z_test) / Z_test, (2.*x_test*z_test + 2.*y_test) / Z_test},
                             {(2.*x_test*y_test + 2.*z_test) / Z_test, (1. + y_test*y_test - x_test*x_test - z_test*z_test) / Z_test, (2.*y_test*z_test - 2.*x_test) / Z_test},
                             {(2.*x_test*z_test - 2.*y_test) / Z_test, (2.*y_test*z_test + 2.*x_test) / Z_test, (1. + z_test*z_test - x_test*x_test - y_test*y_test) / Z_test}};

    double a = functions::rotationDifference(R_q_new_test, R_q);

    int K = int (R_k.size());
    ceres::Problem problem;
    ceres::LossFunction * loss_function = new ceres::HuberLoss(1.0);

    for (int i = 0; i < K; i++) {
        Eigen::Vector3d c_k = - R_k[i].transpose() * t_k[i];
        Eigen::Vector3d t_ = -R_qk[i] * (R_k[i] * c_q + t_k[i]);
        t_.normalize();

        ceres::CostFunction * cost_function = RotationError::Create(c_k, c_q, t_);
        double lambda[1] = {(c_k - c_q).transpose() * R_q.transpose() * t_};
        problem.AddResidualBlock(cost_function, loss_function, R, lambda);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
//    std::cout << summary.FullReport() << "\n";

    double x = R[0];
    double y = R[1];
    double z = R[2];
    double Z = 1. + x*x + y*y + z*z;
    Eigen::Matrix3d R_q_new {{(1. + x*x - y*y - z*z) / Z, (2.*x*y - 2.*z) / Z, (2.*x*z + 2.*y) / Z},
                         {(2.*x*y + 2.*z) / Z, (1. + y*y - x*x - z*z) / Z, (2.*y*z - 2.*x) / Z},
                         {(2.*x*z - 2.*y) / Z, (2.*y*z + 2.*x) / Z, (1. + z*z - x*x - y*y) / Z}};
    return R_q_new;
}

tuple<Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d, int, double, double> pose::hypothesizeRANSAC (
        const double & t_thresh,
        const double & r_thresh,
        const vector<int> & mask,
        const vector<Eigen::Matrix3d> & R_k,
        const vector<Eigen::Vector3d> & t_k,
        const vector<Eigen::Matrix3d> & R_qk,
        const vector<Eigen::Vector3d> & t_qk,
        const Eigen::Vector3d & c_q,
        const Eigen::Matrix3d & R_q) {

    int K = int(R_k.size());
    vector<Eigen::Matrix3d> best_R;
    vector<Eigen::Vector3d> best_t;
    vector<Eigen::Matrix3d> best_R_;
    vector<Eigen::Vector3d> best_t_;
    Eigen::Vector3d best_T_q;
    Eigen::Matrix3d best_R_q;
    double best_total_spread = 0;
    vector<int> best_indices;

    for (int i = 0; i < K - 1; i++) {
        for (int j = i + 1; j < K; j++) {
            vector<Eigen::Matrix3d> R {R_k[i], R_k[j]};
            vector<Eigen::Vector3d> t {t_k[i], t_k[j]};
            vector<Eigen::Matrix3d> R_ {R_qk[i], R_qk[j]};
            vector<Eigen::Vector3d> t_ {t_qk[i], t_qk[j]};
            vector<int> indices {i, j};
            double total_spread = 0;

            Eigen::Matrix3d R_hyp = pose::rotationAverage(vector<Eigen::Matrix3d> {R_qk[i] * R_k[i], R_qk[j] * R_k[j]});

            Eigen::Vector3d c_hyp = hypothesizeQueryCenter(R, t, R_, t_);

//            Eigen::Matrix3d R_hyp = pose::hypothesizeQueryRotation(c_hyp, R_avg, R, t, R_, t_);

            for (int k = 0; k < K; k++) {
                if (k != i && k != j) {

                    Eigen::Matrix3d R_qk_real = R_hyp * R_k[k].transpose();
                    Eigen::Vector3d t_qk_real = -R_qk_real * (R_k[k] * c_hyp + t_k[k]);

                    double r_dist = functions::rotationDifference(R_qk_real, R_qk[k]);
                    double t_dist = functions::getAngleBetween(t_qk_real, t_qk[k]);

                    if (t_dist <= t_thresh && r_dist <= r_thresh) {
                        R.push_back(R_k[k]);
                        t.push_back(t_k[k]);
                        R_.push_back(R_qk[k]);
                        t_.push_back(t_qk[k]);
                        indices.push_back(k);
                        total_spread += t_dist + r_dist;
                    }
                }
            }
            if (R.size() > best_R.size() || (R.size() == best_R.size() && total_spread < best_total_spread)) {
                best_R = R;
                best_t = t;
                best_R_ = R_;
                best_t_ = t_;
                best_indices = indices;
                best_total_spread = total_spread;
                best_R_q = R_hyp;
                best_T_q = - R_hyp * c_hyp;
            }
        }
    }

    double tp = 0, fp = 0;
//    if (best_R.size() > 2) {
//        for (const auto &i: best_indices) {
//            if (mask[i]) {
//                tp++;
//            } else {
//                fp++;
//            }
//        }
//    }

    Eigen::Vector3d c_q_calc = hypothesizeQueryCenter(best_R, best_t, best_R_, best_t_);

    vector<Eigen::Matrix3d> rotations;
    rotations.reserve(best_R.size());
    for(int i = 0; i < best_R.size(); i++) {
        rotations.emplace_back(best_R_[i] * best_R[i]);
    }

    Eigen::Matrix3d R_q_avg = pose::rotationAverage(rotations);

    Eigen::Vector3d t_q_avg = -R_q_avg * c_q_calc;

    Eigen::Matrix3d R_q_calc = hypothesizeQueryRotation(c_q_calc, R_q_avg, best_R, best_t, best_R_, best_t_);

    Eigen::Vector3d t_q_calc = -R_q_calc * c_q_calc;

    Eigen::Vector3d t_q_gov = pose::GovinduTranslationAveraging(best_t, best_R_, best_t_);

    return {best_T_q, best_R_q, t_q_avg, R_q_avg, t_q_calc, R_q_calc, t_q_gov, int (best_R.size()), tp, fp};
}

void pose::adjustHypothesis (const vector<Eigen::Matrix3d> & R_k,
                             const vector<Eigen::Vector3d> & T_k,
                             const vector<vector<tuple<cv::Point2d, cv::Point2d, double>>> & all_points,
                             const double & error_thresh,
                             const double * K,
                             Eigen::Matrix3d & R_q,
                             Eigen::Vector3d & T_q) {

    Eigen::Matrix3d K_eig {{K[0], 0.,   K[2]},
                           {0.,   K[1], K[3]},
                           {0.,   0.,   1.}};
    Eigen::Quaterniond q (R_q);
    q = q.normalized();
    double R [3] = {q.x(), q.y(), q.z()};
    double T [3] = {T_q[0], T_q[1], T_q[2]};


    int k = int (R_k.size());
    ceres::Problem problem;
    ceres::LossFunction * loss_function = new ceres::HuberLoss(1.0);
    for (int i = 0; i < k; i++) {

        Eigen::Matrix3d R_qk_real = R_q * R_k[i].transpose();
        Eigen::Vector3d t_qk_real = T_q - R_qk_real * T_k[i];
        t_qk_real.normalize();

        Eigen::Matrix3d t_cross_real {
                {0,               -t_qk_real(2), t_qk_real(1)},
                {t_qk_real(2),  0,              -t_qk_real(0)},
                {-t_qk_real(1), t_qk_real(0),               0}};
        Eigen::Matrix3d E_real = t_cross_real * R_qk_real;

        Eigen::Matrix3d F = K_eig.inverse().transpose() * E_real * K_eig.inverse();

        for (const auto & match : all_points[i]) {

            Eigen::Vector3d pt_q {get<0>(match).x, get<0>(match).y, 1.};
            Eigen::Vector3d pt_db {get<1>(match).x, get<1>(match).y, 1.};

            Eigen::Vector3d epiline = F * pt_db;

            double error = abs(epiline[0] * pt_q[0] + epiline[1] * pt_q[1] + epiline[2]) /
                    sqrt(epiline[0] * epiline[0] + epiline[1] * epiline[1]);

            if (error <= error_thresh) {

                ceres::CostFunction *cost_function = ReprojectionError::Create(R_k[i], T_k[i], K_eig,
                                                                               get<1>(match),
                                                                               get<0>(match));
                problem.AddResidualBlock(cost_function, loss_function, R, T);
            }
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    int num = problem.NumResidualBlocks();
    if (problem.NumResidualBlocks() > 0) {
        ceres::Solve(options, &problem, &summary);
    } else {
        cout << ", Can't Adjust";
    }
//    std::cout << summary.FullReport() << "\n";

    double x = R[0];
    double y = R[1];
    double z = R[2];
    double Z = 1. + x*x + y*y + z*z;
    R_q = Eigen::Matrix3d {{(1. + x*x - y*y - z*z) / Z, (2.*x*y - 2.*z) / Z, (2.*x*z + 2.*y) / Z},
                           {(2.*x*y + 2.*z) / Z, (1. + y*y - x*x - z*z) / Z, (2.*y*z - 2.*x) / Z},
                           {(2.*x*z - 2.*y) / Z, (2.*y*z + 2.*x) / Z, (1. + z*z - x*x - y*y) / Z}};
    T_q = Eigen::Vector3d {T[0], T[1], T[2]};

}

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

Eigen::Matrix3d pose::rotationAverage(const vector<Eigen::Matrix3d> & rotations) {

    vector<Eigen::Quaterniond> quaternions;
    for (const auto & rot: rotations) {
        Eigen::Quaterniond q(rot);
        quaternions.push_back(q);
    }

    Eigen::Quaterniond avg_Q = averageQuaternions<double>(quaternions.begin(), quaternions.end());
    Eigen::Matrix3d avg_M = avg_Q.toRotationMatrix();
    return avg_M;

}

//The implementation of Govindu's Translation solving scheme
//Listed in the paper: Combining Two-view Constraints For Motion Estimation  (2011)
//Their method uses the data:
//R1: Absolute Rotation of Query
//R2: Absolute Rotation of Database images; We can have multiple R2s
//T1: Our unknown.
//T2: Absolute Translation of Database images;
//R12: Relative Rotation Between Query and Database images, defined as R2 * R1^T
//T12: Relative Translation Between Query and Database images, defined as T2 - R12 * T1
Eigen::Vector3d pose::GovinduTranslationAveraging(const vector<Eigen::Vector3d> & t_k,
                                                  const vector<Eigen::Matrix3d> & R_qk,
                                                  const vector<Eigen::Vector3d> & t_qk)
{
    int nImgs = int(t_k.size());

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
//    while((results_prev - results_curr).norm() > 1e-8 && dist_prev > (results_prev - results_curr).norm())
//    {
        dist_prev = (results_prev - results_curr).norm();
        results_prev = results_curr;
        //Build Linear systems
        for (int i = 0; i < nImgs; i++)
        {

            Eigen::Matrix<double,3,3> R12 = R_qk[i].transpose();
            Eigen::Matrix<double,3,1> T2 = t_k[i];
            Eigen::Matrix<double,3,1> T12 = - R12 * t_qk[i];

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
//            Eigen::Matrix<double,3,3> R12 = R_qk[i].transpose();
//            Eigen::Matrix<double,3,1> T2 = t_k[i];
//
//            lambda[i] = 1 / ((T2 - R12 * results_curr).norm());
//
//        }
//
//    }
    return results_curr;
}
