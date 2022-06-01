
#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include <opencv2/imgcodecs.hpp>
#include <utility>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "glog/logging.h"
#include "../include/OptimalRotationSolver.h"


//The Following is the function for solving the proposition 2.

//In proposition 2, M matrix is a 9x9 symmetric matrix. 
//So M matrix has 45 independent elements. 
//The input of this function is the 45 elements of the M matrix 
//For example, out M matrix is (Using 4x4 as an example)
// 0 1 2 3 
// 1 4 5 6
// 2 5 7 8 
// 3 6 8 9

//Then our input vector is: 0 1 2 3 4 5 6 7 8 9

//Also M matrix is sum(starProduct(T_hat_qk, c_k - c_q);)

//We have 40x4 solutions. Some of them are complex, we can extract the real only solutions whose absolute value of imaginary part is less than some value say 1e-7

//Some test value:
//Say we have 3 dataset images and 1 query

//R_1 = [0.55072336929004850337 -0.82712030047107198971 0.11214178109188482901;
//-0.11722689697755828142 0.056375599209274707135 0.99150372991674018408;
//-0.8264149231123731898 -0.5591903078223356971 -0.065913386309092270032];

//c_1 = [938.1809157763350413 613.74851039172017408 55.011595897134093036]';



//R_2 = [0.088461304994031042526 0.033230569322564879053 0.99552515125497997861
//-0.22987060353240071353 -0.9717843825228138499 0.052864160982681845935
//0.9691925005644660418 -0.2335184000186093789 -0.078326583624883427959];

//c_2 = [-1091.5951653892184368 272.98698735737838206 107.90863343431843191]';


//R_3 = [-0.10387879175474701299 0.8104268925167098514 0.5765565440694283561;
//-0.9498195017811733587 0.091132400555301251721 -0.29922867443677803045;
//-0.29504594669260764128 -0.57870816259556667749 0.76029254362089171426];

//c_3 = [346.58158737322162324 638.29359688719728183 -848.51081310189067608]';

//Ground Truth of query
//R_q = [0.93118482425723736462 0.36262296659846848801 -0.037408650982680001496;
//0.33249792459696492219 -0.8027621803172360071 0.49499294135475135903;
//0.14946555861377897045 -0.47336821391771044532 -0.86809134360424744514];

//c_q = [-157.37370535944677385 551.2138747105557286 985.10057207662759993]';

//The computed M matrix should be:

//   1.0e+06 *

//    1.1429    0.1754   -0.2353    0.1245    0.0631    0.4284    0.2947   -0.0620   -1.0726
//    0.1754    0.0425    0.0779    0.0631    0.0208    0.0942   -0.0620   -0.0280   -0.1696
//   -0.2353    0.0779    1.1801    0.4284    0.0942   -0.2750   -1.0726   -0.1696    0.9669
//    0.1245    0.0631    0.4284    0.1964    0.0498   -0.0482   -0.3716   -0.0799    0.2164
//    0.0631    0.0208    0.0942    0.0498    0.0137    0.0091   -0.0799   -0.0207    0.0061
//    0.4284    0.0942   -0.2750   -0.0482    0.0091    0.7393    0.2164    0.0061   -1.4553
//    0.2947   -0.0620   -1.0726   -0.3716   -0.0799    0.2164    0.9877    0.1472   -0.8400
//   -0.0620   -0.0280   -0.1696   -0.0799   -0.0207    0.0061    0.1472    0.0328   -0.0608
//   -1.0726   -0.1696    0.9669    0.2164    0.0061   -1.4553   -0.8400   -0.0608    3.0773

//The corresponding solution in quaternion is something like [0.2551   -0.9490   -0.1831   -0.0295] or [-0.2551   0.9490   0.1831   0.0295]

//Then you can use the following code to return the solution back to the rotation matrix:
//    Eigen::Quaterniond q;
//    q.w() = -0.2551;
//    q.x() = 0.9490;  
//    q.y() = 0.1831;  
//    q.z() = 0.0295;
//    Eigen::Matrix3d R = q.normalized().toRotationMatrix();

// we have in total 40 solutions; there is only ONE global minimum. 
// To pick the best solution. 
// We need to evaluate cost function in Propostion 2. (Equation 15 in the paper; Equation 4 in the supp.tex with Eqaution 5 substiuted)
//Something like: sum_k[(c_k - c_q)' * (c_k - c_q) - 2 * ((c_k - c_q)' * R_q' * T_hat_qk) * (c_k - c_q)' * R_q' * T_hat_qk + ((c_k - c_q)' * R_q' * T_hat_qk)^2]
//And we need to pick the minimum one among all 40 solutions (actually less than 40, we don't need solutions with imaginary part)

//Remark: Given the data I used is pretty large. The range of M is pretty large. 
//They are all at 1e6 level. 
//This may generate some numerical issue
//If the matrix is pretty large, before feeding it into the function, it will be better to scale the matrix down, for example, M / 1e-5

using namespace Eigen;
using namespace std;

Eigen::Matrix3d rotation::optimal_rotation_using_R_qks(const Eigen::Vector3d &c_q,
                                                       const vector<Eigen::Matrix3d> &R_ks,
                                                       const vector<Eigen::Vector3d> &T_ks,
                                                       const vector<Eigen::Matrix3d> &R_qks,
                                                       const vector<Eigen::Vector3d> &T_qks) {

    int K = int(R_ks.size());

    Matrix<double, 9, 9> M{{0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0}};

    for (int i = 0; i < K; i++) {

        Vector3d c_k = -R_ks[i].transpose() * T_ks[i];

        Vector3d diff = c_k - c_q;

//        Vector3d t_ = t_qk[k];
        Vector3d T_qk = -R_qks[i] * (R_ks[i] * c_q + T_ks[i]);
        T_qk.normalize();

        Matrix<double, 9, 1> star{T_qk[0] * diff[0], T_qk[0] * diff[1], T_qk[0] * diff[2],
                                  T_qk[1] * diff[0], T_qk[1] * diff[1], T_qk[1] * diff[2],
                                  T_qk[2] * diff[0], T_qk[2] * diff[1], T_qk[2] * diff[2]};

        M += star * star.transpose();

    }

//    cout << M << endl;

    Matrix<double, 45, 1> data;
    int idx = 0;
    for (int i = 0; i < 9; i++) {
        for (int j = i; j < 9; j++) {
            data[idx] = M(i, j);
            idx++;
        }
    }

    MatrixXcd sols = rotation::solver_problem_averageRQuatMetric_red(data);

    Matrix3d best_R;
    double best_error = DBL_MAX;
    for (int i = 0; i < 40; i++) {

        Eigen::Quaterniond q;
        if (abs(sols(0, i).imag()) > 1e-6) continue;
        q.w() = sols(0, i).real();
        if (abs(sols(1, i).imag()) > 1e-6) continue;
        q.x() = sols(1, i).real();
        if (abs(sols(2, i).imag()) > 1e-6) continue;
        q.y() = sols(2, i).real();
        if (abs(sols(3, i).imag()) > 1e-6) continue;
        q.z() = sols(3, i).real();

        Eigen::Matrix3d R = q.normalized().toRotationMatrix();

        double error = 0;
        for (int k = 0; k < K; k++) {
            Vector3d c_k = -R_ks[k].transpose() * T_ks[k];

            Vector3d T_qk = -R_qks[k] * (R_ks[k] * c_q + T_ks[k]);
            T_qk.normalize();

            double lambda = (c_k - c_q).transpose() * R.transpose() * T_qk;
            error += pow((c_k - (c_q + lambda * R.transpose() * T_qk)).norm(), 2.);
        }

        if (error < best_error) {
            best_error = error;
            best_R = R;
        }
    }

    return best_R;

}

Eigen::Matrix3d rotation::optimal_rotation_using_T_qks(const Eigen::Vector3d &c_q,
                                                       const vector<Eigen::Matrix3d> &R_ks,
                                                       const vector<Eigen::Vector3d> &T_ks,
                                                       const vector<Eigen::Matrix3d> &R_qks,
                                                       const vector<Eigen::Vector3d> &T_qks) {

    int K = int(R_ks.size());

    Matrix<double, 9, 9> M{{0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0}};

    for (int i = 0; i < K; i++) {

        Vector3d c_k = -R_ks[i].transpose() * T_ks[i];

        Vector3d diff = c_k - c_q;

        Vector3d T_qk = T_qks[i];

        Matrix<double, 9, 1> star{T_qk[0] * diff[0], T_qk[0] * diff[1], T_qk[0] * diff[2],
                                  T_qk[1] * diff[0], T_qk[1] * diff[1], T_qk[1] * diff[2],
                                  T_qk[2] * diff[0], T_qk[2] * diff[1], T_qk[2] * diff[2]};

        M += star * star.transpose();

    }

//    cout << M << endl;

    Matrix<double, 45, 1> data;
    int idx = 0;
    for (int i = 0; i < 9; i++) {
        for (int j = i; j < 9; j++) {
            data[idx] = M(i, j);
            idx++;
        }
    }

    MatrixXcd sols = rotation::solver_problem_averageRQuatMetric_red(data);

    Matrix3d best_R;
    double best_error = DBL_MAX;
    for (int i = 0; i < 40; i++) {

        Eigen::Quaterniond q;
        if (abs(sols(0, i).imag()) > 1e-6) continue;
        q.w() = sols(0, i).real();
        if (abs(sols(1, i).imag()) > 1e-6) continue;
        q.x() = sols(1, i).real();
        if (abs(sols(2, i).imag()) > 1e-6) continue;
        q.y() = sols(2, i).real();
        if (abs(sols(3, i).imag()) > 1e-6) continue;
        q.z() = sols(3, i).real();

        Eigen::Matrix3d R = q.normalized().toRotationMatrix();

        double error = 0;
        for (int k = 0; k < K; k++) {
            Vector3d c_k = -R_ks[k].transpose() * T_ks[k];

            Vector3d T_qk = T_qks[k];

            double lambda = (c_k - c_q).transpose() * R.transpose() * T_qk;
            error += pow((c_k - (c_q + lambda * R.transpose() * T_qk)).norm(), 2.);
        }

        if (error < best_error) {
            best_error = error;
            best_R = R;
        }
    }

    return best_R;

}

Eigen::Matrix3d rotation::NORM_optimal_rotation_using_R_qks(const Eigen::Vector3d &c_q,
                                                            const vector<Eigen::Matrix3d> &R_ks,
                                                            const vector<Eigen::Vector3d> &T_ks,
                                                            const vector<Eigen::Matrix3d> &R_qks,
                                                            const vector<Eigen::Vector3d> &T_qks) {

    int K = int(R_ks.size());

    Matrix<double, 9, 9> M{{0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0}};

    for (int i = 0; i < K; i++) {

        Vector3d c_k = -R_ks[i].transpose() * T_ks[i];

        Vector3d diff = c_k - c_q;
        diff.normalize();

        Vector3d T_qk = -R_qks[i] * (R_ks[i] * c_q + T_ks[i]);
        T_qk.normalize();

        Matrix<double, 9, 1> star{T_qk[0] * diff[0], T_qk[0] * diff[1], T_qk[0] * diff[2],
                                  T_qk[1] * diff[0], T_qk[1] * diff[1], T_qk[1] * diff[2],
                                  T_qk[2] * diff[0], T_qk[2] * diff[1], T_qk[2] * diff[2]};

        M += star * star.transpose();

    }

//    cout << M << endl;

    Matrix<double, 45, 1> data;
    int idx = 0;
    for (int i = 0; i < 9; i++) {
        for (int j = i; j < 9; j++) {
            data[idx] = M(i, j);
            idx++;
        }
    }

    MatrixXcd sols = rotation::solver_problem_averageRQuatMetric_red(data);

    Matrix3d best_R;
    double best_error = DBL_MAX;
    for (int i = 0; i < 40; i++) {

        Eigen::Quaterniond q;
        if (abs(sols(0, i).imag()) > 1e-6) continue;
        q.w() = sols(0, i).real();
        if (abs(sols(1, i).imag()) > 1e-6) continue;
        q.x() = sols(1, i).real();
        if (abs(sols(2, i).imag()) > 1e-6) continue;
        q.y() = sols(2, i).real();
        if (abs(sols(3, i).imag()) > 1e-6) continue;
        q.z() = sols(3, i).real();

        Eigen::Matrix3d R = q.normalized().toRotationMatrix();

        double error = 0;
        for (int k = 0; k < K; k++) {
            Vector3d c_k = -R_ks[k].transpose() * T_ks[k];

            Vector3d T_qk = -R_qks[k] * (R_ks[k] * c_q + T_ks[k]);
            T_qk.normalize();

            double lambda = (c_k - c_q).transpose() * R.transpose() * T_qk;
            error += pow((c_k - (c_q + lambda * R.transpose() * T_qk)).norm() / (c_q - c_k).norm(), 2.);
        }

        if (error < best_error) {
            best_error = error;
            best_R = R;
        }
    }

    return best_R;

}

Eigen::Matrix3d rotation::NORM_optimal_rotation_using_T_qks(const Eigen::Vector3d &c_q,
                                                            const vector<Eigen::Matrix3d> &R_ks,
                                                            const vector<Eigen::Vector3d> &T_ks,
                                                            const vector<Eigen::Matrix3d> &R_qks,
                                                            const vector<Eigen::Vector3d> &T_qks) {

    int K = int(R_ks.size());

    Matrix<double, 9, 9> M{{0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0, 0}};

    for (int i = 0; i < K; i++) {

        Vector3d c_k = -R_ks[i].transpose() * T_ks[i];

        Vector3d diff = c_k - c_q;
        diff.normalize();

        Vector3d T_qk = T_qks[i];

        Matrix<double, 9, 1> star{T_qk[0] * diff[0], T_qk[0] * diff[1], T_qk[0] * diff[2],
                                  T_qk[1] * diff[0], T_qk[1] * diff[1], T_qk[1] * diff[2],
                                  T_qk[2] * diff[0], T_qk[2] * diff[1], T_qk[2] * diff[2]};

        M += star * star.transpose();

    }

//    cout << M << endl;

    Matrix<double, 45, 1> data;
    int idx = 0;
    for (int i = 0; i < 9; i++) {
        for (int j = i; j < 9; j++) {
            data[idx] = M(i, j);
            idx++;
        }
    }

    MatrixXcd sols = rotation::solver_problem_averageRQuatMetric_red(data);

    Matrix3d best_R;
    double best_error = DBL_MAX;
    for (int i = 0; i < 40; i++) {

        Eigen::Quaterniond q;
        if (abs(sols(0, i).imag()) > 1e-6) continue;
        q.w() = sols(0, i).real();
        if (abs(sols(1, i).imag()) > 1e-6) continue;
        q.x() = sols(1, i).real();
        if (abs(sols(2, i).imag()) > 1e-6) continue;
        q.y() = sols(2, i).real();
        if (abs(sols(3, i).imag()) > 1e-6) continue;
        q.z() = sols(3, i).real();

        Eigen::Matrix3d R = q.normalized().toRotationMatrix();

        double error = 0;
        for (int k = 0; k < K; k++) {
            Vector3d c_k = -R_ks[k].transpose() * T_ks[k];

            Vector3d T_qk = T_qks[k];

            double lambda = (c_k - c_q).transpose() * R.transpose() * T_qk;
            error += pow((c_k - (c_q + lambda * R.transpose() * T_qk)).norm() / (c_q - c_k).norm(), 2.);
        }

        if (error < best_error) {
            best_error = error;
            best_R = R;
        }
    }

    return best_R;

}


MatrixXcd rotation::solver_problem_averageRQuatMetric_red(const VectorXd &data) {
    // Compute coefficients
    const double *d = data.data();
    VectorXd coeffs(190);
    coeffs[0] = 2 * d[5] - 2 * d[7] + 2 * d[31] - 2 * d[33] + 2 * d[38] - 2 * d[43];
    coeffs[1] = 4 * d[4] + 4 * d[8] + 4 * d[30] + 8 * d[34] - 4 * d[35] + 8 * d[37] - 4 * d[42] + 4 * d[44];
    coeffs[2] = -12 * d[31] + 12 * d[33] - 12 * d[38] + 12 * d[43];
    coeffs[3] = 4 * d[4] + 4 * d[8] - 4 * d[30] - 8 * d[34] + 4 * d[35] - 8 * d[37] + 4 * d[42] - 4 * d[44];
    coeffs[4] = -2 * d[5] + 2 * d[7] + 2 * d[31] - 2 * d[33] + 2 * d[38] - 2 * d[43];
    coeffs[5] = -2 * d[2] + 2 * d[6] - 2 * d[19] - 2 * d[23] + 2 * d[32] + 2 * d[41];
    coeffs[6] =
            -4 * d[1] - 4 * d[3] - 4 * d[12] - 4 * d[16] + 8 * d[20] - 8 * d[22] - 4 * d[25] - 4 * d[29] - 8 * d[36] +
            8 * d[40];
    coeffs[7] = 12 * d[13] - 12 * d[15] + 12 * d[19] + 12 * d[23] + 12 * d[26] - 12 * d[28] - 12 * d[32] - 12 * d[41];
    coeffs[8] = -4 * d[1] - 4 * d[3] + 12 * d[12] + 12 * d[16] - 8 * d[20] + 8 * d[22] + 12 * d[25] + 12 * d[29] +
                8 * d[36] - 8 * d[40];
    coeffs[9] =
            2 * d[2] - 2 * d[6] - 4 * d[13] + 4 * d[15] - 2 * d[19] - 2 * d[23] - 4 * d[26] + 4 * d[28] + 2 * d[32] +
            2 * d[41];
    coeffs[10] = 4 * d[0] + 4 * d[4] + 8 * d[8] - 4 * d[17] + 8 * d[21] + 4 * d[34] - 4 * d[39] + 4 * d[44];
    coeffs[11] = -12 * d[5] + 12 * d[7] - 12 * d[10] + 12 * d[14] - 12 * d[18] + 12 * d[27] - 12 * d[38] + 12 * d[43];
    coeffs[12] =
            4 * d[0] - 8 * d[4] - 4 * d[8] - 8 * d[9] - 16 * d[11] + 4 * d[17] - 8 * d[21] - 8 * d[24] + 4 * d[30] -
            4 * d[34] + 4 * d[35] - 8 * d[37] + 4 * d[39] + 4 * d[42] - 8 * d[44];
    coeffs[13] = 4 * d[10] - 4 * d[14] + 4 * d[18] - 4 * d[27] + 4 * d[38] - 4 * d[43];
    coeffs[14] = 12 * d[2] - 12 * d[6] + 12 * d[23] - 12 * d[41];
    coeffs[15] = 12 * d[1] + 12 * d[3] - 4 * d[12] + 12 * d[16] - 8 * d[20] + 8 * d[22] - 4 * d[25] + 12 * d[29] +
                 8 * d[36] - 8 * d[40];
    coeffs[16] = -4 * d[13] + 4 * d[15] - 4 * d[23] - 4 * d[26] + 4 * d[28] + 4 * d[41];
    coeffs[17] = -4 * d[0] + 4 * d[4] - 8 * d[8] + 4 * d[17] - 8 * d[21] + 4 * d[34] + 4 * d[39] - 4 * d[44];
    coeffs[18] =
            2 * d[5] - 2 * d[7] + 4 * d[10] - 4 * d[14] + 4 * d[18] - 4 * d[27] - 2 * d[31] + 2 * d[33] + 2 * d[38] -
            2 * d[43];
    coeffs[19] = -2 * d[2] + 2 * d[6] + 2 * d[19] - 2 * d[23] - 2 * d[32] + 2 * d[41];
    coeffs[20] = 2 * d[1] - 2 * d[3] + 2 * d[12] + 2 * d[16] - 2 * d[25] - 2 * d[29];
    coeffs[21] =
            -4 * d[2] - 4 * d[6] - 8 * d[13] + 8 * d[15] - 4 * d[19] - 4 * d[23] + 8 * d[26] - 8 * d[28] - 4 * d[32] -
            4 * d[41];
    coeffs[22] = -12 * d[12] - 12 * d[16] + 12 * d[20] - 12 * d[22] + 12 * d[25] + 12 * d[29] + 12 * d[36] - 12 * d[40];
    coeffs[23] = -4 * d[2] - 4 * d[6] + 8 * d[13] - 8 * d[15] + 12 * d[19] + 12 * d[23] - 8 * d[26] + 8 * d[28] +
                 12 * d[32] + 12 * d[41];
    coeffs[24] =
            -2 * d[1] + 2 * d[3] + 2 * d[12] + 2 * d[16] - 4 * d[20] + 4 * d[22] - 2 * d[25] - 2 * d[29] - 4 * d[36] +
            4 * d[40];
    coeffs[25] =
            -4 * d[5] - 4 * d[7] + 8 * d[10] - 8 * d[14] - 8 * d[18] + 8 * d[27] - 4 * d[31] - 4 * d[33] - 4 * d[38] -
            4 * d[43];
    coeffs[26] = 12 * d[9] - 12 * d[17] - 12 * d[24] + 12 * d[35] + 12 * d[39] - 12 * d[42];
    coeffs[27] = -4 * d[5] - 4 * d[7] - 24 * d[10] - 8 * d[14] - 8 * d[18] - 24 * d[27] + 12 * d[31] + 12 * d[33] +
                 12 * d[38] + 12 * d[43];
    coeffs[28] = -4 * d[9] + 4 * d[17] + 4 * d[24] - 4 * d[35] - 4 * d[39] + 4 * d[42];
    coeffs[29] = -12 * d[1] + 12 * d[3] - 12 * d[16] - 12 * d[20] - 12 * d[22] + 12 * d[29] + 12 * d[36] + 12 * d[40];
    coeffs[30] = 12 * d[2] + 12 * d[6] - 8 * d[13] - 24 * d[15] - 4 * d[19] + 12 * d[23] - 24 * d[26] - 8 * d[28] -
                 4 * d[32] + 12 * d[41];
    coeffs[31] = 4 * d[16] + 8 * d[22] - 4 * d[29] - 8 * d[36];
    coeffs[32] =
            12 * d[5] + 12 * d[7] - 8 * d[10] + 8 * d[14] + 8 * d[18] - 8 * d[27] - 4 * d[31] - 4 * d[33] + 12 * d[38] +
            12 * d[43];
    coeffs[33] =
            2 * d[1] - 2 * d[3] - 2 * d[12] + 2 * d[16] + 4 * d[20] + 4 * d[22] + 2 * d[25] - 2 * d[29] - 4 * d[36] -
            4 * d[40];
    coeffs[34] = 4 * d[0] + 8 * d[4] + 4 * d[8] - 4 * d[9] + 8 * d[11] - 4 * d[24] + 4 * d[30] + 4 * d[34];
    coeffs[35] = -12 * d[5] + 12 * d[7] + 12 * d[10] + 12 * d[14] - 12 * d[18] - 12 * d[27] - 12 * d[31] + 12 * d[33];
    coeffs[36] =
            4 * d[0] - 4 * d[4] - 8 * d[8] + 4 * d[9] - 8 * d[11] - 8 * d[17] - 16 * d[21] + 4 * d[24] - 8 * d[30] -
            4 * d[34] + 4 * d[35] - 8 * d[37] - 8 * d[39] + 4 * d[42] + 4 * d[44];
    coeffs[37] = -4 * d[10] - 4 * d[14] + 4 * d[18] + 4 * d[27] + 4 * d[31] - 4 * d[33];
    coeffs[38] = 12 * d[2] - 12 * d[6] + 12 * d[13] + 12 * d[15] + 12 * d[19] - 12 * d[26] - 12 * d[28] - 12 * d[32];
    coeffs[39] = 12 * d[1] + 12 * d[3] + 12 * d[12] - 4 * d[16] - 24 * d[20] - 8 * d[22] + 12 * d[25] - 4 * d[29] -
                 8 * d[36] - 24 * d[40];
    coeffs[40] = -8 * d[13] - 4 * d[19] + 8 * d[28] + 4 * d[32];
    coeffs[41] =
            -8 * d[0] - 4 * d[4] - 4 * d[8] + 4 * d[9] - 8 * d[11] + 4 * d[17] - 8 * d[21] + 4 * d[24] + 4 * d[30] -
            8 * d[34] - 8 * d[35] - 16 * d[37] + 4 * d[39] - 8 * d[42] + 4 * d[44];
    coeffs[42] = 4 * d[5] - 4 * d[7] - 8 * d[14] + 8 * d[18];
    coeffs[43] = -4 * d[2] + 4 * d[6] - 4 * d[13] - 4 * d[15] + 4 * d[26] + 4 * d[28];
    coeffs[44] = -12 * d[1] + 12 * d[3] - 12 * d[12] + 12 * d[25];
    coeffs[45] = 12 * d[2] + 12 * d[6] + 8 * d[13] - 8 * d[15] + 12 * d[19] - 4 * d[23] - 8 * d[26] + 8 * d[28] +
                 12 * d[32] - 4 * d[41];
    coeffs[46] = 4 * d[12] - 4 * d[20] + 4 * d[22] - 4 * d[25] - 4 * d[36] + 4 * d[40];
    coeffs[47] = 12 * d[5] + 12 * d[7] - 8 * d[10] + 8 * d[14] + 8 * d[18] - 8 * d[27] + 12 * d[31] + 12 * d[33] -
                 4 * d[38] - 4 * d[43];
    coeffs[48] = 4 * d[1] - 4 * d[3] + 4 * d[20] + 4 * d[22] - 4 * d[36] - 4 * d[40];
    coeffs[49] = -4 * d[0] - 8 * d[4] + 4 * d[8] + 4 * d[9] - 8 * d[11] + 4 * d[24] - 4 * d[30] + 4 * d[34];
    coeffs[50] =
            2 * d[5] - 2 * d[7] - 4 * d[10] - 4 * d[14] + 4 * d[18] + 4 * d[27] + 2 * d[31] - 2 * d[33] - 2 * d[38] +
            2 * d[43];
    coeffs[51] =
            -2 * d[2] + 2 * d[6] - 4 * d[13] - 4 * d[15] - 2 * d[19] + 2 * d[23] + 4 * d[26] + 4 * d[28] + 2 * d[32] -
            2 * d[41];
    coeffs[52] = 2 * d[1] - 2 * d[3] + 2 * d[12] - 2 * d[16] - 2 * d[25] + 2 * d[29];
    coeffs[53] = -2 * d[5] + 2 * d[7] - 2 * d[31] + 2 * d[33] - 2 * d[38] + 2 * d[43];
    coeffs[54] = -4 * d[4] - 4 * d[8] - 4 * d[30] - 8 * d[34] + 4 * d[35] - 8 * d[37] + 4 * d[42] - 4 * d[44];
    coeffs[55] = 12 * d[31] - 12 * d[33] + 12 * d[38] - 12 * d[43];
    coeffs[56] = -4 * d[4] - 4 * d[8] + 4 * d[30] + 8 * d[34] - 4 * d[35] + 8 * d[37] - 4 * d[42] + 4 * d[44];
    coeffs[57] = 2 * d[5] - 2 * d[7] - 2 * d[31] + 2 * d[33] - 2 * d[38] + 2 * d[43];
    coeffs[58] =
            2 * d[1] + 2 * d[3] + 2 * d[12] + 2 * d[16] - 4 * d[20] + 4 * d[22] + 2 * d[25] + 2 * d[29] + 4 * d[36] -
            4 * d[40];
    coeffs[59] = -4 * d[2] + 4 * d[6] - 8 * d[13] + 8 * d[15] - 12 * d[19] - 12 * d[23] - 8 * d[26] + 8 * d[28] +
                 12 * d[32] + 12 * d[41];
    coeffs[60] = -12 * d[12] - 12 * d[16] + 12 * d[20] - 12 * d[22] - 12 * d[25] - 12 * d[29] - 12 * d[36] + 12 * d[40];
    coeffs[61] =
            -4 * d[2] + 4 * d[6] + 8 * d[13] - 8 * d[15] + 4 * d[19] + 4 * d[23] + 8 * d[26] - 8 * d[28] - 4 * d[32] -
            4 * d[41];
    coeffs[62] = -2 * d[1] - 2 * d[3] + 2 * d[12] + 2 * d[16] + 2 * d[25] + 2 * d[29];
    coeffs[63] = 4 * d[10] - 4 * d[14] + 4 * d[18] - 4 * d[27] - 4 * d[31] + 4 * d[33];
    coeffs[64] =
            4 * d[0] + 4 * d[4] + 8 * d[8] + 4 * d[9] + 8 * d[11] - 8 * d[17] + 16 * d[21] + 4 * d[24] - 8 * d[30] -
            4 * d[34] + 4 * d[35] - 8 * d[37] - 8 * d[39] + 4 * d[42] + 4 * d[44];
    coeffs[65] = -12 * d[5] + 12 * d[7] - 12 * d[10] + 12 * d[14] - 12 * d[18] + 12 * d[27] + 12 * d[31] - 12 * d[33];
    coeffs[66] = 4 * d[0] - 8 * d[4] - 4 * d[8] - 4 * d[9] - 8 * d[11] - 4 * d[24] + 4 * d[30] + 4 * d[34];
    coeffs[67] = 4 * d[12] - 4 * d[20] + 4 * d[22] + 4 * d[25] + 4 * d[36] - 4 * d[40];
    coeffs[68] = 12 * d[2] - 12 * d[6] - 8 * d[13] + 8 * d[15] - 12 * d[19] + 4 * d[23] - 8 * d[26] + 8 * d[28] +
                 12 * d[32] - 4 * d[41];
    coeffs[69] = 12 * d[1] + 12 * d[3] - 12 * d[12] - 12 * d[25];
    coeffs[70] = -4 * d[0] + 8 * d[4] - 4 * d[8] + 4 * d[9] + 8 * d[11] + 4 * d[24] - 4 * d[30] + 4 * d[34];
    coeffs[71] = -2 * d[1] - 2 * d[3] + 2 * d[12] - 2 * d[16] + 2 * d[25] - 2 * d[29];
    coeffs[72] =
            2 * d[2] + 2 * d[6] + 4 * d[13] - 4 * d[15] + 2 * d[19] + 2 * d[23] - 4 * d[26] + 4 * d[28] + 2 * d[32] +
            2 * d[41];
    coeffs[73] = 4 * d[1] - 4 * d[3] + 12 * d[12] + 12 * d[16] - 8 * d[20] + 8 * d[22] - 12 * d[25] - 12 * d[29] -
                 8 * d[36] + 8 * d[40];
    coeffs[74] = -12 * d[13] + 12 * d[15] - 12 * d[19] - 12 * d[23] + 12 * d[26] - 12 * d[28] - 12 * d[32] - 12 * d[41];
    coeffs[75] =
            4 * d[1] - 4 * d[3] - 4 * d[12] - 4 * d[16] + 8 * d[20] - 8 * d[22] + 4 * d[25] + 4 * d[29] + 8 * d[36] -
            8 * d[40];
    coeffs[76] = -2 * d[2] - 2 * d[6] + 2 * d[19] + 2 * d[23] + 2 * d[32] + 2 * d[41];
    coeffs[77] = -4 * d[5] - 4 * d[7] + 24 * d[10] - 8 * d[14] - 8 * d[18] + 24 * d[27] - 12 * d[31] - 12 * d[33] -
                 12 * d[38] - 12 * d[43];
    coeffs[78] =
            -4 * d[5] - 4 * d[7] - 8 * d[10] - 8 * d[14] - 8 * d[18] - 8 * d[27] + 4 * d[31] + 4 * d[33] + 4 * d[38] +
            4 * d[43];
    coeffs[79] = 8 * d[13] + 4 * d[19] + 8 * d[28] + 4 * d[32];
    coeffs[80] = -12 * d[1] + 12 * d[3] + 12 * d[12] - 4 * d[16] - 24 * d[20] - 8 * d[22] - 12 * d[25] + 4 * d[29] +
                 8 * d[36] + 24 * d[40];
    coeffs[81] = 12 * d[2] + 12 * d[6] - 12 * d[13] - 12 * d[15] - 12 * d[19] - 12 * d[26] - 12 * d[28] - 12 * d[32];
    coeffs[82] = 12 * d[5] + 12 * d[7] + 8 * d[10] + 8 * d[14] + 8 * d[18] + 8 * d[27] - 12 * d[31] - 12 * d[33] +
                 4 * d[38] + 4 * d[43];
    coeffs[83] =
            -2 * d[2] - 2 * d[6] + 4 * d[13] + 4 * d[15] + 2 * d[19] - 2 * d[23] + 4 * d[26] + 4 * d[28] + 2 * d[32] -
            2 * d[41];
    coeffs[84] = -4 * d[10] - 4 * d[14] + 4 * d[18] + 4 * d[27] - 4 * d[38] + 4 * d[43];
    coeffs[85] =
            4 * d[0] + 8 * d[4] + 4 * d[8] - 8 * d[9] + 16 * d[11] + 4 * d[17] + 8 * d[21] - 8 * d[24] + 4 * d[30] -
            4 * d[34] + 4 * d[35] - 8 * d[37] + 4 * d[39] + 4 * d[42] - 8 * d[44];
    coeffs[86] = -12 * d[5] + 12 * d[7] + 12 * d[10] + 12 * d[14] - 12 * d[18] - 12 * d[27] + 12 * d[38] - 12 * d[43];
    coeffs[87] = 4 * d[0] - 4 * d[4] - 8 * d[8] - 4 * d[17] - 8 * d[21] + 4 * d[34] - 4 * d[39] + 4 * d[44];
    coeffs[88] = 4 * d[16] + 8 * d[22] + 4 * d[29] + 8 * d[36];
    coeffs[89] = 12 * d[2] - 12 * d[6] + 8 * d[13] + 24 * d[15] + 4 * d[19] - 12 * d[23] - 24 * d[26] - 8 * d[28] -
                 4 * d[32] + 12 * d[41];
    coeffs[90] = 12 * d[1] + 12 * d[3] - 12 * d[16] - 12 * d[20] - 12 * d[22] - 12 * d[29] - 12 * d[36] - 12 * d[40];
    coeffs[91] =
            -8 * d[0] + 4 * d[4] + 4 * d[8] + 4 * d[9] + 8 * d[11] + 4 * d[17] + 8 * d[21] + 4 * d[24] + 4 * d[30] -
            8 * d[34] - 8 * d[35] - 16 * d[37] + 4 * d[39] - 8 * d[42] + 4 * d[44];
    coeffs[92] = -4 * d[1] - 4 * d[3] + 4 * d[20] + 4 * d[22] + 4 * d[36] + 4 * d[40];
    coeffs[93] = 4 * d[13] - 4 * d[15] + 4 * d[23] - 4 * d[26] + 4 * d[28] + 4 * d[41];
    coeffs[94] = -12 * d[1] + 12 * d[3] - 4 * d[12] + 12 * d[16] - 8 * d[20] + 8 * d[22] + 4 * d[25] - 12 * d[29] -
                 8 * d[36] + 8 * d[40];
    coeffs[95] = 12 * d[2] + 12 * d[6] - 12 * d[23] - 12 * d[41];
    coeffs[96] =
            12 * d[5] + 12 * d[7] + 8 * d[10] + 8 * d[14] + 8 * d[18] + 8 * d[27] + 4 * d[31] + 4 * d[33] - 12 * d[38] -
            12 * d[43];
    coeffs[97] = -4 * d[2] - 4 * d[6] + 4 * d[13] + 4 * d[15] + 4 * d[26] + 4 * d[28];
    coeffs[98] = -4 * d[0] - 4 * d[4] + 8 * d[8] + 4 * d[17] + 8 * d[21] + 4 * d[34] + 4 * d[39] - 4 * d[44];
    coeffs[99] =
            -2 * d[1] - 2 * d[3] - 2 * d[12] + 2 * d[16] + 4 * d[20] + 4 * d[22] - 2 * d[25] + 2 * d[29] + 4 * d[36] +
            4 * d[40];
    coeffs[100] = -2 * d[2] - 2 * d[6] - 2 * d[19] + 2 * d[23] - 2 * d[32] + 2 * d[41];
    coeffs[101] = 2 * d[2] - 2 * d[6] + 2 * d[19] + 2 * d[23] - 2 * d[32] - 2 * d[41];
    coeffs[102] = 4 * d[2] - 4 * d[6] - 4 * d[13] + 4 * d[15] - 4 * d[26] + 4 * d[28];
    coeffs[103] = 4 * d[1] + 4 * d[3] - 4 * d[20] + 4 * d[22] + 4 * d[36] - 4 * d[40];
    coeffs[104] = 2 * d[1] + 2 * d[3] - 2 * d[12] - 2 * d[16] - 2 * d[25] - 2 * d[29];
    coeffs[105] = -4 * d[0] - 4 * d[4] - 8 * d[8] + 4 * d[17] - 8 * d[21] - 4 * d[34] + 4 * d[39] - 4 * d[44];
    coeffs[106] =
            12 * d[5] - 12 * d[7] + 8 * d[10] - 8 * d[14] + 8 * d[18] - 8 * d[27] + 4 * d[31] - 4 * d[33] + 12 * d[38] -
            12 * d[43];
    coeffs[107] =
            -8 * d[0] + 4 * d[4] - 4 * d[8] + 4 * d[9] + 8 * d[11] + 4 * d[17] - 8 * d[21] + 4 * d[24] + 4 * d[30] +
            8 * d[34] - 8 * d[35] + 16 * d[37] + 4 * d[39] - 8 * d[42] + 4 * d[44];
    coeffs[108] = 12 * d[5] - 12 * d[7] + 8 * d[10] - 8 * d[14] + 8 * d[18] - 8 * d[27] - 12 * d[31] + 12 * d[33] -
                  4 * d[38] + 4 * d[43];
    coeffs[109] = -4 * d[0] + 8 * d[4] + 4 * d[8] + 4 * d[9] + 8 * d[11] + 4 * d[24] - 4 * d[30] - 4 * d[34];
    coeffs[110] = -12 * d[2] + 12 * d[6] - 12 * d[23] + 12 * d[41];
    coeffs[111] = -12 * d[1] - 12 * d[3] - 12 * d[16] + 12 * d[20] - 12 * d[22] - 12 * d[29] - 12 * d[36] + 12 * d[40];
    coeffs[112] = -12 * d[2] + 12 * d[6] + 12 * d[13] - 12 * d[15] + 12 * d[19] + 12 * d[26] - 12 * d[28] - 12 * d[32];
    coeffs[113] = -12 * d[1] - 12 * d[3] + 12 * d[12] + 12 * d[25];
    coeffs[114] = 4 * d[0] - 4 * d[4] + 8 * d[8] - 4 * d[17] + 8 * d[21] - 4 * d[34] - 4 * d[39] + 4 * d[44];
    coeffs[115] =
            -4 * d[5] + 4 * d[7] - 8 * d[10] + 8 * d[14] - 8 * d[18] + 8 * d[27] + 4 * d[31] - 4 * d[33] - 4 * d[38] +
            4 * d[43];
    coeffs[116] = 4 * d[0] - 8 * d[4] + 4 * d[8] - 4 * d[9] - 8 * d[11] - 4 * d[24] + 4 * d[30] - 4 * d[34];
    coeffs[117] = 2 * d[2] - 2 * d[6] - 2 * d[19] + 2 * d[23] + 2 * d[32] - 2 * d[41];
    coeffs[118] = 2 * d[1] + 2 * d[3] - 2 * d[12] + 2 * d[16] - 2 * d[25] + 2 * d[29];
    coeffs[119] =
            2 * d[5] + 2 * d[7] - 4 * d[10] + 4 * d[14] + 4 * d[18] - 4 * d[27] + 2 * d[31] + 2 * d[33] + 2 * d[38] +
            2 * d[43];
    coeffs[120] = 4 * d[5] + 4 * d[7] + 8 * d[14] + 8 * d[18];
    coeffs[121] =
            2 * d[5] + 2 * d[7] + 4 * d[10] + 4 * d[14] + 4 * d[18] + 4 * d[27] - 2 * d[31] - 2 * d[33] - 2 * d[38] -
            2 * d[43];
    coeffs[122] = 12 * d[1] - 12 * d[3] + 4 * d[12] + 12 * d[16] + 8 * d[20] + 8 * d[22] - 4 * d[25] - 12 * d[29] -
                  8 * d[36] - 8 * d[40];
    coeffs[123] = -12 * d[2] - 12 * d[6] - 8 * d[13] + 24 * d[15] - 4 * d[19] - 12 * d[23] + 24 * d[26] - 8 * d[28] -
                  4 * d[32] - 12 * d[41];
    coeffs[124] = 12 * d[1] - 12 * d[3] - 12 * d[12] - 4 * d[16] + 24 * d[20] - 8 * d[22] + 12 * d[25] + 4 * d[29] +
                  8 * d[36] - 24 * d[40];
    coeffs[125] = -12 * d[2] - 12 * d[6] + 8 * d[13] + 8 * d[15] + 12 * d[19] + 4 * d[23] + 8 * d[26] + 8 * d[28] +
                  12 * d[32] + 4 * d[41];
    coeffs[126] = -12 * d[5] - 12 * d[7] + 12 * d[10] - 12 * d[14] - 12 * d[18] + 12 * d[27] - 12 * d[38] - 12 * d[43];
    coeffs[127] = -12 * d[5] - 12 * d[7] - 12 * d[10] - 12 * d[14] - 12 * d[18] - 12 * d[27] + 12 * d[31] + 12 * d[33];
    coeffs[128] =
            -4 * d[1] + 4 * d[3] + 4 * d[12] - 4 * d[16] - 8 * d[20] - 8 * d[22] - 4 * d[25] + 4 * d[29] + 8 * d[36] +
            8 * d[40];
    coeffs[129] =
            4 * d[2] + 4 * d[6] - 8 * d[13] - 8 * d[15] - 4 * d[19] + 4 * d[23] - 8 * d[26] - 8 * d[28] - 4 * d[32] +
            4 * d[41];
    coeffs[130] = 2 * d[5] + 2 * d[7] - 2 * d[31] - 2 * d[33] + 2 * d[38] + 2 * d[43];
    coeffs[131] = -4 * d[13] - 4 * d[15] + 4 * d[23] + 4 * d[26] + 4 * d[28] - 4 * d[41];
    coeffs[132] = -4 * d[12] + 4 * d[20] + 4 * d[22] - 4 * d[25] + 4 * d[36] + 4 * d[40];
    coeffs[133] =
            4 * d[0] + 8 * d[4] - 4 * d[8] - 8 * d[9] + 16 * d[11] + 4 * d[17] - 8 * d[21] - 8 * d[24] + 4 * d[30] +
            4 * d[34] + 4 * d[35] + 8 * d[37] + 4 * d[39] + 4 * d[42] - 8 * d[44];
    coeffs[134] = -4 * d[5] + 4 * d[7] + 24 * d[10] + 8 * d[14] - 8 * d[18] - 24 * d[27] - 12 * d[31] + 12 * d[33] +
                  12 * d[38] - 12 * d[43];
    coeffs[135] =
            4 * d[0] + 4 * d[4] - 8 * d[8] + 4 * d[9] + 8 * d[11] - 8 * d[17] - 16 * d[21] + 4 * d[24] - 8 * d[30] +
            4 * d[34] + 4 * d[35] + 8 * d[37] - 8 * d[39] + 4 * d[42] + 4 * d[44];
    coeffs[136] = 12 * d[13] + 12 * d[15] + 12 * d[19] - 12 * d[23] - 12 * d[26] - 12 * d[28] - 12 * d[32] + 12 * d[41];
    coeffs[137] = 12 * d[12] - 12 * d[16] - 12 * d[20] - 12 * d[22] + 12 * d[25] - 12 * d[29] - 12 * d[36] - 12 * d[40];
    coeffs[138] = -4 * d[4] + 4 * d[8] + 4 * d[30] - 8 * d[34] - 4 * d[35] - 8 * d[37] - 4 * d[42] + 4 * d[44];
    coeffs[139] = -4 * d[10] + 4 * d[14] + 4 * d[18] - 4 * d[27] + 4 * d[38] + 4 * d[43];
    coeffs[140] = 4 * d[10] + 4 * d[14] + 4 * d[18] + 4 * d[27] - 4 * d[31] - 4 * d[33];
    coeffs[141] = -4 * d[1] + 4 * d[3] - 12 * d[12] + 12 * d[16] + 8 * d[20] + 8 * d[22] + 12 * d[25] - 12 * d[29] -
                  8 * d[36] - 8 * d[40];
    coeffs[142] =
            4 * d[2] + 4 * d[6] + 8 * d[13] + 8 * d[15] + 12 * d[19] - 12 * d[23] + 8 * d[26] + 8 * d[28] + 12 * d[32] -
            12 * d[41];
    coeffs[143] = 12 * d[31] + 12 * d[33] - 12 * d[38] - 12 * d[43];
    coeffs[144] = -4 * d[4] + 4 * d[8] - 4 * d[30] + 8 * d[34] + 4 * d[35] + 8 * d[37] + 4 * d[42] - 4 * d[44];
    coeffs[145] = -2 * d[5] - 2 * d[7] - 2 * d[31] - 2 * d[33] + 2 * d[38] + 2 * d[43];
    coeffs[146] = -2 * d[1] + 2 * d[3] - 2 * d[12] - 2 * d[16] + 2 * d[25] + 2 * d[29];
    coeffs[147] = -4 * d[1] + 4 * d[3] - 4 * d[20] + 4 * d[22] - 4 * d[36] + 4 * d[40];
    coeffs[148] = 4 * d[2] + 4 * d[6] + 4 * d[13] - 4 * d[15] - 4 * d[26] + 4 * d[28];
    coeffs[149] = 2 * d[2] + 2 * d[6] - 2 * d[19] - 2 * d[23] - 2 * d[32] - 2 * d[41];
    coeffs[150] = -4 * d[12] + 4 * d[20] + 4 * d[22] + 4 * d[25] - 4 * d[36] - 4 * d[40];
    coeffs[151] = 4 * d[13] + 4 * d[15] - 4 * d[23] + 4 * d[26] + 4 * d[28] - 4 * d[41];
    coeffs[152] = -4 * d[10] + 4 * d[14] + 4 * d[18] - 4 * d[27] + 4 * d[31] + 4 * d[33];
    coeffs[153] = 4 * d[10] + 4 * d[14] + 4 * d[18] + 4 * d[27] - 4 * d[38] - 4 * d[43];
    coeffs[154] = -2 * d[5] - 2 * d[7] + 2 * d[31] + 2 * d[33] - 2 * d[38] - 2 * d[43];
    coeffs[155] = -4 * d[0] - 8 * d[4] - 4 * d[8] + 4 * d[9] - 8 * d[11] + 4 * d[24] - 4 * d[30] - 4 * d[34];
    coeffs[156] = 12 * d[5] - 12 * d[7] - 8 * d[10] - 8 * d[14] + 8 * d[18] + 8 * d[27] + 12 * d[31] - 12 * d[33] +
                  4 * d[38] - 4 * d[43];
    coeffs[157] =
            -8 * d[0] - 4 * d[4] + 4 * d[8] + 4 * d[9] - 8 * d[11] + 4 * d[17] + 8 * d[21] + 4 * d[24] + 4 * d[30] +
            8 * d[34] - 8 * d[35] + 16 * d[37] + 4 * d[39] - 8 * d[42] + 4 * d[44];
    coeffs[158] =
            12 * d[5] - 12 * d[7] - 8 * d[10] - 8 * d[14] + 8 * d[18] + 8 * d[27] - 4 * d[31] + 4 * d[33] - 12 * d[38] +
            12 * d[43];
    coeffs[159] = -4 * d[0] + 4 * d[4] + 8 * d[8] + 4 * d[17] + 8 * d[21] - 4 * d[34] + 4 * d[39] - 4 * d[44];
    coeffs[160] = -12 * d[2] + 12 * d[6] - 8 * d[13] - 8 * d[15] - 12 * d[19] - 4 * d[23] + 8 * d[26] + 8 * d[28] +
                  12 * d[32] + 4 * d[41];
    coeffs[161] = -12 * d[1] - 12 * d[3] - 12 * d[12] - 4 * d[16] + 24 * d[20] - 8 * d[22] - 12 * d[25] - 4 * d[29] -
                  8 * d[36] + 24 * d[40];
    coeffs[162] = -12 * d[2] + 12 * d[6] + 8 * d[13] - 24 * d[15] + 4 * d[19] + 12 * d[23] + 24 * d[26] - 8 * d[28] -
                  4 * d[32] - 12 * d[41];
    coeffs[163] = -12 * d[1] - 12 * d[3] + 4 * d[12] + 12 * d[16] + 8 * d[20] + 8 * d[22] + 4 * d[25] + 12 * d[29] +
                  8 * d[36] + 8 * d[40];
    coeffs[164] =
            4 * d[0] - 4 * d[4] + 8 * d[8] + 4 * d[9] - 8 * d[11] - 8 * d[17] + 16 * d[21] + 4 * d[24] - 8 * d[30] +
            4 * d[34] + 4 * d[35] + 8 * d[37] - 8 * d[39] + 4 * d[42] + 4 * d[44];
    coeffs[165] = -4 * d[5] + 4 * d[7] - 24 * d[10] + 8 * d[14] - 8 * d[18] + 24 * d[27] + 12 * d[31] - 12 * d[33] -
                  12 * d[38] + 12 * d[43];
    coeffs[166] =
            4 * d[0] - 8 * d[4] + 4 * d[8] - 8 * d[9] - 16 * d[11] + 4 * d[17] + 8 * d[21] - 8 * d[24] + 4 * d[30] +
            4 * d[34] + 4 * d[35] + 8 * d[37] + 4 * d[39] + 4 * d[42] - 8 * d[44];
    coeffs[167] =
            4 * d[2] - 4 * d[6] - 8 * d[13] - 8 * d[15] - 12 * d[19] + 12 * d[23] + 8 * d[26] + 8 * d[28] + 12 * d[32] -
            12 * d[41];
    coeffs[168] = 4 * d[1] + 4 * d[3] - 12 * d[12] + 12 * d[16] + 8 * d[20] + 8 * d[22] - 12 * d[25] + 12 * d[29] +
                  8 * d[36] + 8 * d[40];
    coeffs[169] = 4 * d[4] - 4 * d[8] - 4 * d[30] + 8 * d[34] + 4 * d[35] + 8 * d[37] + 4 * d[42] - 4 * d[44];
    coeffs[170] = 12 * d[1] - 12 * d[3] + 12 * d[12] - 12 * d[25];
    coeffs[171] = -12 * d[2] - 12 * d[6] - 12 * d[13] + 12 * d[15] - 12 * d[19] + 12 * d[26] - 12 * d[28] - 12 * d[32];
    coeffs[172] = 12 * d[1] - 12 * d[3] - 12 * d[16] + 12 * d[20] - 12 * d[22] + 12 * d[29] + 12 * d[36] - 12 * d[40];
    coeffs[173] = -12 * d[2] - 12 * d[6] + 12 * d[23] + 12 * d[41];
    coeffs[174] = -12 * d[5] - 12 * d[7] + 12 * d[10] - 12 * d[14] - 12 * d[18] + 12 * d[27] - 12 * d[31] - 12 * d[33];
    coeffs[175] = -12 * d[5] - 12 * d[7] - 12 * d[10] - 12 * d[14] - 12 * d[18] - 12 * d[27] + 12 * d[38] + 12 * d[43];
    coeffs[176] = 12 * d[12] - 12 * d[16] - 12 * d[20] - 12 * d[22] - 12 * d[25] + 12 * d[29] + 12 * d[36] + 12 * d[40];
    coeffs[177] =
            -12 * d[13] - 12 * d[15] - 12 * d[19] + 12 * d[23] - 12 * d[26] - 12 * d[28] - 12 * d[32] + 12 * d[41];
    coeffs[178] = -12 * d[31] - 12 * d[33] + 12 * d[38] + 12 * d[43];
    coeffs[179] = 4 * d[0] + 8 * d[4] - 4 * d[8] - 4 * d[9] + 8 * d[11] - 4 * d[24] + 4 * d[30] - 4 * d[34];
    coeffs[180] =
            -4 * d[5] + 4 * d[7] + 8 * d[10] + 8 * d[14] - 8 * d[18] - 8 * d[27] - 4 * d[31] + 4 * d[33] + 4 * d[38] -
            4 * d[43];
    coeffs[181] = 4 * d[0] + 4 * d[4] - 8 * d[8] - 4 * d[17] - 8 * d[21] - 4 * d[34] - 4 * d[39] + 4 * d[44];
    coeffs[182] =
            4 * d[2] - 4 * d[6] + 8 * d[13] + 8 * d[15] + 4 * d[19] - 4 * d[23] - 8 * d[26] - 8 * d[28] - 4 * d[32] +
            4 * d[41];
    coeffs[183] =
            4 * d[1] + 4 * d[3] + 4 * d[12] - 4 * d[16] - 8 * d[20] - 8 * d[22] + 4 * d[25] - 4 * d[29] - 8 * d[36] -
            8 * d[40];
    coeffs[184] = 4 * d[4] - 4 * d[8] + 4 * d[30] - 8 * d[34] - 4 * d[35] - 8 * d[37] - 4 * d[42] + 4 * d[44];
    coeffs[185] = -2 * d[1] + 2 * d[3] - 2 * d[12] + 2 * d[16] + 2 * d[25] - 2 * d[29];
    coeffs[186] = 2 * d[2] + 2 * d[6] + 2 * d[19] - 2 * d[23] + 2 * d[32] - 2 * d[41];
    coeffs[187] = 2 * d[5] + 2 * d[7] + 2 * d[31] + 2 * d[33] - 2 * d[38] - 2 * d[43];
    coeffs[188] = 1;
    coeffs[189] = -1;



    // Setup elimination template
    static const int coeffs0_ind[] = {53, 101, 54, 53, 101, 58, 146, 55, 54, 53, 101, 58, 102, 146, 72, 56, 55, 54, 0,
                                      58, 102, 103, 72, 147, 57, 56, 55, 1, 102, 103, 9, 147, 148, 57, 56, 2, 103, 9,
                                      104, 148, 24, 57, 3, 9, 104, 24, 149, 4, 104, 149, 53, 101, 146, 58, 105, 54, 53,
                                      101, 58, 146, 72, 59, 58, 105, 106, 55, 54, 53, 101, 58, 102, 146, 72, 119, 147,
                                      60, 59, 58, 5, 105, 106, 107, 119, 56, 55, 54, 0, 58, 102, 103, 72, 147, 28, 148,
                                      61, 60, 59, 6, 106, 107, 108, 28, 57, 56, 55, 1, 102, 103, 9, 147, 148, 120, 24,
                                      62, 61, 60, 7, 107, 108, 109, 120, 57, 56, 2, 103, 9, 104, 148, 24, 28, 149, 62,
                                      61, 8, 108, 109, 28, 57, 3, 9, 104, 24, 149, 121, 62, 9, 109, 121, 4, 104, 149,
                                      53, 101, 146, 54, 53, 101, 58, 146, 72, 58, 105, 55, 54, 53, 101, 58, 102, 146,
                                      72, 119, 147, 63, 110, 59, 58, 105, 106, 56, 55, 54, 53, 101, 58, 102, 103, 146,
                                      72, 119, 147, 28, 148, 64, 63, 110, 111, 60, 59, 58, 5, 105, 106, 107, 119, 57,
                                      56, 55, 54, 0, 58, 102, 103, 9, 72, 147, 28, 150, 148, 120, 24, 65, 64, 63, 10,
                                      110, 111, 112, 150, 61, 60, 59, 6, 106, 107, 108, 28, 57, 56, 55, 1, 102, 103, 9,
                                      104, 147, 148, 120, 79, 24, 28, 149, 66, 65, 64, 11, 111, 112, 113, 79, 62, 61,
                                      60, 7, 107, 108, 109, 120, 57, 56, 2, 103, 9, 104, 148, 24, 28, 31, 149, 121, 66,
                                      65, 12, 112, 113, 31, 62, 61, 8, 108, 109, 28, 57, 3, 9, 104, 24, 149, 121, 151,
                                      66, 13, 113, 151, 62, 9, 109, 121, 4, 104, 149, 58, 105, 53, 101, 146, 119, 59,
                                      58, 105, 106, 54, 53, 101, 58, 146, 72, 119, 28, 63, 110, 60, 59, 58, 105, 106,
                                      107, 55, 54, 53, 101, 58, 102, 146, 72, 119, 147, 28, 150, 120, 67, 114, 64, 63,
                                      110, 111, 61, 60, 59, 58, 5, 105, 106, 107, 108, 56, 55, 54, 0, 58, 102, 103, 72,
                                      119, 147, 28, 150, 148, 120, 79, 28, 68, 67, 114, 115, 65, 64, 63, 10, 110, 111,
                                      112, 150, 62, 61, 60, 59, 6, 106, 107, 108, 109, 57, 56, 55, 1, 102, 103, 9, 147,
                                      28, 148, 120, 79, 152, 24, 28, 31, 121, 69, 68, 67, 14, 114, 115, 116, 152, 66,
                                      65, 64, 11, 111, 112, 113, 79, 62, 61, 60, 7, 107, 108, 109, 57, 56, 2, 103, 9,
                                      104, 148, 120, 24, 28, 31, 28, 149, 121, 151, 69, 68, 15, 115, 116, 28, 66, 65,
                                      12, 112, 113, 31, 62, 61, 8, 108, 109, 57, 3, 9, 104, 24, 28, 149, 121, 151, 153,
                                      69, 16, 116, 153, 66, 13, 113, 151, 62, 9, 109, 4, 104, 149, 121, 63, 110, 58,
                                      105, 53, 101, 146, 119, 150, 64, 63, 110, 111, 59, 58, 105, 106, 54, 53, 101, 58,
                                      146, 72, 119, 28, 150, 79, 67, 114, 65, 64, 63, 110, 111, 112, 60, 59, 58, 5, 105,
                                      106, 107, 119, 55, 54, 0, 58, 102, 72, 147, 28, 150, 120, 79, 152, 31, 18, 117,
                                      68, 67, 114, 115, 66, 65, 64, 63, 10, 110, 111, 112, 113, 61, 60, 59, 6, 106, 107,
                                      108, 28, 56, 55, 1, 102, 103, 147, 150, 148, 120, 79, 152, 28, 31, 28, 151, 70,
                                      18, 117, 118, 69, 68, 67, 14, 114, 115, 116, 152, 66, 65, 64, 11, 111, 112, 113,
                                      62, 61, 60, 7, 107, 108, 109, 120, 57, 56, 2, 103, 9, 148, 79, 24, 28, 31, 28, 33,
                                      121, 151, 153, 70, 18, 17, 117, 118, 33, 69, 68, 15, 115, 116, 28, 66, 65, 12,
                                      112, 113, 62, 61, 8, 108, 109, 28, 57, 3, 9, 104, 24, 31, 149, 121, 151, 153, 83,
                                      70, 18, 118, 83, 69, 16, 116, 153, 66, 13, 113, 62, 9, 109, 121, 4, 104, 149, 151,
                                      67, 114, 63, 110, 58, 105, 53, 101, 146, 119, 150, 152, 68, 67, 114, 115, 64, 63,
                                      110, 111, 59, 58, 5, 105, 106, 119, 54, 58, 72, 0, 28, 150, 79, 152, 28, 18, 117,
                                      69, 68, 67, 114, 115, 116, 65, 64, 63, 10, 110, 111, 112, 150, 60, 59, 6, 106,
                                      107, 28, 55, 102, 147, 1, 120, 79, 152, 31, 28, 33, 153, 71, 70, 18, 117, 118, 69,
                                      68, 67, 14, 114, 115, 116, 66, 65, 64, 11, 111, 112, 113, 79, 61, 60, 7, 107, 108,
                                      120, 152, 56, 103, 148, 2, 28, 31, 28, 33, 151, 153, 83, 71, 70, 18, 17, 117, 118,
                                      33, 69, 68, 15, 115, 116, 66, 65, 12, 112, 113, 31, 62, 61, 8, 108, 109, 28, 28,
                                      57, 9, 24, 3, 121, 151, 153, 83, 154, 71, 19, 154, 70, 18, 118, 83, 69, 16, 116,
                                      66, 13, 113, 151, 62, 9, 109, 121, 153, 104, 149, 4, 18, 117, 67, 114, 63, 110,
                                      58, 105, 119, 5, 150, 152, 33, 188, 70, 18, 117, 118, 68, 67, 114, 115, 64, 63,
                                      10, 110, 111, 150, 59, 106, 28, 6, 79, 152, 28, 33, 83, 71, 70, 18, 117, 118, 69,
                                      68, 67, 14, 114, 115, 116, 152, 65, 64, 11, 111, 112, 79, 60, 107, 120, 7, 31, 28,
                                      33, 153, 83, 154, 188, 188, 71, 70, 18, 17, 117, 118, 69, 68, 15, 115, 116, 28,
                                      66, 65, 12, 112, 113, 31, 33, 61, 108, 28, 8, 151, 153, 83, 154, 71, 18, 117, 67,
                                      114, 63, 110, 150, 10, 152, 33, 154, 71, 70, 18, 117, 118, 68, 67, 14, 114, 115,
                                      152, 64, 111, 79, 11, 28, 33, 83, 154, 71, 18, 117, 67, 114, 152, 14, 33, 154,
                                      188, 188, 53, 101, 54, 53, 101, 58, 146, 55, 54, 53, 101, 58, 102, 146, 72, 56,
                                      55, 54, 58, 102, 103, 72, 0, 147, 57, 56, 55, 102, 103, 9, 147, 1, 148, 57, 56,
                                      103, 9, 104, 148, 2, 24, 57, 9, 104, 24, 3, 149, 104, 149, 4, 53, 101, 146, 72,
                                      119, 58, 105, 54, 53, 101, 58, 146, 72, 73, 72, 119, 28, 155, 59, 58, 105, 106,
                                      55, 54, 53, 101, 58, 102, 146, 72, 119, 147, 74, 73, 72, 20, 119, 28, 120, 155,
                                      156, 60, 59, 58, 105, 106, 107, 119, 5, 56, 55, 54, 58, 102, 103, 72, 0, 147, 28,
                                      148, 75, 74, 73, 21, 28, 120, 28, 156, 157, 61, 60, 59, 106, 107, 108, 28, 6, 57,
                                      56, 55, 102, 103, 9, 147, 1, 148, 120, 24, 76, 75, 74, 22, 120, 28, 121, 157, 158,
                                      62, 61, 60, 107, 108, 109, 120, 7, 57, 56, 103, 9, 104, 148, 2, 24, 28, 149, 76,
                                      75, 23, 28, 121, 158, 159, 62, 61, 108, 109, 28, 8, 57, 9, 104, 24, 3, 149, 121,
                                      76, 24, 121, 159, 62, 109, 121, 9, 104, 149, 4, 53, 101, 146, 54, 53, 101, 58,
                                      146, 72, 72, 119, 155, 58, 105, 55, 54, 53, 101, 58, 102, 146, 72, 119, 147, 28,
                                      122, 73, 72, 119, 28, 155, 156, 63, 110, 59, 58, 105, 106, 56, 55, 54, 53, 101,
                                      58, 102, 103, 146, 72, 119, 147, 28, 148, 77, 28, 122, 123, 74, 73, 72, 20, 119,
                                      28, 120, 155, 156, 160, 157, 64, 63, 110, 111, 60, 59, 58, 105, 106, 107, 119, 5,
                                      57, 56, 55, 54, 0, 58, 102, 103, 9, 72, 147, 28, 150, 148, 120, 24, 26, 77, 28,
                                      25, 122, 123, 124, 160, 75, 74, 73, 21, 28, 120, 28, 156, 157, 161, 158, 65, 64,
                                      63, 110, 111, 112, 150, 10, 61, 60, 59, 106, 107, 108, 28, 6, 57, 56, 55, 1, 102,
                                      103, 9, 104, 147, 148, 120, 79, 24, 28, 149, 78, 26, 77, 26, 123, 124, 125, 161,
                                      76, 75, 74, 22, 120, 28, 121, 157, 158, 162, 159, 66, 65, 64, 111, 112, 113, 79,
                                      11, 62, 61, 60, 107, 108, 109, 120, 7, 57, 56, 2, 103, 9, 104, 148, 24, 28, 31,
                                      149, 121, 78, 26, 27, 124, 125, 162, 76, 75, 23, 28, 121, 158, 159, 163, 66, 65,
                                      112, 113, 31, 12, 62, 61, 108, 109, 28, 8, 57, 3, 9, 104, 24, 149, 121, 151, 78,
                                      28, 125, 163, 76, 24, 121, 159, 66, 113, 151, 13, 62, 109, 121, 9, 4, 104, 149,
                                      72, 119, 155, 58, 105, 53, 101, 146, 119, 73, 72, 119, 28, 155, 156, 59, 58, 105,
                                      106, 54, 53, 101, 58, 146, 72, 119, 28, 28, 122, 74, 73, 72, 119, 28, 120, 155,
                                      156, 160, 157, 63, 110, 60, 59, 58, 105, 106, 107, 55, 54, 53, 101, 58, 102, 146,
                                      72, 119, 147, 28, 150, 120, 79, 126, 77, 28, 122, 123, 75, 74, 73, 72, 20, 119,
                                      28, 120, 28, 155, 156, 160, 157, 161, 158, 67, 114, 64, 63, 110, 111, 61, 60, 59,
                                      58, 5, 105, 106, 107, 108, 56, 55, 54, 0, 58, 102, 103, 72, 119, 147, 28, 150,
                                      148, 120, 79, 28, 80, 79, 126, 26, 26, 77, 28, 25, 122, 123, 124, 160, 76, 75, 74,
                                      73, 21, 28, 120, 28, 121, 156, 157, 161, 164, 158, 162, 159, 68, 67, 114, 115, 65,
                                      64, 63, 110, 111, 112, 150, 10, 62, 61, 60, 59, 6, 106, 107, 108, 109, 57, 56, 55,
                                      1, 102, 103, 9, 147, 28, 148, 120, 79, 152, 24, 28, 31, 121, 81, 80, 79, 29, 126,
                                      26, 127, 164, 78, 26, 77, 26, 123, 124, 125, 161, 76, 75, 74, 22, 120, 28, 121,
                                      157, 158, 162, 165, 159, 163, 69, 68, 67, 114, 115, 116, 152, 14, 66, 65, 64, 111,
                                      112, 113, 79, 11, 62, 61, 60, 7, 107, 108, 109, 57, 56, 2, 103, 9, 104, 148, 120,
                                      24, 28, 31, 28, 149, 121, 151, 81, 80, 30, 26, 127, 165, 78, 26, 27, 124, 125,
                                      162, 76, 75, 23, 28, 121, 158, 159, 163, 166, 69, 68, 115, 116, 28, 15, 66, 65,
                                      112, 113, 31, 12, 62, 61, 8, 108, 109, 57, 3, 9, 104, 24, 28, 149, 121, 151, 153,
                                      81, 31, 127, 166, 78, 28, 125, 163, 76, 24, 121, 159, 69, 116, 153, 16, 66, 113,
                                      151, 13, 62, 9, 109, 4, 104, 149, 121, 28, 122, 72, 119, 155, 160, 63, 110, 58,
                                      105, 53, 101, 146, 119, 150, 77, 28, 122, 123, 73, 72, 119, 28, 155, 156, 160,
                                      161, 64, 63, 110, 111, 59, 58, 105, 106, 54, 101, 58, 53, 146, 72, 119, 28, 150,
                                      79, 79, 126, 26, 77, 28, 122, 123, 124, 74, 73, 72, 20, 119, 28, 120, 155, 156,
                                      160, 157, 161, 164, 162, 67, 114, 65, 64, 63, 110, 111, 112, 60, 59, 58, 5, 105,
                                      106, 107, 119, 55, 58, 0, 102, 54, 72, 147, 28, 150, 120, 79, 152, 31, 28, 128,
                                      80, 79, 126, 26, 78, 26, 77, 28, 25, 122, 123, 124, 125, 75, 74, 73, 21, 28, 120,
                                      28, 156, 160, 157, 161, 164, 158, 162, 165, 163, 18, 117, 68, 67, 114, 115, 66,
                                      65, 64, 63, 10, 110, 111, 112, 113, 61, 60, 59, 6, 106, 107, 108, 28, 56, 102, 1,
                                      103, 55, 147, 150, 148, 120, 79, 152, 28, 31, 28, 151, 82, 28, 128, 129, 81, 80,
                                      79, 29, 126, 26, 127, 164, 78, 26, 77, 26, 123, 124, 125, 76, 75, 74, 22, 120, 28,
                                      121, 157, 161, 158, 162, 165, 167, 159, 163, 166, 70, 18, 117, 118, 69, 68, 67,
                                      114, 115, 116, 152, 14, 66, 65, 64, 11, 111, 112, 113, 62, 61, 60, 7, 107, 108,
                                      109, 120, 57, 103, 2, 9, 56, 148, 79, 24, 28, 31, 28, 33, 121, 151, 153, 82, 28,
                                      32, 128, 129, 167, 81, 80, 30, 26, 127, 165, 78, 26, 27, 124, 125, 76, 75, 23, 28,
                                      121, 158, 162, 159, 163, 166, 168, 70, 18, 117, 118, 33, 17, 69, 68, 115, 116, 28,
                                      15, 66, 65, 12, 112, 113, 62, 61, 8, 108, 109, 28, 9, 3, 104, 57, 24, 31, 149,
                                      121, 151, 153, 83, 82, 28, 129, 168, 81, 31, 127, 166, 78, 28, 125, 76, 24, 121,
                                      159, 163, 70, 118, 83, 18, 69, 116, 153, 16, 66, 13, 113, 62, 9, 109, 121, 104, 4,
                                      149, 151, 79, 126, 28, 122, 72, 119, 155, 160, 164, 67, 114, 63, 110, 58, 105, 53,
                                      101, 146, 119, 150, 152, 188, 80, 79, 126, 26, 77, 28, 122, 123, 73, 72, 20, 119,
                                      28, 155, 156, 160, 161, 164, 165, 68, 67, 114, 115, 64, 63, 110, 111, 59, 105, 5,
                                      106, 58, 119, 54, 58, 0, 72, 28, 150, 79, 152, 28, 188, 28, 128, 81, 80, 79, 126,
                                      26, 127, 26, 77, 28, 25, 122, 123, 124, 160, 74, 73, 21, 28, 120, 156, 157, 161,
                                      164, 162, 165, 167, 166, 18, 117, 69, 68, 67, 114, 115, 116, 65, 64, 63, 10, 110,
                                      111, 112, 150, 60, 106, 6, 107, 59, 28, 55, 102, 1, 147, 120, 79, 152, 31, 28, 33,
                                      153, 188, 188, 83, 130, 82, 28, 128, 129, 81, 80, 79, 29, 126, 26, 127, 78, 26,
                                      77, 26, 123, 124, 125, 161, 75, 74, 22, 120, 28, 157, 164, 158, 162, 165, 167,
                                      163, 166, 168, 71, 70, 18, 117, 118, 69, 68, 67, 14, 114, 115, 116, 66, 65, 64,
                                      11, 111, 112, 113, 79, 61, 107, 7, 108, 60, 120, 152, 56, 103, 2, 148, 28, 31, 28,
                                      33, 151, 153, 83, 188, 188, 83, 130, 82, 28, 32, 128, 129, 167, 81, 80, 30, 26,
                                      127, 78, 26, 27, 124, 125, 162, 76, 75, 23, 28, 121, 158, 165, 159, 163, 166, 168,
                                      169, 71, 70, 18, 117, 118, 33, 17, 69, 68, 15, 115, 116, 66, 65, 12, 112, 113, 31,
                                      62, 108, 8, 109, 61, 28, 28, 57, 9, 3, 24, 121, 151, 153, 83, 154, 188, 83, 33,
                                      130, 169, 82, 28, 129, 168, 81, 31, 127, 78, 28, 125, 163, 76, 24, 121, 159, 166,
                                      71, 154, 19, 70, 118, 83, 18, 69, 16, 116, 66, 13, 113, 151, 109, 9, 62, 121, 153,
                                      104, 4, 149, 188, 28, 128, 79, 126, 28, 122, 72, 119, 155, 20, 160, 164, 167, 18,
                                      117, 67, 114, 63, 110, 58, 105, 5, 119, 150, 152, 33, 188, 82, 28, 128, 129, 80,
                                      79, 126, 26, 77, 28, 25, 122, 123, 160, 73, 28, 156, 21, 161, 164, 165, 167, 168,
                                      70, 18, 117, 118, 68, 67, 114, 115, 64, 110, 10, 111, 63, 150, 59, 106, 6, 28, 79,
                                      152, 28, 33, 83, 83, 130, 82, 28, 128, 129, 81, 80, 79, 29, 126, 26, 127, 164, 26,
                                      77, 26, 123, 124, 161, 74, 120, 157, 22, 162, 165, 167, 166, 168, 169, 71, 70, 18,
                                      117, 118, 69, 68, 67, 14, 114, 115, 116, 152, 65, 111, 11, 112, 64, 79, 60, 107,
                                      7, 120, 31, 28, 33, 153, 83, 154, 188, 188, 83, 130, 82, 28, 32, 128, 129, 81, 80,
                                      30, 26, 127, 165, 78, 26, 27, 124, 125, 162, 167, 75, 28, 158, 23, 163, 166, 168,
                                      169, 71, 70, 18, 17, 117, 118, 69, 68, 15, 115, 116, 28, 66, 112, 12, 113, 65, 31,
                                      33, 61, 108, 8, 28, 151, 153, 83, 154, 83, 130, 28, 128, 79, 126, 28, 122, 160,
                                      25, 164, 167, 169, 71, 18, 117, 67, 114, 63, 110, 10, 150, 152, 33, 154, 188, 188,
                                      83, 130, 82, 28, 128, 129, 80, 79, 29, 126, 26, 164, 77, 123, 161, 26, 165, 167,
                                      168, 169, 71, 70, 18, 117, 118, 68, 114, 14, 115, 67, 152, 64, 111, 11, 79, 28,
                                      33, 83, 154, 188, 188, 83, 130, 28, 128, 79, 126, 164, 29, 167, 169, 71, 18, 117,
                                      67, 114, 14, 152, 33, 154, 188, 188, 53, 101, 146, 72, 119, 54, 53, 101, 58, 146,
                                      72, 73, 72, 119, 28, 155, 55, 54, 53, 101, 58, 102, 146, 72, 147, 74, 73, 72, 119,
                                      28, 120, 155, 20, 156, 56, 55, 54, 58, 102, 103, 72, 0, 147, 148, 75, 74, 73, 28,
                                      120, 28, 156, 21, 157, 57, 56, 55, 102, 103, 9, 147, 1, 148, 24, 76, 75, 74, 120,
                                      28, 121, 157, 22, 158, 57, 56, 103, 9, 104, 148, 2, 24, 149, 76, 75, 28, 121, 158,
                                      23, 159, 57, 9, 104, 24, 3, 149, 76, 121, 159, 24, 104, 149, 4, 53, 101, 146, 54,
                                      53, 101, 58, 146, 72, 72, 119, 155, 58, 105, 55, 54, 53, 101, 58, 102, 146, 72,
                                      119, 147, 84, 131, 28, 122, 73, 72, 119, 28, 155, 156, 59, 58, 105, 106, 56, 55,
                                      54, 101, 53, 58, 102, 103, 146, 72, 119, 147, 28, 148, 85, 84, 131, 88, 170, 77,
                                      28, 122, 123, 74, 73, 72, 119, 28, 120, 155, 20, 156, 160, 157, 60, 59, 58, 105,
                                      106, 107, 119, 5, 57, 56, 55, 58, 0, 54, 102, 103, 9, 72, 147, 28, 148, 120, 24,
                                      86, 85, 84, 34, 131, 88, 40, 170, 171, 26, 77, 28, 122, 123, 124, 160, 25, 75, 74,
                                      73, 28, 120, 28, 156, 21, 157, 161, 158, 61, 60, 59, 106, 107, 108, 28, 6, 57, 56,
                                      102, 1, 55, 103, 9, 104, 147, 148, 120, 24, 28, 149, 87, 86, 85, 35, 88, 40, 132,
                                      171, 172, 78, 26, 77, 123, 124, 125, 161, 26, 76, 75, 74, 120, 28, 121, 157, 22,
                                      158, 162, 159, 62, 61, 60, 107, 108, 109, 120, 7, 57, 103, 2, 56, 9, 104, 148, 24,
                                      28, 149, 121, 87, 86, 36, 40, 132, 172, 173, 78, 26, 124, 125, 162, 27, 76, 75,
                                      28, 121, 158, 23, 159, 163, 62, 61, 108, 109, 28, 8, 9, 3, 57, 104, 24, 149, 121,
                                      87, 37, 132, 173, 78, 125, 163, 28, 76, 121, 159, 24, 62, 109, 121, 9, 104, 4,
                                      149, 72, 119, 155, 58, 105, 53, 101, 146, 119, 73, 72, 119, 28, 155, 156, 59, 58,
                                      105, 106, 54, 101, 58, 53, 146, 72, 119, 28, 84, 131, 170, 28, 122, 74, 73, 72,
                                      119, 28, 120, 155, 156, 160, 157, 63, 110, 60, 59, 58, 105, 106, 107, 55, 58, 101,
                                      53, 102, 54, 146, 72, 119, 147, 28, 150, 120, 88, 133, 85, 84, 131, 88, 170, 171,
                                      79, 126, 77, 28, 122, 123, 75, 74, 73, 72, 20, 119, 28, 120, 28, 155, 156, 160,
                                      157, 161, 158, 64, 63, 110, 111, 61, 60, 59, 105, 5, 58, 106, 107, 108, 56, 102,
                                      58, 54, 103, 0, 55, 72, 119, 147, 28, 150, 148, 120, 79, 28, 89, 88, 133, 134, 86,
                                      85, 84, 34, 131, 88, 40, 170, 171, 174, 172, 80, 79, 126, 26, 26, 77, 28, 122,
                                      123, 124, 160, 25, 76, 75, 74, 73, 21, 28, 120, 28, 121, 156, 157, 161, 164, 158,
                                      162, 159, 65, 64, 63, 110, 111, 112, 150, 10, 62, 61, 60, 106, 6, 59, 107, 108,
                                      109, 57, 103, 102, 55, 9, 1, 56, 147, 28, 148, 120, 79, 24, 28, 31, 121, 90, 89,
                                      88, 38, 133, 134, 135, 174, 87, 86, 85, 35, 88, 40, 132, 171, 172, 26, 173, 81,
                                      80, 79, 126, 26, 127, 164, 29, 78, 26, 77, 123, 124, 125, 161, 26, 76, 75, 74, 22,
                                      120, 28, 121, 157, 158, 162, 165, 159, 163, 66, 65, 64, 111, 112, 113, 79, 11, 62,
                                      61, 107, 7, 60, 108, 109, 9, 103, 56, 104, 2, 57, 148, 120, 24, 28, 31, 149, 121,
                                      151, 90, 89, 39, 134, 135, 26, 87, 86, 36, 40, 132, 172, 173, 175, 81, 80, 26,
                                      127, 165, 30, 78, 26, 124, 125, 162, 27, 76, 75, 23, 28, 121, 158, 159, 163, 166,
                                      66, 65, 112, 113, 31, 12, 62, 108, 8, 61, 109, 104, 9, 57, 3, 24, 28, 149, 121,
                                      151, 90, 40, 135, 175, 87, 37, 132, 173, 81, 127, 166, 31, 78, 125, 163, 28, 76,
                                      24, 121, 159, 66, 113, 151, 13, 109, 9, 62, 104, 4, 149, 121, 84, 131, 170, 28,
                                      122, 72, 119, 155, 160, 63, 110, 58, 105, 101, 53, 146, 119, 150, 188, 85, 84,
                                      131, 88, 170, 171, 77, 28, 122, 123, 73, 72, 119, 28, 155, 156, 160, 161, 64, 63,
                                      110, 111, 59, 105, 106, 58, 58, 101, 53, 54, 146, 72, 119, 28, 150, 79, 188, 88,
                                      133, 86, 85, 84, 131, 88, 40, 170, 171, 174, 172, 79, 126, 26, 77, 28, 122, 123,
                                      124, 74, 73, 72, 20, 119, 28, 120, 155, 156, 160, 157, 161, 164, 162, 67, 114, 65,
                                      64, 63, 110, 111, 112, 60, 106, 105, 58, 107, 5, 59, 119, 102, 58, 0, 54, 55, 72,
                                      147, 28, 150, 120, 79, 152, 31, 188, 188, 42, 136, 89, 88, 133, 134, 87, 86, 85,
                                      84, 34, 131, 88, 40, 132, 170, 171, 174, 172, 26, 173, 28, 128, 80, 79, 126, 26,
                                      78, 26, 77, 28, 25, 122, 123, 124, 125, 75, 74, 73, 21, 28, 120, 28, 156, 160,
                                      157, 161, 164, 158, 162, 165, 163, 68, 67, 114, 115, 66, 65, 64, 110, 10, 63, 111,
                                      112, 113, 61, 107, 106, 59, 108, 6, 60, 28, 103, 102, 1, 55, 56, 147, 150, 148,
                                      120, 79, 152, 28, 31, 28, 151, 188, 188, 91, 42, 136, 137, 90, 89, 88, 38, 133,
                                      134, 135, 174, 87, 86, 85, 35, 88, 40, 132, 171, 172, 26, 176, 173, 175, 82, 28,
                                      128, 129, 81, 80, 79, 126, 26, 127, 164, 29, 78, 26, 77, 26, 123, 124, 125, 76,
                                      75, 74, 22, 120, 28, 121, 157, 161, 158, 162, 165, 167, 159, 163, 166, 69, 68, 67,
                                      114, 115, 116, 152, 14, 66, 65, 111, 11, 64, 112, 113, 62, 108, 107, 60, 109, 7,
                                      61, 120, 9, 103, 2, 56, 57, 148, 79, 24, 28, 31, 28, 121, 151, 153, 188, 188, 91,
                                      42, 41, 136, 137, 176, 90, 89, 39, 134, 135, 26, 87, 86, 36, 40, 132, 172, 173,
                                      175, 177, 82, 28, 128, 129, 167, 32, 81, 80, 26, 127, 165, 30, 78, 26, 27, 124,
                                      125, 76, 75, 23, 28, 121, 158, 162, 159, 163, 166, 168, 69, 68, 115, 116, 28, 15,
                                      66, 112, 12, 65, 113, 109, 108, 61, 8, 62, 28, 104, 9, 3, 57, 24, 31, 149, 121,
                                      151, 153, 188, 91, 42, 137, 177, 90, 40, 135, 175, 87, 37, 132, 173, 82, 129, 168,
                                      28, 81, 127, 166, 31, 78, 28, 125, 76, 24, 121, 159, 163, 69, 116, 153, 16, 113,
                                      13, 66, 109, 62, 9, 121, 104, 4, 149, 151, 188, 88, 133, 84, 131, 170, 174, 79,
                                      126, 28, 122, 72, 119, 155, 160, 164, 67, 114, 63, 110, 105, 58, 101, 53, 146,
                                      119, 150, 152, 188, 89, 88, 133, 134, 85, 84, 131, 88, 170, 171, 174, 26, 80, 79,
                                      126, 26, 77, 28, 122, 123, 73, 119, 20, 28, 72, 155, 156, 160, 161, 164, 165, 68,
                                      67, 114, 115, 64, 110, 111, 63, 106, 105, 5, 58, 59, 119, 58, 0, 54, 72, 28, 150,
                                      79, 152, 28, 188, 42, 136, 90, 89, 88, 133, 134, 135, 86, 85, 84, 34, 131, 88, 40,
                                      170, 171, 174, 172, 26, 176, 175, 28, 128, 81, 80, 79, 126, 26, 127, 26, 77, 28,
                                      25, 122, 123, 124, 160, 74, 28, 21, 120, 73, 156, 157, 161, 164, 162, 165, 167,
                                      166, 18, 117, 69, 68, 67, 114, 115, 116, 65, 111, 110, 63, 112, 10, 64, 150, 107,
                                      106, 6, 59, 60, 28, 102, 1, 55, 147, 120, 79, 152, 31, 28, 33, 153, 188, 188, 92,
                                      138, 91, 42, 136, 137, 90, 89, 88, 38, 133, 134, 135, 87, 86, 85, 35, 88, 40, 132,
                                      171, 174, 172, 26, 176, 173, 175, 177, 83, 130, 82, 28, 128, 129, 81, 80, 79, 29,
                                      126, 26, 127, 78, 26, 77, 26, 123, 124, 125, 161, 75, 120, 22, 28, 74, 157, 164,
                                      158, 162, 165, 167, 163, 166, 168, 70, 18, 117, 118, 69, 68, 114, 14, 67, 115,
                                      116, 66, 112, 111, 64, 113, 11, 65, 79, 108, 107, 7, 60, 61, 120, 152, 103, 2, 56,
                                      148, 28, 31, 28, 33, 151, 153, 83, 188, 188, 92, 138, 91, 42, 41, 136, 137, 176,
                                      90, 89, 39, 134, 135, 87, 86, 36, 40, 132, 172, 26, 173, 175, 177, 178, 83, 130,
                                      82, 28, 128, 129, 167, 32, 81, 80, 30, 26, 127, 78, 26, 27, 124, 125, 162, 76, 28,
                                      23, 121, 75, 158, 165, 159, 163, 166, 168, 169, 70, 18, 117, 118, 33, 17, 69, 115,
                                      15, 68, 116, 113, 112, 65, 12, 66, 31, 109, 108, 8, 61, 62, 28, 28, 9, 3, 57, 24,
                                      121, 151, 153, 83, 188, 92, 43, 138, 178, 91, 42, 137, 177, 90, 40, 135, 87, 37,
                                      132, 173, 175, 83, 130, 169, 33, 82, 129, 168, 28, 81, 31, 127, 78, 28, 125, 163,
                                      121, 24, 76, 159, 166, 70, 118, 83, 18, 116, 16, 69, 113, 66, 13, 151, 109, 9, 62,
                                      121, 153, 104, 4, 149, 188, 42, 136, 88, 133, 84, 131, 170, 174, 176, 28, 128, 79,
                                      126, 28, 122, 72, 119, 20, 155, 160, 164, 167, 18, 117, 67, 114, 110, 63, 105, 5,
                                      58, 119, 150, 152, 33, 188, 188, 91, 42, 136, 137, 89, 88, 133, 134, 85, 84, 34,
                                      131, 88, 170, 171, 174, 26, 176, 177, 82, 28, 128, 129, 80, 79, 126, 26, 77, 122,
                                      25, 123, 28, 160, 73, 28, 21, 156, 161, 164, 165, 167, 168, 70, 18, 117, 118, 68,
                                      114, 115, 67, 111, 110, 10, 63, 64, 150, 106, 6, 59, 28, 79, 152, 28, 33, 83, 188,
                                      188, 92, 138, 91, 42, 136, 137, 90, 89, 88, 38, 133, 134, 135, 174, 86, 85, 35,
                                      88, 40, 171, 172, 26, 176, 175, 177, 178, 83, 130, 82, 28, 128, 129, 81, 80, 79,
                                      29, 126, 26, 127, 164, 26, 123, 26, 124, 77, 161, 74, 120, 22, 157, 162, 165, 167,
                                      166, 168, 169, 71, 70, 18, 117, 118, 69, 115, 114, 67, 116, 14, 68, 152, 112, 111,
                                      11, 64, 65, 79, 107, 7, 60, 120, 31, 28, 33, 153, 83, 154, 188, 188, 188, 92, 138,
                                      91, 42, 41, 136, 137, 90, 89, 39, 134, 135, 26, 87, 86, 36, 40, 132, 172, 176,
                                      173, 175, 177, 178, 83, 130, 82, 28, 32, 128, 129, 81, 80, 30, 26, 127, 165, 78,
                                      124, 27, 125, 26, 162, 167, 75, 28, 23, 158, 163, 166, 168, 169, 71, 70, 117, 17,
                                      18, 118, 116, 115, 68, 15, 69, 28, 113, 112, 12, 65, 66, 31, 33, 108, 8, 61, 28,
                                      151, 153, 83, 154, 188, 188, 92, 138, 42, 136, 88, 133, 84, 131, 170, 34, 174,
                                      176, 178, 83, 130, 28, 128, 79, 126, 28, 122, 25, 160, 164, 167, 169, 71, 18, 117,
                                      114, 67, 110, 10, 63, 150, 152, 33, 154, 188, 188, 92, 138, 91, 42, 136, 137, 89,
                                      88, 38, 133, 134, 174, 85, 88, 171, 35, 26, 176, 177, 178, 83, 130, 82, 28, 128,
                                      129, 80, 126, 29, 26, 79, 164, 77, 123, 26, 161, 165, 167, 168, 169, 71, 70, 117,
                                      118, 18, 115, 114, 14, 67, 68, 152, 111, 11, 64, 79, 28, 33, 83, 154, 188, 188,
                                      92, 138, 42, 136, 88, 133, 174, 38, 176, 178, 83, 130, 28, 128, 79, 126, 29, 164,
                                      167, 169, 71, 117, 18, 114, 14, 67, 152, 33, 154, 188, 188, 188, 53, 101, 146, 54,
                                      101, 58, 53, 146, 72, 72, 119, 155, 55, 58, 101, 53, 102, 54, 146, 72, 147, 84,
                                      131, 73, 72, 119, 28, 155, 156, 56, 102, 58, 53, 101, 54, 103, 55, 146, 72, 147,
                                      148, 85, 84, 131, 88, 170, 74, 73, 72, 119, 28, 120, 155, 20, 156, 157, 57, 103,
                                      102, 54, 58, 55, 9, 56, 0, 72, 147, 148, 24, 86, 85, 84, 131, 88, 40, 170, 34,
                                      171, 75, 74, 73, 28, 120, 28, 156, 21, 157, 158, 9, 103, 55, 102, 56, 104, 57, 1,
                                      147, 148, 24, 149, 87, 86, 85, 88, 40, 132, 171, 35, 172, 76, 75, 74, 120, 28,
                                      121, 157, 22, 158, 159, 104, 9, 56, 103, 57, 2, 148, 24, 149, 87, 86, 40, 132,
                                      172, 36, 173, 76, 75, 28, 121, 158, 23, 159, 104, 57, 9, 3, 24, 149, 87, 132, 173,
                                      37, 76, 121, 159, 24, 104, 4, 149, 72, 119, 155, 58, 105, 53, 101, 146, 119, 73,
                                      72, 119, 28, 155, 156, 59, 105, 106, 58, 54, 53, 101, 58, 146, 72, 119, 28, 84,
                                      131, 170, 28, 122, 74, 73, 72, 119, 28, 120, 155, 156, 160, 157, 60, 106, 105, 58,
                                      107, 59, 55, 54, 101, 58, 102, 146, 53, 72, 119, 147, 28, 120, 93, 139, 88, 133,
                                      85, 84, 131, 88, 170, 171, 77, 28, 122, 123, 75, 74, 73, 119, 20, 72, 28, 120, 28,
                                      155, 156, 160, 157, 161, 158, 61, 107, 106, 58, 105, 59, 108, 60, 5, 56, 55, 58,
                                      102, 103, 72, 54, 0, 119, 147, 28, 148, 120, 28, 94, 93, 139, 28, 179, 89, 88,
                                      133, 134, 86, 85, 84, 131, 88, 40, 170, 34, 171, 174, 172, 26, 77, 28, 122, 123,
                                      124, 160, 25, 76, 75, 74, 28, 21, 73, 120, 28, 121, 156, 157, 161, 158, 162, 159,
                                      62, 108, 107, 59, 106, 60, 109, 61, 6, 57, 56, 102, 103, 9, 147, 55, 1, 28, 148,
                                      120, 24, 28, 121, 188, 95, 94, 93, 44, 139, 28, 140, 179, 180, 90, 89, 88, 133,
                                      134, 135, 174, 38, 87, 86, 85, 88, 40, 132, 171, 35, 172, 26, 173, 78, 26, 77,
                                      123, 124, 125, 161, 26, 76, 75, 120, 22, 74, 28, 121, 157, 158, 162, 159, 163,
                                      109, 108, 60, 107, 61, 62, 7, 57, 103, 9, 104, 148, 56, 2, 120, 24, 28, 149, 121,
                                      188, 95, 94, 45, 28, 140, 180, 181, 90, 89, 134, 135, 26, 39, 87, 86, 40, 132,
                                      172, 36, 173, 175, 78, 26, 124, 125, 162, 27, 76, 28, 23, 75, 121, 158, 159, 163,
                                      109, 61, 108, 62, 8, 9, 104, 24, 57, 3, 28, 149, 121, 188, 95, 46, 140, 181, 90,
                                      135, 175, 40, 87, 132, 173, 37, 78, 125, 163, 28, 121, 24, 76, 159, 62, 109, 9,
                                      104, 149, 4, 121, 188, 84, 131, 170, 28, 122, 72, 119, 155, 160, 63, 110, 58, 105,
                                      101, 146, 119, 150, 53, 188, 85, 84, 131, 88, 170, 171, 77, 28, 122, 123, 73, 119,
                                      28, 72, 155, 156, 160, 161, 64, 110, 111, 63, 59, 58, 105, 106, 58, 101, 53, 146,
                                      72, 119, 28, 150, 79, 54, 188, 93, 139, 179, 88, 133, 86, 85, 84, 131, 88, 40,
                                      170, 171, 174, 172, 79, 126, 26, 77, 28, 122, 123, 124, 74, 28, 119, 72, 120, 20,
                                      73, 155, 156, 160, 157, 161, 164, 162, 65, 111, 110, 63, 112, 64, 60, 59, 105,
                                      106, 107, 119, 58, 5, 102, 58, 54, 0, 72, 147, 28, 150, 120, 79, 31, 55, 188, 188,
                                      28, 141, 94, 93, 139, 28, 179, 180, 42, 136, 89, 88, 133, 134, 87, 86, 85, 84, 34,
                                      131, 88, 40, 132, 170, 171, 174, 172, 26, 173, 80, 79, 126, 26, 78, 26, 77, 122,
                                      25, 28, 123, 124, 125, 75, 120, 28, 73, 28, 21, 74, 156, 160, 157, 161, 164, 158,
                                      162, 165, 163, 66, 112, 111, 63, 110, 64, 113, 65, 10, 61, 60, 106, 107, 108, 28,
                                      59, 6, 103, 102, 55, 1, 147, 150, 148, 120, 79, 28, 31, 151, 56, 188, 188, 96, 28,
                                      141, 142, 95, 94, 93, 44, 139, 28, 140, 179, 180, 182, 181, 91, 42, 136, 137, 90,
                                      89, 88, 133, 134, 135, 174, 38, 87, 86, 85, 35, 88, 40, 132, 171, 172, 26, 176,
                                      173, 175, 81, 80, 79, 126, 26, 127, 164, 29, 78, 26, 123, 26, 77, 124, 125, 76,
                                      28, 120, 74, 121, 22, 75, 157, 161, 158, 162, 165, 159, 163, 166, 113, 112, 64,
                                      111, 65, 66, 11, 62, 61, 107, 108, 109, 120, 60, 7, 9, 103, 56, 2, 148, 79, 24,
                                      28, 31, 121, 151, 57, 188, 188, 96, 28, 47, 141, 142, 182, 95, 94, 45, 28, 140,
                                      180, 181, 183, 91, 42, 136, 137, 176, 41, 90, 89, 134, 135, 26, 39, 87, 86, 36,
                                      40, 132, 172, 173, 175, 177, 81, 80, 26, 127, 165, 30, 78, 124, 27, 26, 125, 121,
                                      28, 75, 23, 76, 158, 162, 159, 163, 166, 113, 65, 112, 66, 12, 62, 108, 109, 28,
                                      61, 8, 104, 9, 57, 3, 24, 31, 149, 121, 151, 188, 96, 28, 142, 183, 95, 46, 140,
                                      181, 91, 137, 177, 42, 90, 135, 175, 40, 87, 37, 132, 173, 81, 127, 166, 31, 125,
                                      28, 78, 121, 76, 24, 159, 163, 66, 113, 13, 109, 121, 62, 9, 104, 4, 149, 151,
                                      188, 93, 139, 179, 88, 133, 84, 131, 170, 174, 79, 126, 28, 122, 119, 72, 155,
                                      160, 164, 67, 114, 63, 110, 105, 119, 101, 146, 53, 150, 152, 58, 188, 94, 93,
                                      139, 28, 179, 180, 89, 88, 133, 134, 85, 84, 131, 88, 170, 171, 174, 26, 80, 79,
                                      126, 26, 77, 122, 123, 28, 28, 119, 20, 72, 73, 155, 156, 160, 161, 164, 165, 68,
                                      114, 115, 67, 64, 63, 110, 111, 106, 105, 58, 5, 119, 28, 58, 72, 150, 0, 54, 79,
                                      152, 28, 59, 188, 28, 141, 95, 94, 93, 139, 28, 140, 179, 180, 182, 181, 42, 136,
                                      90, 89, 88, 133, 134, 135, 86, 85, 84, 34, 131, 88, 40, 170, 171, 174, 172, 26,
                                      176, 175, 28, 128, 81, 80, 79, 126, 26, 127, 26, 123, 122, 28, 124, 25, 77, 160,
                                      120, 28, 21, 73, 74, 156, 157, 161, 164, 162, 165, 167, 166, 69, 115, 114, 67,
                                      116, 68, 65, 64, 110, 111, 112, 150, 63, 10, 107, 106, 59, 6, 28, 120, 102, 147,
                                      79, 152, 1, 55, 31, 28, 153, 60, 188, 188, 97, 143, 96, 28, 141, 142, 95, 94, 93,
                                      44, 139, 28, 140, 179, 180, 182, 181, 183, 92, 138, 91, 42, 136, 137, 90, 89, 88,
                                      38, 133, 134, 135, 87, 86, 85, 35, 88, 40, 132, 171, 174, 172, 26, 176, 173, 175,
                                      177, 82, 28, 128, 129, 81, 80, 126, 29, 79, 26, 127, 78, 124, 123, 77, 125, 26,
                                      26, 161, 28, 120, 22, 74, 75, 157, 164, 158, 162, 165, 167, 163, 166, 168, 116,
                                      115, 67, 114, 68, 69, 14, 66, 65, 111, 112, 113, 79, 64, 11, 108, 107, 60, 7, 120,
                                      152, 28, 103, 148, 31, 28, 2, 56, 151, 153, 61, 188, 188, 97, 143, 96, 28, 47,
                                      141, 142, 182, 95, 94, 45, 28, 140, 180, 181, 183, 184, 92, 138, 91, 42, 136, 137,
                                      176, 41, 90, 89, 39, 134, 135, 87, 86, 36, 40, 132, 172, 26, 173, 175, 177, 178,
                                      82, 28, 128, 129, 167, 32, 81, 26, 30, 80, 127, 125, 124, 26, 27, 78, 162, 121,
                                      28, 23, 75, 76, 158, 165, 159, 163, 166, 168, 116, 68, 115, 69, 15, 66, 112, 113,
                                      31, 65, 12, 109, 108, 61, 8, 28, 28, 121, 9, 24, 151, 153, 3, 57, 62, 188, 188,
                                      97, 48, 143, 184, 96, 28, 142, 183, 95, 46, 140, 181, 92, 138, 178, 43, 91, 137,
                                      177, 42, 90, 40, 135, 87, 37, 132, 173, 175, 82, 129, 168, 28, 127, 31, 81, 125,
                                      78, 28, 163, 121, 24, 76, 159, 166, 69, 116, 16, 113, 151, 66, 13, 109, 62, 9,
                                      121, 153, 104, 149, 4, 188, 188, 28, 141, 93, 139, 179, 182, 42, 136, 88, 133, 84,
                                      131, 170, 174, 176, 28, 128, 79, 126, 122, 28, 119, 20, 72, 155, 160, 164, 167,
                                      18, 117, 67, 114, 110, 150, 105, 119, 5, 58, 152, 33, 63, 188, 188, 96, 28, 141,
                                      142, 94, 93, 139, 28, 179, 180, 182, 183, 91, 42, 136, 137, 89, 88, 133, 134, 85,
                                      131, 34, 88, 84, 170, 171, 174, 26, 176, 177, 82, 28, 128, 129, 80, 126, 26, 79,
                                      123, 122, 25, 28, 77, 160, 28, 21, 73, 156, 161, 164, 165, 167, 168, 70, 117, 118,
                                      18, 68, 67, 114, 115, 111, 110, 63, 10, 150, 79, 106, 28, 152, 6, 59, 28, 33, 83,
                                      64, 188, 188, 97, 143, 96, 28, 141, 142, 95, 94, 93, 44, 139, 28, 140, 179, 180,
                                      182, 181, 183, 184, 92, 138, 91, 42, 136, 137, 90, 89, 88, 38, 133, 134, 135, 174,
                                      86, 88, 35, 40, 85, 171, 172, 26, 176, 175, 177, 178, 83, 130, 82, 28, 128, 129,
                                      81, 26, 126, 79, 127, 29, 80, 164, 124, 123, 26, 77, 26, 161, 120, 22, 74, 157,
                                      162, 165, 167, 166, 168, 169, 118, 117, 18, 70, 69, 68, 114, 115, 116, 152, 67,
                                      14, 112, 111, 64, 11, 79, 31, 107, 120, 28, 33, 7, 60, 153, 83, 65, 188, 188, 188,
                                      97, 143, 96, 28, 47, 141, 142, 95, 94, 45, 28, 140, 180, 182, 181, 183, 184, 92,
                                      138, 91, 42, 41, 136, 137, 90, 89, 39, 134, 135, 26, 87, 40, 36, 132, 86, 172,
                                      176, 173, 175, 177, 178, 83, 130, 82, 128, 32, 28, 129, 127, 26, 80, 30, 81, 165,
                                      125, 124, 27, 26, 78, 162, 167, 28, 23, 75, 158, 163, 166, 168, 169, 118, 18, 117,
                                      70, 17, 69, 115, 116, 28, 68, 15, 113, 112, 65, 12, 31, 33, 151, 108, 28, 153, 83,
                                      8, 61, 66, 188, 188, 97, 143, 28, 141, 93, 139, 179, 182, 184, 92, 138, 42, 136,
                                      88, 133, 84, 131, 34, 170, 174, 176, 178, 83, 130, 28, 128, 126, 79, 122, 25, 28,
                                      160, 164, 167, 169, 71, 18, 117, 114, 152, 110, 150, 10, 63, 33, 154, 67, 188,
                                      188, 188, 97, 143, 96, 28, 141, 142, 94, 93, 44, 139, 28, 179, 180, 182, 183, 184,
                                      92, 138, 91, 42, 136, 137, 89, 133, 38, 134, 88, 174, 85, 88, 35, 171, 26, 176,
                                      177, 178, 83, 130, 82, 128, 129, 28, 26, 126, 29, 79, 80, 164, 123, 26, 77, 161,
                                      165, 167, 168, 169, 71, 70, 18, 117, 118, 115, 114, 67, 14, 152, 28, 111, 79, 33,
                                      11, 64, 83, 154, 68, 188, 188, 188, 97, 143, 28, 141, 93, 139, 179, 44, 182, 184,
                                      92, 138, 42, 136, 88, 133, 38, 174, 176, 178, 83, 130, 128, 28, 126, 29, 79, 164,
                                      167, 169, 71, 117, 33, 114, 152, 14, 67, 154, 18, 188, 188, 188, 72, 119, 155, 53,
                                      101, 146, 73, 119, 28, 72, 155, 156, 54, 101, 58, 53, 146, 72, 84, 131, 170, 74,
                                      28, 119, 72, 120, 73, 155, 156, 157, 55, 58, 53, 101, 102, 146, 54, 72, 147, 93,
                                      139, 85, 84, 131, 88, 170, 171, 75, 120, 28, 72, 119, 73, 28, 74, 20, 155, 156,
                                      157, 158, 56, 102, 54, 0, 58, 103, 72, 55, 147, 148, 94, 93, 139, 28, 179, 86, 85,
                                      84, 131, 88, 40, 170, 34, 171, 172, 76, 28, 120, 73, 28, 74, 121, 75, 21, 156,
                                      157, 158, 159, 57, 103, 55, 1, 102, 9, 147, 56, 148, 24, 188, 95, 94, 93, 139, 28,
                                      140, 179, 44, 180, 87, 86, 85, 88, 40, 132, 171, 35, 172, 173, 121, 28, 74, 120,
                                      75, 76, 22, 157, 158, 159, 9, 56, 2, 103, 104, 148, 57, 24, 149, 188, 95, 94, 28,
                                      140, 180, 45, 181, 87, 86, 40, 132, 172, 36, 173, 121, 75, 28, 76, 23, 158, 159,
                                      104, 57, 3, 9, 24, 149, 188, 95, 140, 181, 46, 87, 132, 173, 37, 76, 121, 24, 159,
                                      4, 104, 149, 188, 84, 131, 170, 28, 122, 72, 119, 155, 160, 58, 105, 101, 146,
                                      119, 53, 188, 85, 84, 131, 88, 170, 171, 77, 122, 123, 28, 73, 72, 119, 28, 155,
                                      156, 160, 161, 59, 105, 106, 58, 58, 146, 72, 119, 53, 101, 28, 54, 188, 93, 139,
                                      179, 88, 133, 86, 85, 84, 131, 88, 40, 170, 171, 174, 172, 26, 123, 122, 28, 124,
                                      77, 74, 73, 119, 28, 120, 155, 72, 20, 156, 160, 157, 161, 162, 60, 106, 58, 5,
                                      105, 107, 119, 59, 102, 72, 0, 147, 28, 54, 58, 120, 55, 188, 188, 50, 51, 28,
                                      141, 94, 93, 139, 28, 179, 180, 89, 88, 133, 134, 87, 86, 85, 131, 34, 84, 88, 40,
                                      132, 170, 171, 174, 172, 26, 173, 78, 124, 123, 28, 122, 77, 125, 26, 25, 75, 74,
                                      28, 120, 28, 156, 73, 21, 160, 157, 161, 158, 162, 163, 61, 107, 59, 6, 106, 108,
                                      28, 60, 103, 147, 1, 148, 120, 55, 102, 28, 56, 188, 188, 98, 50, 51, 99, 185, 96,
                                      28, 141, 142, 95, 94, 93, 139, 28, 140, 179, 44, 180, 182, 181, 90, 89, 88, 133,
                                      134, 135, 174, 38, 87, 86, 88, 35, 85, 40, 132, 171, 172, 26, 173, 175, 125, 124,
                                      77, 123, 26, 78, 26, 76, 75, 120, 28, 121, 157, 74, 22, 161, 158, 162, 159, 163,
                                      62, 108, 60, 7, 107, 109, 120, 61, 9, 148, 2, 24, 28, 56, 103, 121, 57, 188, 188,
                                      98, 50, 49, 51, 99, 185, 186, 96, 28, 141, 142, 182, 47, 95, 94, 28, 140, 180, 45,
                                      181, 183, 90, 89, 134, 135, 26, 39, 87, 40, 36, 86, 132, 172, 173, 175, 125, 26,
                                      124, 78, 27, 76, 28, 121, 158, 75, 23, 162, 159, 163, 109, 61, 8, 108, 28, 62,
                                      104, 24, 3, 149, 121, 57, 9, 188, 98, 50, 99, 186, 96, 142, 183, 28, 95, 140, 181,
                                      46, 90, 135, 175, 40, 132, 37, 87, 173, 78, 125, 28, 121, 159, 76, 24, 163, 62, 9,
                                      109, 121, 149, 4, 104, 188, 93, 139, 179, 88, 133, 84, 131, 170, 174, 79, 126, 28,
                                      122, 119, 155, 160, 164, 72, 63, 110, 105, 119, 53, 146, 101, 150, 58, 188, 94,
                                      93, 139, 28, 179, 180, 89, 88, 133, 134, 85, 131, 88, 84, 170, 171, 174, 26, 80,
                                      126, 26, 79, 77, 28, 122, 123, 28, 119, 72, 20, 155, 156, 160, 161, 164, 165, 73,
                                      64, 110, 111, 63, 106, 119, 5, 28, 150, 58, 105, 54, 0, 72, 58, 79, 59, 188, 50,
                                      51, 185, 28, 141, 95, 94, 93, 139, 28, 140, 179, 180, 182, 181, 42, 136, 90, 89,
                                      88, 133, 134, 135, 86, 88, 131, 84, 40, 34, 85, 170, 171, 174, 172, 26, 176, 175,
                                      81, 26, 126, 79, 127, 80, 26, 77, 122, 123, 124, 160, 28, 25, 120, 28, 73, 21,
                                      156, 157, 161, 164, 162, 165, 166, 74, 65, 111, 63, 10, 110, 112, 150, 64, 107,
                                      28, 6, 120, 79, 59, 106, 55, 1, 147, 102, 31, 60, 188, 188, 99, 144, 98, 50, 51,
                                      99, 185, 186, 97, 143, 96, 28, 141, 142, 95, 94, 93, 44, 139, 28, 140, 179, 180,
                                      182, 181, 183, 91, 42, 136, 137, 90, 89, 133, 38, 88, 134, 135, 87, 40, 88, 85,
                                      132, 35, 86, 171, 174, 172, 26, 176, 173, 175, 177, 127, 26, 79, 126, 80, 81, 29,
                                      78, 26, 123, 124, 125, 161, 77, 26, 28, 120, 74, 22, 157, 164, 158, 162, 165, 163,
                                      166, 75, 66, 112, 64, 11, 111, 113, 79, 65, 108, 120, 7, 28, 31, 60, 107, 56, 2,
                                      148, 103, 151, 61, 188, 188, 99, 144, 98, 50, 49, 51, 99, 185, 186, 187, 97, 143,
                                      96, 28, 141, 142, 182, 47, 95, 94, 45, 28, 140, 180, 181, 183, 184, 91, 42, 136,
                                      137, 176, 41, 90, 134, 39, 89, 135, 132, 40, 86, 36, 87, 172, 26, 173, 175, 177,
                                      127, 80, 26, 81, 30, 78, 124, 125, 162, 26, 27, 121, 28, 75, 23, 158, 165, 159,
                                      163, 166, 76, 113, 65, 12, 112, 31, 66, 109, 28, 8, 121, 151, 61, 108, 57, 3, 24,
                                      9, 62, 188, 188, 99, 51, 144, 187, 98, 50, 99, 186, 97, 143, 184, 48, 96, 142,
                                      183, 28, 95, 46, 140, 181, 91, 137, 177, 42, 135, 40, 90, 132, 87, 37, 173, 175,
                                      81, 127, 31, 125, 163, 78, 28, 121, 76, 24, 159, 166, 66, 13, 113, 151, 121, 9,
                                      62, 109, 4, 149, 104, 188, 188, 50, 51, 185, 28, 141, 93, 139, 179, 182, 42, 136,
                                      88, 133, 131, 84, 170, 174, 176, 28, 128, 79, 126, 122, 160, 119, 155, 20, 72,
                                      164, 167, 28, 67, 114, 110, 150, 58, 5, 119, 105, 152, 63, 188, 188, 188, 98, 50,
                                      51, 99, 185, 186, 96, 28, 141, 142, 94, 93, 139, 28, 179, 180, 182, 183, 91, 42,
                                      136, 137, 89, 133, 134, 88, 88, 131, 34, 84, 85, 170, 171, 174, 26, 176, 177, 82,
                                      128, 129, 28, 80, 79, 126, 26, 123, 122, 28, 25, 160, 161, 28, 156, 164, 21, 73,
                                      165, 167, 168, 77, 68, 114, 115, 67, 111, 150, 10, 79, 152, 63, 110, 59, 6, 28,
                                      106, 28, 64, 188, 188, 188, 99, 144, 98, 50, 51, 99, 185, 186, 187, 97, 143, 96,
                                      28, 141, 142, 95, 94, 93, 44, 139, 28, 140, 179, 180, 182, 181, 183, 184, 92, 138,
                                      91, 42, 136, 137, 90, 134, 133, 88, 135, 38, 89, 174, 40, 88, 35, 85, 86, 171,
                                      172, 26, 176, 175, 177, 178, 129, 128, 28, 82, 81, 80, 126, 26, 127, 164, 79, 29,
                                      124, 123, 77, 26, 161, 162, 120, 157, 165, 167, 22, 74, 166, 168, 26, 69, 115, 67,
                                      14, 114, 116, 152, 68, 112, 79, 11, 31, 28, 64, 111, 60, 7, 120, 107, 153, 65,
                                      188, 188, 188, 188, 99, 144, 98, 50, 49, 51, 99, 185, 186, 187, 97, 143, 96, 28,
                                      47, 141, 142, 95, 94, 45, 28, 140, 180, 182, 181, 183, 184, 92, 138, 91, 136, 41,
                                      42, 137, 135, 134, 89, 39, 90, 26, 132, 40, 36, 86, 87, 172, 176, 173, 175, 177,
                                      178, 129, 28, 128, 82, 32, 81, 26, 127, 165, 80, 30, 125, 124, 26, 27, 162, 167,
                                      163, 28, 158, 166, 168, 23, 75, 78, 116, 68, 15, 115, 28, 69, 113, 31, 12, 151,
                                      153, 65, 112, 61, 8, 28, 108, 66, 188, 188, 188, 99, 144, 50, 51, 185, 187, 97,
                                      143, 28, 141, 93, 139, 179, 182, 184, 92, 138, 42, 136, 133, 88, 131, 34, 84, 170,
                                      174, 176, 178, 83, 130, 28, 128, 126, 164, 122, 160, 25, 28, 167, 169, 79, 18,
                                      117, 114, 152, 63, 10, 150, 110, 33, 67, 188, 188, 188, 99, 144, 98, 50, 51, 99,
                                      185, 186, 187, 97, 143, 96, 28, 141, 142, 94, 139, 44, 28, 93, 179, 180, 182, 183,
                                      184, 92, 138, 91, 136, 137, 42, 134, 133, 38, 88, 89, 174, 88, 35, 85, 171, 26,
                                      176, 177, 178, 130, 83, 82, 28, 128, 129, 26, 126, 79, 29, 164, 165, 123, 161,
                                      167, 26, 77, 168, 169, 80, 70, 117, 118, 18, 115, 152, 14, 28, 33, 67, 114, 64,
                                      11, 79, 111, 83, 68, 188, 188, 188, 99, 144, 50, 51, 185, 187, 97, 143, 28, 141,
                                      93, 139, 44, 179, 182, 184, 92, 138, 136, 42, 133, 38, 88, 174, 176, 178, 83, 130,
                                      128, 167, 126, 164, 29, 79, 169, 28, 71, 117, 33, 67, 14, 152, 114, 154, 18, 188,
                                      188, 188, 84, 131, 170, 72, 119, 155, 146, 53, 101, 188, 85, 131, 88, 84, 170,
                                      171, 73, 119, 28, 72, 155, 156, 72, 101, 146, 53, 54, 58, 188, 93, 139, 179, 86,
                                      88, 131, 84, 40, 85, 170, 171, 172, 74, 28, 72, 20, 119, 120, 155, 73, 156, 157,
                                      147, 58, 0, 72, 54, 55, 102, 188, 188, 50, 51, 94, 93, 139, 28, 179, 180, 87, 40,
                                      88, 84, 131, 85, 132, 86, 34, 170, 171, 172, 173, 75, 120, 73, 21, 28, 28, 156,
                                      74, 157, 158, 148, 102, 1, 147, 55, 56, 103, 188, 188, 98, 50, 51, 99, 185, 95,
                                      94, 93, 139, 28, 140, 179, 44, 180, 181, 132, 40, 85, 88, 86, 87, 35, 171, 172,
                                      173, 76, 28, 74, 22, 120, 121, 157, 75, 158, 159, 24, 103, 2, 148, 56, 57, 9, 188,
                                      188, 98, 50, 51, 99, 185, 49, 186, 95, 94, 28, 140, 180, 45, 181, 132, 86, 40, 87,
                                      36, 172, 173, 121, 75, 23, 28, 158, 76, 159, 149, 9, 3, 24, 57, 104, 188, 98, 99,
                                      186, 50, 95, 140, 181, 46, 87, 132, 37, 173, 76, 24, 121, 159, 104, 4, 149, 188,
                                      93, 139, 179, 88, 133, 84, 131, 170, 174, 28, 122, 119, 155, 160, 72, 119, 58,
                                      105, 101, 146, 53, 188, 94, 93, 139, 28, 179, 180, 89, 133, 134, 88, 85, 84, 131,
                                      88, 170, 171, 174, 26, 77, 122, 123, 28, 28, 155, 20, 156, 160, 72, 119, 161, 73,
                                      28, 105, 5, 119, 58, 59, 106, 58, 0, 72, 54, 188, 50, 51, 185, 28, 141, 95, 94,
                                      93, 139, 28, 140, 179, 180, 182, 181, 90, 134, 133, 88, 135, 89, 86, 85, 131, 88,
                                      40, 170, 84, 34, 171, 174, 172, 26, 175, 26, 123, 28, 25, 122, 124, 160, 77, 120,
                                      156, 21, 157, 161, 73, 28, 162, 74, 120, 106, 6, 28, 59, 60, 107, 102, 1, 147, 55,
                                      188, 188, 100, 145, 99, 144, 98, 50, 51, 99, 185, 186, 96, 28, 141, 142, 95, 94,
                                      139, 44, 93, 28, 140, 179, 180, 182, 181, 183, 135, 134, 88, 133, 89, 90, 38, 87,
                                      86, 88, 40, 132, 171, 85, 35, 174, 172, 26, 173, 175, 78, 124, 77, 26, 123, 125,
                                      161, 26, 28, 157, 22, 158, 162, 74, 120, 163, 75, 28, 107, 7, 120, 60, 61, 108,
                                      103, 2, 148, 56, 188, 188, 100, 145, 99, 144, 98, 50, 51, 99, 185, 49, 186, 187,
                                      96, 28, 141, 142, 182, 47, 95, 28, 45, 94, 140, 180, 181, 183, 135, 89, 134, 90,
                                      39, 87, 40, 132, 172, 86, 36, 26, 173, 175, 125, 26, 27, 124, 162, 78, 121, 158,
                                      23, 159, 163, 75, 28, 76, 121, 108, 8, 28, 61, 62, 109, 9, 3, 24, 57, 188, 188,
                                      100, 52, 145, 99, 144, 187, 51, 98, 99, 186, 50, 96, 142, 183, 28, 140, 46, 95,
                                      181, 90, 135, 40, 132, 173, 87, 37, 175, 78, 28, 125, 163, 159, 24, 76, 121, 109,
                                      9, 121, 62, 104, 4, 149, 188, 188, 50, 51, 185, 28, 141, 93, 139, 179, 182, 42,
                                      136, 88, 133, 131, 170, 174, 176, 84, 79, 126, 122, 160, 72, 20, 155, 119, 164,
                                      28, 150, 63, 110, 105, 5, 119, 58, 188, 188, 188, 98, 50, 51, 99, 185, 186, 96,
                                      28, 141, 142, 94, 139, 28, 93, 179, 180, 182, 183, 91, 136, 137, 42, 89, 88, 133,
                                      134, 88, 131, 84, 34, 170, 171, 174, 26, 176, 177, 85, 80, 126, 26, 79, 123, 160,
                                      25, 161, 164, 28, 122, 73, 21, 156, 28, 165, 77, 79, 110, 10, 150, 63, 64, 111,
                                      106, 6, 28, 59, 188, 188, 188, 100, 145, 99, 144, 98, 50, 51, 99, 185, 186, 187,
                                      97, 143, 96, 28, 141, 142, 95, 28, 139, 93, 140, 44, 94, 179, 180, 182, 181, 183,
                                      184, 137, 136, 42, 91, 90, 89, 133, 134, 135, 174, 88, 38, 40, 88, 85, 35, 171,
                                      172, 26, 176, 175, 177, 86, 81, 26, 79, 29, 126, 127, 164, 80, 124, 161, 26, 162,
                                      165, 77, 123, 74, 22, 157, 120, 166, 26, 31, 111, 11, 79, 64, 65, 112, 107, 7,
                                      120, 60, 188, 188, 188, 188, 100, 145, 99, 144, 98, 50, 49, 51, 99, 185, 186, 187,
                                      97, 143, 96, 141, 47, 28, 142, 140, 28, 94, 45, 95, 180, 182, 181, 183, 184, 137,
                                      42, 136, 91, 41, 90, 134, 135, 26, 89, 39, 132, 40, 86, 36, 172, 176, 173, 175,
                                      177, 87, 127, 80, 30, 26, 165, 81, 125, 162, 27, 163, 166, 26, 124, 75, 23, 158,
                                      28, 78, 151, 112, 12, 31, 65, 66, 113, 108, 8, 28, 61, 188, 188, 188, 100, 145,
                                      99, 144, 50, 51, 185, 187, 97, 143, 28, 141, 139, 93, 179, 182, 184, 92, 138, 42,
                                      136, 133, 174, 131, 170, 34, 84, 176, 178, 88, 28, 128, 126, 164, 28, 25, 160,
                                      122, 167, 79, 152, 67, 114, 110, 10, 150, 63, 188, 188, 188, 100, 145, 99, 144,
                                      98, 50, 51, 99, 185, 186, 187, 97, 143, 96, 141, 142, 28, 28, 139, 44, 93, 94,
                                      179, 180, 182, 183, 184, 138, 92, 91, 42, 136, 137, 134, 133, 88, 38, 174, 26, 88,
                                      171, 176, 35, 85, 177, 178, 89, 82, 128, 129, 28, 26, 164, 29, 165, 167, 79, 126,
                                      77, 26, 161, 123, 168, 80, 28, 114, 14, 152, 67, 68, 115, 111, 11, 79, 64, 188,
                                      188, 188, 100, 145, 99, 144, 50, 51, 185, 187, 97, 143, 141, 28, 139, 44, 93, 179,
                                      182, 184, 92, 138, 136, 176, 133, 174, 38, 88, 178, 42, 83, 130, 128, 167, 79, 29,
                                      164, 126, 169, 28, 33, 18, 117, 114, 14, 152, 67, 188, 188, 188, 93, 139, 179, 84,
                                      131, 170, 155, 72, 119, 146, 53, 101, 188, 94, 139, 28, 93, 179, 180, 85, 131, 88,
                                      84, 170, 171, 156, 119, 20, 155, 72, 73, 28, 72, 0, 54, 58, 188, 50, 51, 185, 95,
                                      28, 139, 93, 140, 94, 179, 180, 181, 86, 88, 84, 34, 131, 40, 170, 85, 171, 172,
                                      157, 28, 21, 156, 73, 74, 120, 147, 1, 55, 102, 188, 188, 100, 145, 98, 50, 51,
                                      99, 185, 186, 140, 28, 93, 139, 94, 95, 44, 179, 180, 181, 87, 40, 85, 35, 88,
                                      132, 171, 86, 172, 173, 158, 120, 22, 157, 74, 75, 28, 148, 2, 56, 103, 188, 188,
                                      100, 145, 98, 50, 51, 99, 185, 49, 186, 140, 94, 28, 95, 45, 180, 181, 132, 86,
                                      36, 40, 172, 87, 173, 159, 28, 23, 158, 75, 76, 121, 24, 3, 57, 9, 188, 188, 100,
                                      145, 52, 98, 99, 186, 50, 95, 140, 46, 181, 87, 37, 132, 173, 121, 24, 159, 76,
                                      149, 4, 104, 188, 188, 50, 51, 185, 28, 141, 93, 139, 179, 182, 88, 133, 131, 170,
                                      174, 84, 160, 28, 122, 119, 20, 155, 72, 119, 5, 58, 105, 188, 188, 98, 50, 51,
                                      99, 185, 186, 96, 141, 142, 28, 94, 93, 139, 28, 179, 180, 182, 183, 89, 133, 134,
                                      88, 88, 170, 34, 171, 174, 84, 131, 26, 85, 161, 122, 25, 160, 28, 77, 123, 28,
                                      21, 156, 73, 28, 6, 59, 106, 188, 188, 100, 145, 99, 144, 98, 50, 51, 99, 185,
                                      186, 187, 142, 141, 28, 96, 95, 94, 139, 28, 140, 179, 93, 44, 180, 182, 181, 183,
                                      90, 134, 88, 38, 133, 135, 174, 89, 40, 171, 35, 172, 26, 85, 88, 175, 86, 162,
                                      123, 26, 161, 77, 26, 124, 120, 22, 157, 74, 120, 7, 60, 107, 188, 188, 188, 100,
                                      145, 99, 144, 98, 51, 49, 50, 99, 185, 186, 187, 142, 28, 141, 96, 47, 95, 28,
                                      140, 180, 94, 45, 182, 181, 183, 135, 89, 39, 134, 26, 90, 132, 172, 36, 173, 175,
                                      86, 40, 87, 163, 124, 27, 162, 26, 78, 125, 28, 23, 158, 75, 28, 8, 61, 108, 188,
                                      188, 100, 145, 99, 144, 50, 51, 185, 187, 97, 143, 28, 141, 139, 179, 182, 184,
                                      93, 42, 136, 133, 174, 84, 34, 170, 131, 176, 88, 164, 79, 126, 122, 25, 160, 28,
                                      150, 10, 63, 110, 188, 188, 188, 100, 145, 99, 144, 98, 51, 99, 50, 185, 186, 187,
                                      143, 97, 96, 28, 141, 142, 28, 139, 93, 44, 179, 180, 182, 183, 184, 94, 91, 136,
                                      137, 42, 134, 174, 38, 26, 176, 88, 133, 85, 35, 171, 88, 177, 89, 165, 126, 29,
                                      164, 79, 80, 26, 123, 26, 161, 77, 79, 11, 64, 111, 188, 188, 188, 100, 145, 99,
                                      144, 51, 50, 185, 187, 97, 143, 141, 182, 139, 179, 44, 93, 184, 28, 92, 138, 136,
                                      176, 88, 38, 174, 133, 178, 42, 167, 28, 128, 126, 29, 164, 79, 152, 14, 67, 114,
                                      188, 188, 188, 50, 51, 185, 93, 139, 179, 170, 84, 131, 155, 20, 72, 119, 188,
                                      188, 98, 51, 99, 50, 185, 186, 94, 139, 28, 93, 179, 180, 171, 131, 34, 170, 84,
                                      85, 88, 156, 21, 73, 28, 188, 188, 100, 145, 99, 51, 50, 98, 185, 186, 95, 28, 93,
                                      44, 139, 140, 179, 94, 180, 181, 172, 88, 35, 171, 85, 86, 40, 157, 22, 74, 120,
                                      188, 188, 188, 100, 145, 99, 50, 51, 98, 49, 185, 186, 140, 94, 45, 28, 180, 95,
                                      181, 173, 40, 36, 172, 86, 87, 132, 158, 23, 75, 28, 188, 188, 100, 145, 99, 144,
                                      50, 51, 185, 187, 28, 141, 139, 179, 182, 93, 174, 88, 133, 131, 34, 170, 84, 160,
                                      25, 28, 122, 188, 188, 100, 145, 144, 99, 98, 50, 51, 99, 185, 186, 187, 96, 141,
                                      142, 28, 28, 179, 44, 180, 182, 93, 139, 183, 94, 26, 133, 38, 174, 88, 89, 134,
                                      88, 35, 171, 85, 161, 26, 77, 123, 188, 188, 100, 145, 99, 144, 51, 185, 187, 50,
                                      97, 143, 141, 182, 93, 44, 179, 139, 184, 28, 176, 42, 136, 133, 38, 174, 88, 164,
                                      29, 79, 126, 188, 188, 188, 100, 145, 50, 51, 185, 179, 93, 139, 170, 34, 84, 131,
                                      188, 188, 145, 100, 98, 51, 99, 50, 185, 186, 180, 139, 44, 179, 93, 94, 28, 171,
                                      35, 85, 88, 188, 188, 100, 145, 99, 144, 51, 185, 187, 50, 182, 28, 141, 139, 44,
                                      179, 93, 174, 38, 88, 133, 188, 188, 100, 145, 185, 50, 51, 179, 44, 93, 139, 188,
                                      188, 53, 146, 101, 54, 146, 53, 72, 58, 101, 55, 72, 53, 146, 101, 54, 147, 102,
                                      58, 56, 147, 54, 72, 0, 58, 55, 148, 103, 102, 57, 148, 55, 147, 1, 102, 56, 24,
                                      9, 103, 24, 56, 148, 2, 103, 57, 149, 104, 9, 149, 57, 24, 3, 9, 104, 149, 4, 104,
                                      101, 146, 53, 58, 53, 146, 72, 54, 101, 58, 119, 102, 54, 146, 72, 147, 55, 105,
                                      58, 101, 53, 59, 119, 58, 28, 103, 55, 72, 101, 146, 147, 148, 56, 53, 106, 105,
                                      102, 58, 54, 60, 28, 58, 119, 5, 105, 59, 120, 9, 56, 147, 58, 72, 0, 148, 24, 57,
                                      54, 107, 106, 103, 102, 55, 61, 120, 59, 28, 6, 106, 60, 28, 104, 57, 148, 102,
                                      147, 1, 24, 149, 55, 108, 107, 9, 103, 56, 62, 28, 60, 120, 7, 107, 61, 121, 24,
                                      103, 148, 2, 149, 56, 109, 108, 104, 9, 57, 121, 61, 28, 8, 108, 62, 149, 9, 24,
                                      3, 57, 109, 104, 62, 121, 9, 109, 104, 149, 4, 105, 119, 58, 53, 146, 101, 106,
                                      58, 119, 28, 59, 54, 146, 53, 72, 101, 105, 58, 63, 150, 107, 59, 119, 28, 120,
                                      60, 55, 72, 146, 54, 147, 58, 110, 106, 101, 105, 53, 102, 58, 64, 150, 63, 79,
                                      108, 60, 28, 105, 119, 5, 120, 28, 61, 56, 147, 72, 55, 0, 148, 102, 58, 111, 110,
                                      107, 58, 106, 54, 103, 59, 65, 79, 63, 150, 10, 110, 64, 31, 109, 61, 120, 106,
                                      28, 6, 28, 121, 62, 57, 148, 147, 56, 1, 24, 103, 59, 112, 111, 108, 102, 107, 55,
                                      9, 60, 66, 31, 64, 79, 11, 111, 65, 151, 62, 28, 107, 120, 7, 121, 24, 148, 57, 2,
                                      149, 9, 60, 113, 112, 109, 103, 108, 56, 104, 61, 151, 65, 31, 12, 112, 66, 121,
                                      108, 28, 8, 149, 24, 3, 104, 61, 113, 9, 109, 57, 62, 66, 151, 13, 113, 109, 121,
                                      9, 149, 4, 62, 104, 110, 150, 63, 58, 119, 146, 53, 101, 105, 188, 111, 63, 150,
                                      79, 64, 59, 119, 58, 28, 105, 72, 146, 53, 101, 54, 58, 110, 106, 188, 67, 152,
                                      112, 64, 150, 79, 31, 65, 60, 28, 119, 59, 5, 120, 106, 147, 72, 54, 0, 58, 55,
                                      114, 102, 111, 105, 110, 58, 107, 63, 188, 188, 68, 152, 67, 28, 113, 65, 79, 110,
                                      150, 10, 31, 151, 66, 61, 120, 28, 60, 6, 28, 107, 63, 148, 147, 55, 1, 102, 56,
                                      115, 103, 114, 112, 106, 111, 59, 108, 64, 188, 188, 69, 28, 67, 152, 14, 114, 68,
                                      153, 66, 31, 111, 79, 11, 151, 62, 28, 120, 61, 7, 121, 108, 64, 24, 148, 56, 2,
                                      103, 57, 116, 9, 115, 113, 107, 112, 60, 109, 188, 65, 188, 153, 68, 28, 15, 115,
                                      69, 151, 112, 31, 12, 121, 28, 62, 8, 109, 65, 149, 24, 57, 3, 9, 104, 116, 108,
                                      113, 61, 66, 188, 69, 153, 16, 116, 113, 151, 13, 121, 9, 66, 149, 4, 104, 109,
                                      62, 188, 114, 152, 67, 63, 150, 119, 58, 146, 53, 105, 101, 110, 188, 115, 67,
                                      152, 28, 68, 64, 150, 63, 79, 110, 28, 119, 58, 5, 105, 59, 72, 54, 106, 114, 58,
                                      111, 0, 188, 18, 33, 116, 68, 152, 28, 153, 69, 65, 79, 150, 64, 10, 31, 111, 120,
                                      28, 59, 6, 106, 60, 117, 147, 55, 107, 115, 110, 114, 102, 63, 112, 67, 1, 188,
                                      188, 70, 33, 18, 83, 69, 28, 114, 152, 14, 153, 66, 31, 79, 65, 11, 151, 112, 67,
                                      28, 120, 60, 7, 107, 61, 118, 148, 56, 108, 117, 116, 111, 115, 103, 188, 64, 113,
                                      68, 2, 188, 83, 18, 33, 17, 117, 70, 153, 115, 28, 15, 151, 31, 66, 12, 113, 68,
                                      121, 28, 61, 8, 108, 62, 24, 57, 109, 118, 112, 116, 9, 65, 69, 3, 188, 70, 83,
                                      18, 118, 116, 153, 16, 151, 13, 69, 121, 62, 9, 109, 149, 113, 104, 188, 66, 4,
                                      117, 33, 18, 67, 152, 150, 63, 119, 58, 110, 105, 114, 5, 188, 188, 118, 18, 33,
                                      83, 70, 68, 152, 67, 28, 114, 79, 150, 63, 10, 110, 64, 28, 59, 111, 117, 106,
                                      115, 6, 188, 188, 71, 154, 70, 33, 83, 69, 28, 152, 68, 14, 153, 115, 31, 79, 64,
                                      11, 111, 65, 120, 60, 112, 118, 114, 117, 107, 188, 67, 116, 18, 7, 188, 188, 154,
                                      71, 83, 117, 33, 17, 153, 28, 69, 15, 116, 18, 151, 31, 65, 12, 112, 66, 28, 61,
                                      113, 115, 118, 108, 68, 70, 8, 188, 188, 154, 71, 18, 33, 152, 67, 150, 63, 114,
                                      110, 117, 10, 188, 188, 71, 154, 70, 33, 18, 83, 117, 28, 152, 67, 14, 114, 68,
                                      79, 64, 115, 111, 188, 118, 11, 188, 71, 154, 33, 18, 152, 67, 117, 114, 188, 14,
                                      189, 188, 189, 154, 19, 83, 70, 18, 118, 153, 69, 116, 188, 71, 188, 16, 189, 154,
                                      71, 188, 19, 101, 146, 53, 58, 146, 72, 53, 101, 54, 72, 155, 119, 102, 72, 53,
                                      146, 147, 101, 54, 58, 55, 73, 155, 72, 156, 28, 119, 103, 147, 54, 101, 72, 53,
                                      148, 146, 58, 55, 102, 56, 74, 156, 72, 155, 20, 119, 73, 157, 120, 28, 9, 148,
                                      55, 58, 0, 147, 54, 24, 72, 102, 56, 103, 57, 75, 157, 73, 156, 21, 28, 74, 158,
                                      28, 120, 104, 24, 56, 102, 1, 148, 55, 149, 147, 103, 57, 9, 76, 158, 74, 157, 22,
                                      120, 75, 159, 121, 28, 149, 57, 103, 2, 24, 56, 148, 9, 104, 159, 75, 158, 23, 28,
                                      76, 121, 9, 3, 149, 57, 24, 104, 76, 159, 24, 121, 104, 4, 149, 119, 155, 72, 105,
                                      119, 53, 146, 101, 58, 28, 72, 155, 156, 73, 119, 106, 119, 28, 54, 53, 146, 72,
                                      101, 58, 58, 105, 59, 28, 160, 120, 73, 155, 156, 157, 74, 122, 28, 119, 107, 28,
                                      58, 119, 120, 55, 54, 72, 101, 147, 53, 58, 105, 146, 102, 59, 106, 60, 72, 77,
                                      160, 28, 161, 28, 74, 156, 119, 155, 20, 157, 158, 75, 72, 123, 122, 120, 28, 108,
                                      120, 59, 105, 5, 28, 58, 28, 119, 56, 55, 147, 58, 148, 0, 54, 102, 106, 72, 103,
                                      60, 107, 61, 73, 26, 161, 28, 160, 25, 122, 77, 162, 121, 75, 157, 28, 156, 21,
                                      158, 159, 76, 73, 124, 123, 28, 120, 109, 28, 60, 106, 6, 120, 59, 121, 28, 57,
                                      56, 148, 102, 24, 1, 55, 103, 107, 147, 9, 61, 108, 62, 74, 188, 78, 162, 77, 161,
                                      26, 123, 26, 163, 76, 158, 120, 157, 22, 159, 74, 125, 124, 121, 28, 121, 61, 107,
                                      7, 28, 60, 120, 57, 24, 103, 149, 2, 56, 9, 108, 148, 104, 62, 109, 188, 75, 163,
                                      26, 162, 27, 124, 78, 159, 28, 158, 23, 75, 125, 121, 62, 108, 8, 121, 61, 28,
                                      149, 9, 3, 57, 104, 109, 24, 76, 188, 78, 163, 28, 125, 121, 159, 24, 76, 109, 9,
                                      62, 121, 104, 4, 149, 188, 122, 160, 28, 72, 155, 110, 150, 58, 119, 53, 146, 101,
                                      105, 119, 63, 188, 123, 28, 160, 161, 77, 73, 155, 72, 156, 119, 122, 111, 150,
                                      79, 59, 58, 119, 28, 105, 54, 53, 101, 146, 72, 58, 106, 28, 63, 110, 64, 188, 79,
                                      164, 124, 77, 160, 161, 162, 26, 74, 156, 155, 73, 20, 157, 28, 126, 123, 119,
                                      122, 72, 112, 79, 63, 150, 31, 60, 59, 28, 105, 120, 5, 58, 106, 55, 54, 58, 72,
                                      147, 102, 110, 119, 0, 107, 120, 64, 111, 65, 28, 188, 188, 80, 164, 79, 165, 125,
                                      26, 161, 122, 160, 25, 162, 163, 78, 75, 157, 156, 74, 21, 158, 120, 28, 26, 126,
                                      124, 28, 123, 73, 113, 31, 64, 110, 10, 79, 63, 151, 150, 61, 60, 120, 106, 28, 6,
                                      59, 107, 56, 55, 102, 147, 148, 103, 111, 28, 1, 108, 28, 65, 112, 66, 77, 188,
                                      188, 81, 165, 79, 164, 29, 126, 80, 166, 78, 162, 123, 161, 26, 163, 76, 158, 157,
                                      75, 22, 159, 28, 77, 127, 26, 125, 120, 124, 74, 151, 65, 111, 11, 31, 64, 79, 62,
                                      61, 28, 107, 121, 7, 60, 108, 57, 56, 103, 148, 24, 9, 112, 120, 2, 109, 121, 188,
                                      66, 113, 26, 188, 166, 80, 165, 30, 26, 81, 163, 124, 162, 27, 159, 158, 76, 23,
                                      121, 26, 127, 28, 125, 75, 66, 112, 12, 151, 65, 31, 62, 121, 108, 8, 61, 109, 57,
                                      9, 24, 149, 104, 113, 28, 3, 78, 188, 81, 166, 31, 127, 125, 163, 28, 159, 24, 78,
                                      121, 76, 113, 13, 66, 151, 109, 9, 62, 104, 149, 121, 4, 188, 126, 164, 79, 28,
                                      160, 155, 72, 119, 114, 152, 63, 150, 58, 119, 105, 53, 101, 146, 110, 122, 67,
                                      188, 26, 79, 164, 165, 80, 77, 160, 28, 161, 122, 156, 155, 72, 20, 119, 73, 28,
                                      126, 115, 152, 28, 64, 63, 150, 79, 110, 59, 58, 105, 119, 28, 106, 54, 0, 58, 72,
                                      5, 111, 123, 67, 114, 68, 188, 28, 167, 127, 80, 164, 165, 166, 81, 26, 161, 160,
                                      77, 25, 162, 123, 157, 156, 73, 21, 28, 74, 128, 120, 26, 122, 126, 28, 116, 28,
                                      67, 152, 153, 65, 64, 79, 110, 31, 10, 63, 111, 60, 59, 106, 28, 120, 107, 114,
                                      55, 1, 102, 147, 150, 6, 112, 124, 68, 115, 69, 79, 188, 188, 82, 167, 28, 168,
                                      81, 165, 126, 164, 29, 166, 78, 162, 161, 26, 26, 163, 124, 79, 158, 157, 74, 22,
                                      120, 75, 129, 28, 128, 127, 123, 26, 77, 153, 68, 114, 14, 28, 67, 152, 66, 65,
                                      31, 111, 151, 11, 64, 112, 61, 60, 107, 120, 28, 108, 115, 56, 2, 103, 148, 79,
                                      188, 7, 113, 125, 69, 116, 80, 188, 168, 28, 167, 32, 128, 82, 166, 26, 165, 30,
                                      163, 162, 78, 27, 125, 80, 159, 158, 75, 23, 28, 76, 121, 129, 124, 127, 26, 69,
                                      115, 15, 153, 68, 28, 66, 151, 112, 12, 65, 113, 62, 61, 108, 28, 121, 109, 116,
                                      57, 3, 9, 24, 31, 8, 81, 188, 188, 82, 168, 28, 129, 127, 166, 31, 163, 28, 81,
                                      159, 76, 24, 121, 125, 78, 116, 16, 69, 153, 113, 13, 66, 62, 109, 121, 4, 104,
                                      149, 151, 188, 9, 188, 128, 167, 28, 79, 164, 160, 28, 155, 72, 122, 119, 117, 33,
                                      67, 152, 63, 150, 110, 58, 5, 105, 119, 114, 126, 18, 20, 188, 188, 129, 28, 167,
                                      168, 82, 80, 164, 79, 165, 126, 161, 160, 28, 25, 122, 77, 156, 73, 123, 128, 28,
                                      118, 33, 83, 68, 67, 152, 28, 114, 64, 63, 110, 150, 79, 111, 59, 6, 106, 28, 10,
                                      115, 26, 18, 117, 70, 21, 188, 188, 83, 169, 82, 167, 168, 81, 165, 164, 80, 29,
                                      166, 26, 162, 161, 77, 26, 123, 26, 130, 157, 74, 124, 129, 126, 128, 120, 79, 83,
                                      18, 33, 69, 68, 28, 114, 153, 14, 67, 115, 65, 64, 111, 79, 31, 112, 117, 60, 7,
                                      107, 120, 152, 11, 116, 188, 127, 70, 118, 28, 22, 188, 188, 169, 83, 168, 128,
                                      167, 32, 166, 165, 81, 30, 127, 28, 163, 162, 26, 27, 124, 78, 158, 75, 125, 130,
                                      26, 129, 28, 80, 70, 117, 17, 83, 18, 33, 69, 153, 115, 15, 68, 116, 66, 65, 112,
                                      31, 151, 113, 118, 61, 8, 108, 28, 28, 12, 82, 23, 188, 188, 130, 169, 83, 28,
                                      167, 164, 79, 160, 28, 126, 122, 154, 18, 33, 67, 152, 114, 63, 10, 110, 150, 117,
                                      128, 71, 25, 188, 189, 188, 83, 169, 82, 167, 28, 168, 128, 165, 164, 79, 29, 126,
                                      80, 161, 77, 26, 130, 123, 154, 70, 18, 33, 83, 117, 68, 67, 114, 152, 28, 115,
                                      64, 11, 111, 79, 14, 118, 129, 188, 71, 26, 189, 188, 169, 168, 167, 82, 32, 129,
                                      166, 165, 80, 30, 26, 81, 162, 26, 127, 128, 130, 124, 28, 71, 154, 70, 83, 117,
                                      17, 18, 118, 69, 68, 115, 28, 153, 116, 65, 12, 112, 31, 33, 15, 189, 83, 27, 188,
                                      188, 189, 130, 169, 33, 168, 28, 83, 166, 81, 31, 127, 163, 78, 129, 125, 82, 19,
                                      71, 154, 118, 18, 70, 69, 116, 153, 66, 13, 113, 151, 83, 188, 16, 188, 28, 83,
                                      169, 167, 28, 164, 79, 128, 126, 71, 154, 18, 33, 117, 67, 14, 114, 152, 130, 188,
                                      29, 189, 188, 189, 169, 33, 168, 82, 28, 129, 166, 81, 130, 127, 83, 19, 71, 70,
                                      118, 83, 69, 16, 116, 153, 154, 18, 188, 188, 31, 169, 83, 167, 28, 130, 128, 71,
                                      154, 18, 17, 117, 33, 189, 32, 188, 189, 169, 83, 33, 130, 168, 82, 129, 71, 154,
                                      70, 18, 118, 83, 19, 188, 28, 189, 169, 83, 130, 71, 19, 154, 188, 33, 119, 155,
                                      146, 53, 101, 72, 28, 155, 156, 72, 54, 146, 101, 58, 72, 119, 73, 53, 84, 170,
                                      131, 120, 156, 72, 155, 157, 119, 147, 55, 101, 53, 146, 72, 58, 102, 73, 28, 74,
                                      54, 85, 170, 84, 171, 88, 131, 28, 157, 73, 119, 20, 156, 72, 158, 155, 28, 148,
                                      56, 58, 54, 72, 147, 0, 102, 103, 74, 120, 75, 55, 86, 171, 84, 170, 34, 131, 85,
                                      172, 40, 88, 121, 158, 74, 28, 21, 157, 73, 159, 156, 120, 24, 57, 102, 55, 147,
                                      148, 1, 103, 9, 75, 28, 76, 56, 188, 87, 172, 85, 171, 35, 88, 86, 173, 132, 40,
                                      159, 75, 120, 22, 158, 74, 157, 28, 149, 103, 56, 148, 24, 2, 9, 104, 76, 121,
                                      188, 57, 173, 86, 172, 36, 40, 87, 132, 76, 28, 23, 159, 75, 158, 121, 9, 57, 24,
                                      149, 3, 104, 188, 87, 173, 37, 132, 121, 24, 76, 159, 104, 149, 4, 188, 131, 170,
                                      84, 122, 160, 72, 155, 119, 119, 58, 146, 101, 53, 105, 28, 188, 88, 84, 170, 171,
                                      85, 131, 123, 160, 161, 73, 72, 155, 156, 119, 28, 28, 59, 119, 105, 72, 58, 53,
                                      54, 106, 101, 28, 122, 77, 58, 146, 188, 88, 174, 40, 85, 170, 171, 172, 86, 133,
                                      88, 131, 124, 161, 28, 160, 162, 74, 73, 156, 119, 157, 20, 72, 28, 122, 155, 120,
                                      120, 60, 105, 58, 119, 28, 5, 106, 147, 102, 0, 54, 55, 107, 58, 77, 123, 26, 84,
                                      59, 72, 188, 188, 89, 174, 88, 26, 132, 86, 171, 131, 170, 34, 172, 173, 87, 84,
                                      134, 133, 40, 88, 125, 162, 77, 122, 25, 161, 28, 163, 160, 75, 74, 157, 28, 158,
                                      21, 73, 120, 123, 156, 28, 28, 61, 106, 59, 28, 120, 6, 107, 148, 103, 1, 55, 56,
                                      108, 102, 26, 124, 78, 85, 60, 147, 188, 188, 90, 26, 88, 174, 38, 133, 89, 175,
                                      87, 172, 88, 171, 35, 173, 85, 135, 134, 132, 40, 163, 26, 123, 26, 162, 77, 161,
                                      76, 75, 158, 120, 159, 22, 74, 28, 124, 157, 121, 121, 62, 107, 60, 120, 28, 7,
                                      108, 24, 9, 2, 56, 57, 109, 103, 188, 78, 125, 86, 61, 148, 188, 175, 89, 26, 39,
                                      134, 90, 173, 40, 172, 36, 86, 135, 132, 78, 124, 27, 163, 26, 162, 76, 159, 28,
                                      23, 75, 121, 125, 158, 108, 61, 28, 121, 8, 109, 149, 104, 3, 57, 9, 87, 62, 24,
                                      188, 90, 175, 40, 135, 132, 173, 37, 87, 125, 28, 78, 163, 121, 24, 76, 159, 109,
                                      62, 121, 9, 4, 104, 188, 149, 133, 174, 88, 84, 170, 126, 164, 28, 160, 72, 155,
                                      119, 122, 131, 150, 63, 119, 105, 58, 110, 146, 53, 101, 79, 188, 134, 88, 174,
                                      26, 89, 85, 170, 84, 171, 131, 133, 26, 164, 165, 77, 28, 160, 161, 122, 73, 72,
                                      119, 155, 156, 28, 20, 123, 88, 79, 64, 150, 110, 28, 106, 5, 58, 59, 111, 105,
                                      72, 0, 54, 58, 79, 126, 80, 63, 119, 188, 42, 176, 135, 89, 174, 26, 175, 90, 86,
                                      171, 170, 85, 34, 172, 88, 136, 134, 131, 133, 84, 127, 165, 79, 164, 166, 26, 77,
                                      161, 122, 162, 25, 28, 123, 74, 73, 28, 156, 157, 120, 126, 160, 21, 124, 40, 31,
                                      65, 110, 63, 150, 79, 10, 111, 120, 107, 6, 59, 60, 112, 106, 147, 1, 55, 102, 80,
                                      26, 81, 88, 64, 28, 188, 188, 91, 176, 42, 177, 90, 26, 133, 174, 38, 175, 87,
                                      172, 171, 86, 35, 173, 40, 88, 137, 136, 135, 88, 134, 85, 166, 80, 126, 29, 165,
                                      79, 164, 78, 26, 162, 123, 163, 26, 77, 124, 75, 74, 120, 157, 158, 28, 26, 161,
                                      22, 125, 132, 151, 66, 111, 64, 79, 31, 11, 112, 28, 108, 7, 60, 61, 113, 107,
                                      148, 2, 56, 188, 103, 81, 127, 89, 65, 120, 188, 177, 42, 176, 41, 136, 91, 175,
                                      134, 26, 39, 173, 172, 87, 36, 132, 89, 137, 40, 135, 86, 81, 26, 30, 166, 80,
                                      165, 78, 163, 124, 27, 26, 125, 76, 75, 28, 158, 159, 121, 127, 162, 23, 112, 65,
                                      31, 151, 12, 113, 121, 109, 8, 61, 62, 108, 24, 3, 57, 9, 90, 66, 28, 188, 188,
                                      91, 177, 42, 137, 135, 175, 40, 173, 37, 90, 132, 87, 127, 31, 81, 166, 125, 28,
                                      78, 76, 121, 159, 163, 24, 113, 66, 151, 13, 9, 62, 109, 149, 4, 188, 104, 188,
                                      121, 136, 176, 42, 88, 174, 170, 84, 131, 128, 167, 79, 164, 28, 160, 122, 72, 20,
                                      119, 155, 126, 133, 152, 67, 150, 110, 63, 114, 119, 5, 58, 105, 28, 188, 188,
                                      188, 189, 137, 42, 176, 177, 91, 89, 174, 88, 26, 133, 171, 170, 84, 34, 131, 85,
                                      88, 136, 129, 167, 168, 80, 79, 164, 165, 126, 77, 28, 122, 160, 161, 123, 73, 21,
                                      28, 156, 25, 26, 134, 28, 68, 152, 114, 79, 111, 10, 63, 64, 115, 110, 28, 6, 59,
                                      106, 28, 128, 82, 67, 150, 188, 189, 188, 188, 92, 178, 91, 176, 177, 90, 26, 174,
                                      89, 38, 175, 134, 172, 171, 85, 35, 88, 86, 138, 40, 137, 133, 136, 88, 168, 28,
                                      167, 81, 80, 165, 126, 166, 29, 79, 26, 26, 77, 123, 161, 162, 124, 128, 74, 22,
                                      120, 157, 164, 26, 127, 135, 153, 69, 114, 67, 152, 28, 14, 115, 31, 112, 11, 64,
                                      65, 116, 111, 120, 7, 60, 107, 82, 188, 129, 42, 68, 79, 189, 188, 188, 188, 178,
                                      92, 177, 136, 176, 41, 175, 26, 90, 39, 135, 42, 173, 172, 86, 36, 40, 87, 132,
                                      138, 134, 137, 89, 82, 128, 32, 168, 28, 167, 81, 166, 26, 30, 80, 127, 78, 26,
                                      124, 162, 163, 125, 129, 75, 23, 28, 158, 165, 27, 115, 68, 28, 153, 15, 116, 151,
                                      113, 12, 65, 66, 112, 28, 8, 61, 108, 189, 91, 69, 31, 188, 188, 188, 189, 92,
                                      178, 43, 138, 137, 177, 42, 175, 40, 91, 173, 87, 37, 132, 135, 90, 129, 28, 82,
                                      168, 127, 31, 81, 78, 125, 163, 76, 24, 121, 159, 166, 28, 116, 69, 153, 16, 13,
                                      66, 113, 121, 9, 62, 109, 188, 188, 188, 151, 138, 178, 92, 42, 176, 174, 88, 170,
                                      84, 133, 131, 130, 169, 28, 167, 79, 164, 126, 28, 25, 122, 160, 128, 136, 33, 18,
                                      152, 114, 67, 117, 150, 10, 63, 110, 83, 34, 188, 189, 188, 188, 92, 178, 91, 176,
                                      42, 177, 136, 26, 174, 88, 38, 133, 89, 171, 85, 134, 138, 88, 169, 82, 28, 167,
                                      168, 128, 80, 79, 126, 164, 165, 26, 77, 26, 123, 161, 29, 129, 137, 83, 70, 33,
                                      117, 28, 115, 14, 67, 68, 118, 114, 79, 11, 64, 111, 83, 130, 188, 18, 35, 152,
                                      189, 188, 188, 178, 177, 176, 91, 41, 137, 175, 26, 89, 39, 134, 90, 172, 86, 135,
                                      136, 138, 40, 42, 83, 169, 82, 168, 128, 32, 28, 129, 81, 80, 26, 165, 166, 127,
                                      130, 26, 27, 124, 162, 167, 30, 117, 18, 33, 83, 17, 118, 153, 116, 15, 68, 69,
                                      115, 31, 12, 65, 112, 189, 92, 70, 36, 28, 188, 188, 188, 189, 138, 178, 43, 177,
                                      42, 92, 175, 90, 40, 135, 173, 87, 137, 132, 188, 91, 130, 33, 83, 169, 129, 28,
                                      82, 81, 127, 166, 78, 28, 125, 163, 168, 31, 118, 70, 83, 18, 16, 69, 116, 151,
                                      13, 66, 188, 113, 188, 37, 153, 92, 178, 176, 42, 174, 88, 136, 133, 83, 169, 28,
                                      167, 128, 79, 29, 126, 164, 130, 138, 154, 71, 33, 117, 18, 152, 14, 67, 114, 188,
                                      38, 189, 188, 188, 178, 92, 138, 177, 176, 42, 41, 136, 91, 26, 89, 137, 134, 83,
                                      169, 130, 82, 28, 128, 167, 168, 129, 80, 30, 26, 165, 32, 154, 83, 118, 17, 18,
                                      70, 117, 28, 15, 68, 115, 189, 71, 39, 33, 188, 188, 189, 178, 43, 177, 91, 42,
                                      137, 175, 90, 138, 135, 188, 92, 130, 33, 83, 82, 129, 168, 81, 31, 127, 166, 169,
                                      28, 71, 154, 19, 18, 70, 118, 153, 16, 69, 116, 188, 188, 40, 83, 178, 92, 176,
                                      42, 138, 136, 83, 169, 130, 28, 32, 128, 167, 189, 154, 71, 33, 17, 18, 117, 41,
                                      188, 188, 189, 178, 92, 43, 138, 177, 91, 137, 188, 83, 130, 169, 82, 28, 129,
                                      168, 33, 19, 71, 83, 18, 70, 118, 188, 42, 154, 189, 178, 92, 138, 188, 83, 33,
                                      130, 169, 154, 19, 71, 188, 43, 131, 170, 155, 72, 119, 146, 101, 53, 84, 188, 88,
                                      170, 171, 156, 73, 155, 119, 28, 84, 131, 72, 53, 101, 58, 54, 85, 72, 146, 188,
                                      93, 179, 139, 40, 171, 84, 170, 172, 131, 157, 74, 119, 72, 155, 156, 20, 28, 120,
                                      85, 88, 147, 54, 58, 102, 55, 86, 73, 0, 72, 188, 188, 94, 179, 93, 180, 28, 139,
                                      132, 172, 85, 131, 34, 171, 84, 173, 170, 88, 158, 75, 28, 73, 156, 157, 21, 120,
                                      28, 86, 40, 148, 55, 102, 103, 56, 87, 74, 1, 147, 188, 188, 95, 180, 93, 179, 44,
                                      139, 94, 181, 140, 28, 173, 86, 88, 35, 172, 85, 171, 40, 159, 76, 120, 74, 157,
                                      158, 22, 28, 121, 87, 132, 24, 56, 103, 9, 188, 57, 75, 2, 148, 188, 181, 94, 180,
                                      45, 28, 95, 140, 87, 40, 36, 173, 86, 172, 132, 28, 75, 158, 159, 23, 121, 149,
                                      57, 9, 104, 76, 3, 24, 188, 95, 181, 46, 140, 132, 37, 87, 173, 121, 76, 159, 24,
                                      104, 188, 4, 149, 139, 179, 93, 133, 174, 84, 170, 131, 160, 28, 155, 119, 72,
                                      122, 119, 105, 58, 88, 101, 53, 146, 188, 28, 93, 179, 180, 94, 139, 134, 174, 26,
                                      85, 84, 170, 171, 131, 88, 161, 77, 160, 122, 156, 28, 20, 72, 73, 123, 119, 88,
                                      133, 28, 58, 105, 106, 59, 89, 58, 54, 0, 28, 155, 72, 5, 119, 188, 28, 182, 140,
                                      94, 179, 180, 181, 95, 141, 28, 139, 135, 26, 88, 174, 175, 86, 85, 171, 131, 172,
                                      34, 84, 88, 133, 170, 40, 162, 26, 122, 28, 160, 161, 25, 123, 157, 120, 21, 73,
                                      74, 124, 28, 89, 134, 120, 59, 106, 107, 60, 90, 102, 55, 1, 93, 77, 156, 147, 6,
                                      188, 28, 188, 96, 182, 28, 183, 95, 180, 139, 179, 44, 181, 93, 142, 141, 140, 28,
                                      175, 89, 133, 38, 26, 88, 174, 87, 86, 172, 88, 173, 35, 85, 40, 134, 171, 132,
                                      163, 78, 123, 77, 161, 162, 26, 124, 158, 28, 22, 74, 75, 125, 120, 90, 135, 28,
                                      60, 107, 108, 61, 103, 56, 2, 188, 94, 26, 157, 148, 7, 120, 188, 183, 28, 182,
                                      47, 141, 96, 181, 28, 180, 45, 94, 142, 140, 90, 134, 39, 175, 89, 26, 87, 173,
                                      40, 36, 86, 132, 135, 172, 124, 26, 162, 163, 27, 125, 159, 121, 23, 75, 76, 28,
                                      121, 61, 108, 109, 62, 9, 57, 3, 95, 78, 158, 189, 24, 8, 188, 28, 188, 189, 96,
                                      183, 28, 142, 140, 181, 46, 95, 135, 40, 90, 175, 132, 37, 87, 173, 125, 78, 163,
                                      28, 24, 76, 121, 62, 109, 104, 4, 188, 188, 159, 149, 9, 121, 141, 182, 28, 93,
                                      179, 136, 176, 88, 174, 84, 170, 131, 133, 139, 164, 79, 160, 122, 28, 126, 155,
                                      20, 72, 119, 150, 110, 63, 42, 105, 58, 5, 119, 188, 188, 188, 189, 142, 28, 182,
                                      183, 96, 94, 179, 93, 180, 139, 141, 137, 176, 177, 89, 88, 174, 26, 133, 85, 84,
                                      131, 170, 171, 88, 34, 134, 28, 165, 80, 164, 126, 161, 123, 25, 28, 77, 26, 122,
                                      156, 21, 73, 28, 42, 136, 79, 63, 110, 111, 64, 91, 106, 59, 6, 79, 160, 28, 10,
                                      188, 189, 150, 188, 188, 97, 184, 96, 182, 183, 95, 180, 179, 94, 44, 181, 28,
                                      143, 142, 139, 141, 93, 177, 42, 176, 90, 89, 26, 133, 175, 38, 88, 134, 86, 85,
                                      88, 171, 172, 40, 136, 174, 35, 135, 140, 166, 81, 126, 79, 164, 165, 29, 26, 162,
                                      124, 26, 77, 26, 127, 123, 157, 22, 74, 120, 91, 137, 31, 64, 111, 112, 65, 107,
                                      60, 7, 28, 80, 188, 161, 120, 11, 189, 79, 188, 188, 188, 184, 97, 183, 141, 182,
                                      47, 181, 180, 95, 45, 140, 28, 143, 28, 142, 94, 91, 136, 41, 177, 42, 176, 90,
                                      175, 134, 39, 89, 135, 87, 86, 40, 172, 173, 132, 137, 26, 36, 26, 80, 165, 166,
                                      30, 127, 163, 125, 27, 26, 78, 124, 158, 23, 75, 28, 151, 65, 112, 113, 66, 108,
                                      61, 8, 96, 81, 162, 189, 28, 12, 188, 31, 188, 188, 189, 97, 184, 48, 143, 142,
                                      183, 28, 181, 46, 96, 140, 95, 137, 42, 91, 177, 135, 40, 90, 87, 132, 173, 175,
                                      37, 188, 127, 81, 166, 31, 28, 78, 125, 159, 24, 76, 121, 66, 113, 188, 109, 62,
                                      9, 188, 163, 121, 13, 151, 143, 184, 97, 28, 182, 179, 93, 139, 138, 178, 42, 176,
                                      88, 174, 133, 84, 34, 131, 170, 136, 141, 167, 28, 164, 126, 79, 128, 160, 25, 28,
                                      122, 152, 114, 67, 92, 110, 63, 10, 188, 150, 189, 188, 188, 97, 184, 96, 182, 28,
                                      183, 141, 180, 179, 93, 44, 139, 94, 28, 143, 178, 91, 42, 176, 177, 136, 89, 88,
                                      133, 174, 26, 134, 85, 35, 88, 171, 38, 137, 142, 168, 82, 167, 128, 165, 26, 29,
                                      79, 80, 129, 126, 161, 26, 77, 123, 92, 138, 28, 67, 114, 115, 68, 111, 64, 11,
                                      28, 164, 188, 79, 14, 189, 152, 188, 188, 184, 183, 182, 96, 47, 142, 181, 180,
                                      94, 45, 28, 95, 140, 141, 143, 28, 92, 178, 91, 177, 136, 41, 42, 137, 90, 89,
                                      134, 26, 175, 135, 138, 86, 36, 40, 172, 176, 39, 128, 28, 167, 168, 32, 129, 166,
                                      127, 30, 80, 81, 26, 162, 27, 26, 124, 153, 68, 115, 116, 69, 112, 65, 12, 97, 82,
                                      165, 188, 189, 31, 15, 188, 188, 28, 189, 143, 184, 48, 183, 28, 97, 181, 95, 46,
                                      140, 142, 96, 138, 43, 92, 178, 137, 42, 91, 90, 135, 175, 87, 37, 132, 173, 177,
                                      188, 40, 129, 82, 168, 28, 31, 81, 127, 163, 28, 78, 125, 69, 116, 113, 66, 13,
                                      188, 166, 188, 151, 16, 153, 97, 184, 182, 28, 179, 93, 141, 139, 92, 178, 42,
                                      176, 136, 88, 38, 133, 174, 138, 143, 169, 83, 167, 128, 28, 130, 164, 29, 79,
                                      126, 33, 117, 18, 114, 67, 14, 44, 188, 152, 189, 188, 188, 184, 97, 143, 183,
                                      182, 28, 47, 141, 96, 180, 94, 142, 28, 92, 178, 138, 91, 42, 136, 176, 177, 137,
                                      89, 39, 134, 26, 41, 169, 130, 168, 129, 32, 28, 82, 128, 165, 30, 80, 26, 189,
                                      83, 18, 117, 118, 70, 115, 68, 15, 83, 45, 167, 28, 17, 188, 188, 33, 189, 184,
                                      48, 183, 96, 28, 142, 181, 95, 143, 140, 97, 138, 43, 92, 91, 137, 177, 90, 40,
                                      135, 175, 178, 42, 188, 130, 83, 169, 33, 28, 82, 129, 166, 31, 81, 127, 70, 118,
                                      116, 69, 16, 46, 188, 168, 188, 153, 18, 83, 184, 97, 182, 28, 143, 141, 92, 178,
                                      138, 42, 41, 136, 176, 189, 169, 130, 83, 167, 32, 28, 128, 154, 71, 117, 18, 17,
                                      47, 188, 33, 188, 189, 184, 97, 48, 143, 183, 96, 142, 92, 138, 178, 91, 42, 137,
                                      177, 43, 188, 33, 83, 130, 168, 28, 82, 129, 71, 118, 70, 18, 28, 169, 188, 83,
                                      19, 154, 189, 184, 97, 143, 92, 43, 138, 178, 188, 169, 33, 83, 130, 71, 19, 48,
                                      188, 154, 139, 179, 170, 84, 131, 155, 119, 72, 93, 101, 53, 146, 188, 28, 179,
                                      180, 171, 85, 170, 131, 88, 93, 139, 156, 72, 119, 28, 73, 94, 84, 58, 54, 20, 0,
                                      155, 72, 188, 50, 185, 51, 140, 180, 93, 179, 181, 139, 172, 86, 131, 84, 170,
                                      171, 34, 88, 40, 94, 28, 157, 73, 28, 120, 74, 95, 85, 102, 55, 21, 1, 188, 156,
                                      147, 188, 98, 185, 50, 186, 99, 51, 181, 94, 139, 44, 180, 93, 179, 28, 173, 87,
                                      88, 85, 171, 172, 35, 40, 132, 95, 140, 158, 74, 120, 28, 75, 86, 103, 56, 188,
                                      22, 2, 157, 148, 188, 186, 50, 185, 49, 51, 98, 99, 95, 28, 45, 181, 94, 180, 140,
                                      40, 86, 172, 173, 36, 132, 159, 75, 28, 121, 76, 87, 189, 9, 57, 23, 3, 188, 158,
                                      24, 188, 189, 98, 186, 50, 99, 140, 46, 95, 181, 132, 87, 173, 37, 188, 76, 121,
                                      104, 188, 24, 4, 159, 149, 51, 185, 50, 141, 182, 93, 179, 139, 174, 88, 170, 131,
                                      84, 133, 160, 122, 28, 28, 119, 72, 20, 105, 58, 155, 5, 119, 188, 188, 189, 99,
                                      50, 185, 186, 98, 51, 142, 182, 183, 94, 93, 179, 180, 139, 28, 26, 89, 174, 133,
                                      171, 88, 34, 84, 85, 134, 131, 28, 141, 161, 28, 122, 123, 77, 96, 28, 73, 21, 88,
                                      170, 106, 59, 156, 25, 6, 188, 189, 160, 28, 188, 99, 187, 98, 185, 186, 144, 99,
                                      51, 183, 28, 182, 95, 94, 180, 139, 181, 44, 93, 28, 141, 179, 140, 175, 90, 133,
                                      88, 174, 26, 38, 134, 172, 40, 35, 85, 86, 135, 88, 96, 142, 162, 77, 123, 124,
                                      26, 120, 74, 22, 50, 89, 171, 107, 60, 157, 26, 7, 188, 189, 161, 120, 188, 188,
                                      187, 99, 186, 51, 185, 49, 50, 144, 99, 96, 141, 47, 183, 28, 182, 95, 181, 28,
                                      45, 94, 140, 142, 180, 134, 89, 26, 175, 39, 135, 173, 132, 36, 86, 87, 40, 163,
                                      26, 124, 125, 78, 28, 75, 23, 98, 90, 172, 189, 108, 61, 158, 27, 8, 188, 162, 28,
                                      188, 189, 99, 187, 51, 144, 99, 186, 50, 98, 142, 28, 96, 183, 140, 46, 95, 181,
                                      135, 90, 175, 40, 37, 87, 132, 188, 78, 125, 121, 76, 24, 173, 109, 62, 159, 28,
                                      9, 188, 163, 121, 144, 187, 99, 50, 185, 143, 184, 28, 182, 93, 179, 139, 141, 51,
                                      176, 42, 174, 133, 88, 136, 170, 34, 84, 131, 164, 126, 79, 97, 122, 28, 25, 110,
                                      63, 160, 10, 188, 189, 150, 188, 188, 99, 187, 98, 185, 50, 186, 51, 144, 184, 96,
                                      28, 182, 183, 141, 94, 93, 139, 179, 180, 28, 44, 142, 99, 177, 91, 176, 136, 26,
                                      134, 38, 88, 89, 137, 133, 171, 35, 85, 88, 97, 143, 165, 79, 126, 26, 80, 123,
                                      77, 26, 42, 174, 111, 64, 161, 29, 11, 188, 189, 164, 79, 188, 188, 187, 186, 185,
                                      98, 49, 99, 51, 144, 50, 97, 184, 96, 183, 141, 47, 28, 142, 95, 94, 28, 180, 181,
                                      140, 143, 182, 45, 136, 42, 176, 177, 41, 137, 175, 135, 39, 89, 90, 134, 172, 36,
                                      86, 40, 166, 80, 26, 127, 81, 124, 26, 27, 99, 91, 26, 189, 112, 65, 162, 30, 12,
                                      188, 188, 188, 165, 31, 189, 144, 187, 51, 186, 50, 99, 99, 98, 143, 48, 97, 184,
                                      142, 28, 96, 95, 140, 181, 183, 46, 137, 91, 177, 42, 40, 90, 135, 173, 37, 87,
                                      188, 132, 81, 127, 125, 78, 28, 175, 113, 66, 163, 188, 31, 13, 188, 166, 151, 99,
                                      187, 185, 50, 51, 97, 184, 28, 182, 141, 93, 44, 139, 179, 143, 144, 178, 92, 176,
                                      136, 42, 138, 174, 38, 88, 133, 167, 128, 28, 126, 79, 29, 114, 67, 164, 14, 188,
                                      189, 152, 188, 188, 187, 99, 144, 186, 185, 50, 49, 51, 98, 99, 97, 184, 143, 96,
                                      28, 141, 182, 183, 142, 94, 45, 28, 180, 47, 178, 138, 177, 137, 41, 42, 91, 136,
                                      26, 39, 89, 134, 189, 168, 28, 128, 129, 82, 26, 80, 30, 92, 176, 115, 68, 165,
                                      32, 15, 188, 188, 167, 28, 189, 187, 51, 186, 98, 50, 99, 144, 99, 143, 48, 97,
                                      96, 142, 183, 95, 46, 140, 181, 184, 28, 138, 92, 178, 43, 42, 91, 137, 175, 40,
                                      90, 135, 188, 82, 129, 127, 81, 31, 177, 116, 69, 166, 28, 16, 188, 188, 168, 153,
                                      187, 99, 185, 50, 144, 51, 97, 184, 143, 28, 47, 141, 182, 189, 178, 138, 92, 176,
                                      41, 42, 136, 169, 130, 83, 128, 28, 32, 49, 117, 18, 167, 188, 17, 188, 33, 189,
                                      187, 99, 51, 144, 186, 98, 99, 97, 143, 184, 96, 28, 142, 183, 48, 43, 92, 138,
                                      177, 42, 91, 137, 188, 83, 130, 129, 82, 28, 50, 178, 118, 70, 168, 33, 18, 188,
                                      169, 83, 189, 187, 99, 144, 97, 48, 143, 184, 178, 43, 92, 138, 188, 130, 83, 33,
                                      51, 71, 169, 19, 188, 154, 51, 185, 179, 93, 139, 170, 131, 84, 50, 119, 72, 20,
                                      155, 188, 189, 188, 99, 185, 186, 180, 94, 179, 139, 28, 50, 51, 171, 84, 131, 88,
                                      85, 98, 93, 28, 73, 34, 21, 188, 189, 170, 156, 188, 100, 145, 186, 50, 185, 51,
                                      181, 95, 139, 93, 179, 180, 44, 28, 140, 98, 99, 172, 85, 88, 40, 86, 94, 120, 74,
                                      35, 22, 189, 171, 157, 188, 188, 188, 100, 145, 98, 51, 49, 186, 50, 185, 99, 28,
                                      94, 180, 181, 45, 140, 173, 86, 40, 132, 87, 95, 189, 28, 75, 36, 23, 188, 172,
                                      158, 188, 189, 100, 52, 145, 99, 50, 98, 186, 140, 95, 181, 46, 87, 132, 188, 121,
                                      76, 37, 24, 173, 159, 188, 145, 100, 144, 187, 50, 185, 51, 182, 28, 179, 139, 93,
                                      141, 174, 133, 88, 99, 131, 84, 34, 122, 28, 170, 25, 188, 189, 160, 188, 100,
                                      145, 187, 98, 50, 185, 186, 51, 99, 183, 96, 182, 141, 180, 28, 44, 93, 94, 142,
                                      139, 99, 144, 26, 88, 133, 134, 89, 88, 85, 35, 28, 179, 123, 77, 171, 38, 26,
                                      189, 174, 161, 188, 188, 145, 99, 187, 98, 186, 51, 49, 50, 99, 144, 185, 141, 28,
                                      182, 183, 47, 142, 181, 140, 45, 94, 95, 28, 175, 89, 134, 135, 90, 40, 86, 36,
                                      100, 96, 180, 189, 124, 26, 172, 39, 27, 188, 188, 26, 162, 189, 145, 52, 100,
                                      144, 51, 99, 187, 99, 50, 98, 186, 142, 96, 183, 28, 46, 95, 140, 90, 135, 132,
                                      87, 37, 188, 181, 125, 78, 173, 40, 28, 175, 163, 188, 100, 99, 187, 50, 185, 51,
                                      144, 145, 184, 97, 182, 141, 28, 143, 179, 44, 93, 139, 176, 136, 42, 133, 88, 38,
                                      126, 79, 174, 29, 189, 164, 188, 188, 188, 100, 145, 99, 187, 144, 98, 50, 51,
                                      185, 186, 99, 49, 184, 143, 183, 142, 47, 28, 96, 141, 180, 45, 94, 28, 189, 177,
                                      42, 136, 137, 91, 134, 89, 39, 97, 182, 26, 80, 26, 41, 30, 188, 188, 176, 165,
                                      189, 52, 145, 100, 144, 51, 99, 98, 99, 186, 187, 50, 143, 97, 184, 48, 28, 96,
                                      142, 181, 46, 95, 140, 91, 137, 135, 90, 40, 188, 183, 127, 81, 175, 42, 31, 177,
                                      166, 188, 188, 100, 145, 189, 99, 187, 144, 50, 49, 51, 185, 184, 143, 97, 182,
                                      47, 28, 141, 178, 138, 92, 136, 42, 41, 188, 128, 28, 176, 32, 188, 167, 189, 100,
                                      52, 145, 99, 144, 187, 98, 50, 99, 186, 51, 48, 97, 143, 183, 28, 96, 142, 92,
                                      138, 137, 91, 42, 184, 188, 129, 82, 177, 43, 28, 178, 168, 188, 189, 100, 145,
                                      99, 51, 144, 187, 184, 48, 97, 143, 138, 92, 43, 52, 188, 130, 83, 178, 33, 169,
                                      188, 145, 185, 50, 51, 179, 139, 93, 100, 131, 84, 34, 188, 189, 170, 188, 186,
                                      98, 185, 51, 99, 100, 145, 180, 93, 139, 28, 94, 50, 88, 85, 44, 35, 189, 188,
                                      179, 171, 188, 100, 145, 51, 50, 185, 186, 49, 99, 189, 181, 94, 28, 140, 95, 98,
                                      40, 86, 45, 36, 188, 188, 180, 172, 189, 145, 52, 100, 99, 98, 186, 50, 95, 140,
                                      132, 87, 188, 46, 37, 188, 181, 173, 100, 145, 187, 99, 185, 51, 50, 144, 182,
                                      141, 28, 139, 93, 44, 133, 88, 179, 38, 189, 188, 174, 188, 100, 145, 187, 144,
                                      186, 99, 49, 50, 98, 51, 189, 183, 28, 141, 142, 96, 28, 94, 45, 99, 185, 134, 89,
                                      180, 47, 39, 188, 182, 26, 189, 145, 52, 100, 144, 99, 187, 51, 50, 98, 99, 96,
                                      142, 140, 95, 46, 186, 135, 90, 181, 28, 40, 188, 188, 183, 175, 189, 100, 145,
                                      187, 144, 99, 185, 49, 50, 51, 184, 143, 97, 141, 28, 47, 136, 42, 182, 41, 188,
                                      188, 176, 189, 100, 145, 52, 51, 99, 144, 186, 50, 98, 99, 97, 143, 142, 96, 28,
                                      187, 137, 91, 183, 48, 42, 188, 188, 184, 177, 189, 100, 52, 145, 187, 51, 99,
                                      144, 143, 97, 48, 138, 92, 184, 43, 188, 188, 178, 100, 145, 185, 51, 50, 139, 93,
                                      44, 189, 188, 179, 188, 189, 145, 186, 50, 51, 99, 98, 100, 28, 94, 49, 45, 188,
                                      185, 180, 189, 145, 100, 52, 98, 99, 140, 95, 50, 46, 188, 186, 181, 188, 189,
                                      145, 100, 187, 144, 99, 51, 50, 49, 141, 28, 185, 47, 188, 182, 189, 52, 100, 145,
                                      99, 144, 99, 98, 50, 142, 96, 186, 51, 28, 187, 183, 188, 189, 52, 100, 145, 144,
                                      99, 51, 143, 97, 187, 48, 188, 184, 188, 146, 101, 53, 72, 101, 146, 53, 58, 54,
                                      147, 58, 53, 72, 146, 54, 102, 101, 55, 148, 102, 54, 0, 147, 72, 55, 103, 58, 56,
                                      24, 103, 55, 1, 148, 147, 56, 9, 102, 188, 57, 149, 9, 56, 2, 24, 188, 148, 57,
                                      104, 103, 104, 57, 3, 149, 24, 9, 188, 4, 188, 149, 104, 119, 146, 101, 53, 105,
                                      58, 188, 28, 105, 119, 58, 72, 58, 54, 146, 106, 101, 59, 53, 188, 120, 106, 58,
                                      5, 28, 119, 59, 147, 102, 0, 55, 72, 107, 58, 105, 188, 60, 54, 188, 28, 107, 59,
                                      6, 120, 28, 60, 148, 103, 1, 56, 147, 108, 102, 106, 188, 61, 55, 188, 121, 108,
                                      60, 7, 28, 120, 61, 24, 9, 2, 57, 188, 148, 109, 103, 107, 188, 62, 56, 109, 61,
                                      8, 121, 28, 62, 149, 104, 3, 24, 9, 108, 188, 57, 62, 9, 121, 4, 188, 149, 104,
                                      109, 150, 119, 105, 58, 110, 53, 146, 63, 188, 101, 79, 110, 150, 63, 28, 106, 5,
                                      59, 119, 111, 54, 0, 105, 72, 188, 64, 58, 58, 31, 111, 63, 10, 79, 150, 64, 120,
                                      107, 6, 60, 28, 112, 55, 1, 106, 110, 188, 147, 65, 188, 102, 59, 151, 112, 64,
                                      11, 31, 79, 65, 28, 108, 7, 61, 120, 113, 56, 2, 107, 111, 148, 188, 188, 66, 103,
                                      60, 113, 65, 12, 151, 31, 66, 121, 109, 8, 62, 28, 57, 3, 108, 112, 188, 188, 24,
                                      9, 61, 66, 13, 188, 151, 9, 121, 4, 109, 113, 149, 188, 104, 62, 152, 150, 110,
                                      63, 114, 58, 5, 188, 119, 67, 105, 188, 189, 28, 114, 152, 67, 79, 111, 10, 64,
                                      150, 115, 59, 6, 110, 188, 28, 68, 106, 63, 188, 189, 153, 115, 67, 14, 28, 152,
                                      68, 31, 112, 11, 65, 79, 116, 60, 7, 111, 114, 188, 120, 188, 188, 189, 69, 107,
                                      64, 116, 68, 15, 153, 28, 69, 151, 113, 12, 66, 31, 61, 8, 112, 115, 188, 188,
                                      189, 28, 108, 65, 189, 69, 16, 153, 13, 188, 151, 62, 9, 113, 116, 121, 188, 109,
                                      66, 33, 152, 114, 67, 117, 63, 10, 188, 150, 18, 188, 110, 189, 83, 117, 33, 18,
                                      28, 115, 14, 68, 152, 118, 64, 11, 114, 188, 79, 188, 189, 70, 111, 67, 118, 18,
                                      17, 83, 33, 70, 153, 116, 15, 69, 28, 65, 12, 115, 117, 188, 188, 189, 31, 112,
                                      68, 189, 70, 18, 83, 16, 153, 66, 13, 116, 118, 188, 151, 188, 113, 69, 154, 33,
                                      117, 18, 67, 14, 188, 188, 152, 189, 71, 114, 189, 154, 71, 83, 118, 17, 70, 33,
                                      68, 15, 117, 188, 28, 115, 18, 189, 71, 19, 154, 18, 83, 69, 16, 118, 188, 153,
                                      188, 116, 70, 189, 154, 71, 18, 17, 188, 33, 117, 189, 19, 154, 70, 18, 188, 83,
                                      118, 71, 189, 71, 19, 188, 154, 155, 119, 53, 146, 72, 101, 188, 156, 119, 155,
                                      72, 28, 54, 72, 101, 53, 146, 73, 188, 58, 157, 28, 72, 20, 156, 155, 73, 120,
                                      119, 55, 147, 0, 58, 54, 72, 188, 74, 102, 188, 158, 120, 73, 21, 157, 156, 74,
                                      28, 28, 56, 148, 1, 188, 102, 55, 147, 75, 188, 103, 159, 28, 74, 22, 158, 157,
                                      75, 121, 120, 57, 24, 2, 103, 56, 148, 188, 76, 9, 188, 121, 75, 23, 159, 158, 76,
                                      28, 149, 3, 188, 9, 57, 24, 104, 76, 24, 159, 121, 4, 104, 149, 188, 160, 155,
                                      119, 72, 122, 58, 119, 53, 146, 101, 188, 28, 105, 161, 122, 160, 28, 156, 28, 20,
                                      73, 155, 123, 119, 59, 28, 5, 105, 54, 72, 0, 188, 58, 119, 58, 77, 72, 106, 162,
                                      123, 28, 25, 161, 160, 77, 157, 120, 21, 74, 156, 124, 28, 122, 60, 120, 6, 106,
                                      55, 147, 188, 1, 59, 28, 102, 188, 26, 73, 107, 163, 124, 77, 26, 162, 161, 26,
                                      158, 28, 22, 75, 157, 125, 120, 123, 61, 28, 7, 107, 56, 148, 2, 188, 60, 120,
                                      103, 188, 78, 74, 108, 125, 26, 27, 163, 162, 78, 159, 121, 23, 76, 158, 28, 124,
                                      62, 121, 8, 108, 57, 24, 188, 3, 189, 61, 28, 9, 75, 109, 189, 78, 28, 163, 24,
                                      159, 121, 125, 9, 109, 149, 4, 62, 121, 104, 188, 76, 164, 160, 122, 28, 126, 72,
                                      20, 155, 63, 150, 58, 119, 5, 188, 105, 79, 119, 110, 188, 189, 165, 126, 164, 79,
                                      161, 123, 25, 77, 160, 26, 73, 21, 122, 156, 64, 79, 10, 110, 59, 28, 6, 188, 63,
                                      150, 106, 80, 28, 28, 188, 111, 189, 166, 26, 79, 29, 165, 164, 80, 162, 124, 26,
                                      26, 161, 127, 74, 22, 123, 126, 157, 65, 31, 11, 111, 60, 120, 7, 188, 188, 64,
                                      79, 107, 188, 189, 81, 120, 77, 112, 127, 80, 30, 166, 165, 81, 163, 125, 27, 78,
                                      162, 75, 23, 124, 26, 158, 66, 151, 12, 188, 112, 61, 28, 8, 188, 189, 65, 31,
                                      108, 28, 26, 113, 189, 81, 31, 166, 28, 163, 76, 24, 125, 127, 159, 13, 113, 62,
                                      121, 9, 188, 66, 151, 109, 121, 78, 188, 167, 164, 126, 79, 128, 28, 25, 160, 67,
                                      152, 63, 150, 10, 188, 110, 188, 28, 122, 189, 114, 168, 128, 167, 28, 165, 26,
                                      29, 80, 164, 129, 77, 26, 126, 161, 68, 28, 14, 114, 64, 79, 11, 188, 188, 67,
                                      152, 189, 111, 82, 123, 79, 115, 129, 28, 32, 168, 167, 82, 166, 127, 30, 81, 165,
                                      26, 27, 26, 128, 189, 162, 69, 153, 15, 115, 65, 31, 188, 12, 188, 68, 28, 112,
                                      124, 80, 116, 189, 82, 28, 168, 31, 166, 78, 28, 127, 129, 163, 16, 116, 66, 151,
                                      13, 188, 69, 153, 113, 188, 125, 81, 169, 167, 128, 28, 130, 79, 29, 164, 18, 33,
                                      67, 152, 14, 188, 188, 189, 114, 83, 126, 117, 189, 130, 169, 83, 168, 129, 32,
                                      82, 167, 80, 30, 128, 165, 70, 83, 17, 117, 68, 28, 15, 188, 18, 33, 115, 26, 28,
                                      118, 189, 83, 33, 169, 28, 168, 81, 31, 129, 130, 166, 18, 118, 69, 153, 16, 188,
                                      188, 70, 83, 116, 127, 82, 189, 169, 130, 83, 28, 32, 167, 71, 154, 18, 33, 17,
                                      188, 117, 128, 189, 33, 169, 82, 28, 130, 168, 19, 70, 83, 18, 188, 71, 154, 118,
                                      129, 83, 189, 83, 33, 169, 71, 154, 19, 188, 130, 170, 131, 72, 155, 101, 146,
                                      188, 84, 53, 119, 171, 131, 170, 84, 88, 73, 156, 20, 119, 72, 58, 0, 72, 188,
                                      155, 85, 54, 28, 172, 88, 84, 34, 171, 170, 85, 40, 131, 74, 157, 21, 28, 73, 102,
                                      1, 188, 147, 156, 188, 86, 55, 120, 173, 40, 85, 35, 172, 171, 86, 132, 88, 75,
                                      158, 22, 120, 74, 103, 2, 148, 188, 157, 188, 87, 56, 28, 132, 86, 36, 173, 172,
                                      87, 40, 188, 76, 159, 23, 28, 189, 75, 9, 3, 188, 24, 158, 57, 121, 189, 87, 37,
                                      188, 173, 132, 24, 121, 76, 104, 4, 149, 159, 188, 174, 170, 131, 84, 133, 28,
                                      160, 72, 155, 20, 105, 5, 119, 188, 119, 88, 58, 122, 188, 189, 26, 133, 174, 88,
                                      171, 88, 34, 85, 170, 134, 131, 77, 161, 25, 122, 73, 156, 21, 28, 106, 6, 28,
                                      188, 160, 28, 89, 84, 59, 123, 188, 189, 175, 134, 88, 38, 26, 174, 89, 172, 40,
                                      35, 86, 171, 135, 88, 133, 26, 162, 26, 123, 74, 157, 22, 77, 107, 7, 120, 188,
                                      188, 161, 120, 188, 189, 90, 85, 60, 124, 135, 89, 39, 175, 26, 90, 173, 132, 36,
                                      87, 172, 40, 134, 188, 78, 163, 27, 124, 75, 158, 23, 189, 26, 108, 8, 28, 188,
                                      162, 28, 86, 61, 125, 189, 90, 40, 175, 37, 188, 173, 132, 135, 28, 125, 76, 159,
                                      24, 78, 109, 9, 121, 188, 163, 121, 87, 62, 176, 174, 133, 88, 136, 84, 34, 170,
                                      79, 164, 28, 160, 25, 110, 10, 150, 188, 122, 188, 42, 188, 189, 131, 63, 126,
                                      177, 136, 176, 42, 26, 134, 38, 89, 174, 137, 85, 35, 133, 171, 80, 165, 29, 126,
                                      77, 161, 26, 79, 111, 11, 79, 188, 188, 188, 189, 164, 123, 91, 88, 88, 64, 26,
                                      137, 42, 41, 177, 176, 91, 175, 135, 39, 90, 26, 86, 36, 134, 136, 188, 189, 172,
                                      81, 166, 30, 26, 26, 162, 27, 80, 112, 12, 188, 31, 188, 165, 124, 40, 89, 65,
                                      127, 189, 91, 42, 177, 40, 175, 87, 37, 135, 137, 173, 31, 127, 78, 163, 28, 81,
                                      113, 13, 151, 188, 166, 125, 188, 188, 132, 90, 66, 178, 176, 136, 42, 138, 88,
                                      38, 188, 174, 28, 167, 79, 164, 29, 114, 14, 152, 188, 189, 188, 126, 92, 133, 67,
                                      128, 189, 138, 178, 92, 177, 137, 41, 91, 176, 89, 39, 136, 188, 26, 82, 168, 32,
                                      128, 80, 165, 30, 28, 115, 15, 28, 188, 167, 26, 134, 42, 68, 129, 189, 92, 43,
                                      178, 42, 177, 90, 40, 137, 138, 175, 28, 129, 188, 81, 166, 31, 82, 116, 16, 153,
                                      188, 168, 188, 127, 135, 91, 69, 189, 178, 138, 92, 42, 41, 188, 176, 83, 169, 28,
                                      167, 32, 117, 17, 33, 188, 128, 136, 18, 130, 189, 43, 178, 91, 42, 138, 188, 177,
                                      33, 130, 82, 168, 28, 83, 118, 18, 83, 188, 169, 129, 137, 92, 70, 189, 92, 43,
                                      188, 178, 83, 169, 33, 19, 154, 188, 130, 138, 71, 179, 139, 84, 170, 119, 20,
                                      155, 188, 93, 72, 131, 188, 189, 180, 139, 179, 93, 28, 85, 171, 34, 131, 84, 28,
                                      21, 156, 170, 188, 94, 73, 188, 189, 88, 181, 28, 93, 44, 180, 179, 94, 140, 139,
                                      86, 172, 35, 88, 85, 120, 22, 157, 171, 188, 188, 189, 95, 74, 40, 188, 140, 94,
                                      45, 181, 180, 95, 28, 87, 173, 36, 188, 40, 189, 86, 28, 23, 158, 172, 188, 75,
                                      132, 189, 95, 46, 181, 140, 37, 132, 87, 121, 24, 159, 173, 76, 188, 188, 182,
                                      179, 139, 93, 141, 88, 174, 84, 170, 34, 122, 25, 160, 131, 188, 188, 189, 28, 28,
                                      133, 183, 141, 182, 28, 180, 28, 44, 94, 179, 142, 139, 89, 26, 38, 133, 85, 171,
                                      35, 188, 88, 123, 26, 161, 174, 88, 189, 188, 96, 93, 77, 134, 142, 28, 47, 183,
                                      182, 96, 181, 140, 45, 95, 180, 28, 141, 189, 90, 175, 39, 134, 86, 172, 188, 36,
                                      89, 124, 27, 162, 26, 40, 188, 94, 26, 135, 189, 96, 28, 183, 46, 181, 140, 142,
                                      40, 135, 87, 173, 37, 90, 125, 28, 163, 175, 132, 188, 188, 95, 78, 184, 182, 141,
                                      28, 143, 93, 44, 179, 42, 176, 88, 174, 38, 188, 126, 29, 164, 189, 188, 133, 188,
                                      97, 139, 79, 136, 189, 143, 184, 97, 183, 142, 47, 96, 182, 94, 45, 141, 180, 91,
                                      177, 41, 136, 89, 26, 39, 188, 42, 26, 30, 165, 176, 188, 134, 28, 28, 80, 137,
                                      189, 97, 48, 184, 28, 183, 95, 46, 142, 143, 181, 42, 137, 90, 175, 40, 188, 91,
                                      127, 31, 166, 177, 135, 188, 140, 96, 81, 188, 189, 184, 143, 97, 28, 47, 182, 92,
                                      178, 42, 176, 41, 188, 128, 32, 167, 136, 188, 141, 28, 138, 189, 48, 184, 96, 28,
                                      143, 183, 43, 138, 91, 177, 42, 188, 92, 129, 28, 168, 178, 137, 188, 142, 97, 82,
                                      189, 97, 48, 184, 92, 178, 43, 188, 130, 33, 169, 138, 188, 143, 83, 185, 51, 93,
                                      179, 131, 34, 170, 189, 188, 188, 50, 84, 139, 186, 51, 185, 50, 99, 94, 180, 44,
                                      139, 93, 88, 35, 171, 188, 189, 179, 188, 98, 85, 28, 187, 185, 51, 50, 144, 28,
                                      182, 93, 179, 44, 189, 133, 38, 174, 188, 139, 188, 99, 88, 141, 145, 50, 185,
                                      189, 139, 44, 179, 188, 188, 100, 93, 51, 101, 188, 53, 146, 58, 188, 0, 54, 72,
                                      102, 188, 1, 188, 55, 147, 103, 188, 2, 188, 56, 148, 189, 9, 188, 3, 57, 24, 189,
                                      104, 4, 188, 149, 105, 5, 188, 58, 119, 189, 106, 6, 188, 59, 189, 28, 189, 107,
                                      7, 188, 188, 60, 120, 189, 108, 8, 188, 61, 28, 189, 109, 9, 188, 62, 121, 110,
                                      10, 188, 188, 189, 63, 150, 189, 111, 188, 11, 188, 64, 79, 189, 112, 188, 12,
                                      188, 65, 31, 113, 13, 188, 188, 189, 66, 151, 189, 114, 14, 188, 188, 67, 152,
                                      189, 115, 15, 188, 68, 28, 189, 116, 16, 188, 188, 69, 153, 189, 117, 17, 188, 18,
                                      33, 189, 118, 18, 188, 70, 83, 189, 19, 188, 71, 154, 119, 20, 188, 72, 189, 155,
                                      28, 21, 188, 73, 189, 156, 189, 120, 22, 188, 188, 74, 157, 189, 28, 23, 188, 75,
                                      158, 121, 24, 188, 76, 189, 159, 122, 25, 189, 188, 28, 160, 189, 123, 26, 188,
                                      77, 161, 189, 124, 27, 188, 26, 162, 125, 28, 189, 188, 78, 163, 189, 126, 29,
                                      188, 79, 188, 164, 189, 26, 30, 188, 80, 165, 189, 127, 31, 188, 81, 188, 166,
                                      189, 128, 32, 188, 28, 167, 189, 129, 28, 188, 82, 168, 189, 130, 33, 83, 188,
                                      169, 189, 131, 34, 188, 84, 188, 170, 189, 88, 188, 35, 85, 188, 171, 189, 133,
                                      38, 188, 88, 188, 174, 189, 139, 44, 188, 93, 188, 179, 189, 188, 189, 188, 189,
                                      188, 189, 188, 189, 99, 50, 49, 186, 185, 98, 51, 95, 181, 45, 28, 94, 40, 36,
                                      188, 172, 180, 188, 86, 140, 189, 144, 187, 99, 186, 99, 49, 98, 185, 51, 96, 183,
                                      47, 141, 94, 180, 45, 28, 134, 39, 26, 188, 182, 28, 50, 89, 142, 189, 145, 100,
                                      98, 186, 49, 51, 50, 28, 45, 180, 185, 188, 94, 99, 189, 187, 144, 99, 50, 49,
                                      185, 97, 184, 28, 182, 47, 136, 41, 176, 188, 141, 188, 51, 42, 143, 189, 145,
                                      100, 99, 187, 50, 185, 49, 141, 47, 182, 51, 188, 28, 144, 189, 145, 100, 51, 50,
                                      49, 188, 185, 189, 98, 50, 186, 99, 46, 140, 95, 132, 37, 173, 181, 188, 188, 87,
                                      189, 99, 51, 187, 50, 186, 99, 144, 28, 142, 95, 181, 46, 96, 135, 40, 175, 188,
                                      183, 140, 188, 98, 90, 100, 52, 145, 50, 99, 98, 140, 46, 181, 189, 186, 188, 95,
                                      188, 189, 51, 187, 98, 50, 144, 186, 48, 143, 96, 183, 28, 97, 137, 42, 177, 188,
                                      184, 142, 188, 99, 99, 91, 52, 145, 51, 144, 98, 186, 50, 189, 99, 142, 28, 183,
                                      187, 99, 188, 100, 96, 189, 100, 145, 99, 98, 52, 50, 188, 186, 189, 99, 51, 187,
                                      97, 184, 48, 138, 43, 178, 188, 143, 188, 144, 92, 100, 52, 99, 187, 51, 143, 48,
                                      184, 189, 144, 188, 188, 145, 97, 189, 145, 100, 52, 144, 99, 51, 188, 187, 189,
                                      145, 100, 52, 188};
    static const int coeffs1_ind[] = {189, 189, 188, 189, 40, 188, 36, 86, 188, 172, 189, 189, 134, 39, 188, 89, 26,
                                      189, 189, 28, 45, 188, 94, 180, 189, 188, 189, 136, 41, 188, 42, 188, 176, 189,
                                      189, 141, 47, 188, 28, 182, 189, 188, 189, 51, 49, 50, 188, 185, 189, 100, 51, 49,
                                      185, 188, 50, 145, 189, 189, 188, 189, 132, 37, 188, 87, 188, 173, 189, 188, 189,
                                      135, 40, 188, 90, 188, 175, 189, 188, 140, 46, 188, 95, 188, 189, 181, 189, 188,
                                      189, 137, 42, 188, 91, 188, 177, 189, 189, 142, 28, 188, 96, 183, 189, 188, 189,
                                      99, 50, 98, 188, 186, 189, 52, 145, 100, 99, 50, 186, 188, 98, 189, 189, 188, 189,
                                      138, 43, 188, 92, 188, 178, 189, 188, 189, 143, 48, 97, 188, 188, 184, 189, 188,
                                      189, 144, 51, 99, 188, 187, 189, 100, 52, 144, 51, 187, 145, 188, 99, 189, 189,
                                      188, 145, 52, 189, 100, 188, 189, 145, 52, 188, 100};

    static const int C0_ind[] = {0, 6, 630, 631, 635, 636, 678, 1260, 1261, 1262, 1264, 1265, 1266, 1267, 1308, 1890,
                                 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1938, 2520, 2521, 2522, 2523, 2524, 2525,
                                 2526, 2527, 2568, 3151, 3152, 3153, 3154, 3155, 3156, 3157, 3198, 3782, 3783, 3784,
                                 3785, 3787, 3828, 4413, 4414, 4417, 5048, 5054, 5091, 5670, 5676, 5678, 5679, 5683,
                                 5684, 5717, 5721, 6300, 6301, 6305, 6306, 6308, 6309, 6310, 6312, 6313, 6314, 6315,
                                 6347, 6348, 6351, 6930, 6931, 6932, 6933, 6934, 6935, 6936, 6937, 6938, 6939, 6940,
                                 6941, 6942, 6943, 6944, 6945, 6977, 6978, 6981, 7560, 7561, 7562, 7563, 7564, 7565,
                                 7566, 7567, 7568, 7569, 7570, 7571, 7572, 7573, 7574, 7575, 7607, 7608, 7611, 8190,
                                 8191, 8192, 8193, 8194, 8195, 8196, 8197, 8199, 8200, 8201, 8202, 8203, 8204, 8205,
                                 8237, 8238, 8241, 8821, 8822, 8823, 8824, 8825, 8827, 8830, 8831, 8832, 8833, 8835,
                                 8867, 8868, 9452, 9453, 9454, 9457, 9461, 9462, 9465, 10096, 10104, 10132, 10726,
                                 10727, 10733, 10734, 10760, 10762, 11348, 11354, 11356, 11357, 11358, 11362, 11363,
                                 11364, 11386, 11390, 11391, 11392, 11970, 11976, 11978, 11979, 11983, 11984, 11986,
                                 11987, 11988, 11989, 11991, 11992, 11993, 11994, 12009, 12016, 12017, 12020, 12021,
                                 12022, 12600, 12601, 12605, 12606, 12608, 12609, 12610, 12611, 12612, 12613, 12614,
                                 12615, 12616, 12617, 12618, 12619, 12620, 12621, 12622, 12623, 12624, 12639, 12646,
                                 12647, 12648, 12650, 12651, 12652, 13230, 13231, 13232, 13233, 13234, 13235, 13236,
                                 13237, 13238, 13239, 13240, 13241, 13242, 13243, 13244, 13245, 13247, 13248, 13249,
                                 13250, 13251, 13252, 13253, 13254, 13269, 13276, 13277, 13278, 13280, 13281, 13282,
                                 13860, 13861, 13862, 13863, 13864, 13865, 13866, 13867, 13868, 13869, 13870, 13871,
                                 13872, 13873, 13874, 13875, 13878, 13879, 13880, 13881, 13882, 13883, 13899, 13906,
                                 13907, 13908, 13910, 13911, 14491, 14492, 14493, 14494, 14495, 14497, 14499, 14500,
                                 14501, 14502, 14503, 14505, 14509, 14510, 14511, 14512, 14529, 14536, 14537, 14538,
                                 15122, 15123, 15124, 15127, 15130, 15131, 15132, 15135, 15140, 15141, 15159, 15766,
                                 15774, 15775, 15781, 15799, 15802, 16396, 16397, 16403, 16404, 16405, 16406, 16410,
                                 16411, 16425, 16429, 16430, 16432, 17018, 17024, 17026, 17027, 17028, 17032, 17033,
                                 17034, 17035, 17036, 17037, 17039, 17040, 17041, 17042, 17055, 17056, 17059, 17060,
                                 17061, 17062, 17640, 17646, 17648, 17649, 17653, 17654, 17656, 17657, 17658, 17659,
                                 17660, 17661, 17662, 17663, 17664, 17665, 17666, 17667, 17668, 17669, 17670, 17671,
                                 17672, 17679, 17685, 17686, 17687, 17689, 17690, 17691, 17692, 18270, 18271, 18275,
                                 18276, 18278, 18279, 18280, 18281, 18282, 18283, 18284, 18285, 18286, 18287, 18288,
                                 18289, 18290, 18291, 18292, 18293, 18294, 18295, 18296, 18297, 18298, 18299, 18300,
                                 18301, 18302, 18309, 18315, 18316, 18317, 18318, 18319, 18320, 18321, 18322, 18900,
                                 18901, 18902, 18903, 18904, 18905, 18906, 18907, 18908, 18909, 18910, 18911, 18912,
                                 18913, 18914, 18915, 18917, 18918, 18919, 18920, 18921, 18922, 18923, 18926, 18927,
                                 18928, 18929, 18930, 18931, 18932, 18939, 18945, 18946, 18947, 18948, 18949, 18950,
                                 18951, 19531, 19532, 19533, 19534, 19535, 19537, 19539, 19540, 19541, 19542, 19543,
                                 19545, 19548, 19549, 19550, 19551, 19552, 19557, 19558, 19559, 19560, 19562, 19569,
                                 19575, 19576, 19577, 19578, 20162, 20163, 20164, 20167, 20170, 20171, 20172, 20175,
                                 20179, 20180, 20181, 20188, 20189, 20192, 20199, 20806, 20814, 20815, 20821, 20823,
                                 20827, 20834, 20839, 20842, 21436, 21437, 21443, 21444, 21445, 21446, 21450, 21451,
                                 21453, 21454, 21456, 21457, 21458, 21464, 21465, 21469, 21470, 21472, 22058, 22064,
                                 22066, 22067, 22068, 22072, 22073, 22074, 22075, 22076, 22077, 22078, 22079, 22080,
                                 22081, 22082, 22083, 22084, 22085, 22086, 22087, 22088, 22094, 22095, 22096, 22099,
                                 22100, 22101, 22102, 22680, 22686, 22688, 22689, 22693, 22694, 22696, 22697, 22698,
                                 22699, 22700, 22701, 22702, 22703, 22704, 22705, 22706, 22707, 22708, 22709, 22710,
                                 22711, 22712, 22713, 22714, 22715, 22716, 22717, 22718, 22719, 22724, 22725, 22726,
                                 22727, 22729, 22730, 22731, 22732, 23310, 23311, 23315, 23316, 23318, 23319, 23320,
                                 23321, 23322, 23323, 23324, 23325, 23327, 23328, 23329, 23330, 23331, 23332, 23333,
                                 23335, 23336, 23337, 23338, 23339, 23340, 23341, 23342, 23343, 23344, 23345, 23346,
                                 23347, 23348, 23349, 23354, 23355, 23356, 23357, 23358, 23359, 23360, 23361, 23941,
                                 23942, 23943, 23944, 23945, 23947, 23949, 23950, 23951, 23952, 23953, 23955, 23958,
                                 23959, 23960, 23961, 23962, 23966, 23967, 23968, 23969, 23970, 23972, 23974, 23975,
                                 23976, 23977, 23978, 23979, 23984, 23985, 23986, 23987, 23988, 24572, 24573, 24574,
                                 24577, 24580, 24581, 24582, 24585, 24589, 24590, 24591, 24597, 24598, 24599, 24602,
                                 24605, 24606, 24608, 24609, 25216, 25224, 25225, 25231, 25233, 25237, 25240, 25241,
                                 25242, 25244, 25249, 25252, 25846, 25847, 25853, 25854, 25855, 25856, 25860, 25861,
                                 25863, 25864, 25865, 25866, 25867, 25868, 25870, 25871, 25872, 25873, 25874, 25875,
                                 25879, 25880, 25882, 26468, 26474, 26476, 26477, 26478, 26482, 26483, 26484, 26485,
                                 26486, 26487, 26488, 26489, 26490, 26491, 26492, 26493, 26494, 26495, 26496, 26497,
                                 26498, 26500, 26501, 26502, 26503, 26504, 26505, 26506, 26509, 26510, 26511, 26512,
                                 27090, 27098, 27099, 27103, 27104, 27107, 27108, 27109, 27110, 27111, 27112, 27113,
                                 27115, 27116, 27117, 27118, 27119, 27120, 27121, 27122, 27123, 27124, 27125, 27126,
                                 27127, 27128, 27129, 27130, 27131, 27132, 27133, 27134, 27135, 27136, 27137, 27139,
                                 27140, 27141, 27721, 27729, 27730, 27731, 27732, 27733, 27735, 27738, 27739, 27740,
                                 27741, 27742, 27746, 27747, 27748, 27749, 27750, 27752, 27753, 27754, 27755, 27756,
                                 27757, 27758, 27759, 27760, 27761, 27762, 27763, 27764, 27765, 27766, 27767, 27768,
                                 28352, 28353, 28357, 28360, 28361, 28362, 28365, 28369, 28370, 28371, 28377, 28378,
                                 28379, 28382, 28384, 28385, 28386, 28388, 28389, 28391, 28392, 28393, 28996, 29004,
                                 29005, 29011, 29013, 29017, 29020, 29021, 29022, 29023, 29024, 29029, 29032, 29414,
                                 29626, 29627, 29633, 29634, 29635, 29636, 29640, 29641, 29643, 29644, 29645, 29646,
                                 29647, 29648, 29650, 29651, 29652, 29653, 29654, 29655, 29659, 29660, 29662, 30248,
                                 30257, 30258, 30262, 30263, 30265, 30266, 30267, 30268, 30269, 30270, 30271, 30272,
                                 30273, 30274, 30275, 30276, 30277, 30278, 30280, 30281, 30282, 30283, 30284, 30285,
                                 30286, 30289, 30290, 30291, 30469, 30674, 30879, 30888, 30889, 30890, 30891, 30892,
                                 30896, 30897, 30898, 30899, 30900, 30902, 30903, 30904, 30905, 30906, 30907, 30908,
                                 30909, 30910, 30911, 30912, 30913, 30914, 30915, 30916, 30917, 31516, 31525, 31531,
                                 31533, 31537, 31540, 31541, 31542, 31543, 31544, 31549, 31552, 32147, 32155, 32156,
                                 32160, 32161, 32163, 32164, 32165, 32166, 32167, 32168, 32170, 32171, 32172, 32173,
                                 32174, 32175, 32179, 32180, 32785, 32793, 32797, 32800, 32801, 32802, 32803, 32804,
                                 32809, 32946, 33194, 33443, 33448, 34073, 34074, 34077, 34078, 34121, 34703, 34704,
                                 34705, 34706, 34707, 34708, 34709, 34751, 35333, 35334, 35335, 35336, 35337, 35338,
                                 35339, 35340, 35381, 35963, 35964, 35965, 35966, 35967, 35968, 35969, 35970, 36011,
                                 36594, 36595, 36596, 36597, 36598, 36599, 36600, 36641, 37225, 37226, 37227, 37229,
                                 37230, 37271, 37856, 37859, 37860, 38491, 38496, 38534, 39060, 39066, 39113, 39118,
                                 39121, 39122, 39125, 39126, 39160, 39164, 39690, 39691, 39695, 39696, 39738, 39743,
                                 39744, 39747, 39748, 39751, 39752, 39753, 39754, 39755, 39756, 39757, 39790, 39791,
                                 39794, 40320, 40321, 40322, 40323, 40324, 40325, 40326, 40327, 40368, 40373, 40374,
                                 40375, 40376, 40377, 40378, 40379, 40380, 40381, 40382, 40383, 40384, 40385, 40386,
                                 40387, 40388, 40420, 40421, 40424, 40950, 40951, 40952, 40953, 40954, 40955, 40956,
                                 40957, 40998, 41003, 41004, 41005, 41006, 41007, 41008, 41009, 41010, 41011, 41012,
                                 41013, 41014, 41015, 41016, 41017, 41018, 41050, 41051, 41054, 41580, 41581, 41582,
                                 41583, 41584, 41585, 41586, 41587, 41628, 41633, 41634, 41635, 41636, 41637, 41638,
                                 41639, 41640, 41642, 41643, 41644, 41645, 41646, 41647, 41648, 41680, 41681, 41684,
                                 42211, 42212, 42213, 42214, 42215, 42217, 42258, 42264, 42265, 42266, 42267, 42269,
                                 42270, 42273, 42274, 42275, 42277, 42278, 42310, 42311, 42842, 42843, 42844, 42847,
                                 42895, 42896, 42899, 42900, 42904, 42907, 42908, 43539, 43547, 43575, 44169, 44170,
                                 44176, 44177, 44203, 44205, 44738, 44744, 44781, 44791, 44796, 44799, 44800, 44801,
                                 44805, 44806, 44807, 44829, 44833, 44834, 44835, 45360, 45366, 45368, 45369, 45373,
                                 45374, 45407, 45411, 45413, 45418, 45421, 45422, 45425, 45426, 45429, 45430, 45431,
                                 45432, 45434, 45435, 45436, 45437, 45452, 45459, 45460, 45463, 45464, 45465, 45990,
                                 45991, 45995, 45996, 45998, 45999, 46000, 46001, 46002, 46003, 46004, 46005, 46037,
                                 46038, 46041, 46043, 46044, 46047, 46048, 46051, 46052, 46053, 46054, 46055, 46056,
                                 46057, 46058, 46059, 46060, 46061, 46062, 46063, 46064, 46065, 46066, 46067, 46082,
                                 46089, 46090, 46091, 46093, 46094, 46095, 46620, 46621, 46622, 46623, 46624, 46625,
                                 46626, 46627, 46628, 46629, 46630, 46631, 46632, 46633, 46634, 46635, 46667, 46668,
                                 46671, 46673, 46674, 46675, 46676, 46677, 46678, 46679, 46680, 46681, 46682, 46683,
                                 46684, 46685, 46686, 46687, 46688, 46690, 46691, 46692, 46693, 46694, 46695, 46696,
                                 46697, 46712, 46719, 46720, 46721, 46723, 46724, 46725, 47250, 47251, 47252, 47253,
                                 47254, 47255, 47256, 47257, 47258, 47259, 47260, 47261, 47262, 47263, 47264, 47265,
                                 47297, 47298, 47301, 47303, 47304, 47305, 47306, 47307, 47308, 47309, 47310, 47311,
                                 47312, 47313, 47314, 47315, 47316, 47317, 47318, 47321, 47322, 47323, 47324, 47325,
                                 47326, 47342, 47349, 47350, 47351, 47353, 47354, 47881, 47882, 47883, 47884, 47885,
                                 47887, 47889, 47890, 47891, 47892, 47893, 47895, 47927, 47928, 47934, 47935, 47936,
                                 47937, 47939, 47940, 47942, 47943, 47944, 47945, 47947, 47948, 47952, 47953, 47954,
                                 47955, 47972, 47979, 47980, 47981, 48512, 48513, 48514, 48517, 48520, 48521, 48522,
                                 48525, 48565, 48566, 48569, 48570, 48573, 48574, 48577, 48578, 48583, 48584, 48602,
                                 49156, 49164, 49192, 49209, 49217, 49218, 49224, 49242, 49245, 49786, 49787, 49793,
                                 49794, 49820, 49822, 49839, 49840, 49846, 49847, 49848, 49849, 49853, 49854, 49868,
                                 49872, 49873, 49875, 50408, 50414, 50416, 50417, 50418, 50422, 50423, 50424, 50446,
                                 50450, 50451, 50452, 50461, 50466, 50469, 50470, 50471, 50475, 50476, 50477, 50478,
                                 50479, 50480, 50482, 50483, 50484, 50485, 50498, 50499, 50502, 50503, 50504, 50505,
                                 51030, 51036, 51038, 51039, 51043, 51044, 51046, 51047, 51048, 51049, 51050, 51051,
                                 51052, 51053, 51054, 51069, 51076, 51077, 51080, 51081, 51082, 51083, 51088, 51091,
                                 51092, 51095, 51096, 51099, 51100, 51101, 51102, 51103, 51104, 51105, 51106, 51107,
                                 51108, 51109, 51110, 51111, 51112, 51113, 51114, 51115, 51122, 51128, 51129, 51130,
                                 51132, 51133, 51134, 51135, 51660, 51661, 51665, 51666, 51668, 51669, 51670, 51671,
                                 51672, 51673, 51674, 51675, 51676, 51677, 51678, 51679, 51680, 51681, 51682, 51683,
                                 51684, 51699, 51706, 51707, 51708, 51710, 51711, 51712, 51713, 51714, 51717, 51718,
                                 51721, 51722, 51723, 51724, 51725, 51726, 51727, 51728, 51729, 51730, 51731, 51732,
                                 51733, 51734, 51735, 51736, 51737, 51738, 51739, 51740, 51741, 51742, 51743, 51744,
                                 51745, 51752, 51758, 51759, 51760, 51761, 51762, 51763, 51764, 51765, 52290, 52291,
                                 52292, 52293, 52294, 52295, 52296, 52297, 52298, 52299, 52300, 52301, 52302, 52303,
                                 52304, 52305, 52307, 52308, 52309, 52310, 52311, 52312, 52313, 52329, 52336, 52337,
                                 52338, 52340, 52341, 52343, 52344, 52345, 52346, 52347, 52348, 52349, 52350, 52351,
                                 52352, 52353, 52354, 52355, 52356, 52357, 52358, 52360, 52361, 52362, 52363, 52364,
                                 52365, 52366, 52369, 52370, 52371, 52372, 52373, 52374, 52375, 52382, 52388, 52389,
                                 52390, 52391, 52392, 52393, 52394, 52921, 52922, 52923, 52924, 52925, 52927, 52929,
                                 52930, 52931, 52932, 52933, 52935, 52938, 52939, 52940, 52941, 52942, 52959, 52966,
                                 52967, 52968, 52974, 52975, 52976, 52977, 52979, 52980, 52982, 52983, 52984, 52985,
                                 52987, 52988, 52991, 52992, 52993, 52994, 52995, 53000, 53001, 53002, 53003, 53005,
                                 53012, 53018, 53019, 53020, 53021, 53552, 53553, 53554, 53557, 53560, 53561, 53562,
                                 53565, 53569, 53570, 53571, 53589, 53605, 53606, 53609, 53610, 53613, 53614, 53617,
                                 53618, 53622, 53623, 53624, 53631, 53632, 53635, 53642, 54196, 54204, 54205, 54211,
                                 54229, 54232, 54249, 54257, 54258, 54264, 54266, 54269, 54277, 54282, 54285, 54826,
                                 54827, 54833, 54834, 54835, 54836, 54840, 54841, 54855, 54859, 54860, 54862, 54879,
                                 54880, 54886, 54887, 54888, 54889, 54893, 54894, 54896, 54897, 54899, 54900, 54901,
                                 54907, 54908, 54912, 54913, 54915, 55448, 55454, 55456, 55457, 55458, 55462, 55463,
                                 55464, 55465, 55466, 55467, 55468, 55469, 55470, 55471, 55472, 55485, 55486, 55489,
                                 55490, 55491, 55492, 55501, 55506, 55509, 55510, 55511, 55515, 55516, 55517, 55518,
                                 55519, 55520, 55521, 55522, 55523, 55524, 55525, 55526, 55527, 55528, 55529, 55530,
                                 55531, 55537, 55538, 55539, 55542, 55543, 55544, 55545, 56070, 56076, 56078, 56079,
                                 56083, 56084, 56086, 56087, 56088, 56089, 56090, 56091, 56092, 56093, 56094, 56095,
                                 56096, 56097, 56098, 56099, 56100, 56101, 56102, 56109, 56115, 56116, 56117, 56119,
                                 56120, 56121, 56122, 56123, 56128, 56131, 56132, 56135, 56136, 56139, 56140, 56141,
                                 56142, 56143, 56144, 56145, 56146, 56147, 56148, 56149, 56150, 56151, 56152, 56153,
                                 56154, 56155, 56156, 56157, 56158, 56159, 56160, 56161, 56162, 56167, 56168, 56169,
                                 56170, 56172, 56173, 56174, 56175, 56700, 56701, 56705, 56706, 56708, 56709, 56710,
                                 56711, 56712, 56713, 56714, 56715, 56717, 56718, 56719, 56720, 56721, 56722, 56723,
                                 56725, 56726, 56727, 56728, 56729, 56730, 56731, 56732, 56739, 56745, 56746, 56747,
                                 56748, 56749, 56750, 56751, 56753, 56754, 56757, 56758, 56761, 56762, 56763, 56764,
                                 56765, 56766, 56767, 56768, 56770, 56771, 56772, 56773, 56774, 56775, 56776, 56778,
                                 56779, 56780, 56781, 56782, 56783, 56784, 56785, 56786, 56787, 56788, 56789, 56790,
                                 56791, 56792, 56797, 56798, 56799, 56800, 56801, 56802, 56803, 56804, 57331, 57332,
                                 57333, 57334, 57335, 57337, 57339, 57340, 57341, 57342, 57343, 57345, 57348, 57349,
                                 57350, 57351, 57352, 57356, 57357, 57358, 57359, 57360, 57362, 57369, 57375, 57376,
                                 57377, 57378, 57384, 57385, 57386, 57387, 57389, 57390, 57392, 57393, 57394, 57395,
                                 57397, 57398, 57401, 57402, 57403, 57404, 57405, 57409, 57410, 57411, 57412, 57413,
                                 57415, 57417, 57418, 57419, 57420, 57421, 57422, 57427, 57428, 57429, 57430, 57431,
                                 57962, 57963, 57964, 57967, 57970, 57971, 57972, 57975, 57979, 57980, 57981, 57987,
                                 57988, 57989, 57992, 57999, 58015, 58016, 58019, 58020, 58023, 58024, 58027, 58028,
                                 58032, 58033, 58034, 58040, 58041, 58042, 58045, 58047, 58048, 58051, 58052, 58606,
                                 58614, 58615, 58621, 58623, 58627, 58634, 58639, 58642, 58659, 58667, 58668, 58674,
                                 58676, 58679, 58683, 58684, 58686, 58687, 58692, 58695, 59051, 59236, 59237, 59243,
                                 59244, 59245, 59246, 59250, 59251, 59253, 59254, 59255, 59256, 59257, 59258, 59264,
                                 59265, 59269, 59270, 59272, 59289, 59290, 59296, 59297, 59298, 59299, 59303, 59304,
                                 59306, 59307, 59308, 59309, 59310, 59311, 59313, 59314, 59315, 59316, 59317, 59318,
                                 59322, 59323, 59325, 59667, 59858, 59864, 59866, 59867, 59868, 59872, 59873, 59874,
                                 59875, 59876, 59877, 59878, 59879, 59880, 59881, 59882, 59883, 59884, 59885, 59886,
                                 59887, 59888, 59894, 59895, 59896, 59899, 59900, 59901, 59902, 59911, 59916, 59919,
                                 59920, 59921, 59925, 59926, 59927, 59928, 59929, 59930, 59931, 59932, 59933, 59934,
                                 59935, 59936, 59937, 59938, 59939, 59940, 59941, 59943, 59944, 59945, 59946, 59947,
                                 59948, 59949, 59952, 59953, 59954, 59955, 60193, 60311, 60480, 60486, 60488, 60489,
                                 60493, 60494, 60497, 60498, 60499, 60500, 60501, 60502, 60503, 60505, 60506, 60507,
                                 60508, 60509, 60510, 60511, 60512, 60513, 60514, 60515, 60516, 60517, 60518, 60519,
                                 60524, 60525, 60526, 60527, 60529, 60530, 60531, 60533, 60541, 60542, 60545, 60546,
                                 60550, 60551, 60552, 60553, 60554, 60555, 60556, 60558, 60559, 60560, 60561, 60562,
                                 60563, 60564, 60565, 60566, 60567, 60568, 60569, 60570, 60571, 60572, 60573, 60574,
                                 60575, 60576, 60577, 60578, 60579, 60580, 60582, 60583, 60584, 60722, 60927, 61111,
                                 61115, 61119, 61120, 61121, 61122, 61123, 61125, 61128, 61129, 61130, 61131, 61132,
                                 61136, 61137, 61138, 61139, 61140, 61142, 61143, 61144, 61145, 61146, 61147, 61148,
                                 61149, 61154, 61155, 61156, 61157, 61158, 61164, 61172, 61173, 61174, 61175, 61177,
                                 61178, 61181, 61182, 61183, 61184, 61185, 61189, 61190, 61191, 61192, 61193, 61195,
                                 61196, 61197, 61198, 61199, 61200, 61201, 61202, 61203, 61204, 61205, 61206, 61207,
                                 61208, 61209, 61210, 61211, 61453, 61742, 61743, 61744, 61747, 61750, 61751, 61752,
                                 61755, 61759, 61760, 61761, 61767, 61768, 61769, 61772, 61774, 61775, 61776, 61778,
                                 61779, 61795, 61799, 61800, 61803, 61804, 61807, 61808, 61812, 61813, 61814, 61820,
                                 61821, 61822, 61825, 61827, 61828, 61830, 61831, 61832, 61834, 61835, 61836, 61982,
                                 62386, 62394, 62395, 62401, 62403, 62407, 62410, 62411, 62412, 62413, 62414, 62419,
                                 62422, 62439, 62447, 62448, 62454, 62456, 62459, 62463, 62464, 62465, 62466, 62467,
                                 62472, 62475, 62803, 63016, 63017, 63023, 63024, 63025, 63026, 63030, 63031, 63033,
                                 63034, 63035, 63036, 63037, 63038, 63040, 63041, 63042, 63043, 63044, 63045, 63049,
                                 63050, 63052, 63069, 63070, 63076, 63077, 63078, 63079, 63083, 63084, 63086, 63087,
                                 63088, 63089, 63090, 63091, 63093, 63094, 63095, 63096, 63097, 63098, 63102, 63103,
                                 63105, 63638, 63644, 63647, 63648, 63652, 63653, 63655, 63656, 63657, 63658, 63659,
                                 63660, 63661, 63662, 63663, 63664, 63665, 63666, 63667, 63668, 63670, 63671, 63672,
                                 63673, 63674, 63675, 63676, 63679, 63680, 63681, 63691, 63700, 63701, 63705, 63706,
                                 63708, 63709, 63710, 63711, 63712, 63713, 63714, 63715, 63716, 63717, 63718, 63719,
                                 63720, 63721, 63723, 63724, 63725, 63726, 63727, 63728, 63729, 63732, 63733, 63734,
                                 63858, 64063, 64269, 64273, 64278, 64279, 64280, 64281, 64282, 64286, 64287, 64288,
                                 64289, 64290, 64292, 64293, 64294, 64295, 64296, 64297, 64298, 64299, 64300, 64301,
                                 64302, 64303, 64304, 64305, 64306, 64307, 64322, 64331, 64332, 64333, 64334, 64335,
                                 64339, 64340, 64341, 64342, 64343, 64345, 64346, 64347, 64348, 64349, 64350, 64351,
                                 64352, 64353, 64354, 64355, 64356, 64357, 64358, 64359, 64360, 64906, 64914, 64915,
                                 64921, 64923, 64927, 64930, 64931, 64932, 64933, 64934, 64939, 64942, 64959, 64968,
                                 64974, 64976, 64979, 64983, 64984, 64985, 64986, 64987, 64992, 64995, 65237, 65351,
                                 65537, 65543, 65545, 65546, 65550, 65551, 65553, 65554, 65555, 65556, 65557, 65558,
                                 65560, 65561, 65562, 65563, 65564, 65565, 65569, 65570, 65590, 65598, 65599, 65603,
                                 65604, 65606, 65607, 65608, 65609, 65610, 65611, 65613, 65614, 65615, 65616, 65617,
                                 65618, 65622, 65623, 65731, 65967, 66175, 66181, 66183, 66187, 66190, 66191, 66192,
                                 66193, 66194, 66199, 66228, 66236, 66239, 66243, 66244, 66245, 66246, 66247, 66252,
                                 66335, 66583, 66886, 66891, 66928, 67463, 67468, 67516, 67517, 67520, 67521, 67555,
                                 67558, 68093, 68094, 68097, 68098, 68141, 68146, 68147, 68148, 68149, 68150, 68151,
                                 68152, 68185, 68188, 68723, 68724, 68725, 68726, 68727, 68728, 68729, 68730, 68771,
                                 68776, 68777, 68778, 68779, 68780, 68781, 68782, 68783, 68815, 68818, 69353, 69354,
                                 69355, 69356, 69357, 69358, 69359, 69360, 69401, 69406, 69407, 69408, 69409, 69410,
                                 69411, 69412, 69413, 69445, 69448, 69983, 69984, 69985, 69986, 69987, 69988, 69989,
                                 69990, 70031, 70037, 70038, 70039, 70040, 70041, 70042, 70043, 70075, 70078, 70614,
                                 70615, 70616, 70617, 70619, 70620, 70661, 70668, 70669, 70670, 70672, 70673, 70705,
                                 71245, 71246, 71249, 71250, 71299, 71302, 71303, 71934, 71942, 71969, 72564, 72565,
                                 72571, 72572, 72597, 72599, 73141, 73146, 73184, 73186, 73191, 73194, 73195, 73196,
                                 73200, 73201, 73202, 73224, 73227, 73228, 73229, 73710, 73716, 73763, 73768, 73771,
                                 73772, 73775, 73776, 73810, 73814, 73816, 73817, 73820, 73821, 73824, 73825, 73826,
                                 73827, 73829, 73830, 73831, 73832, 73847, 73854, 73855, 73857, 73858, 73859, 74340,
                                 74341, 74345, 74346, 74388, 74393, 74394, 74397, 74398, 74401, 74402, 74403, 74404,
                                 74405, 74406, 74407, 74408, 74440, 74441, 74444, 74446, 74447, 74448, 74449, 74450,
                                 74451, 74452, 74453, 74454, 74455, 74456, 74457, 74458, 74459, 74460, 74461, 74462,
                                 74477, 74484, 74485, 74487, 74488, 74489, 74970, 74971, 74972, 74973, 74974, 74975,
                                 74976, 74977, 75018, 75023, 75024, 75025, 75026, 75027, 75028, 75029, 75030, 75031,
                                 75032, 75033, 75034, 75035, 75036, 75037, 75038, 75070, 75071, 75074, 75076, 75077,
                                 75078, 75079, 75080, 75081, 75082, 75083, 75085, 75086, 75087, 75088, 75089, 75090,
                                 75091, 75092, 75107, 75114, 75115, 75117, 75118, 75119, 75600, 75601, 75602, 75603,
                                 75604, 75605, 75606, 75607, 75648, 75653, 75654, 75655, 75656, 75657, 75658, 75659,
                                 75660, 75661, 75662, 75663, 75664, 75665, 75666, 75667, 75668, 75700, 75701, 75704,
                                 75706, 75707, 75708, 75709, 75710, 75711, 75712, 75713, 75716, 75717, 75718, 75719,
                                 75720, 75721, 75737, 75744, 75745, 75747, 75748, 76231, 76232, 76233, 76234, 76235,
                                 76237, 76278, 76284, 76285, 76286, 76287, 76289, 76290, 76292, 76293, 76294, 76295,
                                 76297, 76298, 76330, 76331, 76337, 76338, 76339, 76340, 76342, 76343, 76347, 76348,
                                 76349, 76350, 76367, 76374, 76375, 76862, 76863, 76864, 76867, 76915, 76916, 76919,
                                 76920, 76923, 76924, 76927, 76928, 76968, 76969, 76972, 76973, 76977, 76978, 76997,
                                 77559, 77567, 77595, 77604, 77612, 77613, 77617, 77636, 77639, 78189, 78190, 78196,
                                 78197, 78223, 78225, 78234, 78235, 78241, 78242, 78243, 78244, 78247, 78249, 78263,
                                 78266, 78267, 78269, 78758, 78764, 78801, 78811, 78816, 78819, 78820, 78821, 78825,
                                 78826, 78827, 78849, 78853, 78854, 78855, 78856, 78861, 78864, 78865, 78866, 78870,
                                 78871, 78872, 78873, 78874, 78875, 78876, 78877, 78879, 78880, 78893, 78894, 78896,
                                 78897, 78898, 78899, 79380, 79386, 79388, 79389, 79393, 79394, 79427, 79431, 79433,
                                 79438, 79441, 79442, 79445, 79446, 79449, 79450, 79451, 79452, 79453, 79454, 79455,
                                 79456, 79457, 79472, 79479, 79480, 79483, 79484, 79485, 79486, 79487, 79490, 79491,
                                 79494, 79495, 79496, 79497, 79498, 79499, 79500, 79501, 79502, 79503, 79504, 79505,
                                 79506, 79507, 79508, 79509, 79510, 79517, 79523, 79524, 79525, 79526, 79527, 79528,
                                 79529, 80010, 80011, 80015, 80016, 80018, 80019, 80020, 80021, 80022, 80023, 80024,
                                 80025, 80057, 80058, 80061, 80063, 80064, 80067, 80068, 80071, 80072, 80073, 80074,
                                 80075, 80076, 80077, 80078, 80079, 80080, 80081, 80082, 80083, 80084, 80085, 80086,
                                 80087, 80102, 80109, 80110, 80111, 80113, 80114, 80115, 80116, 80117, 80118, 80119,
                                 80120, 80121, 80122, 80123, 80124, 80125, 80126, 80127, 80128, 80129, 80130, 80131,
                                 80132, 80133, 80134, 80135, 80136, 80137, 80138, 80139, 80140, 80147, 80153, 80154,
                                 80155, 80156, 80157, 80158, 80159, 80640, 80641, 80642, 80643, 80644, 80645, 80646,
                                 80647, 80648, 80649, 80650, 80651, 80652, 80653, 80654, 80655, 80687, 80688, 80691,
                                 80693, 80694, 80695, 80696, 80697, 80698, 80699, 80700, 80701, 80702, 80703, 80704,
                                 80705, 80706, 80707, 80708, 80710, 80711, 80712, 80713, 80714, 80715, 80716, 80732,
                                 80739, 80740, 80741, 80743, 80744, 80746, 80747, 80748, 80749, 80750, 80751, 80752,
                                 80753, 80755, 80756, 80757, 80758, 80759, 80760, 80761, 80764, 80765, 80766, 80767,
                                 80768, 80769, 80770, 80777, 80783, 80784, 80785, 80786, 80787, 80788, 81271, 81272,
                                 81273, 81274, 81275, 81277, 81279, 81280, 81281, 81282, 81283, 81285, 81317, 81318,
                                 81324, 81325, 81326, 81327, 81329, 81330, 81332, 81333, 81334, 81335, 81337, 81338,
                                 81341, 81342, 81343, 81344, 81345, 81362, 81369, 81370, 81371, 81377, 81378, 81379,
                                 81380, 81382, 81383, 81386, 81387, 81388, 81389, 81390, 81394, 81395, 81396, 81398,
                                 81400, 81407, 81413, 81414, 81415, 81902, 81903, 81904, 81907, 81910, 81911, 81912,
                                 81915, 81955, 81956, 81959, 81960, 81963, 81964, 81967, 81968, 81972, 81973, 81974,
                                 81992, 82008, 82009, 82012, 82013, 82017, 82018, 82019, 82025, 82028, 82030, 82037,
                                 82546, 82554, 82582, 82599, 82607, 82608, 82614, 82632, 82635, 82644, 82652, 82653,
                                 82657, 82661, 82665, 82672, 82676, 82679, 83159, 83176, 83177, 83183, 83184, 83210,
                                 83212, 83229, 83230, 83236, 83237, 83238, 83239, 83243, 83244, 83258, 83262, 83263,
                                 83265, 83274, 83275, 83281, 83282, 83283, 83284, 83287, 83289, 83291, 83292, 83294,
                                 83295, 83296, 83302, 83303, 83306, 83307, 83309, 83625, 83798, 83804, 83806, 83807,
                                 83808, 83812, 83813, 83814, 83836, 83840, 83841, 83842, 83851, 83856, 83859, 83860,
                                 83861, 83865, 83866, 83867, 83868, 83869, 83870, 83871, 83872, 83873, 83874, 83875,
                                 83888, 83889, 83892, 83893, 83894, 83895, 83896, 83901, 83904, 83905, 83906, 83910,
                                 83911, 83912, 83913, 83914, 83915, 83916, 83917, 83918, 83919, 83920, 83921, 83922,
                                 83923, 83924, 83925, 83926, 83932, 83933, 83934, 83936, 83937, 83938, 83939, 84243,
                                 84419, 84420, 84426, 84428, 84429, 84433, 84434, 84436, 84437, 84438, 84439, 84440,
                                 84441, 84442, 84443, 84444, 84459, 84466, 84467, 84470, 84471, 84472, 84473, 84478,
                                 84481, 84482, 84485, 84486, 84489, 84490, 84491, 84492, 84493, 84494, 84495, 84496,
                                 84497, 84498, 84499, 84500, 84501, 84502, 84503, 84504, 84505, 84512, 84518, 84519,
                                 84520, 84522, 84523, 84524, 84525, 84526, 84527, 84530, 84531, 84534, 84535, 84536,
                                 84537, 84538, 84539, 84540, 84541, 84542, 84543, 84544, 84545, 84546, 84547, 84548,
                                 84549, 84550, 84551, 84552, 84553, 84554, 84555, 84556, 84557, 84562, 84563, 84564,
                                 84565, 84566, 84567, 84568, 84569, 84805, 84885, 85050, 85051, 85055, 85056, 85058,
                                 85059, 85060, 85061, 85062, 85063, 85064, 85065, 85067, 85068, 85069, 85070, 85071,
                                 85072, 85073, 85089, 85096, 85097, 85098, 85100, 85101, 85103, 85104, 85107, 85108,
                                 85111, 85112, 85113, 85114, 85115, 85116, 85117, 85118, 85120, 85121, 85122, 85123,
                                 85124, 85125, 85126, 85128, 85129, 85130, 85131, 85132, 85133, 85134, 85135, 85142,
                                 85148, 85149, 85150, 85151, 85152, 85153, 85154, 85156, 85157, 85158, 85159, 85160,
                                 85161, 85162, 85163, 85165, 85166, 85167, 85168, 85169, 85170, 85171, 85173, 85174,
                                 85175, 85176, 85177, 85178, 85179, 85180, 85181, 85182, 85183, 85184, 85185, 85186,
                                 85187, 85192, 85193, 85194, 85195, 85196, 85197, 85198, 85299, 85503, 85681, 85682,
                                 85683, 85684, 85685, 85687, 85689, 85690, 85691, 85692, 85693, 85695, 85698, 85699,
                                 85700, 85701, 85702, 85719, 85726, 85727, 85728, 85734, 85735, 85736, 85737, 85739,
                                 85740, 85742, 85743, 85744, 85745, 85747, 85748, 85751, 85752, 85753, 85754, 85755,
                                 85759, 85760, 85761, 85762, 85763, 85765, 85772, 85778, 85779, 85780, 85781, 85787,
                                 85788, 85789, 85790, 85792, 85793, 85796, 85797, 85798, 85799, 85800, 85804, 85805,
                                 85806, 85808, 85809, 85810, 85811, 85812, 85813, 85814, 85816, 85817, 85822, 85823,
                                 85824, 85825, 86065, 86312, 86313, 86314, 86317, 86320, 86321, 86322, 86325, 86329,
                                 86330, 86331, 86349, 86365, 86366, 86369, 86370, 86373, 86374, 86377, 86378, 86382,
                                 86383, 86384, 86390, 86391, 86392, 86395, 86402, 86418, 86419, 86422, 86423, 86427,
                                 86428, 86429, 86435, 86436, 86438, 86440, 86442, 86443, 86446, 86447, 86559, 86956,
                                 86964, 86965, 86971, 86989, 86992, 87009, 87017, 87018, 87024, 87026, 87029, 87037,
                                 87042, 87045, 87054, 87062, 87063, 87067, 87071, 87075, 87078, 87080, 87081, 87082,
                                 87086, 87089, 87399, 87586, 87587, 87593, 87594, 87595, 87596, 87600, 87601, 87615,
                                 87619, 87620, 87622, 87639, 87640, 87646, 87647, 87648, 87649, 87653, 87654, 87656,
                                 87657, 87658, 87659, 87660, 87661, 87667, 87668, 87672, 87673, 87675, 87684, 87685,
                                 87691, 87692, 87693, 87694, 87697, 87699, 87701, 87702, 87703, 87704, 87705, 87706,
                                 87708, 87709, 87710, 87711, 87712, 87713, 87716, 87717, 87719, 88016, 88208, 88214,
                                 88216, 88217, 88218, 88222, 88223, 88224, 88225, 88226, 88227, 88228, 88229, 88230,
                                 88231, 88232, 88245, 88246, 88249, 88250, 88251, 88252, 88261, 88266, 88269, 88270,
                                 88271, 88275, 88276, 88277, 88278, 88279, 88280, 88281, 88282, 88283, 88284, 88285,
                                 88286, 88287, 88288, 88289, 88290, 88291, 88297, 88298, 88299, 88302, 88303, 88304,
                                 88305, 88306, 88311, 88314, 88315, 88316, 88320, 88321, 88322, 88323, 88324, 88325,
                                 88326, 88327, 88328, 88329, 88330, 88331, 88332, 88333, 88334, 88335, 88336, 88338,
                                 88339, 88340, 88341, 88342, 88343, 88344, 88346, 88347, 88348, 88349, 88578, 88659,
                                 88830, 88836, 88838, 88839, 88843, 88844, 88847, 88848, 88849, 88850, 88851, 88852,
                                 88853, 88855, 88856, 88857, 88858, 88859, 88860, 88861, 88862, 88869, 88875, 88876,
                                 88877, 88879, 88880, 88881, 88883, 88888, 88891, 88892, 88895, 88896, 88900, 88901,
                                 88902, 88903, 88904, 88905, 88906, 88908, 88909, 88910, 88911, 88912, 88913, 88914,
                                 88915, 88916, 88917, 88918, 88919, 88920, 88921, 88922, 88927, 88928, 88929, 88930,
                                 88932, 88933, 88934, 88936, 88937, 88940, 88941, 88945, 88946, 88947, 88948, 88949,
                                 88950, 88951, 88953, 88954, 88955, 88956, 88957, 88958, 88959, 88960, 88961, 88962,
                                 88963, 88964, 88965, 88966, 88967, 88968, 88969, 88970, 88971, 88972, 88973, 88974,
                                 88975, 88976, 88977, 88978, 89071, 89276, 89461, 89465, 89469, 89470, 89471, 89472,
                                 89473, 89475, 89478, 89479, 89480, 89481, 89482, 89486, 89487, 89488, 89489, 89490,
                                 89492, 89499, 89505, 89506, 89507, 89508, 89514, 89517, 89522, 89523, 89524, 89525,
                                 89527, 89528, 89531, 89532, 89533, 89534, 89535, 89539, 89540, 89541, 89542, 89543,
                                 89545, 89546, 89547, 89548, 89549, 89550, 89551, 89552, 89557, 89558, 89559, 89560,
                                 89561, 89567, 89568, 89569, 89570, 89572, 89573, 89576, 89577, 89578, 89579, 89580,
                                 89584, 89585, 89586, 89588, 89589, 89590, 89591, 89592, 89593, 89594, 89595, 89596,
                                 89597, 89598, 89599, 89600, 89601, 89602, 89603, 89604, 89605, 89838, 90092, 90093,
                                 90094, 90097, 90100, 90101, 90102, 90105, 90109, 90110, 90111, 90117, 90118, 90119,
                                 90122, 90129, 90145, 90146, 90149, 90150, 90153, 90154, 90157, 90158, 90162, 90163,
                                 90164, 90170, 90171, 90172, 90175, 90177, 90178, 90180, 90181, 90182, 90198, 90199,
                                 90202, 90203, 90207, 90208, 90209, 90215, 90216, 90218, 90220, 90222, 90223, 90224,
                                 90226, 90227, 90228, 90229, 90231, 90331, 90736, 90744, 90745, 90751, 90753, 90757,
                                 90764, 90769, 90772, 90789, 90797, 90798, 90804, 90806, 90809, 90813, 90814, 90815,
                                 90816, 90817, 90822, 90825, 90834, 90842, 90843, 90847, 90851, 90855, 90858, 90859,
                                 90860, 90861, 90862, 90866, 90869, 91152, 91349, 91366, 91367, 91373, 91374, 91375,
                                 91376, 91380, 91381, 91383, 91384, 91385, 91386, 91387, 91388, 91394, 91395, 91399,
                                 91400, 91402, 91419, 91420, 91426, 91427, 91428, 91429, 91433, 91434, 91436, 91437,
                                 91438, 91439, 91440, 91441, 91443, 91444, 91445, 91446, 91447, 91448, 91452, 91453,
                                 91455, 91464, 91465, 91471, 91472, 91473, 91474, 91477, 91479, 91481, 91482, 91483,
                                 91484, 91485, 91486, 91488, 91489, 91490, 91491, 91492, 91493, 91496, 91497, 91499,
                                 91731, 91815, 91988, 91994, 91997, 91998, 92002, 92003, 92005, 92006, 92007, 92008,
                                 92009, 92010, 92011, 92012, 92013, 92014, 92015, 92016, 92017, 92018, 92024, 92025,
                                 92026, 92029, 92030, 92031, 92041, 92046, 92050, 92051, 92055, 92056, 92058, 92059,
                                 92060, 92061, 92062, 92063, 92064, 92065, 92066, 92067, 92068, 92069, 92070, 92071,
                                 92073, 92074, 92075, 92076, 92077, 92078, 92079, 92082, 92083, 92084, 92086, 92095,
                                 92096, 92100, 92101, 92103, 92104, 92105, 92106, 92107, 92108, 92109, 92110, 92111,
                                 92112, 92113, 92114, 92115, 92116, 92118, 92119, 92120, 92121, 92122, 92123, 92124,
                                 92126, 92127, 92128, 92207, 92412, 92433, 92619, 92623, 92628, 92629, 92630, 92631,
                                 92632, 92636, 92637, 92638, 92639, 92640, 92642, 92643, 92644, 92645, 92646, 92647,
                                 92648, 92649, 92654, 92655, 92656, 92657, 92672, 92675, 92681, 92682, 92683, 92684,
                                 92685, 92689, 92690, 92691, 92692, 92693, 92695, 92696, 92697, 92698, 92699, 92700,
                                 92701, 92702, 92703, 92704, 92705, 92706, 92707, 92708, 92709, 92710, 92717, 92726,
                                 92727, 92728, 92729, 92730, 92734, 92735, 92736, 92738, 92739, 92740, 92741, 92742,
                                 92743, 92744, 92745, 92746, 92747, 92748, 92749, 92750, 92751, 92752, 92753, 92754,
                                 92755, 92991, 92995, 93256, 93264, 93265, 93271, 93273, 93277, 93280, 93281, 93282,
                                 93283, 93284, 93289, 93292, 93309, 93317, 93318, 93324, 93326, 93329, 93333, 93334,
                                 93335, 93336, 93337, 93342, 93345, 93354, 93363, 93367, 93371, 93375, 93378, 93379,
                                 93380, 93381, 93382, 93386, 93389, 93588, 93699, 93887, 93893, 93895, 93896, 93900,
                                 93901, 93903, 93904, 93905, 93906, 93907, 93908, 93910, 93911, 93912, 93913, 93914,
                                 93915, 93919, 93920, 93940, 93946, 93948, 93949, 93953, 93954, 93956, 93957, 93958,
                                 93959, 93960, 93961, 93963, 93964, 93965, 93966, 93967, 93968, 93972, 93973, 93985,
                                 93993, 93994, 93997, 93999, 94001, 94002, 94003, 94004, 94005, 94006, 94008, 94009,
                                 94010, 94011, 94012, 94013, 94016, 94017, 94080, 94316, 94525, 94531, 94533, 94537,
                                 94540, 94541, 94542, 94543, 94544, 94549, 94578, 94584, 94586, 94589, 94593, 94594,
                                 94595, 94596, 94597, 94602, 94623, 94631, 94635, 94638, 94639, 94640, 94641, 94642,
                                 94646, 94650, 94932, 94934, 95281, 95287, 95313, 95911, 95912, 95917, 95918, 95942,
                                 95943, 96496, 96501, 96538, 96541, 96542, 96543, 96546, 96547, 96548, 96568, 96572,
                                 96573, 97073, 97078, 97126, 97127, 97130, 97131, 97165, 97168, 97171, 97172, 97173,
                                 97174, 97175, 97176, 97177, 97178, 97193, 97198, 97202, 97203, 97703, 97704, 97707,
                                 97708, 97751, 97756, 97757, 97758, 97759, 97760, 97761, 97762, 97763, 97795, 97798,
                                 97801, 97802, 97803, 97804, 97805, 97806, 97807, 97808, 97809, 97823, 97828, 97832,
                                 97833, 98333, 98334, 98335, 98336, 98337, 98338, 98339, 98340, 98381, 98386, 98387,
                                 98388, 98389, 98390, 98391, 98392, 98393, 98425, 98428, 98432, 98433, 98434, 98435,
                                 98436, 98437, 98438, 98439, 98453, 98458, 98462, 98463, 98963, 98964, 98965, 98966,
                                 98967, 98968, 98969, 98970, 99011, 99016, 99017, 99018, 99019, 99020, 99021, 99022,
                                 99023, 99055, 99058, 99062, 99063, 99064, 99065, 99066, 99069, 99083, 99088, 99092,
                                 99594, 99595, 99596, 99597, 99599, 99600, 99641, 99647, 99648, 99649, 99650, 99652,
                                 99653, 99685, 99693, 99694, 99695, 99699, 99713, 99718, 100225, 100226, 100229, 100230,
                                 100278, 100279, 100282, 100283, 100325, 100329, 100343, 100914, 100922, 100949, 100951,
                                 100957, 100960, 100964, 100981, 100983, 101544, 101545, 101551, 101552, 101577, 101579,
                                 101581, 101582, 101587, 101588, 101590, 101591, 101593, 101594, 101607, 101611, 101612,
                                 101613, 102121, 102126, 102164, 102166, 102171, 102174, 102175, 102176, 102180, 102181,
                                 102182, 102204, 102207, 102208, 102209, 102211, 102212, 102213, 102216, 102217, 102218,
                                 102220, 102221, 102222, 102223, 102224, 102225, 102226, 102237, 102238, 102241, 102242,
                                 102243, 102690, 102696, 102743, 102748, 102751, 102752, 102755, 102756, 102790, 102794,
                                 102796, 102797, 102800, 102801, 102804, 102805, 102806, 102807, 102808, 102809, 102810,
                                 102811, 102812, 102827, 102834, 102835, 102837, 102838, 102839, 102841, 102842, 102843,
                                 102844, 102845, 102846, 102847, 102848, 102849, 102850, 102851, 102852, 102853, 102854,
                                 102855, 102856, 102857, 102863, 102867, 102868, 102871, 102872, 102873, 103320, 103321,
                                 103325, 103326, 103368, 103373, 103374, 103377, 103378, 103381, 103382, 103383, 103384,
                                 103385, 103386, 103387, 103388, 103420, 103421, 103424, 103426, 103427, 103428, 103429,
                                 103430, 103431, 103432, 103433, 103434, 103435, 103436, 103437, 103438, 103439, 103440,
                                 103441, 103442, 103457, 103464, 103465, 103467, 103468, 103469, 103471, 103472, 103473,
                                 103474, 103475, 103476, 103477, 103478, 103479, 103480, 103481, 103482, 103483, 103484,
                                 103485, 103486, 103487, 103493, 103497, 103498, 103501, 103502, 103503, 103733, 103950,
                                 103951, 103952, 103953, 103954, 103955, 103956, 103957, 103998, 104003, 104004, 104005,
                                 104006, 104007, 104008, 104009, 104010, 104011, 104012, 104013, 104014, 104015, 104016,
                                 104017, 104018, 104050, 104051, 104054, 104056, 104057, 104058, 104059, 104060, 104061,
                                 104062, 104063, 104065, 104066, 104067, 104068, 104069, 104070, 104071, 104087, 104094,
                                 104095, 104097, 104098, 104102, 104103, 104104, 104105, 104106, 104108, 104109, 104111,
                                 104112, 104113, 104114, 104115, 104116, 104117, 104123, 104127, 104128, 104131, 104132,
                                 104202, 104581, 104582, 104583, 104584, 104585, 104587, 104628, 104634, 104635, 104636,
                                 104637, 104639, 104640, 104642, 104643, 104644, 104645, 104647, 104648, 104680, 104681,
                                 104687, 104688, 104689, 104690, 104692, 104693, 104696, 104697, 104698, 104699, 104700,
                                 104717, 104724, 104725, 104733, 104734, 104735, 104736, 104739, 104742, 104743, 104745,
                                 104746, 104747, 104753, 104757, 104758, 104993, 105212, 105213, 105214, 105217, 105265,
                                 105266, 105269, 105270, 105273, 105274, 105277, 105278, 105318, 105319, 105322, 105323,
                                 105327, 105328, 105329, 105347, 105364, 105365, 105369, 105372, 105375, 105377, 105383,
                                 105462, 105909, 105917, 105945, 105954, 105962, 105963, 105967, 105986, 105989, 105991,
                                 105997, 106000, 106004, 106008, 106014, 106021, 106023, 106024, 106467, 106539, 106540,
                                 106546, 106547, 106573, 106575, 106584, 106585, 106591, 106592, 106593, 106594, 106597,
                                 106599, 106613, 106616, 106617, 106619, 106621, 106622, 106627, 106628, 106630, 106631,
                                 106633, 106634, 106638, 106639, 106640, 106642, 106644, 106647, 106651, 106652, 106653,
                                 106654, 106934, 107108, 107114, 107151, 107161, 107166, 107169, 107170, 107171, 107175,
                                 107176, 107177, 107199, 107203, 107204, 107205, 107206, 107211, 107214, 107215, 107216,
                                 107220, 107221, 107222, 107223, 107224, 107225, 107226, 107227, 107228, 107229, 107230,
                                 107243, 107244, 107246, 107247, 107248, 107249, 107251, 107252, 107253, 107256, 107257,
                                 107258, 107260, 107261, 107262, 107263, 107264, 107265, 107266, 107267, 107268, 107269,
                                 107270, 107271, 107272, 107274, 107277, 107278, 107281, 107282, 107283, 107284, 107552,
                                 107727, 107730, 107736, 107738, 107739, 107743, 107744, 107777, 107781, 107783, 107788,
                                 107791, 107792, 107795, 107796, 107799, 107800, 107801, 107802, 107803, 107804, 107805,
                                 107806, 107807, 107822, 107829, 107830, 107833, 107834, 107835, 107836, 107837, 107840,
                                 107841, 107844, 107845, 107846, 107847, 107848, 107849, 107850, 107851, 107852, 107853,
                                 107854, 107855, 107856, 107857, 107858, 107859, 107860, 107867, 107873, 107874, 107875,
                                 107876, 107877, 107878, 107879, 107881, 107882, 107883, 107884, 107885, 107886, 107887,
                                 107888, 107889, 107890, 107891, 107892, 107893, 107894, 107895, 107896, 107897, 107898,
                                 107899, 107900, 107901, 107902, 107903, 107904, 107907, 107908, 107911, 107912, 107913,
                                 107914, 108141, 108194, 108360, 108361, 108365, 108366, 108368, 108369, 108370, 108371,
                                 108372, 108373, 108374, 108375, 108407, 108408, 108411, 108413, 108414, 108417, 108418,
                                 108421, 108422, 108423, 108424, 108425, 108426, 108427, 108428, 108430, 108431, 108432,
                                 108433, 108434, 108435, 108436, 108452, 108459, 108460, 108461, 108463, 108464, 108466,
                                 108467, 108468, 108469, 108470, 108471, 108472, 108473, 108475, 108476, 108477, 108478,
                                 108479, 108480, 108481, 108483, 108484, 108485, 108486, 108487, 108488, 108489, 108490,
                                 108497, 108503, 108504, 108505, 108506, 108507, 108508, 108512, 108513, 108514, 108515,
                                 108516, 108518, 108519, 108520, 108521, 108522, 108523, 108524, 108525, 108526, 108527,
                                 108528, 108529, 108530, 108531, 108532, 108533, 108534, 108537, 108538, 108541, 108542,
                                 108544, 108606, 108812, 108991, 108992, 108993, 108994, 108995, 108997, 108999, 109000,
                                 109001, 109002, 109003, 109005, 109037, 109038, 109044, 109045, 109046, 109047, 109049,
                                 109050, 109052, 109053, 109054, 109055, 109057, 109058, 109061, 109062, 109063, 109064,
                                 109065, 109082, 109089, 109090, 109091, 109097, 109098, 109099, 109100, 109102, 109103,
                                 109106, 109107, 109108, 109109, 109110, 109114, 109115, 109116, 109118, 109119, 109120,
                                 109127, 109133, 109134, 109135, 109143, 109144, 109145, 109146, 109149, 109151, 109152,
                                 109153, 109155, 109156, 109157, 109158, 109159, 109160, 109161, 109162, 109163, 109164,
                                 109167, 109168, 109401, 109622, 109623, 109624, 109627, 109630, 109631, 109632, 109635,
                                 109675, 109676, 109679, 109680, 109683, 109684, 109687, 109688, 109692, 109693, 109694,
                                 109712, 109728, 109729, 109732, 109733, 109737, 109738, 109739, 109745, 109746, 109748,
                                 109750, 109757, 109774, 109775, 109779, 109782, 109785, 109786, 109787, 109789, 109791,
                                 109792, 109793, 109866, 110266, 110274, 110302, 110319, 110327, 110328, 110334, 110352,
                                 110355, 110364, 110372, 110373, 110377, 110381, 110385, 110392, 110396, 110399, 110401,
                                 110407, 110410, 110414, 110418, 110424, 110425, 110426, 110430, 110431, 110433, 110434,
                                 110708, 110896, 110897, 110903, 110904, 110930, 110932, 110949, 110950, 110956, 110957,
                                 110958, 110959, 110963, 110964, 110978, 110982, 110983, 110985, 110994, 110995, 111001,
                                 111002, 111003, 111004, 111007, 111009, 111011, 111012, 111013, 111014, 111015, 111016,
                                 111022, 111023, 111026, 111027, 111029, 111031, 111032, 111037, 111038, 111040, 111041,
                                 111043, 111044, 111048, 111049, 111050, 111051, 111052, 111054, 111055, 111056, 111057,
                                 111059, 111060, 111061, 111062, 111063, 111064, 111325, 111518, 111524, 111526, 111527,
                                 111528, 111532, 111533, 111534, 111556, 111560, 111561, 111562, 111571, 111576, 111579,
                                 111580, 111581, 111585, 111586, 111587, 111588, 111589, 111590, 111591, 111592, 111593,
                                 111594, 111595, 111608, 111609, 111612, 111613, 111614, 111615, 111616, 111621, 111624,
                                 111625, 111626, 111630, 111631, 111632, 111633, 111634, 111635, 111636, 111637, 111638,
                                 111639, 111640, 111641, 111642, 111643, 111644, 111645, 111646, 111652, 111653, 111654,
                                 111656, 111657, 111658, 111659, 111661, 111662, 111663, 111666, 111667, 111668, 111670,
                                 111671, 111672, 111673, 111674, 111675, 111676, 111677, 111678, 111679, 111680, 111681,
                                 111682, 111684, 111685, 111686, 111687, 111688, 111689, 111690, 111691, 111692, 111693,
                                 111694, 111917, 111968, 112140, 112146, 112148, 112149, 112153, 112154, 112157, 112158,
                                 112159, 112160, 112161, 112162, 112163, 112179, 112186, 112187, 112190, 112191, 112193,
                                 112198, 112201, 112202, 112205, 112206, 112210, 112211, 112212, 112213, 112214, 112215,
                                 112216, 112218, 112219, 112220, 112221, 112222, 112223, 112224, 112225, 112232, 112238,
                                 112239, 112240, 112242, 112243, 112244, 112246, 112247, 112250, 112251, 112255, 112256,
                                 112257, 112258, 112259, 112260, 112261, 112263, 112264, 112265, 112266, 112267, 112268,
                                 112269, 112270, 112271, 112272, 112273, 112274, 112275, 112276, 112277, 112282, 112283,
                                 112284, 112285, 112286, 112287, 112288, 112292, 112293, 112294, 112295, 112296, 112298,
                                 112299, 112300, 112301, 112302, 112303, 112304, 112305, 112306, 112307, 112308, 112309,
                                 112310, 112311, 112312, 112313, 112314, 112315, 112316, 112317, 112318, 112319, 112320,
                                 112321, 112322, 112324, 112373, 112585, 112771, 112775, 112779, 112780, 112781, 112782,
                                 112783, 112785, 112788, 112789, 112790, 112791, 112792, 112809, 112816, 112817, 112818,
                                 112824, 112827, 112832, 112833, 112834, 112835, 112837, 112838, 112841, 112842, 112843,
                                 112844, 112845, 112849, 112850, 112851, 112852, 112853, 112855, 112862, 112868, 112869,
                                 112870, 112871, 112877, 112878, 112879, 112880, 112882, 112883, 112886, 112887, 112888,
                                 112889, 112890, 112894, 112895, 112896, 112898, 112899, 112900, 112901, 112902, 112903,
                                 112904, 112905, 112906, 112907, 112912, 112913, 112914, 112915, 112923, 112924, 112925,
                                 112926, 112929, 112931, 112932, 112933, 112935, 112936, 112937, 112938, 112939, 112940,
                                 112941, 112942, 112943, 112944, 112945, 112946, 112947, 112948, 112949, 112950, 112954,
                                 113177, 113183, 113402, 113403, 113404, 113407, 113410, 113411, 113412, 113415, 113419,
                                 113420, 113421, 113439, 113455, 113456, 113459, 113460, 113463, 113464, 113467, 113468,
                                 113472, 113473, 113474, 113480, 113481, 113482, 113485, 113492, 113508, 113509, 113512,
                                 113513, 113517, 113518, 113519, 113525, 113526, 113528, 113530, 113532, 113533, 113534,
                                 113536, 113537, 113554, 113555, 113559, 113562, 113565, 113566, 113567, 113569, 113570,
                                 113571, 113572, 113573, 113575, 113576, 113579, 113633, 113652, 114046, 114054, 114055,
                                 114061, 114079, 114082, 114099, 114107, 114108, 114114, 114116, 114119, 114127, 114132,
                                 114135, 114144, 114152, 114153, 114157, 114161, 114165, 114168, 114169, 114170, 114171,
                                 114172, 114176, 114179, 114181, 114187, 114190, 114194, 114198, 114204, 114205, 114206,
                                 114209, 114210, 114211, 114213, 114214, 114461, 114657, 114676, 114677, 114683, 114684,
                                 114685, 114686, 114690, 114691, 114705, 114709, 114710, 114712, 114729, 114730, 114736,
                                 114737, 114738, 114739, 114743, 114744, 114746, 114747, 114748, 114749, 114750, 114751,
                                 114757, 114758, 114762, 114763, 114765, 114774, 114775, 114781, 114782, 114783, 114784,
                                 114787, 114789, 114791, 114792, 114793, 114794, 114795, 114796, 114798, 114799, 114800,
                                 114801, 114802, 114803, 114806, 114807, 114809, 114811, 114812, 114817, 114818, 114820,
                                 114821, 114823, 114824, 114828, 114829, 114830, 114831, 114832, 114834, 114835, 114836,
                                 114837, 114839, 114840, 114841, 114842, 114843, 114844, 115043, 115124, 115298, 115304,
                                 115307, 115308, 115312, 115313, 115315, 115316, 115317, 115318, 115319, 115320, 115321,
                                 115322, 115335, 115336, 115339, 115340, 115341, 115351, 115356, 115360, 115361, 115365,
                                 115366, 115368, 115369, 115370, 115371, 115372, 115373, 115374, 115375, 115376, 115377,
                                 115378, 115379, 115380, 115381, 115387, 115388, 115389, 115392, 115393, 115394, 115396,
                                 115401, 115405, 115406, 115410, 115411, 115413, 115414, 115415, 115416, 115417, 115418,
                                 115419, 115420, 115421, 115422, 115423, 115424, 115425, 115426, 115428, 115429, 115430,
                                 115431, 115432, 115433, 115434, 115436, 115437, 115438, 115442, 115443, 115446, 115448,
                                 115450, 115451, 115452, 115453, 115454, 115455, 115456, 115457, 115458, 115459, 115460,
                                 115461, 115462, 115464, 115465, 115466, 115467, 115468, 115469, 115470, 115471, 115472,
                                 115474, 115503, 115721, 115742, 115929, 115933, 115938, 115939, 115940, 115941, 115942,
                                 115946, 115947, 115948, 115949, 115950, 115952, 115959, 115965, 115966, 115967, 115982,
                                 115985, 115991, 115992, 115993, 115994, 115995, 115999, 116000, 116001, 116002, 116003,
                                 116005, 116006, 116007, 116008, 116009, 116010, 116011, 116012, 116017, 116018, 116019,
                                 116020, 116027, 116030, 116036, 116037, 116038, 116039, 116040, 116044, 116045, 116046,
                                 116048, 116049, 116050, 116051, 116052, 116053, 116054, 116055, 116056, 116057, 116058,
                                 116059, 116060, 116061, 116062, 116063, 116064, 116065, 116073, 116074, 116075, 116076,
                                 116079, 116081, 116082, 116083, 116085, 116086, 116087, 116088, 116089, 116090, 116091,
                                 116092, 116093, 116094, 116095, 116096, 116097, 116098, 116099, 116100, 116104, 116303,
                                 116331, 116566, 116574, 116575, 116581, 116583, 116587, 116594, 116599, 116602, 116619,
                                 116627, 116628, 116634, 116636, 116639, 116643, 116644, 116645, 116646, 116647, 116652,
                                 116655, 116664, 116672, 116673, 116677, 116681, 116685, 116688, 116689, 116690, 116691,
                                 116692, 116696, 116699, 116701, 116710, 116714, 116718, 116724, 116725, 116726, 116729,
                                 116730, 116731, 116733, 116734, 116896, 117008, 117011, 117197, 117203, 117205, 117206,
                                 117210, 117211, 117213, 117214, 117215, 117216, 117217, 117218, 117224, 117225, 117229,
                                 117230, 117250, 117256, 117258, 117259, 117263, 117264, 117266, 117267, 117268, 117269,
                                 117270, 117271, 117273, 117274, 117275, 117276, 117277, 117278, 117282, 117283, 117295,
                                 117301, 117303, 117304, 117307, 117309, 117311, 117312, 117313, 117314, 117315, 117316,
                                 117318, 117319, 117320, 117321, 117322, 117323, 117326, 117327, 117338, 117340, 117341,
                                 117343, 117344, 117348, 117349, 117350, 117351, 117352, 117354, 117355, 117356, 117357,
                                 117359, 117360, 117361, 117362, 117364, 117389, 117625, 117627, 117835, 117841, 117843,
                                 117847, 117850, 117851, 117852, 117853, 117854, 117859, 117888, 117894, 117896, 117899,
                                 117903, 117904, 117905, 117906, 117907, 117912, 117933, 117937, 117941, 117945, 117948,
                                 117949, 117950, 117951, 117952, 117956, 117970, 117978, 117984, 117985, 117986, 117989,
                                 117990, 117991, 117994, 117997, 118241, 118243, 118591, 118597, 118623, 118628, 118633,
                                 118647, 119221, 119222, 119227, 119228, 119252, 119253, 119258, 119259, 119263, 119265,
                                 119270, 119277, 119806, 119811, 119848, 119851, 119852, 119853, 119856, 119857, 119858,
                                 119878, 119882, 119883, 119888, 119889, 119890, 119892, 119893, 119894, 119895, 119900,
                                 119907, 120383, 120388, 120436, 120437, 120440, 120441, 120475, 120478, 120481, 120482,
                                 120483, 120484, 120485, 120486, 120487, 120488, 120489, 120503, 120508, 120512, 120513,
                                 120518, 120519, 120520, 120521, 120522, 120523, 120524, 120525, 120530, 120537, 121013,
                                 121014, 121017, 121018, 121061, 121066, 121067, 121068, 121069, 121070, 121071, 121072,
                                 121073, 121105, 121108, 121111, 121112, 121113, 121114, 121115, 121116, 121117, 121118,
                                 121119, 121133, 121138, 121142, 121143, 121148, 121149, 121150, 121151, 121152, 121153,
                                 121154, 121155, 121160, 121167, 121372, 121643, 121644, 121645, 121646, 121647, 121648,
                                 121649, 121650, 121691, 121696, 121697, 121698, 121699, 121700, 121701, 121702, 121703,
                                 121735, 121738, 121742, 121743, 121744, 121745, 121746, 121748, 121749, 121763, 121768,
                                 121772, 121779, 121780, 121781, 121782, 121783, 121784, 121785, 121790, 121797, 121840,
                                 122274, 122275, 122276, 122277, 122279, 122280, 122321, 122327, 122328, 122329, 122330,
                                 122332, 122333, 122365, 122373, 122374, 122375, 122376, 122379, 122393, 122398, 122409,
                                 122410, 122411, 122412, 122414, 122420, 122632, 122905, 122906, 122909, 122910, 122958,
                                 122959, 122962, 122963, 123004, 123005, 123009, 123023, 123041, 123042, 123044, 123100,
                                 123594, 123602, 123629, 123631, 123637, 123640, 123644, 123661, 123663, 123668, 123673,
                                 123676, 123679, 123687, 123688, 124105, 124224, 124225, 124231, 124232, 124257, 124259,
                                 124261, 124262, 124267, 124268, 124270, 124271, 124273, 124274, 124287, 124291, 124292,
                                 124293, 124298, 124299, 124303, 124305, 124306, 124307, 124309, 124310, 124311, 124312,
                                 124317, 124318, 124573, 124801, 124806, 124844, 124846, 124851, 124854, 124855, 124856,
                                 124860, 124861, 124862, 124884, 124887, 124888, 124889, 124891, 124892, 124893, 124896,
                                 124897, 124898, 124900, 124901, 124902, 124903, 124904, 124905, 124906, 124907, 124917,
                                 124918, 124921, 124922, 124923, 124928, 124929, 124930, 124931, 124932, 124933, 124934,
                                 124935, 124936, 124937, 124938, 124939, 124940, 124941, 124942, 124947, 124948, 125189,
                                 125365, 125370, 125376, 125423, 125428, 125431, 125432, 125435, 125436, 125470, 125474,
                                 125476, 125477, 125480, 125481, 125484, 125485, 125486, 125487, 125488, 125489, 125490,
                                 125491, 125492, 125507, 125514, 125515, 125517, 125518, 125519, 125521, 125522, 125523,
                                 125524, 125525, 125526, 125527, 125528, 125529, 125530, 125531, 125532, 125533, 125534,
                                 125535, 125536, 125537, 125543, 125547, 125548, 125551, 125552, 125553, 125558, 125559,
                                 125560, 125561, 125562, 125563, 125564, 125565, 125566, 125567, 125568, 125569, 125570,
                                 125571, 125572, 125577, 125578, 125780, 125833, 126000, 126001, 126005, 126006, 126048,
                                 126053, 126054, 126057, 126058, 126061, 126062, 126063, 126064, 126065, 126066, 126067,
                                 126068, 126100, 126101, 126104, 126106, 126107, 126108, 126109, 126110, 126111, 126112,
                                 126113, 126115, 126116, 126117, 126118, 126119, 126120, 126121, 126137, 126144, 126145,
                                 126147, 126148, 126152, 126153, 126154, 126155, 126156, 126158, 126159, 126160, 126161,
                                 126162, 126163, 126164, 126165, 126166, 126167, 126173, 126177, 126178, 126181, 126182,
                                 126188, 126189, 126190, 126191, 126192, 126193, 126194, 126195, 126196, 126197, 126198,
                                 126199, 126200, 126201, 126202, 126207, 126208, 126245, 126449, 126631, 126632, 126633,
                                 126634, 126635, 126637, 126678, 126684, 126685, 126686, 126687, 126689, 126690, 126692,
                                 126693, 126694, 126695, 126697, 126698, 126730, 126731, 126737, 126738, 126739, 126740,
                                 126742, 126743, 126746, 126747, 126748, 126749, 126750, 126767, 126774, 126775, 126783,
                                 126784, 126785, 126786, 126789, 126791, 126792, 126793, 126795, 126796, 126797, 126803,
                                 126807, 126808, 126819, 126820, 126821, 126822, 126824, 126825, 126826, 126827, 126828,
                                 126829, 126830, 126831, 126832, 127040, 127262, 127263, 127264, 127267, 127315, 127316,
                                 127319, 127320, 127323, 127324, 127327, 127328, 127368, 127369, 127372, 127373, 127377,
                                 127378, 127379, 127397, 127414, 127415, 127419, 127422, 127425, 127426, 127427, 127433,
                                 127450, 127451, 127452, 127454, 127457, 127458, 127462, 127505, 127959, 127967, 127995,
                                 128004, 128012, 128013, 128017, 128036, 128039, 128041, 128047, 128050, 128054, 128058,
                                 128064, 128071, 128073, 128074, 128078, 128083, 128086, 128089, 128093, 128095, 128096,
                                 128097, 128098, 128347, 128589, 128590, 128596, 128597, 128623, 128625, 128634, 128635,
                                 128641, 128642, 128643, 128644, 128647, 128649, 128663, 128666, 128667, 128669, 128671,
                                 128672, 128677, 128678, 128680, 128681, 128683, 128684, 128688, 128689, 128690, 128691,
                                 128692, 128694, 128697, 128701, 128702, 128703, 128704, 128708, 128709, 128713, 128715,
                                 128716, 128717, 128718, 128719, 128720, 128721, 128722, 128723, 128724, 128725, 128726,
                                 128727, 128728, 128963, 129158, 129164, 129201, 129211, 129216, 129219, 129220, 129221,
                                 129225, 129226, 129227, 129249, 129253, 129254, 129255, 129256, 129261, 129264, 129265,
                                 129266, 129270, 129271, 129272, 129273, 129274, 129275, 129276, 129277, 129278, 129279,
                                 129280, 129293, 129294, 129296, 129297, 129298, 129299, 129301, 129302, 129303, 129306,
                                 129307, 129308, 129310, 129311, 129312, 129313, 129314, 129315, 129316, 129317, 129318,
                                 129319, 129320, 129321, 129322, 129324, 129327, 129328, 129331, 129332, 129333, 129334,
                                 129338, 129339, 129340, 129341, 129342, 129343, 129344, 129345, 129346, 129347, 129348,
                                 129349, 129350, 129351, 129352, 129353, 129354, 129355, 129356, 129357, 129358, 129556,
                                 129607, 129780, 129786, 129788, 129789, 129793, 129794, 129827, 129831, 129833, 129838,
                                 129841, 129842, 129845, 129846, 129850, 129851, 129852, 129853, 129854, 129855, 129856,
                                 129872, 129879, 129880, 129883, 129884, 129886, 129887, 129890, 129891, 129895, 129896,
                                 129897, 129898, 129899, 129900, 129901, 129903, 129904, 129905, 129906, 129907, 129908,
                                 129909, 129910, 129917, 129923, 129924, 129925, 129926, 129927, 129928, 129932, 129933,
                                 129934, 129935, 129936, 129938, 129939, 129940, 129941, 129942, 129943, 129944, 129945,
                                 129946, 129947, 129948, 129949, 129950, 129951, 129952, 129953, 129954, 129957, 129958,
                                 129961, 129962, 129964, 129968, 129969, 129970, 129971, 129972, 129973, 129974, 129975,
                                 129976, 129977, 129978, 129979, 129980, 129981, 129982, 129983, 129984, 129985, 129986,
                                 129987, 129988, 130012, 130223, 130411, 130415, 130419, 130420, 130421, 130422, 130423,
                                 130425, 130457, 130458, 130464, 130467, 130472, 130473, 130474, 130475, 130477, 130478,
                                 130481, 130482, 130483, 130484, 130485, 130502, 130509, 130510, 130511, 130517, 130518,
                                 130519, 130520, 130522, 130523, 130526, 130527, 130528, 130529, 130530, 130534, 130535,
                                 130536, 130538, 130539, 130540, 130547, 130553, 130554, 130555, 130563, 130564, 130565,
                                 130566, 130569, 130571, 130572, 130573, 130575, 130576, 130577, 130578, 130579, 130580,
                                 130581, 130582, 130583, 130584, 130587, 130588, 130594, 130599, 130600, 130601, 130602,
                                 130604, 130605, 130606, 130607, 130608, 130609, 130610, 130611, 130612, 130613, 130614,
                                 130615, 130616, 130618, 130816, 130822, 131042, 131043, 131044, 131047, 131050, 131051,
                                 131052, 131055, 131095, 131096, 131099, 131100, 131103, 131104, 131107, 131108, 131112,
                                 131113, 131114, 131132, 131148, 131149, 131152, 131153, 131157, 131158, 131159, 131165,
                                 131166, 131168, 131170, 131177, 131194, 131195, 131199, 131202, 131205, 131206, 131207,
                                 131209, 131210, 131211, 131212, 131213, 131230, 131231, 131232, 131234, 131237, 131238,
                                 131241, 131242, 131244, 131245, 131246, 131272, 131290, 131686, 131694, 131722, 131739,
                                 131747, 131748, 131754, 131772, 131775, 131784, 131792, 131793, 131797, 131801, 131805,
                                 131812, 131816, 131819, 131821, 131827, 131830, 131834, 131838, 131844, 131845, 131846,
                                 131849, 131850, 131851, 131853, 131854, 131858, 131863, 131866, 131869, 131873, 131874,
                                 131875, 131876, 131877, 131878, 132100, 132295, 132299, 132316, 132317, 132323, 132324,
                                 132350, 132352, 132369, 132370, 132376, 132377, 132378, 132379, 132383, 132384, 132398,
                                 132402, 132403, 132405, 132414, 132415, 132421, 132422, 132423, 132424, 132427, 132429,
                                 132431, 132432, 132433, 132434, 132435, 132436, 132442, 132443, 132446, 132447, 132449,
                                 132451, 132452, 132457, 132458, 132460, 132461, 132463, 132464, 132468, 132469, 132470,
                                 132471, 132472, 132474, 132475, 132476, 132477, 132479, 132480, 132481, 132482, 132483,
                                 132484, 132488, 132489, 132493, 132495, 132496, 132497, 132498, 132499, 132500, 132501,
                                 132502, 132503, 132504, 132505, 132506, 132507, 132508, 132682, 132763, 132765, 132938,
                                 132944, 132947, 132948, 132952, 132953, 132976, 132980, 132981, 132991, 132996, 133000,
                                 133001, 133005, 133006, 133008, 133009, 133010, 133011, 133012, 133013, 133014, 133015,
                                 133028, 133029, 133032, 133033, 133034, 133036, 133041, 133045, 133046, 133050, 133051,
                                 133053, 133054, 133055, 133056, 133057, 133058, 133059, 133060, 133061, 133062, 133063,
                                 133064, 133065, 133066, 133072, 133073, 133074, 133076, 133077, 133078, 133082, 133083,
                                 133086, 133088, 133090, 133091, 133092, 133093, 133094, 133095, 133096, 133097, 133098,
                                 133099, 133100, 133101, 133102, 133104, 133105, 133106, 133107, 133108, 133109, 133110,
                                 133111, 133112, 133114, 133118, 133119, 133120, 133121, 133122, 133123, 133124, 133125,
                                 133126, 133127, 133128, 133129, 133130, 133131, 133132, 133133, 133134, 133135, 133136,
                                 133137, 133138, 133142, 133360, 133379, 133383, 133569, 133573, 133578, 133579, 133580,
                                 133581, 133582, 133599, 133606, 133607, 133622, 133625, 133631, 133632, 133633, 133634,
                                 133635, 133639, 133640, 133641, 133642, 133643, 133645, 133652, 133658, 133659, 133660,
                                 133667, 133670, 133676, 133677, 133678, 133679, 133680, 133684, 133685, 133686, 133688,
                                 133689, 133690, 133691, 133692, 133693, 133694, 133695, 133696, 133697, 133702, 133703,
                                 133704, 133705, 133713, 133714, 133715, 133716, 133719, 133721, 133722, 133723, 133725,
                                 133726, 133727, 133728, 133729, 133730, 133731, 133732, 133733, 133734, 133735, 133736,
                                 133737, 133738, 133739, 133740, 133744, 133749, 133750, 133751, 133752, 133754, 133755,
                                 133756, 133757, 133758, 133759, 133760, 133761, 133762, 133763, 133764, 133765, 133766,
                                 133768, 133942, 133945, 133970, 134206, 134214, 134215, 134221, 134239, 134242, 134259,
                                 134267, 134268, 134274, 134276, 134279, 134287, 134292, 134295, 134304, 134312, 134313,
                                 134317, 134321, 134325, 134328, 134329, 134330, 134331, 134332, 134336, 134339, 134341,
                                 134347, 134350, 134354, 134358, 134364, 134365, 134366, 134369, 134370, 134371, 134373,
                                 134374, 134378, 134383, 134386, 134389, 134393, 134394, 134395, 134396, 134397, 134398,
                                 134534, 134647, 134649, 134837, 134843, 134845, 134846, 134850, 134851, 134865, 134869,
                                 134870, 134890, 134896, 134898, 134899, 134903, 134904, 134906, 134907, 134908, 134909,
                                 134910, 134911, 134917, 134918, 134922, 134923, 134935, 134941, 134943, 134944, 134947,
                                 134949, 134951, 134952, 134953, 134954, 134955, 134956, 134958, 134959, 134960, 134961,
                                 134962, 134963, 134966, 134967, 134972, 134978, 134980, 134981, 134983, 134984, 134988,
                                 134989, 134990, 134991, 134992, 134994, 134995, 134996, 134997, 134999, 135000, 135001,
                                 135002, 135004, 135008, 135009, 135013, 135015, 135016, 135017, 135018, 135019, 135020,
                                 135021, 135022, 135023, 135024, 135025, 135026, 135027, 135028, 135034, 135263, 135266,
                                 135475, 135481, 135483, 135487, 135494, 135499, 135528, 135534, 135536, 135539, 135543,
                                 135544, 135545, 135546, 135547, 135552, 135573, 135577, 135581, 135585, 135588, 135589,
                                 135590, 135591, 135592, 135596, 135610, 135614, 135618, 135624, 135625, 135626, 135629,
                                 135630, 135631, 135634, 135638, 135646, 135649, 135653, 135654, 135655, 135656, 135657,
                                 135658, 135665, 135880, 135882, 136231, 136237, 136263, 136268, 136273, 136287, 136296,
                                 136301, 136302, 136699, 136861, 136862, 136867, 136868, 136892, 136893, 136898, 136899,
                                 136903, 136905, 136910, 136917, 136926, 136927, 136929, 136930, 136931, 136932, 137172,
                                 137446, 137451, 137488, 137491, 137492, 137493, 137496, 137497, 137498, 137518, 137522,
                                 137523, 137528, 137529, 137530, 137531, 137532, 137533, 137534, 137535, 137540, 137547,
                                 137556, 137557, 137558, 137559, 137560, 137561, 137562, 137788, 137959, 138023, 138028,
                                 138076, 138077, 138080, 138081, 138115, 138118, 138121, 138122, 138123, 138124, 138125,
                                 138126, 138127, 138128, 138129, 138143, 138148, 138152, 138153, 138158, 138159, 138160,
                                 138161, 138162, 138163, 138164, 138165, 138170, 138177, 138186, 138187, 138188, 138189,
                                 138190, 138191, 138192, 138379, 138432, 138653, 138654, 138657, 138658, 138701, 138706,
                                 138707, 138708, 138709, 138710, 138711, 138712, 138713, 138745, 138748, 138752, 138753,
                                 138754, 138755, 138756, 138758, 138759, 138773, 138778, 138782, 138788, 138789, 138790,
                                 138791, 138792, 138793, 138794, 138795, 138800, 138807, 138816, 138817, 138818, 138819,
                                 138820, 138821, 138822, 138843, 139048, 139284, 139285, 139286, 139287, 139289, 139290,
                                 139331, 139337, 139338, 139339, 139340, 139342, 139343, 139375, 139383, 139384, 139385,
                                 139386, 139389, 139403, 139408, 139419, 139420, 139421, 139422, 139424, 139425, 139430,
                                 139446, 139447, 139448, 139449, 139450, 139452, 139639, 139915, 139916, 139919, 139920,
                                 139968, 139969, 139972, 139973, 140014, 140015, 140019, 140033, 140050, 140051, 140052,
                                 140054, 140077, 140078, 140079, 140103, 140604, 140612, 140639, 140641, 140647, 140650,
                                 140654, 140671, 140673, 140678, 140683, 140686, 140689, 140697, 140698, 140706, 140711,
                                 140712, 140713, 140715, 140716, 140946, 141234, 141235, 141241, 141242, 141267, 141269,
                                 141271, 141272, 141277, 141278, 141280, 141281, 141283, 141284, 141297, 141301, 141302,
                                 141303, 141308, 141309, 141313, 141315, 141316, 141317, 141318, 141319, 141320, 141321,
                                 141322, 141327, 141328, 141336, 141337, 141338, 141339, 141340, 141341, 141342, 141343,
                                 141344, 141345, 141346, 141562, 141811, 141816, 141854, 141856, 141861, 141864, 141865,
                                 141866, 141870, 141871, 141872, 141894, 141897, 141898, 141899, 141901, 141902, 141903,
                                 141906, 141907, 141908, 141910, 141911, 141912, 141913, 141914, 141915, 141916, 141917,
                                 141927, 141928, 141931, 141932, 141933, 141938, 141939, 141940, 141941, 141942, 141943,
                                 141944, 141945, 141946, 141947, 141948, 141949, 141950, 141951, 141952, 141957, 141958,
                                 141966, 141967, 141968, 141969, 141970, 141971, 141972, 141973, 141974, 141975, 141976,
                                 142153, 142206, 142380, 142386, 142433, 142438, 142441, 142442, 142445, 142446, 142480,
                                 142484, 142486, 142487, 142490, 142491, 142495, 142496, 142497, 142498, 142499, 142500,
                                 142501, 142517, 142524, 142525, 142527, 142528, 142532, 142533, 142534, 142535, 142536,
                                 142538, 142539, 142540, 142541, 142542, 142543, 142544, 142545, 142546, 142547, 142553,
                                 142557, 142558, 142561, 142562, 142568, 142569, 142570, 142571, 142572, 142573, 142574,
                                 142575, 142576, 142577, 142578, 142579, 142580, 142581, 142582, 142587, 142588, 142596,
                                 142597, 142598, 142599, 142600, 142601, 142602, 142603, 142604, 142605, 142606, 142610,
                                 142822, 143011, 143015, 143064, 143067, 143072, 143073, 143074, 143075, 143077, 143078,
                                 143110, 143111, 143117, 143118, 143119, 143120, 143122, 143123, 143126, 143127, 143128,
                                 143129, 143130, 143147, 143154, 143155, 143163, 143164, 143165, 143166, 143169, 143171,
                                 143172, 143173, 143175, 143176, 143177, 143183, 143187, 143188, 143199, 143200, 143201,
                                 143202, 143204, 143205, 143206, 143207, 143208, 143209, 143210, 143211, 143212, 143218,
                                 143226, 143227, 143228, 143229, 143230, 143231, 143232, 143233, 143234, 143235, 143236,
                                 143413, 143423, 143642, 143643, 143644, 143695, 143696, 143699, 143700, 143703, 143704,
                                 143707, 143708, 143748, 143749, 143752, 143753, 143757, 143758, 143759, 143777, 143794,
                                 143795, 143799, 143802, 143805, 143806, 143807, 143813, 143830, 143831, 143832, 143834,
                                 143837, 143838, 143841, 143842, 143857, 143858, 143859, 143860, 143863, 143864, 143865,
                                 143870, 143892, 144339, 144347, 144375, 144384, 144392, 144393, 144397, 144416, 144419,
                                 144421, 144427, 144430, 144434, 144438, 144444, 144451, 144453, 144454, 144458, 144463,
                                 144466, 144469, 144473, 144474, 144475, 144476, 144477, 144478, 144486, 144491, 144492,
                                 144493, 144494, 144495, 144496, 144698, 144889, 144897, 144969, 144970, 144976, 144977,
                                 145003, 145005, 145014, 145015, 145021, 145022, 145023, 145024, 145027, 145029, 145043,
                                 145046, 145047, 145049, 145051, 145052, 145057, 145058, 145060, 145061, 145063, 145064,
                                 145068, 145069, 145070, 145071, 145072, 145074, 145077, 145081, 145082, 145083, 145084,
                                 145088, 145089, 145093, 145095, 145096, 145097, 145098, 145099, 145100, 145101, 145102,
                                 145103, 145104, 145105, 145106, 145107, 145108, 145116, 145117, 145118, 145119, 145120,
                                 145121, 145122, 145123, 145124, 145125, 145126, 145279, 145362, 145364, 145538, 145544,
                                 145591, 145596, 145600, 145601, 145605, 145606, 145629, 145633, 145634, 145636, 145641,
                                 145645, 145646, 145650, 145651, 145653, 145654, 145655, 145656, 145657, 145658, 145659,
                                 145660, 145673, 145674, 145676, 145677, 145678, 145682, 145683, 145686, 145688, 145690,
                                 145691, 145692, 145693, 145694, 145695, 145696, 145697, 145698, 145699, 145700, 145701,
                                 145702, 145704, 145707, 145708, 145711, 145712, 145714, 145718, 145719, 145720, 145721,
                                 145722, 145723, 145724, 145725, 145726, 145727, 145728, 145729, 145730, 145731, 145732,
                                 145733, 145734, 145735, 145736, 145737, 145738, 145746, 145747, 145748, 145749, 145750,
                                 145751, 145752, 145753, 145754, 145755, 145756, 145761, 145958, 145978, 145982, 146169,
                                 146173, 146222, 146225, 146231, 146232, 146233, 146234, 146235, 146252, 146259, 146260,
                                 146267, 146270, 146276, 146277, 146278, 146279, 146280, 146284, 146285, 146286, 146288,
                                 146289, 146290, 146297, 146303, 146304, 146305, 146313, 146314, 146315, 146316, 146319,
                                 146321, 146322, 146323, 146325, 146326, 146327, 146328, 146329, 146330, 146331, 146332,
                                 146333, 146334, 146337, 146338, 146344, 146349, 146350, 146351, 146352, 146354, 146355,
                                 146356, 146357, 146358, 146359, 146360, 146361, 146362, 146363, 146364, 146365, 146366,
                                 146368, 146376, 146377, 146378, 146379, 146380, 146381, 146382, 146383, 146384, 146385,
                                 146386, 146539, 146569, 146571, 146806, 146814, 146859, 146867, 146868, 146874, 146892,
                                 146895, 146904, 146912, 146913, 146917, 146921, 146925, 146932, 146936, 146939, 146941,
                                 146947, 146950, 146954, 146958, 146964, 146965, 146966, 146969, 146970, 146971, 146973,
                                 146974, 146978, 146983, 146986, 146989, 146993, 146994, 146995, 146996, 146997, 146998,
                                 147006, 147011, 147012, 147013, 147014, 147015, 147016, 147096, 147246, 147248, 147437,
                                 147443, 147490, 147496, 147498, 147499, 147503, 147504, 147518, 147522, 147523, 147535,
                                 147541, 147543, 147544, 147547, 147549, 147551, 147552, 147553, 147554, 147555, 147556,
                                 147562, 147563, 147566, 147567, 147572, 147578, 147580, 147581, 147583, 147584, 147588,
                                 147589, 147590, 147591, 147592, 147594, 147595, 147596, 147597, 147599, 147600, 147601,
                                 147602, 147604, 147608, 147609, 147613, 147615, 147616, 147617, 147618, 147619, 147620,
                                 147621, 147622, 147623, 147624, 147625, 147626, 147627, 147628, 147636, 147637, 147638,
                                 147639, 147640, 147641, 147642, 147643, 147644, 147645, 147646, 147654, 147862, 147865,
                                 148075, 148081, 148128, 148134, 148136, 148139, 148147, 148152, 148173, 148177, 148181,
                                 148185, 148188, 148189, 148190, 148191, 148192, 148196, 148210, 148214, 148218, 148224,
                                 148225, 148226, 148229, 148230, 148231, 148234, 148238, 148243, 148246, 148249, 148253,
                                 148254, 148255, 148256, 148257, 148258, 148266, 148271, 148272, 148273, 148274, 148275,
                                 148276, 148285, 148478, 148481, 148831, 148837, 148863, 148868, 148873, 148887, 148896,
                                 148901, 148902, 148916, 148918, 148919, 149135, 149461, 149462, 149467, 149468, 149492,
                                 149493, 149498, 149499, 149503, 149505, 149510, 149517, 149526, 149527, 149528, 149529,
                                 149530, 149531, 149532, 149546, 149547, 149548, 149549, 149749, 150046, 150051, 150088,
                                 150091, 150092, 150093, 150096, 150097, 150098, 150118, 150122, 150123, 150128, 150129,
                                 150130, 150131, 150132, 150133, 150134, 150135, 150140, 150147, 150156, 150157, 150158,
                                 150159, 150160, 150161, 150162, 150176, 150177, 150178, 150179, 150324, 150395, 150623,
                                 150628, 150676, 150677, 150680, 150681, 150715, 150718, 150722, 150723, 150724, 150725,
                                 150726, 150728, 150729, 150743, 150748, 150752, 150758, 150759, 150760, 150761, 150762,
                                 150763, 150764, 150765, 150770, 150777, 150786, 150787, 150788, 150789, 150790, 150791,
                                 150792, 150806, 150807, 150808, 150809, 150810, 151009, 151254, 151257, 151307, 151308,
                                 151309, 151310, 151312, 151313, 151345, 151353, 151354, 151355, 151356, 151359, 151373,
                                 151378, 151389, 151390, 151391, 151392, 151394, 151395, 151400, 151416, 151417, 151418,
                                 151419, 151420, 151421, 151422, 151436, 151437, 151438, 151439, 151584, 151612, 151885,
                                 151886, 151890, 151938, 151939, 151942, 151943, 151984, 151985, 151989, 152003, 152020,
                                 152021, 152022, 152024, 152047, 152048, 152049, 152050, 152066, 152067, 152069, 152070,
                                 152080, 152574, 152582, 152609, 152611, 152617, 152620, 152624, 152641, 152643, 152648,
                                 152653, 152656, 152659, 152667, 152668, 152676, 152681, 152682, 152683, 152684, 152685,
                                 152686, 152696, 152697, 152698, 152699, 152887, 153085, 153204, 153205, 153211, 153212,
                                 153237, 153239, 153241, 153242, 153247, 153248, 153250, 153251, 153253, 153254, 153267,
                                 153271, 153272, 153273, 153278, 153279, 153283, 153285, 153286, 153287, 153288, 153289,
                                 153290, 153291, 153292, 153297, 153298, 153306, 153307, 153308, 153309, 153310, 153311,
                                 153312, 153313, 153314, 153315, 153316, 153326, 153327, 153328, 153329, 153465, 153553,
                                 153781, 153786, 153826, 153831, 153835, 153836, 153840, 153841, 153864, 153867, 153868,
                                 153872, 153873, 153876, 153878, 153880, 153881, 153882, 153883, 153884, 153885, 153886,
                                 153887, 153897, 153898, 153901, 153902, 153908, 153909, 153910, 153911, 153912, 153913,
                                 153914, 153915, 153916, 153917, 153918, 153919, 153920, 153921, 153922, 153927, 153928,
                                 153936, 153937, 153938, 153939, 153940, 153941, 153942, 153943, 153944, 153945, 153946,
                                 153956, 153957, 153958, 153959, 153964, 154147, 154169, 154412, 154415, 154457, 154460,
                                 154466, 154467, 154468, 154469, 154470, 154487, 154494, 154495, 154503, 154504, 154505,
                                 154506, 154509, 154511, 154512, 154513, 154515, 154516, 154517, 154523, 154527, 154528,
                                 154539, 154540, 154541, 154542, 154544, 154545, 154546, 154547, 154548, 154549, 154550,
                                 154551, 154552, 154558, 154566, 154567, 154568, 154569, 154570, 154571, 154572, 154573,
                                 154574, 154575, 154576, 154586, 154587, 154588, 154589, 154725, 154760, 155049, 155057,
                                 155094, 155102, 155103, 155107, 155126, 155129, 155131, 155137, 155140, 155144, 155148,
                                 155154, 155161, 155163, 155164, 155168, 155173, 155176, 155179, 155183, 155184, 155185,
                                 155186, 155187, 155188, 155196, 155201, 155202, 155203, 155204, 155205, 155206, 155216,
                                 155217, 155218, 155219, 155285, 155435, 155437, 155680, 155686, 155725, 155731, 155733,
                                 155734, 155737, 155739, 155753, 155756, 155757, 155762, 155768, 155770, 155771, 155773,
                                 155774, 155778, 155779, 155780, 155781, 155782, 155784, 155787, 155791, 155792, 155794,
                                 155798, 155799, 155803, 155805, 155806, 155807, 155808, 155809, 155810, 155811, 155812,
                                 155813, 155814, 155815, 155816, 155817, 155818, 155826, 155827, 155828, 155829, 155830,
                                 155831, 155832, 155833, 155834, 155835, 155836, 155846, 155847, 155848, 155849, 155857,
                                 156049, 156053, 156318, 156324, 156363, 156367, 156371, 156375, 156382, 156386, 156400,
                                 156404, 156408, 156414, 156415, 156416, 156419, 156420, 156421, 156424, 156428, 156433,
                                 156436, 156439, 156443, 156444, 156445, 156446, 156447, 156448, 156456, 156461, 156462,
                                 156463, 156464, 156465, 156466, 156476, 156477, 156478, 156479, 156488, 156667, 156670,
                                 157021, 157027, 157053, 157058, 157063, 157077, 157086, 157091, 157092, 157106, 157107,
                                 157108, 157109, 157294, 157489, 157651, 157652, 157657, 157658, 157682, 157683, 157688,
                                 157689, 157693, 157695, 157700, 157707, 157716, 157717, 157718, 157719, 157720, 157721,
                                 157722, 157736, 157737, 157738, 157739, 157849, 157962, 158236, 158241, 158282, 158283,
                                 158286, 158288, 158308, 158312, 158318, 158319, 158320, 158321, 158322, 158323, 158324,
                                 158325, 158330, 158337, 158346, 158347, 158348, 158349, 158350, 158351, 158352, 158366,
                                 158367, 158368, 158369, 158381, 158554, 158578, 158867, 158870, 158913, 158914, 158915,
                                 158916, 158919, 158933, 158938, 158949, 158950, 158951, 158952, 158954, 158955, 158960,
                                 158976, 158977, 158978, 158979, 158980, 158981, 158982, 158996, 158997, 158998, 158999,
                                 159109, 159169, 159504, 159512, 159541, 159547, 159550, 159554, 159571, 159573, 159578,
                                 159583, 159586, 159589, 159597, 159598, 159606, 159611, 159612, 159613, 159614, 159615,
                                 159616, 159626, 159627, 159628, 159629, 159643, 159846, 160135, 160141, 160172, 160178,
                                 160180, 160181, 160183, 160184, 160197, 160201, 160202, 160208, 160209, 160213, 160215,
                                 160216, 160217, 160218, 160219, 160220, 160221, 160222, 160227, 160228, 160236, 160237,
                                 160238, 160239, 160240, 160241, 160242, 160243, 160244, 160245, 160246, 160256, 160257,
                                 160258, 160259, 160274, 160462, 160773, 160777, 160810, 160814, 160818, 160824, 160831,
                                 160834, 160838, 160843, 160846, 160849, 160853, 160854, 160855, 160856, 160857, 160858,
                                 160866, 160871, 160872, 160873, 160874, 160875, 160876, 160886, 160887, 160888, 160889,
                                 160905, 161074, 161078, 161431, 161437, 161468, 161473, 161487, 161496, 161501, 161502,
                                 161516, 161517, 161518, 161519, 161536, 161735, 162062, 162068, 162098, 162099, 162103,
                                 162105, 162110, 162117, 162126, 162127, 162128, 162129, 162130, 162131, 162132, 162146,
                                 162147, 162148, 162149, 162167, 162349, 162700, 162704, 162728, 162733, 162736, 162739,
                                 162747, 162748, 162756, 162761, 162762, 162763, 162764, 162765, 162766, 162776, 162777,
                                 162778, 162779, 162798, 162967, 163358, 163363, 163386, 163391, 163392, 163406, 163407,
                                 163408, 163409, 163429, 163594, 164060, 164067, 164091, 164690, 164691, 164696, 164697,
                                 164721, 164725, 165320, 165321, 165322, 165323, 165325, 165326, 165327, 165351, 165355,
                                 165950, 165951, 165952, 165953, 165954, 165955, 165956, 165957, 165981, 165985, 166580,
                                 166581, 166582, 166583, 166584, 166585, 166586, 166587, 166611, 166615, 167211, 167212,
                                 167213, 167214, 167215, 167216, 167217, 167241, 167245, 167841, 167842, 167843, 167844,
                                 167845, 167875, 168473, 168474, 168475, 169108, 169115, 169116, 169738, 169739, 169744,
                                 169745, 169746, 169766, 170360, 170367, 170368, 170369, 170370, 170374, 170375, 170376,
                                 170391, 170396, 170398, 170498, 170990, 170991, 170996, 170997, 170998, 170999, 171000,
                                 171001, 171002, 171004, 171005, 171006, 171014, 171021, 171025, 171026, 171028, 171128,
                                 171620, 171621, 171622, 171623, 171624, 171625, 171626, 171627, 171628, 171629, 171630,
                                 171631, 171632, 171633, 171634, 171635, 171636, 171644, 171651, 171655, 171656, 171658,
                                 171758, 172250, 172251, 172252, 172253, 172254, 172255, 172256, 172257, 172258, 172259,
                                 172260, 172261, 172262, 172263, 172264, 172265, 172274, 172281, 172285, 172286, 172288,
                                 172388, 172880, 172881, 172882, 172883, 172884, 172885, 172886, 172887, 172890, 172891,
                                 172892, 172893, 172894, 172904, 172911, 172915, 172916, 172918, 173018, 173511, 173512,
                                 173513, 173514, 173515, 173516, 173520, 173521, 173522, 173523, 173534, 173545, 173548,
                                 174142, 174143, 174144, 174145, 174151, 174152, 174153, 174778, 174785, 174786, 174787,
                                 174792, 174850, 175408, 175409, 175414, 175415, 175416, 175417, 175418, 175420, 175422,
                                 175423, 175436, 175480, 176030, 176037, 176038, 176039, 176040, 176044, 176045, 176046,
                                 176047, 176048, 176049, 176050, 176052, 176053, 176061, 176066, 176067, 176068, 176072,
                                 176110, 176168, 176660, 176661, 176666, 176667, 176668, 176669, 176670, 176671, 176672,
                                 176673, 176674, 176675, 176676, 176677, 176678, 176679, 176680, 176681, 176682, 176683,
                                 176684, 176691, 176695, 176696, 176697, 176698, 176702, 176740, 176798, 177290, 177291,
                                 177292, 177293, 177294, 177295, 177296, 177297, 177298, 177299, 177300, 177301, 177302,
                                 177303, 177304, 177305, 177306, 177307, 177308, 177309, 177310, 177311, 177312, 177313,
                                 177314, 177321, 177325, 177326, 177327, 177328, 177332, 177370, 177428, 177920, 177921,
                                 177922, 177923, 177924, 177925, 177926, 177927, 177929, 177930, 177931, 177932, 177933,
                                 177934, 177938, 177939, 177940, 177941, 177942, 177943, 177944, 177951, 177955, 177956,
                                 177957, 177958, 177962, 178000, 178058, 178551, 178552, 178553, 178554, 178555, 178556,
                                 178560, 178561, 178562, 178563, 178568, 178569, 178571, 178573, 178574, 178585, 178587,
                                 178588, 178592, 178688, 179182, 179183, 179184, 179185, 179191, 179192, 179193, 179199,
                                 179201, 179204, 179217, 179818, 179825, 179826, 179827, 179832, 179835, 179840, 179844,
                                 179890, 180178, 180448, 180449, 180454, 180455, 180456, 180457, 180458, 180460, 180462,
                                 180463, 180465, 180466, 180467, 180469, 180470, 180474, 180476, 180520, 180798, 181070,
                                 181077, 181078, 181079, 181080, 181084, 181085, 181086, 181087, 181088, 181089, 181090,
                                 181091, 181092, 181093, 181095, 181096, 181097, 181098, 181099, 181100, 181101, 181104,
                                 181106, 181107, 181108, 181112, 181150, 181208, 181374, 181438, 181700, 181701, 181706,
                                 181707, 181708, 181709, 181710, 181711, 181712, 181713, 181714, 181715, 181716, 181717,
                                 181718, 181719, 181720, 181721, 181722, 181723, 181724, 181725, 181726, 181727, 181728,
                                 181729, 181730, 181731, 181734, 181735, 181736, 181737, 181738, 181742, 181780, 181838,
                                 181930, 182058, 182330, 182331, 182332, 182333, 182334, 182335, 182336, 182337, 182339,
                                 182340, 182341, 182342, 182343, 182344, 182347, 182348, 182349, 182350, 182351, 182352,
                                 182353, 182354, 182355, 182356, 182357, 182358, 182359, 182360, 182361, 182364, 182365,
                                 182366, 182367, 182368, 182372, 182410, 182442, 182468, 182634, 182961, 182962, 182963,
                                 182964, 182965, 182966, 182970, 182971, 182972, 182973, 182978, 182979, 182980, 182981,
                                 182983, 182984, 182985, 182986, 182987, 182988, 182989, 182994, 182995, 182997, 182998,
                                 183002, 183098, 183190, 183592, 183593, 183594, 183595, 183601, 183602, 183603, 183609,
                                 183611, 183614, 183616, 183618, 183619, 183627, 183632, 183702, 184228, 184235, 184236,
                                 184237, 184242, 184245, 184250, 184252, 184253, 184254, 184259, 184300, 184558, 184858,
                                 184859, 184864, 184865, 184866, 184867, 184868, 184870, 184872, 184873, 184875, 184876,
                                 184877, 184878, 184879, 184880, 184882, 184883, 184884, 184886, 184889, 184930, 184990,
                                 185140, 185480, 185487, 185488, 185489, 185490, 185494, 185495, 185496, 185497, 185498,
                                 185499, 185500, 185501, 185502, 185503, 185505, 185506, 185507, 185508, 185509, 185510,
                                 185511, 185512, 185513, 185514, 185516, 185517, 185518, 185519, 185522, 185560, 185618,
                                 185620, 185714, 185818, 186110, 186111, 186116, 186117, 186119, 186120, 186121, 186122,
                                 186123, 186124, 186127, 186128, 186129, 186130, 186131, 186132, 186133, 186134, 186135,
                                 186136, 186137, 186138, 186139, 186140, 186141, 186142, 186143, 186144, 186145, 186146,
                                 186147, 186148, 186149, 186151, 186152, 186190, 186248, 186250, 186400, 186741, 186742,
                                 186743, 186744, 186745, 186746, 186750, 186751, 186752, 186753, 186758, 186759, 186760,
                                 186761, 186763, 186764, 186765, 186766, 186767, 186768, 186769, 186770, 186772, 186773,
                                 186774, 186775, 186777, 186778, 186779, 186782, 186878, 186880, 186974, 187372, 187373,
                                 187374, 187375, 187381, 187382, 187383, 187389, 187391, 187394, 187396, 187397, 187398,
                                 187399, 187402, 187407, 187409, 187411, 187412, 187510, 188008, 188015, 188016, 188017,
                                 188022, 188025, 188030, 188032, 188033, 188034, 188039, 188080, 188140, 188247, 188368,
                                 188638, 188639, 188644, 188645, 188646, 188647, 188648, 188650, 188652, 188653, 188655,
                                 188656, 188657, 188658, 188659, 188660, 188662, 188663, 188664, 188666, 188669, 188710,
                                 188770, 188814, 188988, 189260, 189267, 189269, 189270, 189274, 189277, 189278, 189279,
                                 189280, 189281, 189282, 189283, 189285, 189286, 189287, 189288, 189289, 189290, 189292,
                                 189293, 189294, 189296, 189297, 189298, 189299, 189300, 189302, 189340, 189398, 189400,
                                 189507, 189564, 189891, 189896, 189900, 189901, 189902, 189903, 189908, 189909, 189910,
                                 189911, 189913, 189914, 189915, 189916, 189917, 189918, 189919, 189920, 189922, 189923,
                                 189924, 189927, 189928, 189929, 189932, 190028, 190030, 190074, 190120, 190535, 190536,
                                 190537, 190542, 190545, 190550, 190552, 190553, 190554, 190559, 190600, 190660, 190683,
                                 190858, 191159, 191164, 191167, 191168, 191170, 191172, 191173, 191175, 191176, 191177,
                                 191178, 191179, 191180, 191182, 191183, 191184, 191189, 191193, 191230, 191290, 191440,
                                 191797, 191802, 191805, 191810, 191812, 191813, 191814, 191819, 191824, 191920, 191954,
                                 192027, 192379, 192429, 192431, 192436, 192437, 192438, 192439, 192442, 192443, 192449,
                                 192450, 192452, 192454, 192550, 192966, 193072, 193073, 193084, 193180, 193717, 193724,
                                 193802, 194347, 194348, 194354, 194411, 194414, 194432, 194930, 194937, 194961, 194977,
                                 194978, 194979, 194982, 194984, 195000, 195041, 195044, 195062, 195560, 195561, 195566,
                                 195567, 195591, 195595, 195607, 195608, 195609, 195610, 195612, 195613, 195614, 195615,
                                 195630, 195671, 195674, 195692, 196190, 196191, 196192, 196193, 196194, 196195, 196196,
                                 196197, 196221, 196225, 196237, 196238, 196239, 196240, 196241, 196242, 196243, 196244,
                                 196245, 196260, 196301, 196304, 196322, 196820, 196821, 196822, 196823, 196824, 196825,
                                 196826, 196827, 196851, 196855, 196867, 196868, 196869, 196870, 196871, 196872, 196873,
                                 196874, 196875, 196890, 196931, 196934, 197450, 197451, 197452, 197453, 197454, 197455,
                                 197456, 197457, 197481, 197485, 197498, 197499, 197500, 197501, 197502, 197503, 197505,
                                 197520, 197564, 198081, 198082, 198083, 198084, 198085, 198086, 198115, 198130, 198131,
                                 198132, 198133, 198135, 198150, 198712, 198713, 198714, 198715, 198760, 198761, 198765,
                                 199348, 199355, 199356, 199387, 199394, 199396, 199400, 199418, 199472, 199978, 199979,
                                 199984, 199985, 199986, 200006, 200017, 200018, 200024, 200026, 200027, 200028, 200030,
                                 200033, 200048, 200081, 200084, 200102, 200600, 200607, 200608, 200609, 200610, 200614,
                                 200615, 200616, 200631, 200636, 200638, 200647, 200648, 200649, 200652, 200654, 200656,
                                 200657, 200658, 200659, 200660, 200662, 200663, 200670, 200675, 200678, 200711, 200714,
                                 200732, 200738, 201230, 201231, 201236, 201237, 201238, 201239, 201240, 201241, 201242,
                                 201243, 201244, 201245, 201246, 201254, 201261, 201265, 201266, 201268, 201277, 201278,
                                 201279, 201280, 201281, 201282, 201283, 201284, 201285, 201286, 201287, 201288, 201289,
                                 201290, 201291, 201292, 201293, 201300, 201305, 201308, 201341, 201344, 201362, 201368,
                                 201860, 201861, 201862, 201863, 201864, 201865, 201866, 201867, 201868, 201869, 201870,
                                 201871, 201872, 201873, 201874, 201875, 201876, 201884, 201891, 201895, 201896, 201898,
                                 201907, 201908, 201909, 201910, 201911, 201912, 201913, 201914, 201915, 201916, 201917,
                                 201918, 201919, 201920, 201921, 201922, 201923, 201930, 201935, 201938, 201971, 201974,
                                 201992, 201998, 202113, 202490, 202491, 202492, 202493, 202494, 202495, 202496, 202497,
                                 202499, 202500, 202501, 202502, 202503, 202504, 202514, 202521, 202525, 202526, 202528,
                                 202538, 202539, 202540, 202541, 202542, 202543, 202545, 202547, 202548, 202549, 202550,
                                 202551, 202552, 202553, 202560, 202565, 202568, 202601, 202604, 202627, 202628, 203121,
                                 203122, 203123, 203124, 203125, 203126, 203130, 203131, 203132, 203133, 203144, 203155,
                                 203158, 203169, 203170, 203171, 203172, 203173, 203175, 203178, 203179, 203181, 203182,
                                 203183, 203190, 203195, 203258, 203373, 203752, 203753, 203754, 203755, 203761, 203762,
                                 203763, 203774, 203800, 203801, 203803, 203805, 203809, 203811, 203825, 203887, 204388,
                                 204395, 204396, 204397, 204402, 204427, 204434, 204436, 204440, 204444, 204448, 204449,
                                 204458, 204460, 204512, 204746, 205018, 205019, 205024, 205025, 205026, 205027, 205028,
                                 205030, 205032, 205033, 205046, 205057, 205058, 205064, 205066, 205067, 205068, 205070,
                                 205073, 205074, 205075, 205076, 205077, 205078, 205079, 205088, 205090, 205121, 205124,
                                 205142, 205366, 205640, 205647, 205648, 205649, 205650, 205654, 205655, 205656, 205657,
                                 205658, 205659, 205660, 205661, 205662, 205663, 205671, 205676, 205677, 205678, 205682,
                                 205687, 205688, 205689, 205692, 205694, 205696, 205697, 205698, 205699, 205700, 205701,
                                 205702, 205703, 205704, 205705, 205706, 205707, 205708, 205709, 205710, 205715, 205717,
                                 205718, 205720, 205751, 205754, 205772, 205778, 205943, 206006, 206270, 206271, 206276,
                                 206277, 206278, 206279, 206280, 206281, 206282, 206283, 206284, 206285, 206286, 206287,
                                 206288, 206289, 206290, 206291, 206292, 206293, 206294, 206301, 206305, 206306, 206307,
                                 206308, 206312, 206317, 206318, 206319, 206320, 206321, 206322, 206323, 206324, 206325,
                                 206326, 206327, 206328, 206329, 206330, 206331, 206332, 206333, 206334, 206335, 206336,
                                 206337, 206338, 206339, 206340, 206345, 206347, 206348, 206350, 206381, 206384, 206402,
                                 206408, 206528, 206626, 206900, 206901, 206902, 206903, 206904, 206905, 206906, 206907,
                                 206909, 206910, 206911, 206912, 206913, 206914, 206917, 206918, 206919, 206920, 206921,
                                 206922, 206923, 206924, 206931, 206935, 206936, 206937, 206938, 206942, 206948, 206949,
                                 206950, 206951, 206952, 206953, 206955, 206956, 206957, 206958, 206959, 206960, 206961,
                                 206962, 206963, 206964, 206965, 206966, 206967, 206968, 206969, 206970, 206975, 206977,
                                 206978, 206980, 206981, 207011, 207014, 207038, 207203, 207531, 207532, 207533, 207534,
                                 207535, 207536, 207540, 207541, 207542, 207543, 207548, 207549, 207550, 207551, 207553,
                                 207554, 207565, 207567, 207568, 207572, 207579, 207580, 207581, 207582, 207583, 207585,
                                 207587, 207588, 207589, 207591, 207592, 207593, 207595, 207596, 207597, 207598, 207599,
                                 207600, 207605, 207607, 207668, 207788, 208162, 208163, 208164, 208165, 208171, 208172,
                                 208173, 208179, 208181, 208184, 208197, 208202, 208210, 208211, 208213, 208215, 208219,
                                 208221, 208222, 208226, 208227, 208235, 208237, 208241, 208798, 208805, 208806, 208807,
                                 208812, 208815, 208820, 208824, 208837, 208844, 208846, 208850, 208854, 208858, 208859,
                                 208861, 208863, 208864, 208868, 208870, 208922, 209124, 209428, 209429, 209434, 209435,
                                 209436, 209437, 209438, 209440, 209442, 209443, 209445, 209446, 209447, 209448, 209449,
                                 209450, 209454, 209456, 209467, 209468, 209474, 209476, 209477, 209478, 209480, 209483,
                                 209484, 209485, 209486, 209487, 209488, 209489, 209491, 209492, 209493, 209494, 209497,
                                 209498, 209500, 209531, 209534, 209552, 209708, 210050, 210057, 210058, 210059, 210060,
                                 210064, 210065, 210066, 210067, 210068, 210069, 210070, 210071, 210072, 210073, 210075,
                                 210076, 210077, 210078, 210079, 210080, 210081, 210084, 210086, 210087, 210088, 210092,
                                 210097, 210098, 210099, 210102, 210104, 210106, 210107, 210108, 210109, 210110, 210111,
                                 210112, 210113, 210114, 210115, 210116, 210117, 210118, 210119, 210120, 210121, 210122,
                                 210123, 210124, 210125, 210127, 210128, 210130, 210161, 210164, 210182, 210188, 210287,
                                 210384, 210680, 210681, 210686, 210687, 210689, 210690, 210691, 210692, 210693, 210694,
                                 210697, 210698, 210699, 210700, 210701, 210702, 210703, 210704, 210705, 210706, 210707,
                                 210708, 210709, 210710, 210711, 210714, 210715, 210716, 210717, 210718, 210722, 210728,
                                 210729, 210730, 210731, 210732, 210733, 210735, 210736, 210737, 210738, 210739, 210740,
                                 210741, 210742, 210743, 210744, 210745, 210746, 210747, 210748, 210749, 210750, 210751,
                                 210752, 210753, 210754, 210755, 210756, 210757, 210758, 210760, 210791, 210794, 210818,
                                 210968, 211311, 211312, 211313, 211314, 211315, 211316, 211320, 211321, 211322, 211323,
                                 211328, 211329, 211330, 211331, 211333, 211334, 211335, 211336, 211337, 211338, 211339,
                                 211340, 211344, 211345, 211347, 211348, 211352, 211359, 211360, 211361, 211362, 211363,
                                 211365, 211367, 211368, 211369, 211371, 211372, 211373, 211374, 211375, 211376, 211377,
                                 211378, 211379, 211380, 211381, 211382, 211383, 211384, 211385, 211387, 211448, 211547,
                                 211563, 211942, 211943, 211944, 211945, 211951, 211952, 211953, 211959, 211961, 211964,
                                 211966, 211967, 211968, 211969, 211977, 211982, 211990, 211991, 211993, 211995, 211999,
                                 212001, 212002, 212005, 212006, 212007, 212012, 212013, 212014, 212015, 212016, 212017,
                                 212077, 212578, 212585, 212586, 212587, 212592, 212595, 212600, 212602, 212603, 212604,
                                 212609, 212617, 212624, 212626, 212630, 212634, 212638, 212639, 212641, 212642, 212643,
                                 212644, 212648, 212650, 212702, 212710, 212837, 212936, 213208, 213209, 213214, 213215,
                                 213216, 213217, 213218, 213220, 213222, 213223, 213225, 213226, 213227, 213228, 213229,
                                 213230, 213232, 213233, 213234, 213236, 213239, 213247, 213248, 213254, 213256, 213257,
                                 213258, 213260, 213263, 213264, 213265, 213266, 213267, 213268, 213269, 213271, 213272,
                                 213273, 213274, 213277, 213278, 213280, 213311, 213314, 213332, 213340, 213381, 213556,
                                 213830, 213837, 213839, 213840, 213844, 213847, 213848, 213849, 213850, 213851, 213852,
                                 213853, 213855, 213856, 213857, 213858, 213859, 213860, 213861, 213862, 213863, 213864,
                                 213866, 213867, 213868, 213869, 213872, 213878, 213879, 213882, 213886, 213887, 213888,
                                 213889, 213890, 213891, 213892, 213893, 213894, 213895, 213896, 213897, 213898, 213899,
                                 213900, 213901, 213902, 213903, 213904, 213905, 213907, 213908, 213909, 213910, 213941,
                                 213944, 213968, 213970, 214097, 214133, 214461, 214466, 214470, 214471, 214472, 214473,
                                 214478, 214479, 214480, 214481, 214483, 214484, 214485, 214486, 214487, 214488, 214489,
                                 214490, 214492, 214493, 214494, 214495, 214497, 214498, 214499, 214502, 214509, 214510,
                                 214511, 214512, 214513, 214515, 214517, 214518, 214519, 214521, 214522, 214523, 214524,
                                 214525, 214526, 214527, 214528, 214529, 214530, 214531, 214532, 214533, 214534, 214535,
                                 214537, 214598, 214600, 214641, 214718, 215098, 215105, 215106, 215107, 215112, 215115,
                                 215120, 215122, 215123, 215124, 215129, 215144, 215146, 215150, 215154, 215158, 215159,
                                 215161, 215162, 215163, 215164, 215168, 215170, 215222, 215230, 215250, 215291, 215424,
                                 215729, 215734, 215737, 215738, 215740, 215742, 215743, 215745, 215746, 215747, 215748,
                                 215749, 215750, 215752, 215753, 215754, 215756, 215759, 215768, 215776, 215777, 215778,
                                 215780, 215783, 215784, 215785, 215786, 215787, 215788, 215789, 215791, 215792, 215793,
                                 215794, 215797, 215798, 215800, 215802, 215831, 215860, 215907, 216008, 216360, 216368,
                                 216369, 216370, 216371, 216373, 216375, 216376, 216377, 216378, 216379, 216380, 216382,
                                 216383, 216384, 216387, 216388, 216389, 216392, 216399, 216402, 216407, 216408, 216409,
                                 216411, 216412, 216413, 216414, 216415, 216416, 216417, 216418, 216419, 216421, 216422,
                                 216423, 216424, 216425, 216427, 216433, 216488, 216490, 216510, 216587, 216962, 216991,
                                 216992, 216993, 216999, 217001, 217004, 217006, 217007, 217008, 217009, 217012, 217013,
                                 217017, 217019, 217022, 217031, 217033, 217035, 217039, 217041, 217042, 217045, 217046,
                                 217047, 217051, 217052, 217053, 217054, 217055, 217056, 217057, 217062, 217120, 217627,
                                 217632, 217635, 217640, 217642, 217643, 217644, 217649, 217666, 217670, 217674, 217678,
                                 217679, 217681, 217682, 217683, 217684, 217690, 217695, 217750, 217783, 217877, 218208,
                                 218259, 218261, 218266, 218267, 218268, 218269, 218272, 218273, 218277, 218279, 218282,
                                 218301, 218302, 218305, 218306, 218307, 218311, 218312, 218313, 218314, 218315, 218317,
                                 218319, 218325, 218380, 218895, 218900, 218902, 218903, 218904, 218909, 218934, 218938,
                                 218941, 218942, 218943, 218944, 218957, 219010, 219030, 219451, 219526, 219527, 219528,
                                 219529, 219532, 219533, 219539, 219565, 219567, 219571, 219572, 219573, 219574, 219577,
                                 219582, 219640, 220055, 220162, 220163, 220169, 220201, 220202, 220204, 220215, 220270,
                                 220807, 220814, 220850, 220851, 220863, 220892, 221437, 221438, 221444, 221480, 221481,
                                 221485, 221487, 221493, 221501, 221504, 221522, 221529, 222020, 222027, 222051, 222067,
                                 222068, 222069, 222072, 222074, 222090, 222110, 222111, 222112, 222113, 222114, 222115,
                                 222117, 222123, 222131, 222134, 222152, 222159, 222650, 222651, 222656, 222657, 222681,
                                 222685, 222697, 222698, 222699, 222700, 222701, 222702, 222703, 222704, 222705, 222720,
                                 222740, 222741, 222742, 222743, 222744, 222745, 222746, 222747, 222753, 222761, 222764,
                                 222782, 222789, 223280, 223281, 223282, 223283, 223284, 223285, 223286, 223287, 223311,
                                 223315, 223327, 223328, 223329, 223330, 223331, 223332, 223333, 223334, 223335, 223350,
                                 223370, 223371, 223372, 223373, 223374, 223375, 223376, 223377, 223383, 223391, 223394,
                                 223412, 223419, 223543, 223910, 223911, 223912, 223913, 223914, 223915, 223916, 223917,
                                 223941, 223945, 223958, 223959, 223960, 223961, 223962, 223963, 223965, 223980, 224000,
                                 224002, 224003, 224004, 224005, 224006, 224007, 224013, 224021, 224024, 224026, 224049,
                                 224541, 224542, 224543, 224544, 224545, 224546, 224575, 224589, 224590, 224591, 224592,
                                 224593, 224595, 224610, 224632, 224633, 224634, 224635, 224636, 224637, 224803, 225172,
                                 225173, 225174, 225175, 225220, 225221, 225223, 225225, 225262, 225264, 225266, 225286,
                                 225808, 225815, 225816, 225847, 225854, 225856, 225860, 225878, 225890, 225891, 225898,
                                 225899, 225902, 225903, 225932, 226164, 226438, 226439, 226444, 226445, 226446, 226466,
                                 226477, 226478, 226484, 226486, 226487, 226488, 226490, 226493, 226508, 226520, 226521,
                                 226525, 226527, 226528, 226529, 226531, 226532, 226533, 226534, 226541, 226544, 226562,
                                 226569, 226572, 226784, 227060, 227067, 227068, 227069, 227070, 227074, 227075, 227076,
                                 227091, 227096, 227098, 227107, 227108, 227109, 227112, 227114, 227116, 227117, 227118,
                                 227119, 227120, 227121, 227122, 227123, 227130, 227135, 227138, 227150, 227151, 227152,
                                 227153, 227154, 227155, 227156, 227157, 227158, 227159, 227160, 227161, 227162, 227163,
                                 227164, 227171, 227174, 227192, 227198, 227199, 227202, 227361, 227424, 227690, 227691,
                                 227696, 227697, 227698, 227699, 227700, 227701, 227702, 227703, 227704, 227705, 227706,
                                 227714, 227721, 227725, 227726, 227728, 227737, 227738, 227739, 227740, 227741, 227742,
                                 227743, 227744, 227745, 227746, 227747, 227748, 227749, 227750, 227751, 227752, 227753,
                                 227760, 227765, 227768, 227780, 227781, 227782, 227783, 227784, 227785, 227786, 227787,
                                 227788, 227789, 227790, 227791, 227792, 227793, 227794, 227801, 227804, 227822, 227828,
                                 227829, 227832, 227949, 228044, 228320, 228321, 228322, 228323, 228324, 228325, 228326,
                                 228327, 228329, 228330, 228331, 228332, 228333, 228334, 228344, 228351, 228355, 228356,
                                 228358, 228368, 228369, 228370, 228371, 228372, 228373, 228375, 228376, 228377, 228378,
                                 228379, 228380, 228381, 228382, 228383, 228390, 228395, 228398, 228410, 228411, 228412,
                                 228413, 228414, 228415, 228416, 228417, 228418, 228419, 228420, 228421, 228422, 228423,
                                 228424, 228430, 228431, 228434, 228458, 228459, 228462, 228621, 228951, 228952, 228953,
                                 228954, 228955, 228956, 228960, 228961, 228962, 228963, 228974, 228985, 228988, 228999,
                                 229000, 229001, 229002, 229003, 229005, 229007, 229008, 229009, 229011, 229012, 229013,
                                 229020, 229025, 229042, 229043, 229044, 229045, 229046, 229047, 229048, 229049, 229050,
                                 229051, 229054, 229088, 229089, 229092, 229209, 229582, 229583, 229584, 229585, 229591,
                                 229592, 229593, 229604, 229630, 229631, 229633, 229635, 229639, 229641, 229642, 229655,
                                 229672, 229673, 229674, 229676, 229680, 229684, 229690, 229722, 230218, 230225, 230226,
                                 230227, 230232, 230257, 230264, 230266, 230270, 230274, 230278, 230279, 230288, 230290,
                                 230300, 230301, 230308, 230309, 230312, 230313, 230315, 230317, 230319, 230342, 230539,
                                 230848, 230849, 230854, 230855, 230856, 230857, 230858, 230860, 230862, 230863, 230876,
                                 230887, 230888, 230894, 230896, 230897, 230898, 230900, 230903, 230904, 230905, 230906,
                                 230907, 230908, 230909, 230917, 230918, 230920, 230930, 230931, 230935, 230937, 230938,
                                 230939, 230940, 230941, 230942, 230943, 230944, 230945, 230946, 230947, 230949, 230951,
                                 230954, 230972, 230979, 230982, 231126, 231470, 231477, 231478, 231479, 231480, 231484,
                                 231485, 231486, 231487, 231488, 231489, 231490, 231491, 231492, 231493, 231501, 231506,
                                 231507, 231508, 231512, 231517, 231518, 231519, 231522, 231524, 231526, 231527, 231528,
                                 231529, 231530, 231531, 231532, 231533, 231534, 231535, 231536, 231537, 231538, 231539,
                                 231540, 231545, 231547, 231548, 231550, 231560, 231561, 231562, 231563, 231564, 231565,
                                 231566, 231567, 231568, 231569, 231570, 231571, 231572, 231573, 231574, 231575, 231576,
                                 231577, 231579, 231581, 231584, 231602, 231608, 231609, 231612, 231705, 231799, 232100,
                                 232101, 232106, 232107, 232109, 232110, 232111, 232112, 232113, 232114, 232117, 232118,
                                 232119, 232120, 232121, 232122, 232123, 232124, 232131, 232135, 232136, 232137, 232138,
                                 232142, 232148, 232149, 232150, 232151, 232152, 232153, 232155, 232156, 232157, 232158,
                                 232159, 232160, 232161, 232162, 232163, 232164, 232165, 232166, 232167, 232168, 232169,
                                 232170, 232175, 232177, 232178, 232180, 232190, 232191, 232192, 232193, 232194, 232195,
                                 232196, 232197, 232198, 232199, 232200, 232201, 232202, 232203, 232204, 232205, 232206,
                                 232207, 232208, 232209, 232211, 232214, 232238, 232239, 232242, 232386, 232731, 232732,
                                 232733, 232734, 232735, 232736, 232740, 232741, 232742, 232743, 232748, 232749, 232750,
                                 232751, 232753, 232754, 232765, 232767, 232768, 232772, 232779, 232780, 232781, 232782,
                                 232783, 232785, 232787, 232788, 232789, 232791, 232792, 232793, 232794, 232795, 232796,
                                 232797, 232798, 232799, 232800, 232805, 232807, 232822, 232823, 232824, 232825, 232826,
                                 232827, 232828, 232829, 232830, 232831, 232832, 232834, 232835, 232836, 232837, 232839,
                                 232868, 232869, 232872, 232965, 232993, 233362, 233363, 233364, 233365, 233371, 233372,
                                 233373, 233379, 233381, 233384, 233397, 233402, 233410, 233411, 233413, 233415, 233419,
                                 233421, 233422, 233425, 233426, 233427, 233435, 233437, 233452, 233453, 233454, 233456,
                                 233460, 233461, 233464, 233465, 233466, 233468, 233469, 233476, 233502, 233998, 234005,
                                 234006, 234007, 234012, 234015, 234020, 234024, 234037, 234044, 234046, 234050, 234054,
                                 234058, 234059, 234061, 234062, 234063, 234064, 234068, 234070, 234080, 234081, 234088,
                                 234089, 234092, 234093, 234095, 234096, 234097, 234099, 234122, 234269, 234354, 234358,
                                 234359, 234628, 234629, 234634, 234635, 234636, 234637, 234638, 234640, 234642, 234643,
                                 234645, 234646, 234647, 234648, 234649, 234650, 234654, 234656, 234667, 234668, 234674,
                                 234676, 234677, 234678, 234680, 234683, 234684, 234685, 234686, 234687, 234688, 234689,
                                 234691, 234692, 234693, 234694, 234697, 234698, 234700, 234710, 234711, 234715, 234717,
                                 234718, 234719, 234720, 234721, 234722, 234723, 234724, 234725, 234726, 234727, 234729,
                                 234731, 234734, 234752, 234759, 234762, 234798, 234825, 234974, 234978, 235250, 235257,
                                 235259, 235260, 235264, 235267, 235268, 235269, 235270, 235271, 235272, 235273, 235275,
                                 235276, 235277, 235278, 235279, 235280, 235281, 235284, 235286, 235287, 235288, 235292,
                                 235298, 235299, 235302, 235306, 235307, 235308, 235309, 235310, 235311, 235312, 235313,
                                 235314, 235315, 235316, 235317, 235318, 235319, 235320, 235321, 235322, 235323, 235324,
                                 235325, 235327, 235328, 235330, 235340, 235341, 235342, 235343, 235344, 235345, 235346,
                                 235347, 235348, 235349, 235350, 235351, 235352, 235353, 235354, 235355, 235356, 235357,
                                 235359, 235361, 235363, 235364, 235388, 235389, 235392, 235443, 235529, 235551, 235554,
                                 235881, 235886, 235890, 235891, 235892, 235893, 235898, 235899, 235900, 235901, 235903,
                                 235904, 235905, 235906, 235907, 235908, 235909, 235910, 235914, 235915, 235917, 235918,
                                 235922, 235929, 235930, 235931, 235932, 235933, 235935, 235937, 235938, 235939, 235941,
                                 235942, 235943, 235944, 235945, 235946, 235947, 235948, 235949, 235950, 235951, 235952,
                                 235953, 235954, 235955, 235957, 235972, 235973, 235974, 235975, 235976, 235977, 235978,
                                 235979, 235980, 235981, 235982, 235984, 235985, 235986, 235987, 235989, 236005, 236018,
                                 236019, 236022, 236058, 236110, 236139, 236499, 236512, 236513, 236514, 236515, 236521,
                                 236522, 236523, 236529, 236531, 236534, 236536, 236537, 236538, 236539, 236547, 236552,
                                 236560, 236561, 236563, 236565, 236569, 236571, 236572, 236575, 236576, 236577, 236581,
                                 236582, 236583, 236584, 236585, 236587, 236602, 236603, 236604, 236606, 236610, 236611,
                                 236614, 236615, 236616, 236617, 236619, 236620, 236622, 236623, 236652, 237148, 237155,
                                 237156, 237157, 237162, 237165, 237170, 237172, 237173, 237174, 237179, 237187, 237194,
                                 237196, 237200, 237204, 237208, 237209, 237211, 237212, 237213, 237214, 237218, 237220,
                                 237230, 237231, 237238, 237239, 237242, 237243, 237245, 237246, 237247, 237249, 237272,
                                 237280, 237298, 237339, 237469, 237478, 237779, 237784, 237787, 237788, 237790, 237792,
                                 237793, 237795, 237796, 237797, 237798, 237799, 237800, 237802, 237803, 237804, 237806,
                                 237809, 237818, 237826, 237827, 237828, 237830, 237833, 237834, 237835, 237836, 237837,
                                 237838, 237839, 237841, 237842, 237843, 237844, 237847, 237848, 237850, 237860, 237861,
                                 237865, 237867, 237868, 237869, 237870, 237871, 237872, 237873, 237874, 237875, 237876,
                                 237877, 237879, 237881, 237884, 237887, 237909, 237910, 237912, 237956, 238056, 238060,
                                 238410, 238418, 238419, 238420, 238421, 238423, 238425, 238426, 238427, 238428, 238429,
                                 238430, 238432, 238433, 238434, 238437, 238438, 238439, 238442, 238449, 238452, 238457,
                                 238458, 238459, 238461, 238462, 238463, 238464, 238465, 238466, 238467, 238468, 238469,
                                 238470, 238471, 238472, 238473, 238474, 238475, 238477, 238492, 238493, 238494, 238495,
                                 238496, 238497, 238498, 238499, 238500, 238501, 238502, 238504, 238505, 238506, 238507,
                                 238509, 238518, 238538, 238539, 238540, 238542, 238558, 238634, 238635, 239011, 239041,
                                 239042, 239043, 239049, 239051, 239054, 239056, 239057, 239058, 239059, 239062, 239063,
                                 239067, 239069, 239071, 239072, 239080, 239081, 239083, 239085, 239089, 239091, 239092,
                                 239095, 239096, 239097, 239101, 239102, 239103, 239104, 239105, 239107, 239122, 239123,
                                 239124, 239126, 239130, 239131, 239134, 239135, 239136, 239137, 239138, 239139, 239147,
                                 239170, 239172, 239677, 239682, 239685, 239690, 239692, 239693, 239694, 239699, 239716,
                                 239720, 239724, 239728, 239729, 239731, 239732, 239733, 239734, 239738, 239740, 239750,
                                 239751, 239758, 239759, 239762, 239765, 239766, 239767, 239769, 239780, 239800, 239832,
                                 239907, 239939, 240308, 240310, 240313, 240315, 240316, 240317, 240318, 240319, 240320,
                                 240322, 240323, 240324, 240329, 240347, 240348, 240353, 240354, 240355, 240356, 240357,
                                 240358, 240359, 240361, 240362, 240363, 240364, 240367, 240385, 240388, 240389, 240390,
                                 240391, 240392, 240394, 240395, 240396, 240397, 240399, 240411, 240429, 240430, 240432,
                                 240468, 240474, 240887, 240939, 240941, 240946, 240947, 240948, 240949, 240952, 240953,
                                 240957, 240959, 240960, 240962, 240979, 240981, 240982, 240985, 240986, 240987, 240991,
                                 240992, 240993, 240994, 240995, 240997, 241013, 241014, 241016, 241020, 241021, 241024,
                                 241025, 241026, 241027, 241029, 241033, 241040, 241060, 241062, 241575, 241580, 241582,
                                 241583, 241584, 241589, 241614, 241618, 241619, 241621, 241622, 241623, 241624, 241638,
                                 241648, 241652, 241655, 241656, 241657, 241659, 241690, 241708, 241713, 242130, 242206,
                                 242207, 242208, 242209, 242212, 242213, 242219, 242223, 242245, 242246, 242247, 242251,
                                 242252, 242253, 242254, 242257, 242280, 242281, 242285, 242286, 242287, 242289, 242297,
                                 242320, 242322, 242700, 242842, 242843, 242849, 242854, 242881, 242882, 242883, 242884,
                                 242915, 242916, 242917, 242930, 242950, 243487, 243494, 243530, 243531, 243543, 243566,
                                 243569, 243571, 243572, 243801, 244117, 244118, 244124, 244160, 244161, 244165, 244167,
                                 244173, 244181, 244184, 244196, 244197, 244198, 244199, 244201, 244202, 244209, 244309,
                                 244413, 244700, 244707, 244731, 244747, 244748, 244749, 244752, 244754, 244770, 244790,
                                 244791, 244792, 244793, 244794, 244795, 244796, 244797, 244803, 244811, 244814, 244826,
                                 244827, 244828, 244829, 244831, 244832, 244839, 244859, 244939, 244998, 245061, 245330,
                                 245331, 245336, 245337, 245361, 245365, 245377, 245378, 245379, 245380, 245381, 245382,
                                 245383, 245384, 245385, 245400, 245420, 245421, 245422, 245423, 245424, 245425, 245426,
                                 245427, 245433, 245441, 245444, 245456, 245457, 245458, 245459, 245461, 245462, 245469,
                                 245489, 245569, 245584, 245673, 245960, 245961, 245962, 245963, 245964, 245965, 245966,
                                 245967, 245991, 245995, 246008, 246009, 246010, 246011, 246012, 246013, 246015, 246030,
                                 246050, 246051, 246052, 246053, 246054, 246055, 246056, 246057, 246063, 246071, 246074,
                                 246086, 246087, 246088, 246089, 246090, 246091, 246099, 246119, 246199, 246258, 246591,
                                 246592, 246593, 246594, 246595, 246596, 246625, 246639, 246640, 246641, 246642, 246643,
                                 246645, 246660, 246682, 246683, 246684, 246685, 246686, 246687, 246716, 246717, 246718,
                                 246719, 246729, 246749, 246829, 246844, 247222, 247223, 247224, 247225, 247270, 247271,
                                 247273, 247275, 247312, 247313, 247314, 247316, 247348, 247350, 247379, 247459, 247858,
                                 247865, 247866, 247897, 247904, 247906, 247910, 247928, 247940, 247941, 247948, 247949,
                                 247952, 247953, 247976, 247979, 247981, 247982, 247983, 247984, 248006, 248176, 248488,
                                 248489, 248494, 248495, 248496, 248516, 248527, 248528, 248534, 248536, 248537, 248538,
                                 248540, 248543, 248558, 248570, 248571, 248575, 248577, 248578, 248579, 248580, 248581,
                                 248582, 248583, 248584, 248591, 248594, 248606, 248607, 248608, 248609, 248611, 248612,
                                 248613, 248614, 248615, 248619, 248622, 248636, 248639, 248719, 248774, 249110, 249117,
                                 249118, 249119, 249120, 249124, 249125, 249126, 249141, 249146, 249148, 249157, 249158,
                                 249159, 249162, 249164, 249166, 249167, 249168, 249169, 249170, 249171, 249172, 249173,
                                 249180, 249185, 249188, 249200, 249201, 249202, 249203, 249204, 249205, 249206, 249207,
                                 249208, 249209, 249210, 249211, 249212, 249213, 249214, 249221, 249224, 249236, 249237,
                                 249238, 249239, 249241, 249242, 249243, 249244, 249245, 249248, 249249, 249252, 249266,
                                 249269, 249341, 249349, 249436, 249740, 249741, 249746, 249747, 249749, 249750, 249751,
                                 249752, 249753, 249754, 249764, 249771, 249775, 249776, 249778, 249788, 249789, 249790,
                                 249791, 249792, 249793, 249795, 249796, 249797, 249798, 249799, 249800, 249801, 249802,
                                 249803, 249810, 249815, 249818, 249830, 249831, 249832, 249833, 249834, 249835, 249836,
                                 249837, 249838, 249839, 249840, 249841, 249842, 249843, 249844, 249851, 249854, 249866,
                                 249867, 249868, 249869, 249871, 249873, 249874, 249875, 249876, 249878, 249879, 249882,
                                 249896, 249899, 249979, 250034, 250371, 250372, 250373, 250374, 250375, 250376, 250380,
                                 250381, 250382, 250383, 250394, 250405, 250408, 250419, 250420, 250421, 250422, 250423,
                                 250425, 250427, 250428, 250429, 250431, 250432, 250433, 250440, 250445, 250462, 250463,
                                 250464, 250465, 250466, 250467, 250468, 250469, 250470, 250471, 250472, 250474, 250496,
                                 250497, 250498, 250499, 250501, 250503, 250504, 250505, 250508, 250509, 250512, 250523,
                                 250526, 250529, 250601, 250609, 250623, 250992, 251002, 251003, 251004, 251005, 251011,
                                 251012, 251013, 251024, 251050, 251051, 251053, 251055, 251059, 251061, 251062, 251075,
                                 251092, 251093, 251094, 251096, 251100, 251101, 251104, 251127, 251128, 251133, 251135,
                                 251136, 251137, 251142, 251156, 251159, 251239, 251638, 251645, 251646, 251647, 251652,
                                 251677, 251684, 251686, 251690, 251694, 251698, 251699, 251708, 251710, 251720, 251721,
                                 251728, 251729, 251732, 251733, 251735, 251736, 251737, 251739, 251756, 251759, 251761,
                                 251762, 251763, 251764, 251765, 251786, 251905, 251991, 251996, 251997, 252268, 252269,
                                 252274, 252275, 252276, 252277, 252278, 252280, 252282, 252283, 252296, 252307, 252308,
                                 252314, 252316, 252317, 252318, 252320, 252323, 252324, 252325, 252326, 252327, 252328,
                                 252329, 252337, 252338, 252340, 252350, 252351, 252355, 252357, 252358, 252359, 252360,
                                 252361, 252362, 252363, 252364, 252365, 252366, 252367, 252369, 252371, 252374, 252386,
                                 252387, 252388, 252389, 252391, 252392, 252393, 252394, 252395, 252399, 252402, 252416,
                                 252419, 252437, 252464, 252499, 252603, 252616, 252890, 252897, 252899, 252900, 252904,
                                 252907, 252908, 252909, 252910, 252911, 252912, 252913, 252921, 252926, 252927, 252928,
                                 252932, 252938, 252939, 252942, 252946, 252947, 252948, 252949, 252950, 252951, 252952,
                                 252953, 252954, 252955, 252956, 252957, 252958, 252959, 252960, 252965, 252967, 252968,
                                 252970, 252980, 252981, 252982, 252983, 252984, 252985, 252986, 252987, 252988, 252989,
                                 252990, 252991, 252992, 252993, 252994, 252995, 252996, 252997, 252999, 253001, 253004,
                                 253016, 253017, 253018, 253019, 253021, 253023, 253024, 253025, 253028, 253029, 253031,
                                 253032, 253046, 253049, 253082, 253129, 253165, 253188, 253193, 253521, 253526, 253530,
                                 253531, 253532, 253533, 253538, 253539, 253540, 253541, 253543, 253544, 253555, 253557,
                                 253558, 253562, 253569, 253570, 253571, 253572, 253573, 253575, 253577, 253578, 253579,
                                 253581, 253582, 253583, 253584, 253585, 253586, 253587, 253588, 253589, 253590, 253595,
                                 253597, 253612, 253613, 253614, 253615, 253616, 253617, 253618, 253619, 253620, 253621,
                                 253622, 253624, 253625, 253626, 253627, 253629, 253646, 253647, 253648, 253649, 253651,
                                 253653, 253654, 253655, 253658, 253659, 253662, 253671, 253676, 253679, 253697, 253759,
                                 253774, 253778, 254136, 254152, 254153, 254154, 254155, 254161, 254162, 254163, 254169,
                                 254171, 254174, 254187, 254192, 254200, 254201, 254203, 254205, 254209, 254211, 254212,
                                 254215, 254216, 254217, 254225, 254227, 254231, 254242, 254243, 254244, 254246, 254250,
                                 254251, 254254, 254255, 254256, 254257, 254259, 254277, 254278, 254280, 254283, 254284,
                                 254285, 254291, 254292, 254306, 254309, 254389, 254788, 254795, 254796, 254797, 254802,
                                 254805, 254810, 254814, 254827, 254834, 254836, 254840, 254844, 254848, 254849, 254851,
                                 254852, 254853, 254854, 254858, 254860, 254870, 254871, 254878, 254879, 254882, 254883,
                                 254885, 254886, 254887, 254889, 254906, 254909, 254911, 254912, 254913, 254914, 254915,
                                 254924, 254936, 254978, 255106, 255114, 255419, 255424, 255427, 255428, 255430, 255432,
                                 255433, 255435, 255436, 255437, 255438, 255439, 255440, 255444, 255446, 255458, 255466,
                                 255467, 255468, 255470, 255473, 255474, 255475, 255476, 255477, 255478, 255479, 255481,
                                 255482, 255483, 255484, 255487, 255488, 255490, 255500, 255501, 255505, 255507, 255508,
                                 255509, 255510, 255511, 255512, 255513, 255514, 255515, 255516, 255517, 255519, 255521,
                                 255524, 255536, 255537, 255538, 255539, 255541, 255543, 255544, 255545, 255549, 255552,
                                 255555, 255566, 255569, 255595, 255649, 255698, 255704, 256050, 256058, 256059, 256060,
                                 256061, 256063, 256065, 256066, 256067, 256068, 256069, 256070, 256074, 256077, 256078,
                                 256082, 256089, 256092, 256097, 256098, 256099, 256101, 256102, 256103, 256104, 256105,
                                 256106, 256107, 256108, 256109, 256110, 256111, 256112, 256113, 256114, 256115, 256117,
                                 256132, 256133, 256134, 256135, 256136, 256137, 256138, 256139, 256140, 256141, 256142,
                                 256144, 256145, 256146, 256147, 256149, 256166, 256167, 256168, 256169, 256171, 256173,
                                 256174, 256175, 256178, 256179, 256182, 256184, 256187, 256196, 256199, 256271, 256277,
                                 256279, 256643, 256681, 256682, 256683, 256689, 256691, 256694, 256696, 256697, 256698,
                                 256699, 256707, 256712, 256720, 256721, 256723, 256725, 256729, 256731, 256732, 256735,
                                 256736, 256737, 256741, 256742, 256743, 256744, 256745, 256746, 256747, 256762, 256763,
                                 256764, 256766, 256770, 256771, 256774, 256775, 256776, 256777, 256779, 256797, 256798,
                                 256803, 256804, 256805, 256806, 256812, 256815, 256826, 256829, 256909, 257317, 257322,
                                 257325, 257330, 257332, 257333, 257334, 257339, 257356, 257360, 257364, 257368, 257369,
                                 257371, 257372, 257373, 257374, 257378, 257380, 257390, 257391, 257398, 257399, 257402,
                                 257403, 257405, 257406, 257407, 257409, 257426, 257429, 257431, 257433, 257434, 257435,
                                 257440, 257448, 257456, 257471, 257567, 257575, 257948, 257950, 257953, 257955, 257956,
                                 257957, 257958, 257959, 257960, 257962, 257963, 257964, 257969, 257987, 257988, 257993,
                                 257994, 257995, 257996, 257997, 257998, 257999, 258001, 258002, 258003, 258004, 258007,
                                 258025, 258027, 258028, 258029, 258030, 258031, 258032, 258034, 258035, 258036, 258037,
                                 258039, 258053, 258056, 258057, 258058, 258059, 258061, 258063, 258064, 258065, 258069,
                                 258070, 258072, 258086, 258089, 258107, 258111, 258169, 258513, 258579, 258581, 258586,
                                 258587, 258588, 258589, 258592, 258593, 258597, 258599, 258602, 258619, 258621, 258622,
                                 258625, 258626, 258627, 258631, 258632, 258633, 258634, 258635, 258637, 258639, 258652,
                                 258653, 258654, 258656, 258660, 258661, 258664, 258665, 258666, 258667, 258669, 258687,
                                 258688, 258693, 258694, 258695, 258700, 258701, 258702, 258708, 258716, 258719, 258799,
                                 259215, 259220, 259222, 259223, 259224, 259229, 259254, 259258, 259259, 259261, 259262,
                                 259263, 259264, 259276, 259288, 259289, 259292, 259295, 259296, 259297, 259299, 259316,
                                 259321, 259323, 259324, 259325, 259330, 259334, 259346, 259350, 259769, 259846, 259847,
                                 259848, 259849, 259852, 259853, 259859, 259885, 259886, 259887, 259891, 259892, 259893,
                                 259894, 259897, 259902, 259920, 259921, 259924, 259925, 259926, 259927, 259929, 259947,
                                 259953, 259954, 259955, 259960, 259962, 259965, 259976, 259979, 260059, 260377, 260482,
                                 260483, 260489, 260521, 260522, 260523, 260524, 260535, 260555, 260556, 260557, 260559,
                                 260584, 260585, 260590, 260598, 260606, 261127, 261134, 261170, 261171, 261183, 261206,
                                 261209, 261211, 261212, 261234, 261235, 261320, 261385, 261757, 261758, 261764, 261800,
                                 261801, 261805, 261807, 261813, 261821, 261824, 261836, 261837, 261838, 261839, 261841,
                                 261842, 261849, 261864, 261865, 261869, 261871, 261949, 261950, 261994, 262340, 262347,
                                 262371, 262387, 262388, 262389, 262392, 262394, 262410, 262430, 262431, 262432, 262433,
                                 262434, 262435, 262436, 262437, 262443, 262451, 262454, 262466, 262467, 262468, 262469,
                                 262471, 262472, 262479, 262494, 262495, 262499, 262501, 262531, 262579, 262580, 262645,
                                 262970, 262971, 262976, 262977, 263001, 263005, 263018, 263019, 263020, 263021, 263022,
                                 263023, 263025, 263040, 263060, 263061, 263062, 263063, 263064, 263065, 263066, 263067,
                                 263073, 263081, 263084, 263096, 263097, 263098, 263099, 263101, 263109, 263124, 263125,
                                 263127, 263129, 263131, 263209, 263210, 263254, 263601, 263602, 263603, 263604, 263605,
                                 263606, 263635, 263649, 263650, 263651, 263652, 263653, 263655, 263670, 263692, 263693,
                                 263694, 263695, 263696, 263697, 263726, 263727, 263728, 263729, 263731, 263739, 263752,
                                 263754, 263755, 263759, 263761, 263791, 263839, 263840, 263863, 264220, 264232, 264233,
                                 264234, 264235, 264280, 264281, 264283, 264285, 264322, 264323, 264324, 264326, 264346,
                                 264357, 264358, 264384, 264387, 264389, 264391, 264469, 264470, 264868, 264875, 264876,
                                 264907, 264914, 264916, 264920, 264938, 264950, 264951, 264958, 264959, 264962, 264963,
                                 264986, 264989, 264991, 264992, 264993, 264994, 264995, 265014, 265015, 265016, 265021,
                                 265100, 265126, 265224, 265225, 265498, 265499, 265504, 265505, 265506, 265526, 265537,
                                 265538, 265544, 265546, 265547, 265548, 265550, 265553, 265568, 265580, 265581, 265585,
                                 265587, 265588, 265589, 265590, 265591, 265592, 265593, 265594, 265601, 265604, 265616,
                                 265617, 265618, 265619, 265621, 265622, 265623, 265624, 265625, 265629, 265632, 265644,
                                 265645, 265646, 265649, 265651, 265665, 265693, 265729, 265730, 265844, 266120, 266127,
                                 266129, 266130, 266134, 266151, 266156, 266158, 266168, 266169, 266172, 266176, 266177,
                                 266178, 266179, 266180, 266181, 266182, 266183, 266190, 266195, 266198, 266210, 266211,
                                 266212, 266213, 266214, 266215, 266216, 266217, 266218, 266219, 266220, 266221, 266222,
                                 266223, 266224, 266231, 266234, 266246, 266247, 266248, 266249, 266251, 266253, 266254,
                                 266255, 266258, 266259, 266262, 266274, 266275, 266276, 266279, 266281, 266282, 266309,
                                 266359, 266360, 266386, 266421, 266751, 266756, 266760, 266761, 266762, 266763, 266774,
                                 266785, 266788, 266799, 266800, 266801, 266802, 266803, 266805, 266807, 266808, 266809,
                                 266811, 266812, 266813, 266820, 266825, 266842, 266843, 266844, 266845, 266846, 266847,
                                 266848, 266849, 266850, 266851, 266852, 266854, 266876, 266877, 266878, 266879, 266881,
                                 266883, 266884, 266885, 266888, 266889, 266892, 266900, 266904, 266905, 266906, 266909,
                                 266911, 266925, 266989, 266990, 267009, 267365, 267382, 267383, 267384, 267385, 267391,
                                 267392, 267393, 267404, 267430, 267431, 267433, 267435, 267439, 267441, 267442, 267455,
                                 267472, 267473, 267474, 267476, 267480, 267481, 267484, 267490, 267507, 267508, 267513,
                                 267514, 267515, 267522, 267534, 267535, 267536, 267539, 267541, 267542, 267619, 267620,
                                 268018, 268025, 268026, 268027, 268032, 268057, 268064, 268066, 268070, 268074, 268078,
                                 268079, 268088, 268090, 268100, 268101, 268108, 268109, 268112, 268113, 268115, 268116,
                                 268117, 268119, 268136, 268139, 268141, 268142, 268143, 268144, 268145, 268164, 268165,
                                 268166, 268171, 268175, 268207, 268250, 268315, 268339, 268649, 268654, 268657, 268658,
                                 268660, 268662, 268663, 268676, 268688, 268696, 268697, 268698, 268700, 268703, 268704,
                                 268705, 268706, 268707, 268708, 268709, 268717, 268718, 268720, 268730, 268731, 268735,
                                 268737, 268738, 268739, 268740, 268741, 268742, 268743, 268744, 268745, 268746, 268747,
                                 268749, 268751, 268754, 268766, 268767, 268768, 268769, 268771, 268773, 268774, 268775,
                                 268779, 268782, 268794, 268795, 268796, 268799, 268801, 268806, 268823, 268879, 268880,
                                 268924, 268926, 269280, 269288, 269289, 269290, 269291, 269293, 269307, 269308, 269312,
                                 269319, 269322, 269327, 269328, 269329, 269331, 269332, 269333, 269334, 269335, 269336,
                                 269337, 269338, 269339, 269340, 269345, 269347, 269362, 269363, 269364, 269365, 269366,
                                 269367, 269368, 269369, 269370, 269371, 269372, 269374, 269375, 269376, 269377, 269379,
                                 269396, 269397, 269398, 269399, 269401, 269403, 269404, 269405, 269408, 269409, 269412,
                                 269416, 269424, 269425, 269426, 269429, 269431, 269435, 269461, 269505, 269509, 269510,
                                 269872, 269911, 269912, 269913, 269919, 269921, 269924, 269937, 269942, 269950, 269951,
                                 269953, 269955, 269959, 269961, 269962, 269965, 269966, 269967, 269975, 269977, 269992,
                                 269993, 269994, 269996, 270000, 270001, 270004, 270005, 270006, 270007, 270008, 270009,
                                 270027, 270028, 270033, 270034, 270035, 270042, 270054, 270055, 270056, 270057, 270059,
                                 270061, 270066, 270139, 270140, 270547, 270552, 270555, 270560, 270564, 270586, 270590,
                                 270594, 270598, 270599, 270601, 270602, 270603, 270604, 270608, 270610, 270620, 270621,
                                 270628, 270629, 270632, 270633, 270635, 270636, 270637, 270639, 270656, 270659, 270661,
                                 270663, 270664, 270665, 270684, 270685, 270686, 270691, 270699, 270700, 270770, 270796,
                                 270809, 271178, 271180, 271183, 271185, 271186, 271187, 271188, 271189, 271190, 271194,
                                 271217, 271218, 271223, 271224, 271225, 271226, 271227, 271228, 271229, 271231, 271232,
                                 271233, 271234, 271237, 271255, 271257, 271258, 271259, 271260, 271261, 271262, 271264,
                                 271265, 271266, 271267, 271269, 271282, 271286, 271287, 271288, 271289, 271291, 271293,
                                 271294, 271295, 271299, 271302, 271314, 271315, 271316, 271319, 271321, 271335, 271338,
                                 271399, 271400, 271742, 271809, 271811, 271816, 271817, 271818, 271819, 271827, 271832,
                                 271849, 271851, 271852, 271855, 271856, 271857, 271861, 271862, 271863, 271864, 271865,
                                 271867, 271882, 271883, 271884, 271886, 271890, 271891, 271894, 271895, 271896, 271897,
                                 271899, 271903, 271917, 271918, 271923, 271924, 271925, 271932, 271944, 271945, 271946,
                                 271949, 271951, 271952, 271959, 272029, 272030, 272445, 272450, 272452, 272453, 272454,
                                 272459, 272484, 272488, 272489, 272491, 272492, 272493, 272494, 272504, 272518, 272519,
                                 272522, 272525, 272526, 272527, 272529, 272546, 272549, 272551, 272553, 272554, 272555,
                                 272560, 272574, 272575, 272576, 272578, 272581, 272585, 272660, 273004, 273076, 273077,
                                 273078, 273079, 273082, 273083, 273089, 273115, 273116, 273117, 273121, 273122, 273123,
                                 273124, 273127, 273150, 273151, 273154, 273155, 273156, 273157, 273159, 273167, 273177,
                                 273178, 273183, 273184, 273185, 273190, 273192, 273204, 273205, 273206, 273209, 273211,
                                 273216, 273289, 273290, 273635, 273712, 273713, 273719, 273751, 273752, 273753, 273754,
                                 273785, 273786, 273787, 273789, 273800, 273813, 273814, 273815, 273820, 273835, 273836,
                                 273841, 273849, 273920, 274357, 274364, 274400, 274401, 274413, 274436, 274439, 274441,
                                 274442, 274464, 274465, 274471, 274550, 274575, 274669, 274671, 274987, 274988, 274994,
                                 275030, 275031, 275035, 275037, 275043, 275051, 275054, 275066, 275067, 275068, 275069,
                                 275071, 275072, 275079, 275094, 275095, 275099, 275101, 275116, 275142, 275179, 275180,
                                 275283, 275570, 275601, 275618, 275619, 275622, 275640, 275660, 275661, 275662, 275663,
                                 275664, 275665, 275666, 275667, 275673, 275681, 275684, 275696, 275697, 275698, 275699,
                                 275701, 275709, 275724, 275725, 275729, 275731, 275758, 275809, 275810, 275835, 275850,
                                 275868, 276206, 276235, 276249, 276250, 276251, 276252, 276253, 276255, 276270, 276292,
                                 276293, 276294, 276295, 276296, 276297, 276326, 276327, 276328, 276329, 276331, 276339,
                                 276349, 276354, 276355, 276359, 276361, 276376, 276439, 276440, 276454, 276813, 276832,
                                 276834, 276835, 276880, 276881, 276883, 276885, 276922, 276923, 276924, 276926, 276957,
                                 276958, 276960, 276984, 276985, 276989, 276991, 277069, 277070, 277110, 277468, 277476,
                                 277507, 277514, 277516, 277520, 277538, 277550, 277551, 277558, 277559, 277562, 277563,
                                 277586, 277589, 277591, 277592, 277593, 277594, 277595, 277614, 277615, 277616, 277621,
                                 277640, 277656, 277700, 277786, 278099, 278126, 278138, 278146, 278147, 278148, 278150,
                                 278153, 278168, 278180, 278181, 278185, 278187, 278188, 278189, 278190, 278191, 278192,
                                 278193, 278194, 278201, 278204, 278216, 278217, 278218, 278219, 278221, 278223, 278224,
                                 278225, 278229, 278232, 278244, 278245, 278246, 278249, 278251, 278272, 278329, 278330,
                                 278352, 278384, 278758, 278769, 278772, 278777, 278778, 278779, 278781, 278782, 278783,
                                 278790, 278795, 278812, 278813, 278814, 278815, 278816, 278817, 278818, 278819, 278820,
                                 278821, 278822, 278824, 278846, 278847, 278848, 278849, 278851, 278853, 278854, 278855,
                                 278858, 278859, 278862, 278863, 278874, 278875, 278876, 278879, 278881, 278900, 278951,
                                 278959, 278960, 279320, 279361, 279363, 279374, 279400, 279401, 279403, 279405, 279409,
                                 279411, 279412, 279425, 279442, 279443, 279444, 279446, 279450, 279451, 279454, 279477,
                                 279478, 279483, 279484, 279485, 279486, 279492, 279504, 279505, 279506, 279509, 279511,
                                 279589, 279590, 279612, 279997, 280036, 280040, 280044, 280048, 280049, 280058, 280060,
                                 280070, 280071, 280078, 280079, 280082, 280083, 280085, 280086, 280087, 280089, 280106,
                                 280109, 280111, 280113, 280114, 280115, 280134, 280135, 280136, 280141, 280148, 280220,
                                 280245, 280254, 280255, 280630, 280633, 280667, 280668, 280673, 280674, 280675, 280676,
                                 280677, 280678, 280679, 280687, 280705, 280707, 280708, 280709, 280710, 280711, 280712,
                                 280714, 280715, 280716, 280717, 280719, 280729, 280736, 280737, 280738, 280739, 280741,
                                 280743, 280744, 280745, 280749, 280752, 280764, 280765, 280766, 280769, 280771, 280786,
                                 280787, 280849, 280850, 281211, 281261, 281277, 281282, 281299, 281301, 281302, 281305,
                                 281306, 281307, 281315, 281317, 281332, 281333, 281334, 281336, 281340, 281341, 281344,
                                 281345, 281346, 281347, 281349, 281367, 281368, 281373, 281374, 281375, 281381, 281382,
                                 281394, 281395, 281396, 281399, 281401, 281479, 281480, 281514, 281520, 281900, 281904,
                                 281916, 281934, 281938, 281939, 281941, 281942, 281943, 281944, 281968, 281969, 281972,
                                 281975, 281976, 281977, 281979, 281996, 281999, 282001, 282003, 282004, 282005, 282014,
                                 282024, 282025, 282026, 282031, 282050, 282110, 282474, 282527, 282528, 282529, 282565,
                                 282566, 282567, 282571, 282572, 282573, 282574, 282577, 282600, 282601, 282604, 282605,
                                 282606, 282607, 282609, 282627, 282628, 282633, 282634, 282635, 282642, 282645, 282654,
                                 282655, 282656, 282659, 282661, 282739, 282740, 282762, 283105, 283163, 283169, 283201,
                                 283202, 283203, 283204, 283235, 283236, 283237, 283239, 283263, 283264, 283265, 283270,
                                 283278, 283284, 283285, 283286, 283291, 283370, 283404, 283807, 283850, 283851, 283863,
                                 283886, 283889, 283891, 283892, 283914, 283915, 283921, 283950, 283955, 284000, 284065,
                                 284480, 284481, 284485, 284487, 284493, 284501, 284504, 284516, 284517, 284518, 284519,
                                 284521, 284529, 284544, 284545, 284549, 284551, 284569, 284628, 284629, 284630, 284674,
                                 285069, 285090, 285112, 285113, 285114, 285115, 285116, 285117, 285144, 285146, 285147,
                                 285148, 285149, 285151, 285159, 285174, 285175, 285179, 285181, 285210, 285211, 285259,
                                 285260, 285630, 285700, 285701, 285703, 285742, 285743, 285744, 285746, 285777, 285778,
                                 285804, 285805, 285807, 285809, 285811, 285888, 285889, 285890, 286336, 286358, 286370,
                                 286371, 286378, 286379, 286382, 286383, 286406, 286409, 286411, 286413, 286414, 286415,
                                 286434, 286435, 286436, 286441, 286447, 286474, 286520, 286546, 286967, 286973, 287005,
                                 287007, 287008, 287009, 287010, 287011, 287012, 287014, 287025, 287036, 287037, 287038,
                                 287039, 287041, 287043, 287044, 287045, 287049, 287052, 287064, 287065, 287066, 287069,
                                 287071, 287085, 287149, 287150, 287524, 287599, 287601, 287602, 287632, 287633, 287634,
                                 287636, 287640, 287641, 287644, 287667, 287668, 287673, 287674, 287675, 287682, 287694,
                                 287695, 287696, 287699, 287701, 287702, 287734, 287779, 287780, 288215, 288234, 288239,
                                 288268, 288269, 288272, 288275, 288276, 288277, 288279, 288296, 288299, 288301, 288303,
                                 288304, 288305, 288324, 288325, 288326, 288331, 288335, 288360, 288410, 288787, 288865,
                                 288866, 288877, 288900, 288901, 288904, 288905, 288906, 288907, 288909, 288927, 288928,
                                 288933, 288934, 288935, 288942, 288954, 288955, 288956, 288959, 288961, 288966, 289038,
                                 289039, 289040, 289418, 289501, 289502, 289503, 289535, 289536, 289537, 289539, 289563,
                                 289564, 289565, 289584, 289585, 289586, 289591, 289599, 289624, 289670, 290151, 290163,
                                 290186, 290189, 290191, 290214, 290215, 290221, 290224, 290260, 290300, 290325, 290779,
                                 290787, 290816, 290817, 290818, 290819, 290821, 290829, 290844, 290845, 290849, 290851,
                                 290866, 290929, 290930, 291311, 291412, 291413, 291416, 291447, 291448, 291474, 291475,
                                 291479, 291481, 291520, 291559, 291560, 291600, 291943, 292049, 292052, 292076, 292079,
                                 292081, 292083, 292084, 292085, 292104, 292105, 292106, 292111, 292130, 292190, 292574,
                                 292680, 292681, 292684, 292707, 292708, 292713, 292714, 292715, 292734, 292735, 292736,
                                 292739, 292741, 292819, 292820, 292842, 293205, 293316, 293317, 293319, 293343, 293344,
                                 293345, 293364, 293365, 293366, 293371, 293410, 293450, 293484, 294046, 294060, 294167,
                                 294676, 294677, 294680, 294683, 294690, 294797, 295306, 295307, 295308, 295310, 295312,
                                 295313, 295320, 295324, 295427, 295936, 295937, 295938, 295939, 295940, 295942, 295943,
                                 295950, 295954, 296057, 296566, 296567, 296568, 296569, 296570, 296572, 296573, 296580,
                                 296584, 296587, 296687, 297196, 297197, 297198, 297199, 297200, 297201, 297202, 297203,
                                 297210, 297214, 297827, 297828, 297829, 297830, 297832, 297844, 297847, 298459, 298461,
                                 298462, 298474, 299086, 299094, 299095, 299097, 299100, 299207, 299243, 299716, 299717,
                                 299720, 299723, 299724, 299725, 299727, 299729, 299730, 299733, 299837, 299845, 299855,
                                 300346, 300347, 300348, 300349, 300350, 300352, 300353, 300354, 300355, 300356, 300357,
                                 300359, 300360, 300363, 300364, 300439, 300467, 300475, 300503, 300976, 300977, 300978,
                                 300979, 300980, 300982, 300983, 300984, 300985, 300986, 300987, 300989, 300990, 300993,
                                 300994, 300996, 301097, 301105, 301115, 301606, 301607, 301608, 301609, 301610, 301612,
                                 301613, 301614, 301615, 301616, 301617, 301618, 301619, 301620, 301623, 301624, 301699,
                                 301727, 301735, 302237, 302238, 302239, 302240, 302242, 302243, 302244, 302245, 302246,
                                 302249, 302253, 302254, 302256, 302365, 302868, 302869, 302872, 302876, 302878, 302879,
                                 302883, 302884, 303496, 303504, 303505, 303507, 303510, 303511, 303531, 303617, 303618,
                                 303620, 304126, 304127, 304130, 304133, 304134, 304135, 304136, 304137, 304139, 304140,
                                 304141, 304142, 304143, 304161, 304205, 304247, 304250, 304255, 304756, 304757, 304758,
                                 304759, 304760, 304762, 304763, 304764, 304765, 304766, 304767, 304769, 304770, 304771,
                                 304772, 304773, 304774, 304775, 304791, 304877, 304878, 304880, 304885, 305386, 305387,
                                 305388, 305389, 305390, 305392, 305393, 305394, 305395, 305396, 305397, 305399, 305400,
                                 305401, 305402, 305403, 305404, 305421, 305465, 305503, 305507, 305510, 305515, 306017,
                                 306018, 306019, 306020, 306022, 306023, 306024, 306025, 306026, 306027, 306029, 306031,
                                 306032, 306033, 306034, 306035, 306037, 306051, 306140, 306145, 306648, 306649, 306651,
                                 306652, 306656, 306659, 306662, 306663, 306664, 306681, 306763, 306770, 306775, 307276,
                                 307284, 307285, 307287, 307290, 307291, 307292, 307299, 307311, 307397, 307400, 307433,
                                 307438, 307906, 307907, 307910, 307913, 307914, 307915, 307916, 307917, 307919, 307920,
                                 307921, 307922, 307923, 307928, 307941, 308027, 308030, 308035, 308045, 308058, 308536,
                                 308537, 308538, 308539, 308540, 308542, 308543, 308544, 308545, 308546, 308547, 308549,
                                 308550, 308551, 308552, 308553, 308554, 308559, 308571, 308578, 308629, 308634, 308657,
                                 308660, 308665, 309167, 309168, 309169, 309170, 309172, 309173, 309174, 309175, 309176,
                                 309177, 309179, 309181, 309182, 309183, 309184, 309186, 309188, 309190, 309201, 309290,
                                 309295, 309702, 309798, 309799, 309802, 309806, 309808, 309809, 309811, 309812, 309813,
                                 309814, 309831, 309838, 309920, 309925, 310426, 310434, 310435, 310437, 310440, 310441,
                                 310442, 310452, 310461, 310547, 310548, 310550, 310558, 311056, 311057, 311060, 311063,
                                 311064, 311065, 311066, 311067, 311069, 311070, 311071, 311072, 311073, 311083, 311091,
                                 311135, 311140, 311177, 311180, 311185, 311687, 311688, 311689, 311690, 311692, 311693,
                                 311694, 311695, 311696, 311697, 311699, 311701, 311702, 311703, 311704, 311705, 311712,
                                 311714, 311721, 311810, 311815, 312151, 312318, 312319, 312322, 312326, 312329, 312331,
                                 312332, 312333, 312334, 312343, 312351, 312433, 312440, 312445, 312946, 312954, 312955,
                                 312957, 312961, 312962, 312969, 312976, 312981, 312987, 313067, 313070, 313554, 313580,
                                 313583, 313584, 313585, 313586, 313587, 313589, 313591, 313592, 313593, 313598, 313611,
                                 313700, 313705, 314040, 314208, 314209, 314212, 314216, 314219, 314221, 314222, 314223,
                                 314236, 314241, 314248, 314330, 314335, 314793, 314844, 314847, 314851, 314852, 314862,
                                 314871, 314960, 315303, 315476, 315479, 315481, 315482, 315493, 315501, 315590, 315595,
                                 315934, 316111, 316112, 316126, 316131, 316726, 316740, 316762, 316763, 316847, 316864,
                                 316880, 317356, 317357, 317360, 317363, 317370, 317392, 317393, 317396, 317419, 317437,
                                 317477, 317492, 317494, 317986, 317987, 317988, 317989, 317990, 317992, 317993, 318000,
                                 318004, 318022, 318023, 318024, 318026, 318049, 318067, 318075, 318107, 318124, 318140,
                                 318616, 318617, 318618, 318619, 318620, 318622, 318623, 318630, 318634, 318652, 318653,
                                 318654, 318655, 318656, 318679, 318697, 318737, 318752, 318754, 319246, 319247, 319248,
                                 319249, 319250, 319252, 319253, 319260, 319264, 319282, 319283, 319284, 319286, 319309,
                                 319327, 319335, 319367, 319384, 319395, 319877, 319878, 319879, 319880, 319882, 319883,
                                 319894, 319913, 319914, 319915, 319916, 319939, 319957, 320014, 320508, 320509, 320512,
                                 320524, 320544, 320546, 320587, 320655, 321136, 321144, 321145, 321147, 321150, 321172,
                                 321173, 321179, 321180, 321223, 321254, 321257, 321274, 321766, 321767, 321770, 321773,
                                 321774, 321775, 321776, 321777, 321779, 321780, 321783, 321802, 321803, 321804, 321806,
                                 321809, 321810, 321812, 321828, 321829, 321847, 321853, 321887, 321895, 321904, 322396,
                                 322397, 322398, 322399, 322400, 322402, 322403, 322404, 322405, 322406, 322407, 322409,
                                 322410, 322413, 322414, 322432, 322433, 322434, 322436, 322439, 322440, 322441, 322442,
                                 322459, 322477, 322483, 322514, 322517, 322525, 322534, 323026, 323027, 323028, 323029,
                                 323030, 323032, 323033, 323034, 323035, 323036, 323037, 323039, 323040, 323043, 323044,
                                 323062, 323063, 323064, 323066, 323069, 323070, 323072, 323088, 323089, 323107, 323113,
                                 323138, 323147, 323155, 323164, 323657, 323658, 323659, 323660, 323662, 323663, 323664,
                                 323665, 323666, 323667, 323669, 323673, 323674, 323692, 323693, 323694, 323696, 323699,
                                 323700, 323701, 323702, 323703, 323719, 323737, 323743, 323785, 323794, 324217, 324288,
                                 324289, 324292, 324296, 324299, 324303, 324304, 324324, 324326, 324330, 324332, 324349,
                                 324367, 324373, 324398, 324415, 324916, 324924, 324925, 324927, 324930, 324931, 324932,
                                 324951, 324952, 324953, 324959, 324960, 324962, 324965, 325003, 325037, 325040, 325054,
                                 325070, 325076, 325546, 325547, 325550, 325553, 325554, 325555, 325556, 325557, 325559,
                                 325560, 325561, 325562, 325563, 325581, 325582, 325583, 325584, 325586, 325589, 325590,
                                 325592, 325596, 325609, 325627, 325633, 325667, 325670, 325675, 325682, 325684, 325696,
                                 326176, 326177, 326178, 326179, 326180, 326182, 326183, 326184, 326185, 326186, 326187,
                                 326189, 326190, 326191, 326192, 326193, 326194, 326211, 326212, 326213, 326214, 326216,
                                 326219, 326220, 326222, 326225, 326227, 326239, 326257, 326263, 326265, 326273, 326297,
                                 326300, 326305, 326314, 326807, 326808, 326809, 326810, 326812, 326813, 326814, 326815,
                                 326816, 326817, 326819, 326821, 326822, 326823, 326824, 326841, 326842, 326843, 326844,
                                 326845, 326846, 326849, 326850, 326852, 326856, 326858, 326869, 326887, 326893, 326930,
                                 326935, 326944, 327311, 327438, 327439, 327442, 327446, 327449, 327451, 327452, 327453,
                                 327454, 327471, 327474, 327476, 327479, 327480, 327482, 327487, 327499, 327517, 327523,
                                 327560, 327565, 327585, 328066, 328074, 328075, 328077, 328080, 328081, 328082, 328101,
                                 328102, 328103, 328109, 328110, 328112, 328120, 328153, 328184, 328187, 328190, 328194,
                                 328204, 328696, 328697, 328700, 328703, 328704, 328705, 328706, 328707, 328709, 328710,
                                 328711, 328712, 328713, 328731, 328732, 328733, 328734, 328736, 328739, 328740, 328742,
                                 328751, 328758, 328759, 328777, 328778, 328783, 328817, 328820, 328825, 328834, 329327,
                                 329328, 329329, 329330, 329332, 329333, 329334, 329335, 329336, 329337, 329339, 329341,
                                 329342, 329343, 329344, 329357, 329361, 329362, 329363, 329364, 329366, 329369, 329370,
                                 329371, 329372, 329380, 329389, 329407, 329413, 329450, 329455, 329464, 329826, 329958,
                                 329959, 329962, 329966, 329969, 329971, 329972, 329973, 329974, 329991, 329994, 329996,
                                 329999, 330000, 330002, 330011, 330019, 330037, 330043, 330068, 330080, 330085, 330586,
                                 330594, 330595, 330597, 330600, 330601, 330602, 330621, 330622, 330623, 330629, 330630,
                                 330632, 330635, 330644, 330647, 330673, 330707, 330710, 330724, 331191, 331217, 331220,
                                 331223, 331224, 331225, 331226, 331227, 331229, 331231, 331232, 331233, 331251, 331252,
                                 331253, 331254, 331256, 331259, 331260, 331262, 331266, 331279, 331297, 331303, 331340,
                                 331345, 331354, 331719, 331848, 331849, 331852, 331856, 331859, 331861, 331862, 331863,
                                 331864, 331881, 331884, 331886, 331889, 331890, 331892, 331897, 331904, 331909, 331927,
                                 331933, 331970, 331975, 332430, 332484, 332485, 332487, 332491, 332492, 332511, 332512,
                                 332513, 332519, 332520, 332522, 332530, 332563, 332600, 332982, 333116, 333119, 333121,
                                 333122, 333123, 333141, 333144, 333149, 333150, 333152, 333161, 333169, 333187, 333193,
                                 333230, 333235, 333615, 333751, 333752, 333771, 333779, 333780, 333782, 333794, 333860,
                                 334366, 334380, 334402, 334403, 334430, 334433, 334469, 334487, 334501, 334504, 334996,
                                 334997, 335000, 335003, 335010, 335032, 335033, 335034, 335036, 335059, 335060, 335061,
                                 335063, 335073, 335077, 335117, 335131, 335134, 335626, 335627, 335628, 335629, 335630,
                                 335632, 335633, 335640, 335644, 335662, 335663, 335664, 335666, 335689, 335690, 335691,
                                 335692, 335693, 335707, 335729, 335747, 335761, 335764, 336256, 336257, 336258, 336259,
                                 336260, 336262, 336263, 336270, 336274, 336292, 336293, 336294, 336296, 336319, 336320,
                                 336321, 336323, 336333, 336337, 336352, 336377, 336391, 336394, 336887, 336888, 336889,
                                 336890, 336892, 336893, 336904, 336907, 336922, 336923, 336924, 336926, 336943, 336949,
                                 336950, 336951, 336952, 336953, 336967, 337021, 337024, 337426, 337518, 337519, 337521,
                                 337522, 337534, 337554, 337556, 337579, 337580, 337581, 337583, 337597, 337612, 338146,
                                 338154, 338155, 338157, 338160, 338182, 338183, 338189, 338190, 338192, 338210, 338211,
                                 338213, 338216, 338233, 338267, 338281, 338284, 338303, 338304, 338776, 338777, 338780,
                                 338783, 338784, 338785, 338786, 338787, 338789, 338790, 338793, 338812, 338813, 338814,
                                 338816, 338819, 338820, 338822, 338839, 338840, 338841, 338843, 338847, 338857, 338863,
                                 338897, 338905, 338911, 338914, 338915, 338924, 339406, 339407, 339408, 339409, 339410,
                                 339412, 339413, 339414, 339415, 339416, 339417, 339419, 339420, 339423, 339424, 339442,
                                 339443, 339444, 339446, 339449, 339450, 339452, 339469, 339470, 339471, 339473, 339476,
                                 339478, 339487, 339493, 339499, 339501, 339527, 339535, 339541, 339544, 340037, 340038,
                                 340039, 340040, 340042, 340043, 340044, 340045, 340046, 340047, 340049, 340053, 340054,
                                 340056, 340072, 340073, 340074, 340076, 340079, 340080, 340082, 340089, 340099, 340100,
                                 340101, 340103, 340107, 340117, 340123, 340165, 340171, 340174, 340570, 340668, 340669,
                                 340672, 340676, 340678, 340679, 340683, 340684, 340704, 340706, 340709, 340710, 340712,
                                 340729, 340730, 340731, 340733, 340738, 340747, 340753, 340795, 340801, 341296, 341304,
                                 341305, 341307, 341310, 341311, 341312, 341331, 341332, 341333, 341339, 341340, 341342,
                                 341360, 341361, 341363, 341371, 341383, 341399, 341417, 341418, 341419, 341420, 341431,
                                 341434, 341926, 341927, 341930, 341933, 341934, 341935, 341936, 341937, 341939, 341940,
                                 341941, 341942, 341943, 341961, 341962, 341963, 341964, 341966, 341969, 341970, 341972,
                                 341989, 341990, 341991, 341993, 342002, 342003, 342005, 342006, 342007, 342013, 342047,
                                 342050, 342055, 342061, 342064, 342557, 342558, 342559, 342560, 342562, 342563, 342564,
                                 342565, 342566, 342567, 342569, 342571, 342572, 342573, 342574, 342575, 342585, 342591,
                                 342592, 342593, 342594, 342596, 342599, 342600, 342602, 342619, 342620, 342621, 342622,
                                 342623, 342631, 342637, 342643, 342680, 342685, 342691, 342694, 343088, 343188, 343189,
                                 343192, 343196, 343199, 343201, 343202, 343203, 343204, 343221, 343224, 343226, 343229,
                                 343230, 343232, 343249, 343250, 343251, 343253, 343262, 343267, 343273, 343282, 343303,
                                 343310, 343315, 343321, 343816, 343824, 343825, 343827, 343830, 343831, 343832, 343839,
                                 343851, 343852, 343853, 343859, 343860, 343862, 343880, 343881, 343883, 343886, 343889,
                                 343899, 343903, 343937, 343940, 343951, 343954, 344418, 344447, 344450, 344453, 344454,
                                 344455, 344456, 344457, 344459, 344461, 344462, 344463, 344468, 344481, 344482, 344483,
                                 344484, 344486, 344489, 344490, 344492, 344509, 344510, 344511, 344513, 344517, 344527,
                                 344533, 344570, 344575, 344581, 344584, 344983, 345078, 345079, 345082, 345086, 345089,
                                 345091, 345092, 345093, 345094, 345111, 345114, 345116, 345118, 345119, 345120, 345122,
                                 345139, 345140, 345141, 345143, 345148, 345157, 345159, 345163, 345200, 345205, 345211,
                                 345658, 345714, 345715, 345717, 345721, 345722, 345732, 345741, 345742, 345743, 345749,
                                 345750, 345752, 345770, 345771, 345773, 345781, 345793, 345830, 345841, 345844, 346247,
                                 346346, 346349, 346351, 346352, 346353, 346363, 346371, 346374, 346376, 346379, 346380,
                                 346382, 346399, 346400, 346401, 346403, 346412, 346417, 346423, 346460, 346465, 346471,
                                 346880, 346981, 346982, 346996, 347001, 347009, 347010, 347012, 347031, 347033, 347049,
                                 347053, 347090, 347101, 347596, 347610, 347632, 347633, 347660, 347661, 347663, 347681,
                                 347717, 347731, 347734, 347750, 347751, 348226, 348227, 348230, 348233, 348240, 348262,
                                 348263, 348264, 348266, 348289, 348290, 348291, 348293, 348307, 348312, 348347, 348361,
                                 348362, 348363, 348364, 348856, 348857, 348858, 348859, 348860, 348862, 348863, 348870,
                                 348874, 348892, 348893, 348894, 348896, 348919, 348920, 348921, 348923, 348937, 348941,
                                 348945, 348948, 348977, 348991, 348994, 349007, 349487, 349488, 349489, 349490, 349492,
                                 349493, 349504, 349522, 349523, 349524, 349525, 349526, 349534, 349549, 349550, 349551,
                                 349553, 349567, 349572, 349621, 349624, 350040, 350118, 350119, 350122, 350134, 350154,
                                 350156, 350179, 350180, 350181, 350183, 350197, 350251, 350265, 350267, 350746, 350754,
                                 350755, 350757, 350760, 350782, 350783, 350789, 350790, 350792, 350810, 350811, 350813,
                                 350833, 350836, 350864, 350866, 350867, 350881, 350884, 351376, 351377, 351380, 351383,
                                 351384, 351385, 351386, 351387, 351389, 351390, 351393, 351412, 351413, 351414, 351416,
                                 351419, 351420, 351422, 351438, 351439, 351440, 351441, 351443, 351457, 351463, 351464,
                                 351467, 351497, 351505, 351511, 351514, 352007, 352008, 352009, 352010, 352012, 352013,
                                 352014, 352015, 352016, 352017, 352019, 352023, 352024, 352031, 352042, 352043, 352044,
                                 352046, 352049, 352050, 352051, 352052, 352069, 352070, 352071, 352073, 352087, 352093,
                                 352096, 352135, 352141, 352144, 352566, 352638, 352639, 352642, 352646, 352649, 352653,
                                 352654, 352674, 352676, 352679, 352680, 352682, 352699, 352700, 352701, 352703, 352717,
                                 352723, 352727, 352748, 352765, 352771, 353266, 353274, 353275, 353277, 353280, 353281,
                                 353282, 353301, 353302, 353303, 353309, 353310, 353312, 353315, 353330, 353331, 353333,
                                 353335, 353351, 353353, 353360, 353387, 353390, 353401, 353404, 353867, 353897, 353900,
                                 353903, 353904, 353905, 353906, 353907, 353909, 353911, 353912, 353913, 353931, 353932,
                                 353933, 353934, 353936, 353939, 353940, 353942, 353946, 353959, 353960, 353961, 353963,
                                 353977, 353982, 353983, 354020, 354025, 354031, 354034, 354461, 354528, 354529, 354532,
                                 354536, 354539, 354541, 354542, 354543, 354544, 354561, 354564, 354566, 354569, 354570,
                                 354572, 354577, 354589, 354590, 354591, 354593, 354607, 354613, 354620, 354650, 354655,
                                 354661, 354677, 355094, 355164, 355165, 355167, 355171, 355172, 355191, 355192, 355193,
                                 355199, 355200, 355202, 355210, 355220, 355221, 355223, 355243, 355246, 355280, 355291,
                                 355294, 355725, 355796, 355799, 355801, 355802, 355803, 355821, 355824, 355826, 355829,
                                 355830, 355832, 355841, 355849, 355850, 355851, 355853, 355867, 355873, 355877, 355910,
                                 355915, 355921, 356358, 356431, 356432, 356451, 356459, 356460, 356462, 356474, 356480,
                                 356481, 356483, 356503, 356510, 356540, 356551, 357046, 357060, 357082, 357083, 357110,
                                 357111, 357113, 357145, 357146, 357149, 357167, 357181, 357184, 357676, 357677, 357680,
                                 357683, 357690, 357712, 357713, 357714, 357716, 357739, 357740, 357741, 357743, 357753,
                                 357754, 357757, 357777, 357797, 357811, 357814, 358306, 358314, 358315, 358317, 358320,
                                 358342, 358343, 358349, 358350, 358352, 358366, 358370, 358371, 358373, 358376, 358393,
                                 358408, 358427, 358441, 358444, 358950, 358972, 358973, 358995, 359000, 359001, 359003,
                                 359021, 359045, 359057, 359071, 359074, 359670, 359676, 359699, 359722, 360300, 360301,
                                 360303, 360329, 360352, 360930, 360932, 360933, 360936, 360959, 360982, 361560, 361561,
                                 361563, 361564, 361589, 361612, 362107, 362190, 362192, 362193, 362219, 362242, 362721,
                                 362820, 362823, 362824, 362872, 363450, 363453, 363460, 363479, 363502, 363503, 364080,
                                 364083, 364087, 364109, 364115, 364132, 364699, 364710, 364713, 364719, 364720, 364739,
                                 364762, 365256, 365340, 365343, 365347, 365369, 365392, 365878, 365970, 365973, 365979,
                                 365999, 366022, 366600, 366603, 366606, 366611, 366618, 366629, 366652, 367205, 367230,
                                 367231, 367233, 367242, 367259, 367282, 367775, 367860, 367862, 367863, 367871, 367889,
                                 367912, 368490, 368493, 368494, 368502, 368503, 368519, 368542, 369039, 369120, 369123,
                                 369130, 369135, 369149, 369172, 369668, 369750, 369753, 369757, 369779, 369802, 370318,
                                 370380, 370383, 370389, 370395, 370409, 370432, 370932, 371010, 371013, 371021, 371039,
                                 371062, 371563, 371640, 371643, 371652, 371669, 371692, 372196, 372273, 372285, 372299,
                                 372322, 372900, 372903, 372921, 372929, 372950, 372952, 373530, 373533, 373552, 373559,
                                 373562, 373582, 374145, 374160, 374163, 374181, 374183, 374189, 374212, 374725, 374790,
                                 374793, 374812, 374819, 374842, 375420, 375423, 375443, 375449, 375465, 375472, 376050,
                                 376053, 376064, 376076, 376079, 376102, 376638, 376680, 376683, 376707, 376709, 376732,
                                 377251, 377310, 377313, 377336, 377339, 377362, 377940, 377943, 377948, 377967, 377969,
                                 377992, 378515, 378570, 378573, 378591, 378599, 378600, 378622, 379146, 379200, 379203,
                                 379222, 379229, 379252, 379777, 379830, 379833, 379853, 379859, 379860, 379882, 380410,
                                 380460, 380463, 380486, 380489, 380512, 381041, 381090, 381093, 381117, 381119, 381142,
                                 381674, 381720, 381723, 381749, 381750, 381772, 382349, 382350, 382353, 382356, 382379,
                                 382386, 382402, 382953, 382980, 382981, 382983, 383009, 383017, 383032, 383576, 383610,
                                 383613, 383620, 383639, 383648, 383662, 384221, 384240, 384243, 384261, 384269, 384279,
                                 384292, 384876, 384910, 385501, 385541, 386140, 386172, 386781, 386803, 387271, 387287,
                                 387288, 387289, 387290, 387292, 387293, 387304, 387322, 387323, 387324, 387326, 387349,
                                 387350, 387351, 387352, 387353, 387367, 387386, 387421, 387424, 387885, 387917, 387920,
                                 387923, 387924, 387925, 387926, 387927, 387929, 387933, 387952, 387953, 387954, 387956,
                                 387959, 387960, 387962, 387979, 387980, 387981, 387983, 387987, 387997, 388003, 388045,
                                 388051, 388054, 388516, 388547, 388553, 388582, 388583, 388584, 388586, 388609, 388610,
                                 388611, 388613, 388627, 388632, 388681, 388684, 389135, 389184, 389185, 389187, 389191,
                                 389192, 389211, 389212, 389213, 389219, 389220, 389222, 389240, 389241, 389243, 389251,
                                 389263, 389276, 389300, 389311, 389314, 389780, 389815, 389817, 389842, 389843, 389849,
                                 389850, 389852, 389870, 389871, 389873, 389893, 389896, 389941, 389944, 390226, 390359,
                                 390361, 390384, 390385, 390391, 390420, 390470, 391017, 391068, 391069, 391072, 391084,
                                 391104, 391106, 391129, 391130, 391131, 391133, 391147, 391162, 391167, 391201, 391652,
                                 391698, 391699, 391702, 391706, 391709, 391713, 391714, 391734, 391736, 391739, 391740,
                                 391742, 391759, 391760, 391761, 391763, 391768, 391777, 391783, 391798, 391825, 391831,
                                 392328, 392329, 392344, 392364, 392366, 392389, 392390, 392391, 392393, 392400, 392407,
                                 392435, 392461, 392477, 392916, 392966, 392969, 392971, 392972, 392973, 392991, 392994,
                                 392996, 392999, 393000, 393002, 393019, 393020, 393021, 393023, 393032, 393037, 393043,
                                 393057, 393080, 393085, 393091, 393596, 393603, 393624, 393626, 393629, 393630, 393632,
                                 393642, 393649, 393650, 393651, 393653, 393667, 393673, 393677, 393715, 393721, 394007,
                                 394137, 394138, 394164, 394165, 394169, 394171, 394248, 394250, 394809, 394861, 394862,
                                 394881, 394889, 394890, 394892, 394910, 394911, 394913, 394929, 394933, 394948, 394970,
                                 394981, 395491, 395492, 395519, 395520, 395522, 395540, 395541, 395543, 395544, 395563,
                                 395570, 395585, 395600, 395611, 395898, 396033, 396034, 396035, 396054, 396055, 396061,
                                 396094, 396140, 396529, 396684, 396685, 396691, 396730};
    static const int C1_ind[] = {610, 1202, 1240, 1792, 1830, 1832, 1833, 1859, 1866, 1882, 2467, 3057, 3090, 3093,
                                 3097, 3119, 3142, 3742, 4332, 4350, 4353, 4372, 4379, 4402, 4991, 5020, 5581, 5610,
                                 5613, 5621, 5639, 5646, 5662, 6266, 6856, 6870, 6873, 6896, 6899, 6922, 7536, 7540,
                                 8126, 8130, 8133, 8159, 8166, 8182, 8640, 8692, 8720, 8721, 8723, 8756, 8791, 8794,
                                 9431, 10024, 10061, 10642, 10650, 10653, 10654, 10679, 10687, 10702, 11289, 11322,
                                 11878, 11910, 11913, 11919, 11939, 11948, 11962, 12563, 12583, 13170, 13173, 13193,
                                 13199, 13209, 13217, 13222, 13812, 13841, 14402, 14430, 14433, 14442, 14459, 14467,
                                 14482, 15087, 15677, 15690, 15693, 15717, 15719, 15742, 16357, 16361, 16947, 16950,
                                 16953, 16979, 16987, 17002, 17508, 17514, 17516, 17539, 17540, 17541, 17543, 17577,
                                 17611, 18252, 18855, 18882, 19449, 19470, 19473, 19485, 19499, 19508, 19522, 20130,
                                 20143, 20720, 20730, 20733, 20759, 20760, 20769, 20782, 21398, 21402, 21988, 21990,
                                 21993, 22019, 22028, 22042, 22504, 22559, 22562, 22580, 22581, 22583, 22603, 22618,
                                 22651, 23293, 23919, 23923, 24510, 24513, 24515, 24539, 24549, 25030, 25100, 25101,
                                 25145, 25171};

    MatrixXd C0 = MatrixXd::Zero(630, 630);
    MatrixXd C1 = MatrixXd::Zero(630, 40);
    for (int i = 0; i < 18764; i++) { C0(C0_ind[i]) = coeffs(coeffs0_ind[i]); }
    for (int i = 0; i < 166; i++) { C1(C1_ind[i]) = coeffs(coeffs1_ind[i]); }

    MatrixXd C12 = C0.partialPivLu().solve(C1);



    // Setup action matrix
    Matrix<double, 56, 40> RR;
    RR << -C12.bottomRows(16), Matrix<double, 40, 40>::Identity(40, 40);

    static const int AM_ind[] = {27, 18, 0, 20, 1, 22, 2, 24, 3, 26, 4, 28, 29, 5, 41, 32, 6, 34, 7, 36, 8, 38, 9, 40,
                                 10, 42, 43, 11, 49, 46, 12, 48, 13, 50, 51, 14, 53, 54, 55, 15};
    Matrix<double, 40, 40> AM;
    for (int i = 0; i < 40; i++) {
        AM.row(i) = RR.row(AM_ind[i]);
    }

    Matrix<std::complex<double>, 4, 40> sols;
    sols.setZero();

    // Solve eigenvalue problem
    EigenSolver<Matrix<double, 40, 40> > es(AM);
    ArrayXcd D = es.eigenvalues();
    ArrayXXcd V = es.eigenvectors();
    ArrayXXcd scale = (D.transpose() / (V.row(36).array() * V.row(36).array())).sqrt();
    V = (V * scale.replicate(40, 1)).eval();


    sols.row(0) = V.row(0).array();
    sols.row(1) = V.row(14).array();
    sols.row(2) = V.row(28).array();
    sols.row(3) = V.row(36).array();


    return sols;
}

Eigen::Matrix3d rotation::optimal_rotation_using_R_qks_NEW(const Eigen::Vector3d &c_q,
                                                           const vector<Eigen::Matrix3d> &R_ks,
                                                           const vector<Eigen::Vector3d> &T_ks,
                                                           const vector<Eigen::Matrix3d> &R_qks,
                                                           const vector<Eigen::Vector3d> &T_qks) {

    int K = int(R_ks.size());

    double overall_best_error = DBL_MAX;
    Eigen::Matrix3d overall_best_R;

    for(int b = 0; b < pow(2, K); b++) {

        vector<int> betas (K);
        Eigen::Vector<double, 9> M{0,0,0,0,0,0,0,0,0};

        for(int k = 0; k < K; k++) {

            betas[k] = (b & int(pow(2, k))) ? 1 : -1;

            Vector3d c_k = -R_ks[k].transpose() * T_ks[k];
            Vector3d diff = c_k - c_q;
            diff.normalize();

            Vector3d T_qk = -R_qks[k] * (R_ks[k] * c_q + T_ks[k]);
            T_qk.normalize();

            Eigen::Vector<double, 9> star{T_qk[0] * diff[0], T_qk[0] * diff[1], T_qk[0] * diff[2],
                                          T_qk[1] * diff[0], T_qk[1] * diff[1], T_qk[1] * diff[2],
                                          T_qk[2] * diff[0], T_qk[2] * diff[1], T_qk[2] * diff[2]};

            M = M + betas[k] * star;
        }

        MatrixXcd sols = rotation::solver_problem_averageRQuatMetricAlter_red_new(M);


        Matrix3d best_R;
        double best_error = DBL_MAX;
        for (int i = 0; i < 4; i++) {

            Eigen::Quaterniond q;
            if (abs(sols(0, i).imag()) > 1e-6) continue;
            q.w() = sols(0, i).real();
            if (abs(sols(1, i).imag()) > 1e-6) continue;
            q.x() = sols(1, i).real();
            if (abs(sols(2, i).imag()) > 1e-6) continue;
            q.y() = sols(2, i).real();
            if (abs(sols(3, i).imag()) > 1e-6) continue;
            q.z() = sols(3, i).real();

            Eigen::Matrix3d R = q.normalized().toRotationMatrix();

            double error = 0;
            for (int k = 0; k < K; k++) {
                Vector3d c_k = -R_ks[k].transpose() * T_ks[k];

                Vector3d T_qk = -R_qks[k] * (R_ks[k] * c_q + T_ks[k]);
                T_qk.normalize();

                error += pow((R.transpose() * T_qk - betas[k] * (c_k - c_q).normalized()).norm(), 2.);
            }

            if (error < best_error) {
                best_error = error;
                best_R = R;
            }
        }

        if (best_error < overall_best_error) {
            overall_best_error = best_error;
            overall_best_R = best_R;
        }
    }

    return overall_best_R;
}

Eigen::Matrix3d rotation::optimal_rotation_using_T_qks_NEW(const Eigen::Vector3d &c_q,
                                                                const vector<Eigen::Matrix3d> &R_ks,
                                                                const vector<Eigen::Vector3d> &T_ks,
                                                                const vector<Eigen::Matrix3d> &R_qks,
                                                                const vector<Eigen::Vector3d> &T_qks) {

    int K = int(R_ks.size());

    double overall_best_error = DBL_MAX;
    Eigen::Matrix3d overall_best_R;

    for(int b = 0; b < pow(2, K); b++) {

        vector<int> betas (K);
        Eigen::Vector<double, 9> M{0,0,0,0,0,0,0,0,0};

        for(int k = 0; k < K; k++) {

            betas[k] = (b & int(pow(2, k))) ? 1 : -1;

            Vector3d c_k = -R_ks[k].transpose() * T_ks[k];
            Vector3d diff = c_k - c_q;
            diff.normalize();

            Vector3d T_qk = T_qks[k];
            Eigen::Vector<double, 9> star{T_qk[0] * diff[0], T_qk[0] * diff[1], T_qk[0] * diff[2],
                                          T_qk[1] * diff[0], T_qk[1] * diff[1], T_qk[1] * diff[2],
                                          T_qk[2] * diff[0], T_qk[2] * diff[1], T_qk[2] * diff[2]};

            M = M + betas[k] * star;
        }

        MatrixXcd sols = rotation::solver_problem_averageRQuatMetricAlter_red_new(M);


        Matrix3d best_R;
        double best_error = DBL_MAX;
        for (int i = 0; i < 4; i++) {

            Eigen::Quaterniond q;
            if (abs(sols(0, i).imag()) > 1e-6) continue;
            q.w() = sols(0, i).real();
            if (abs(sols(1, i).imag()) > 1e-6) continue;
            q.x() = sols(1, i).real();
            if (abs(sols(2, i).imag()) > 1e-6) continue;
            q.y() = sols(2, i).real();
            if (abs(sols(3, i).imag()) > 1e-6) continue;
            q.z() = sols(3, i).real();

            Eigen::Matrix3d R = q.normalized().toRotationMatrix();

            double error = 0;
            for (int k = 0; k < K; k++) {
                Vector3d c_k = -R_ks[k].transpose() * T_ks[k];
                Vector3d T_qk = T_qks[k];

                error += pow((R.transpose() * T_qk - betas[k] * (c_k - c_q).normalized()).norm(), 2.);
            }

            if (error < best_error) {
                best_error = error;
                best_R = R;
            }
        }

        if (best_error < overall_best_error) {
            overall_best_error = best_error;
            overall_best_R = best_R;
        }
    }

    return overall_best_R;
}

Eigen::Matrix3d rotation::optimal_rotation_using_R_qks_NEW_2(const Eigen::Vector3d &c_q,
                                                           const vector<Eigen::Matrix3d> &R_ks,
                                                           const vector<Eigen::Vector3d> &T_ks,
                                                           const vector<Eigen::Matrix3d> &R_qks,
                                                           const vector<Eigen::Vector3d> &T_qks) {

    int K = int(R_ks.size());

    double overall_best_error = DBL_MAX;
    Eigen::Matrix3d overall_best_R;

    for(int b = 0; b < pow(2, K); b++) {

        vector<int> betas (K);
        Eigen::Vector<double, 9> M{0,0,0,0,0,0,0,0,0};

        for(int k = 0; k < K; k++) {

            betas[k] = (b & int(pow(2, k))) ? 1 : -1;

            Vector3d c_k = -R_ks[k].transpose() * T_ks[k];
            Vector3d diff = c_k - c_q;
            diff = diff.norm() * diff;

            Vector3d T_qk = -R_qks[k] * (R_ks[k] * c_q + T_ks[k]);
            T_qk.normalize();

            Eigen::Vector<double, 9> star{T_qk[0] * diff[0], T_qk[0] * diff[1], T_qk[0] * diff[2],
                                          T_qk[1] * diff[0], T_qk[1] * diff[1], T_qk[1] * diff[2],
                                          T_qk[2] * diff[0], T_qk[2] * diff[1], T_qk[2] * diff[2]};

            M = M + betas[k] * star;
        }

        MatrixXcd sols = rotation::solver_problem_averageRQuatMetricAlter_red_new(M);


        Matrix3d best_R;
        double best_error = DBL_MAX;
        for (int i = 0; i < 4; i++) {

            Eigen::Quaterniond q;
            if (abs(sols(0, i).imag()) > 1e-6) continue;
            q.w() = sols(0, i).real();
            if (abs(sols(1, i).imag()) > 1e-6) continue;
            q.x() = sols(1, i).real();
            if (abs(sols(2, i).imag()) > 1e-6) continue;
            q.y() = sols(2, i).real();
            if (abs(sols(3, i).imag()) > 1e-6) continue;
            q.z() = sols(3, i).real();

            Eigen::Matrix3d R = q.normalized().toRotationMatrix();

            double error = 0;
            for (int k = 0; k < K; k++) {
                Vector3d c_k = -R_ks[k].transpose() * T_ks[k];
                Vector3d diff = c_k - c_q;

                Vector3d T_qk = -R_qks[k] * (R_ks[k] * c_q + T_ks[k]);
                T_qk.normalize();

                error += pow((diff.norm() * R.transpose() * T_qk - betas[k] * diff).norm(), 2.);
            }

            if (error < best_error) {
                best_error = error;
                best_R = R;
            }
        }

        if (best_error < overall_best_error) {
            overall_best_error = best_error;
            overall_best_R = best_R;
        }
    }

    return overall_best_R;
}

Eigen::Matrix3d rotation::optimal_rotation_using_T_qks_NEW_2(const Eigen::Vector3d &c_q,
                                                           const vector<Eigen::Matrix3d> &R_ks,
                                                           const vector<Eigen::Vector3d> &T_ks,
                                                           const vector<Eigen::Matrix3d> &R_qks,
                                                           const vector<Eigen::Vector3d> &T_qks) {

    int K = int(R_ks.size());

    double overall_best_error = DBL_MAX;
    Eigen::Matrix3d overall_best_R;

    for(int b = 0; b < pow(2, K); b++) {

        vector<int> betas (K);
        Eigen::Vector<double, 9> M{0,0,0,0,0,0,0,0,0};

        for(int k = 0; k < K; k++) {

            betas[k] = (b & int(pow(2, k))) ? 1 : -1;

            Vector3d c_k = -R_ks[k].transpose() * T_ks[k];
            Vector3d diff = c_k - c_q;
            diff = diff.norm() * diff;

            Vector3d T_qk = T_qks[k];

            Eigen::Vector<double, 9> star{T_qk[0] * diff[0], T_qk[0] * diff[1], T_qk[0] * diff[2],
                                          T_qk[1] * diff[0], T_qk[1] * diff[1], T_qk[1] * diff[2],
                                          T_qk[2] * diff[0], T_qk[2] * diff[1], T_qk[2] * diff[2]};

            M = M + betas[k] * star;
        }

        MatrixXcd sols = rotation::solver_problem_averageRQuatMetricAlter_red_new(M);


        Matrix3d best_R;
        double best_error = DBL_MAX;
        for (int i = 0; i < 4; i++) {

            Eigen::Quaterniond q;
            if (abs(sols(0, i).imag()) > 1e-6) continue;
            q.w() = sols(0, i).real();
            if (abs(sols(1, i).imag()) > 1e-6) continue;
            q.x() = sols(1, i).real();
            if (abs(sols(2, i).imag()) > 1e-6) continue;
            q.y() = sols(2, i).real();
            if (abs(sols(3, i).imag()) > 1e-6) continue;
            q.z() = sols(3, i).real();

            Eigen::Matrix3d R = q.normalized().toRotationMatrix();

            double error = 0;
            for (int k = 0; k < K; k++) {
                Vector3d c_k = -R_ks[k].transpose() * T_ks[k];
                Vector3d diff = c_k - c_q;

                Vector3d T_qk = T_qks[k];

                error += pow((diff.norm() * R.transpose() * T_qk - betas[k] * diff).norm(), 2.);
            }

            if (error < best_error) {
                best_error = error;
                best_R = R;
            }
        }

        if (best_error < overall_best_error) {
            overall_best_error = best_error;
            overall_best_R = best_R;
        }
    }

    return overall_best_R;
}

MatrixXcd rotation::solver_problem_averageRQuatMetricAlter_red_new(const VectorXd& data)
{
    // Compute coefficients
    const double* d = data.data();
    VectorXd coeffs(32);
    coeffs[0] = 2*d[5] - 2*d[7];
    coeffs[1] = 4*d[4] + 4*d[8];
    coeffs[2] = -2*d[5] + 2*d[7];
    coeffs[3] = -2*d[2] + 2*d[6];
    coeffs[4] = -4*d[1] - 4*d[3];
    coeffs[5] = 2*d[2] - 2*d[6];
    coeffs[6] = 4*d[0] + 4*d[8];
    coeffs[7] = 2*d[1] - 2*d[3];
    coeffs[8] = -4*d[2] - 4*d[6];
    coeffs[9] = -2*d[1] + 2*d[3];
    coeffs[10] = -4*d[5] - 4*d[7];
    coeffs[11] = 4*d[0] + 4*d[4];
    coeffs[12] = -4*d[4] - 4*d[8];
    coeffs[13] = 2*d[1] + 2*d[3];
    coeffs[14] = -4*d[2] + 4*d[6];
    coeffs[15] = -2*d[1] - 2*d[3];
    coeffs[16] = 4*d[0] - 4*d[4];
    coeffs[17] = 2*d[2] + 2*d[6];
    coeffs[18] = 4*d[1] - 4*d[3];
    coeffs[19] = -2*d[2] - 2*d[6];
    coeffs[20] = 4*d[0] - 4*d[8];
    coeffs[21] = -4*d[0] - 4*d[8];
    coeffs[22] = 4*d[5] - 4*d[7];
    coeffs[23] = -4*d[0] + 4*d[4];
    coeffs[24] = 2*d[5] + 2*d[7];
    coeffs[25] = -2*d[5] - 2*d[7];
    coeffs[26] = 4*d[4] - 4*d[8];
    coeffs[27] = -4*d[0] - 4*d[4];
    coeffs[28] = -4*d[0] + 4*d[8];
    coeffs[29] = -4*d[4] + 4*d[8];
    coeffs[30] = 1;
    coeffs[31] = -1;



    // Setup elimination template
    static const int coeffs0_ind[] = { 2,5,9,30,12,5,13,9,17,30,0,13,0,5,17,9,30,30,5,1,13,9,17,30,13,21,2,9,24,30,14,21,3,22,12,0,17,24,30,15,22,4,23,0,1,9,24,30,23,5,2,17,24,30,2,3,13,3,24,9,30,30,5,9,2,30,13,5,9,17,12,30,5,13,0,17,9,0,30,30,13,5,1,9,17,30,17,24,27,21,9,2,5,24,13,30,18,24,7,27,22,22,21,3,24,17,0,12,13,14,30,19,8,24,22,28,23,22,4,9,1,0,5,24,15,30,24,9,28,23,5,24,17,2,13,30,18,17,7,27,14,3,24,3,13,21,9,2,30,30,10,18,10,8,18,8,22,14,4,15,3,6,9,4,14,22,17,16,30,24,27,17,5,2,30,24,7,27,22,18,13,0,12,30,24,8,22,28,19,5,1,0,30,24,9,28,13,2,30,2,5,7,18,27,7,17,24,14,21,3,13,30,30,20,5,11,13,7,19,8,18,10,14,22,8,18,4,10,22,4,14,30,5,7,2,24,7,17,30,30,5,9,30,13,17,30,23,24,30,31,15,17,30,31,31,24,30,24,27,30,31,22,31,24,28,30,31,8,4,31,31,25,29,30,5,7,31,30,13,19,30,31,31,26,25,30,31,24,30,21,24,30,31,22,31,3,9,30,31,18,14,31 };
    static const int coeffs1_ind[] = { 31,31,31,31 };
    static const int C0_ind[] = { 0,3,8,43,44,45,47,51,52,85,88,89,90,91,95,96,123,131,133,134,135,139,140,173,176,179,180,182,184,215,220,221,222,223,224,225,226,227,252,264,265,266,267,268,269,270,272,303,309,310,313,314,315,340,352,355,356,357,358,360,376,395,405,413,414,438,449,450,452,457,458,480,493,494,495,496,501,502,517,526,537,538,539,540,545,568,572,575,580,581,585,587,588,589,590,610,616,617,618,623,624,625,626,627,628,629,630,631,632,634,650,660,662,663,667,668,669,670,671,673,674,675,676,677,678,698,705,706,711,714,715,716,717,718,720,738,751,752,753,754,756,757,761,762,763,764,765,766,770,790,792,793,794,795,796,797,798,799,800,801,802,803,804,806,807,808,809,810,832,845,853,854,855,857,872,890,891,892,897,898,899,900,901,910,933,935,936,941,942,943,944,945,960,978,979,980,987,988,998,1012,1015,1020,1021,1025,1026,1027,1028,1029,1031,1032,1033,1035,1055,1056,1057,1058,1059,1063,1064,1065,1066,1067,1068,1069,1070,1071,1073,1074,1075,1076,1077,1097,1109,1117,1118,1119,1120,1121,1125,1142,1170,1171,1181,1214,1215,1217,1258,1259,1260,1267,1302,1303,1305,1308,1344,1347,1348,1390,1391,1395,1406,1435,1448,1478,1479,1483,1485,1522,1523,1530,1562,1566,1567,1571,1610,1611,1620,1621,1654,1655,1657,1658,1695,1698,1699,1700,1741,1742,1747,1786,1787,1788,1803,1830,1845,1874,1875,1885,1887,1918,1919,1930 } ;
    static const int C1_ind[] = { 37,73,116,163 };

    Matrix<double,44,44> C0; C0.setZero();
    Matrix<double,44,4> C1; C1.setZero();
    for (int i = 0; i < 302; i++) { C0(C0_ind[i]) = coeffs(coeffs0_ind[i]); }
    for (int i = 0; i < 4; i++) { C1(C1_ind[i]) = coeffs(coeffs1_ind[i]); }

    Matrix<double,44,4> C12 = C0.partialPivLu().solve(C1);




    // Setup action matrix
    Matrix<double,8, 4> RR;
    RR << -C12.bottomRows(4), Matrix<double,4,4>::Identity(4, 4);

    static const int AM_ind[] = { 0,1,2,3 };
    Matrix<double, 4, 4> AM;
    for (int i = 0; i < 4; i++) {
        AM.row(i) = RR.row(AM_ind[i]);
    }

    Matrix<std::complex<double>, 4, 4> sols;
    sols.setZero();

    // Solve eigenvalue problem
    EigenSolver<Matrix<double, 4, 4> > es(AM);
    ArrayXcd D = es.eigenvalues();
    ArrayXXcd V = es.eigenvectors();
    ArrayXXcd scale = (D.transpose() / (V.row(0).array()*V.row(2).array())).sqrt();
    V = (V * scale.replicate(4, 1)).eval();


    sols.row(0) = V.row(0).array();
    sols.row(1) = V.row(1).array();
    sols.row(2) = V.row(2).array();
    sols.row(3) = V.row(3).array();






    return sols;
}