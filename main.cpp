
#include "include/sevenScenes.h"
#include "include/functions.h"
#include "include/Space.h"
#include "include/calibrate.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"


#define FOLDER "/Users/cameronfiore/C++/image_localization_project/data/"
#define SCENE 0
#define IMAGE_LIST "/Users/cameronfiore/C++/image_localization_project/data/images_1000.txt"
#define EXT ".color.png"
#define GENERATE_DATABASE false

using namespace std;
using namespace cv;

int main() {

//    calibrate::run();

//    string query = "/Users/cameronfiore/C++/image_localization_project/data/chess/seq-03/frame-000400";
//    utils::showTop1000(query, 200, 1.6, 20);
//    auto retrieved = utils::retrieveSimilar(query, 200, 1.6);
//    vector<pair<string, cv::Scalar>> seq_colors;
//    utils::projectCentersTo2D(query, retrieved, seq_colors, query + " Retrieved");
//    auto spaced = utils::optimizeSpacing(retrieved, 20);
//    utils::projectCentersTo2D(query, spaced, seq_colors, query + " Spaced");


//    string query = "/Users/cameronfiore/C++/image_localization_project/data/fire/seq-03/frame-000300";
////    showTop1000(query, 200, 1.6, 20);
//    auto retrieved = utils::retrieveSimilar(query, 200, 1.6);
//    vector<pair<string, cv::Scalar>> seq_colors;
//    utils::projectCentersTo2D(query, retrieved, seq_colors, query + " Retrieved");
//    auto spaced = utils::optimizeSpacing(retrieved, 20);
//    utils::projectCentersTo2D(query, spaced, seq_colors, query + " Spaced");
//
//    string query = "/Users/cameronfiore/C++/image_localization_project/data/heads/seq-01/frame-000495";
////   utils::showTop1000(query, 200, 1.6, 20);
//    auto retrieved = utils::retrieveSimilar(query, 200, 1.6);
//    vector<pair<string, cv::Scalar>> seq_colors;
//    utils::projectCentersTo2D(query, retrieved, seq_colors, query + " Retrieved");
//    auto spaced = utils::optimizeSpacing(retrieved, 20);
//    utils::projectCentersTo2D(query, spaced, seq_colors, query + " Spaced");
//
//    string query = "/Users/cameronfiore/C++/image_localization_project/data/office/seq-06/frame-000160";
////    showTop1000(query, 200, 1.6, 20);
//    auto retrieved = retrieveSimilar(query, 200, 1.6);
//    vector<pair<string, cv::Scalar>> seq_colors;
//    projectCentersTo2D(query, retrieved, seq_colors, query + " Retrieved");
//    auto spaced = processing::optimizeSpacing(retrieved, 20);
//    projectCentersTo2D(query, spaced, seq_colors, query + " Spaced");
//
//    string query = "/Users/cameronfiore/C++/image_localization_project/data/pumpkin/seq-07/frame-000490";
////    showTop1000(query, 200, 1.6, 20);
//    auto retrieved = retrieveSimilar(query, 200, 1.6);
//    vector<pair<string, cv::Scalar>> seq_colors;
//    projectCentersTo2D(query, retrieved, seq_colors, query + " Retrieved");
//    auto spaced = processing::optimizeSpacing(retrieved, 20);
//    projectCentersTo2D(query, spaced, seq_colors, query + " Spaced");

//    string query = "/Users/cameronfiore/C++/image_localization_project/data/redkitchen/seq-06/frame-000490";
////    showTop1000(query, 200, 1.6, 20);
//    auto retrieved = retrieveSimilar(query, 200, 1.6);
//    vector<pair<string, cv::Scalar>> seq_colors;
//    projectCentersTo2D(query, retrieved, seq_colors, query + " Retrieved");
//    auto spaced = processing::optimizeSpacing(retrieved, 20);
//    projectCentersTo2D(query, spaced, seq_colors, query + " Spaced");

//    string query = "/Users/cameronfiore/C++/image_localization_project/data/stairs/seq-01/frame-000260";
////    showTop1000(query, 200, 1.6, 20);
//    auto retrieved = retrieveSimilar(query, 200, 1.6);
//    vector<pair<string, cv::Scalar>> seq_colors;
//    projectCentersTo2D(query, retrieved, seq_colors, query + " Retrieved");
//    auto spaced = processing::optimizeSpacing(retrieved, 20);
//    projectCentersTo2D(query, spaced, seq_colors, query + " Spaced");

//    string query = "/Users/cameronfiore/C++/image_localization_project/data/heads/seq-01/frame-000654";
////    showTop1000(query, 200, 1.6, 20);
//    auto retrieved = retrieveSimilar(query, 200, 1.6);
//    vector<pair<string, cv::Scalar>> seq_colors;
//    projectCentersTo2D(query, retrieved, seq_colors, query + " Retrieved");
//    auto spaced = processing::optimizeSpacing(retrieved, 20);
//    projectCentersTo2D(query, spaced, seq_colors, query + " Spaced");

//    string query = "/Users/cameronfiore/C++/image_localization_project/data/office/seq-06/frame-000058";
////    showTop1000(query, 200, 1.6, 20);
//    auto retrieved = retrieveSimilar(query, 200, 1.6);
//    vector<pair<string, cv::Scalar>> seq_colors;
//    projectCentersTo2D(query, retrieved, seq_colors, query + " Retrieved");
//    auto spaced = processing::optimizeSpacing(retrieved, 20);
//    projectCentersTo2D(query, spaced, seq_colors, query + " Spaced");
//
//    string query = "/Users/cameronfiore/C++/image_localization_project/data/pumpkin/seq-07/frame-000647";
////    showTop1000(query, 200, 1.6, 20);
//    auto retrieved = retrieveSimilar(query, 200, 1.6);
//    vector<pair<string, cv::Scalar>> seq_colors;
//    projectCentersTo2D(query, retrieved, seq_colors, query + " Retrieved");
//    auto spaced = processing::optimizeSpacing(retrieved, 20);
//    projectCentersTo2D(query, spaced, seq_colors, query + " Spaced");

//    string query = "/Users/cameronfiore/C++/image_localization_project/data/redkitchen/seq-04/frame-000373";
////    showTop1000(query, 200, 1.6, 20);
//    auto retrieved = retrieveSimilar(query, 200, 1.6);
//    vector<pair<string, cv::Scalar>> seq_colors;
//    projectCentersTo2D(query, retrieved, seq_colors, query + " Retrieved");
//    auto spaced = processing::optimizeSpacing(retrieved, 20);
//    projectCentersTo2D(query, spaced, seq_colors, query + " Spaced");




//    string query = "/Users/cameronfiore/C++/image_localization_project/data/office/seq-01/frame-000200";
//
//    vector<string> ensemble {
//            "/Users/cameronfiore/C++/image_localization_project/data/office/seq-01/frame-000175",
//            "/Users/cameronfiore/C++/image_localization_project/data/office/seq-01/frame-000180",
//            "/Users/cameronfiore/C++/image_localization_project/data/office/seq-01/frame-000185",
//            "/Users/cameronfiore/C++/image_localization_project/data/office/seq-01/frame-000190",
//            "/Users/cameronfiore/C++/image_localization_project/data/office/seq-01/frame-000195",
//            "/Users/cameronfiore/C++/image_localization_project/data/office/seq-01/frame-000205",
//            "/Users/cameronfiore/C++/image_localization_project/data/office/seq-01/frame-000210",
//            "/Users/cameronfiore/C++/image_localization_project/data/office/seq-01/frame-000215",
//            "/Users/cameronfiore/C++/image_localization_project/data/office/seq-01/frame-000220",
//            "/Users/cameronfiore/C++/image_localization_project/data/office/seq-01/frame-000225"
//    };
//
//
//    string query = "/Users/cameronfiore/C++/image_localization_project/data/chess/seq-01/frame-000350";
//
//    vector<string> ensemble {
//            "/Users/cameronfiore/C++/image_localization_project/data/chess/seq-01/frame-000325",
//            "/Users/cameronfiore/C++/image_localization_project/data/chess/seq-01/frame-000330",
//            "/Users/cameronfiore/C++/image_localization_project/data/chess/seq-01/frame-000335",
//            "/Users/cameronfiore/C++/image_localization_project/data/chess/seq-01/frame-000340",
//            "/Users/cameronfiore/C++/image_localization_project/data/chess/seq-01/frame-000345",
//            "/Users/cameronfiore/C++/image_localization_project/data/chess/seq-01/frame-000355",
//            "/Users/cameronfiore/C++/image_localization_project/data/chess/seq-01/frame-000360",
//            "/Users/cameronfiore/C++/image_localization_project/data/chess/seq-01/frame-000365",
//            "/Users/cameronfiore/C++/image_localization_project/data/chess/seq-01/frame-000370",
//            "/Users/cameronfiore/C++/image_localization_project/data/chess/seq-01/frame-000375"
//    };

//    string query = "/Users/cameronfiore/C++/image_localization_project/data/fire/seq-01/frame-000100";
//
//    vector<string> ensemble {
//            "/Users/cameronfiore/C++/image_localization_project/data/fire/seq-01/frame-000075",
//            "/Users/cameronfiore/C++/image_localization_project/data/fire/seq-01/frame-000080",
//            "/Users/cameronfiore/C++/image_localization_project/data/fire/seq-01/frame-000085",
//            "/Users/cameronfiore/C++/image_localization_project/data/fire/seq-01/frame-000090",
//            "/Users/cameronfiore/C++/image_localization_project/data/fire/seq-01/frame-000095",
//            "/Users/cameronfiore/C++/image_localization_project/data/fire/seq-01/frame-000105",
//            "/Users/cameronfiore/C++/image_localization_project/data/fire/seq-01/frame-000110",
//            "/Users/cameronfiore/C++/image_localization_project/data/fire/seq-01/frame-000115",
//            "/Users/cameronfiore/C++/image_localization_project/data/fire/seq-01/frame-000120",
//            "/Users/cameronfiore/C++/image_localization_project/data/fire/seq-01/frame-000125"
//    };

//    string query = "/Users/cameronfiore/C++/image_localization_project/data/heads/seq-01/frame-000100";
//
//    vector<string> ensemble {
//            "/Users/cameronfiore/C++/image_localization_project/data/heads/seq-01/frame-000075",
//            "/Users/cameronfiore/C++/image_localization_project/data/heads/seq-01/frame-000080",
//            "/Users/cameronfiore/C++/image_localization_project/data/heads/seq-01/frame-000085",
//            "/Users/cameronfiore/C++/image_localization_project/data/heads/seq-01/frame-000090",
//            "/Users/cameronfiore/C++/image_localization_project/data/heads/seq-01/frame-000095",
//            "/Users/cameronfiore/C++/image_localization_project/data/heads/seq-01/frame-000105",
//            "/Users/cameronfiore/C++/image_localization_project/data/heads/seq-01/frame-000110",
//            "/Users/cameronfiore/C++/image_localization_project/data/heads/seq-01/frame-000115",
//            "/Users/cameronfiore/C++/image_localization_project/data/heads/seq-01/frame-000120",
//            "/Users/cameronfiore/C++/image_localization_project/data/heads/seq-01/frame-000125"
//    };

//    string query = "/Users/cameronfiore/C++/image_localization_project/data/pumpkin/seq-01/frame-000100";
//
//    vector<string> ensemble {
//            "/Users/cameronfiore/C++/image_localization_project/data/pumpkin/seq-01/frame-000075",
//            "/Users/cameronfiore/C++/image_localization_project/data/pumpkin/seq-01/frame-000080",
//            "/Users/cameronfiore/C++/image_localization_project/data/pumpkin/seq-01/frame-000085",
//            "/Users/cameronfiore/C++/image_localization_project/data/pumpkin/seq-01/frame-000090",
//            "/Users/cameronfiore/C++/image_localization_project/data/pumpkin/seq-01/frame-000095",
//            "/Users/cameronfiore/C++/image_localization_project/data/pumpkin/seq-01/frame-000105",
//            "/Users/cameronfiore/C++/image_localization_project/data/pumpkin/seq-01/frame-000110",
//            "/Users/cameronfiore/C++/image_localization_project/data/pumpkin/seq-01/frame-000115",
//            "/Users/cameronfiore/C++/image_localization_project/data/pumpkin/seq-01/frame-000120",
//            "/Users/cameronfiore/C++/image_localization_project/data/pumpkin/seq-01/frame-000125"
//    };

//    string query = "/Users/cameronfiore/C++/image_localization_project/data/redkitchen/seq-01/frame-000100";
//
//    vector<string> ensemble {
//            "/Users/cameronfiore/C++/image_localization_project/data/redkitchen/seq-01/frame-000075",
//            "/Users/cameronfiore/C++/image_localization_project/data/redkitchen/seq-01/frame-000080",
//            "/Users/cameronfiore/C++/image_localization_project/data/redkitchen/seq-01/frame-000085",
//            "/Users/cameronfiore/C++/image_localization_project/data/redkitchen/seq-01/frame-000090",
//            "/Users/cameronfiore/C++/image_localization_project/data/redkitchen/seq-01/frame-000095",
//            "/Users/cameronfiore/C++/image_localization_project/data/redkitchen/seq-01/frame-000105",
//            "/Users/cameronfiore/C++/image_localization_project/data/redkitchen/seq-01/frame-000110",
//            "/Users/cameronfiore/C++/image_localization_project/data/redkitchen/seq-01/frame-000115",
//            "/Users/cameronfiore/C++/image_localization_project/data/redkitchen/seq-01/frame-000120",
//            "/Users/cameronfiore/C++/image_localization_project/data/redkitchen/seq-01/frame-000125"
//    };

//    string query = "/Users/cameronfiore/C++/image_localization_project/data/stairs/seq-01/frame-000100";
//
//    vector<string> ensemble {
//            "/Users/cameronfiore/C++/image_localization_project/data/stairs/seq-01/frame-000075",
//            "/Users/cameronfiore/C++/image_localization_project/data/stairs/seq-01/frame-000080",
//            "/Users/cameronfiore/C++/image_localization_project/data/stairs/seq-01/frame-000085",
//            "/Users/cameronfiore/C++/image_localization_project/data/stairs/seq-01/frame-000090",
//            "/Users/cameronfiore/C++/image_localization_project/data/stairs/seq-01/frame-000095",
//            "/Users/cameronfiore/C++/image_localization_project/data/stairs/seq-01/frame-000105",
//            "/Users/cameronfiore/C++/image_localization_project/data/stairs/seq-01/frame-000110",
//            "/Users/cameronfiore/C++/image_localization_project/data/stairs/seq-01/frame-000115",
//            "/Users/cameronfiore/C++/image_localization_project/data/stairs/seq-01/frame-000120",
//            "/Users/cameronfiore/C++/image_localization_project/data/stairs/seq-01/frame-000125"
//    };
//
//    Eigen::Matrix3d R = sevenScenes::getR(query).transpose();
//    Eigen::Vector3d t = -R * sevenScenes::getT(query);
//
//    Eigen::Vector3d c_q = processing::hypothesizeQueryCenter(query, ensemble, "7-Scenes");
//
//    vector<Eigen::Matrix3d> rotations;
//    for (const auto & im: ensemble) {
//        Eigen::Matrix3d R_k = sevenScenes::getR(im).transpose();
//        Eigen::Matrix3d R_qk;
//        Eigen::Vector3d t_qk;
//        processing::getRelativePose(im, query, "SURF", R_qk, t_qk);
//        rotations.push_back(R_qk * R_k);
//    }
//    Eigen::Matrix3d R_q = processing::rotationAverage(rotations);
//
//    double c_dist = processing::getDistBetween(c_q, sevenScenes::getT(query));
//    double t_dist = processing::getDistBetween(-R_q * c_q, t);
//    double angle = processing::rotationDifference(R, R_q);
//
//
//    cout << c_dist << endl;
//    cout << t_dist << endl;
//    cout << angle << endl;






//    vector<string> listImage, listQuery;
//    vector<tuple<string, string, vector<string>, vector<string>>> info = sevenScenes::createInfoVector();
//    createImageVector(listImage, info, SCENE);
//    createQueryVector(listQuery, info, SCENE);

    //Loading specific to SURF'''
//    if (GENERATE_DATABASE) {
//        vector<vector<vector<float>>> features;
//        loadFeaturesSURF(listImage, features);
//        DatabaseSaveSURF(features);
//    }
//    string dbFile = "db.yml.gz";
//    imageMatcher_Orb im(IMAGE_LIST, FOLDER + dbFile, "SURF", "vocabularyTree");
//    cout << "Loading done!" << endl;

    //Loading Specific to ORB
//    if (GENERATE_DATABASE)
//    {
//        vector<vector<cv::Mat>> features;
//        loadFeaturesORB(listImage, features);
//        DatabaseSaveORB(features);
//    }

//    string dbFile = "db_orb.yml.gz";
//    imageMatcher_Orb im(IMAGE_LIST, FOLDER + dbFile, "ORB", "vocabularyTree");
//    cout << "Loading done!" << endl;











//    cout << "Running queries..." << endl;
//    int startIdx = 0;
//    double c_error = 0.;
//    double r_error = 0.;
//    double t_error = 0.;
//    int total = 0;
//    int top_k = 50;
//    double threshold = 0.01; //degrees
//    for (int i = startIdx; i < listQuery.size(); ++i) {
//        string query = listQuery[i];
//
//        vector<pair<string, double>> images = getTopN(query, 1000);
//
//
//        vector<Eigen::Matrix3d> R_k;
//        vector<Eigen::Vector3d> t_k;
//        vector<Eigen::Matrix3d> R_qk;
//        vector<Eigen::Vector3d> t_qk;
//        processing::getEnsemble(top_k, "7-Scenes", query, images, R_k, t_k, R_qk, t_qk);
//
//        Eigen::Vector3d c_q;
//        Eigen::Matrix3d R_q;
//        processing::useRANSAC(R_k, t_k, R_qk, t_qk, R_q, c_q, threshold);
//        Eigen::Vector3d t_q = -R_q * c_q;
//
//        Eigen::Vector3d c = sevenScenes::getT(query);
//        Eigen::Matrix3d R = sevenScenes::getR(query).transpose();
//        Eigen::Vector3d t = -R * c;
//
//        double c_dist = processing::getDistBetween(c_q, c);
//        double r_dist = processing::rotationDifference(R_q, R);
//        double t_dist = processing::getDistBetween(t_q, t);
//
//        c_error += c_dist;
//        r_error += r_dist;
//        t_error += t_dist;
//        total++;
//
//        cout
//        << total << "/" << listQuery.size() - startIdx << "  "
//        << "Center error: " << to_string(c_error/total)
//        << ", Rotational error: " << to_string(r_error/total)
//        << ", Translation error: " << to_string(t_error/total) << endl;
//    }
    return 0;
}