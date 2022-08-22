#include "include/aachen.h"
#include "include/sevenScenes.h"
#include "include/synthetic.h"
#include "include/functions.h"
#include "include/poseEstimation.h"
#include "include/calibrate.h"
#include <iostream>
#include <fstream>
#include <Eigen/SVD>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <opencv2/xfeatures2d.hpp>
#include "include/OptimalRotationSolver.h"
#include "include/CambridgeLandmarks.h"


#define PI 3.1415926536

#define FOLDER "/Users/cameronfiore/C++/image_localization_project/data/"
//#define SCENE 2
#define IMAGE_LIST "/Users/cameronfiore/C++/image_localization_project/data/images_1000.txt"
#define EXT ".color.png"
#define GENERATE_DATABASE false

using namespace std;
using namespace chrono;

int main() {

    string folder = "StMarysChurch";
    int num = 150;
    int rows = 10;
    int cols = 15;
    assert(rows*cols == num);

    vector<string> queries;// = cambridge::getTestImages("/Users/cameronfiore/C++/image_localization_project/data/"+folder+"/");
    vector<tuple<double, double, double>> calibrations;
    aachen::createQueryVector(queries, calibrations);

    for (int q = 0; q < queries.size(); q++) {

        cout << q << "/" << int(queries.size()) << ":" << endl;

        string query = queries[q];

//        vector<string> retrieved = cambridge::retrieveSimilar(query, num);
        vector<string> retrieved = aachen::retrieveSimilar(query, num);

//        vector<string> closest = cambridge::findClosest(query, "/Users/cameronfiore/C++/image_localization_project/data/"+folder+"/", num);
        vector<string> closest = aachen::findClosest(retrieved[0], num);

        cv::Mat query_pic = cv::imread(query);
        cv::imshow(query, query_pic);
        cv::waitKey();

        for (const auto & im : retrieved) {
            cv::Mat im_pic = cv::imread(im);
            cv::imshow(im, im_pic);
            cv::waitKey();
        }
        
//        functions::showTop(rows, cols, query, retrieved, "", "Top "+to_string(num));
//        functions::showTop(rows, cols, query, closest, "", "Closest "+to_string(num));

    }

    return 0;
}