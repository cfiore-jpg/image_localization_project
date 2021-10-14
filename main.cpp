#include "include/imageMatcher_Orb.h"
#include "include/imageMatcher.h"
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <base/database.h>
#include <base/image.h>
#include "include/sevenScenes.h"
#include "include/KDTree.h"


#define FOLDER "/Users/cameronfiore/C++/ImageMatcherProject/data/chess/"
#define IMAGE_LIST "/Users/cameronfiore/C++/ImageMatcherProject/data/images_1000.txt"
#define EXT ".color.png"
//#define MIN 0.5
//#define MED 2.0
//#define MAX 5.0
#define GENERATE_DATABASE false

using namespace std;



void createImageVector(vector<string> &listImage, vector<tuple<string, string, vector<string>, vector<string>>> &info, int scene);
void createQueryVector(vector<string> &listQuery, vector<tuple<string, string, vector<string>, vector<string>>> &info, int scene);

//SURF Functions
void loadFeaturesSURF(const vector<string> &listImage, vector<vector<vector<float>>> &features);
void changeStructureSURF(const vector<float> &plain, vector<vector<float>> &out, int L);
void DatabaseSaveSURF(const vector<vector<vector<float>>> &features);

// ORB functions
void loadFeaturesORB(const vector<string> &listImage, vector<vector<cv::Mat>> &features);
void changeStructureORB(const cv::Mat &plain, vector<cv::Mat> &out);
void DatabaseSaveORB(const vector<vector<cv::Mat>> &features);

void saveResultVector(const vector<pair<int, float>>& result, const string& query_image, const string& method);
vector<pair<int, float>> getResultVector(const string& query_image, const string& method);

int main()
{
    vector<string> listImage, listQuery;

    vector<tuple<string, string, vector<string>, vector<string>>> info = sevenScenes::createInfoVector();

    int scene = 0;
    createImageVector(listImage, info, scene);
    createQueryVector(listQuery, info, scene);

//    cout << "Creating tree..." << endl;
//    KDTree tree(listImage);
//    cout << "done" << endl;
//
//    double total;
//    for (string i : listQuery)
//    {
//        Node* trueBest = tree.nearestNeighbor(tree.getRoot(), 0, sevenScenes::getCameraCenter(i), tree.getRoot());
//
//        double dist = node::getDistanceFrom(trueBest, sevenScenes::getCameraCenter(i));
//
//        total += dist;
//
//        cout << "Best match for image " << i << " is " << trueBest->name << " whose distance is " << dist << "m" << endl;
//    }
//
//    cout << "Average true error: " << total/listQuery.size() << endl;

    if (GENERATE_DATABASE) {
        vector<vector<vector<float>>> features;
        loadFeaturesSURF(listImage, features);
        DatabaseSaveSURF(features);
    }

//    string dbFile = "db.yml.gz";
//    imageMatcher im (IMAGE_LIST, FOLDER + dbFile, "SURF", "vocabularyTree");
//    cout << "Loading done!" << endl;
//
//    cout << "Running queries..." << endl;
//    double totalError = 0;
//    int outOfBounds = 0;
//    double min = 0;
//    double med = 0;
//    double max = 0;
//    for (int i = 0; i < listQuery.size(); ++i) {
//        vector<pair<int, float>> result = im.matching_one_image(listQuery[i] + EXT);
//
//        stringstream q (listQuery[i]);
//        stringstream m (listImage[result[0].first]);
//        string segment;
//        vector<std::string> qSeglist;
//        vector<std::string> mSeglist;
//        while(std::getline(q, segment, '/'))
//        {
//            qSeglist.push_back(segment);
//        }
//        while(std::getline(m, segment, '/'))
//        {
//            mSeglist.push_back(segment);
//        }
//        if (qSeglist[6] != mSeglist[6])
//        {
//            ++outOfBounds;
//            continue;
//        }
//
//        Eigen::Vector3d queryCenter = sevenScenes::getCameraCenter(listQuery[i]);
//        Eigen::Vector3d bestMatchCenter = sevenScenes::getCameraCenter(listImage[result[0].first]);
//
//        double distX = queryCenter(0) - bestMatchCenter(0);
//        double distY = queryCenter(1) - bestMatchCenter(1);
//        double distZ = queryCenter(2) - bestMatchCenter(2);
//        double toAdd = sqrt(pow(distX, 2.0) + pow(distY, 2.0) + pow(distZ, 2.0));
//
//        totalError += toAdd;
//
//        for (pair<int, float> match : result)
//        {
//            Eigen::Vector3d imageCenter = sevenScenes::getCameraCenter(match.first, listImage);
//
//            distX = queryCenter(0) - imageCenter(0);
//            distY = queryCenter(1) - imageCenter(1);
//            distZ = queryCenter(2) - imageCenter(2);
//            double dist = sqrt(pow(distX, 2.0) + pow(distY, 2.0) + pow(distZ, 2.0));
//
//            if (dist > MIN)
//            {
//                ++min;
//            }
//            if (dist > MED)
//            {
//                ++med;
//            }
//            if (dist > MAX)
//            {
//                ++max;
//            }
//        }
//    }
//    cout << "Done!" << endl;
//
//    cout << "Average Error [m]: " << totalError / listQuery.size() << endl;
//    cout << "Average Number of Outliers (> 0.5): " << min / listQuery.size() << ", (> 2.0): " << med / listQuery.size() << ", (> 5.0): " << max / listQuery.size() << endl;
//    cout << "Percentage of best matches from outside the correct scene: " << (outOfBounds / listQuery.size()) * 100 << endl;
//


//    if (GENERATE_DATABASE)
//    {
//        vector<vector<cv::Mat>> features;
//        loadFeaturesORB(listImage, features);
//        DatabaseSaveORB(features);
//    }
//
//    string dbFile = "db_orb.yml.gz";
//    imageMatcher_Orb im(IMAGE_LIST, FOLDER + dbFile, "ORB", "vocabularyTree");
//    cout << "Loading done!" << endl;

    cout << "Running queries..." << endl;
    double totalError = 0;
    int totalRels = 0;
    int totalInliers = 0;
    double min = 0;
    double med = 0;
    double max = 0;
    int startIdx = 0;
    for (int i = startIdx; i < listQuery.size(); ++i)
    {
//        vector<pair<int, float>> result = im.matching_one_image(listQuery[i] + EXT);
//        saveResultVector(result, listQuery[i], "surf_jts");
        vector<pair<int, float>> result = getResultVector(listQuery[i], "surf");
        vector<tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d, int>> rel_images = sevenScenes::pairSelection(result, listImage, listQuery[i], 0.02, 10, 2);
        cout << listQuery[i] << " rel_images = " << rel_images.size();
        Eigen::Vector3d h;
        if(rel_images.empty())
        {
            h = sevenScenes::getT(listImage[result[0].first]);
        } else
        {
            h = sevenScenes::useRANSAC(rel_images,10);
        }
        Eigen::Vector3d queryCenter = sevenScenes::getT(listQuery[i]);
        double toAdd = sevenScenes::getDistBetween(queryCenter, h);
        totalError += toAdd;
        cout << ", Cur Error = " << toAdd << ", Avg Error = " << totalError / (i - startIdx + 1) << endl;

//        stringstream q(listQuery[i]);
//        stringstream m(listImage[result[0].first]);
//        string segment;
//        vector<std::string> qSeglist;
//        vector<std::string> mSeglist;
//        while (std::getline(q, segment, '/')) {
//            qSeglist.push_back(segment);
//        }
//        while (std::getline(m, segment, '/')) {
//            mSeglist.push_back(segment);
//        }
//        if (qSeglist[6] != mSeglist[6]) {
//            ++outOfBounds;
//            continue;
//        }

//        Eigen::Vector3d bestMatchCenter = sevenScenes::getT(listImage[result[0].first]);
//
//        double distX = queryCenter(0) - bestMatchCenter(0);
//        double distY = queryCenter(1) - bestMatchCenter(1);
//        double distZ = queryCenter(2) - bestMatchCenter(2);
//        double toAdd = sqrt(pow(distX, 2.0) + pow(distY, 2.0) + pow(distZ, 2.0));
//
//        totalError += toAdd;
//
//        for (pair<int, float> match : result) {
//            Eigen::Vector3d imageCenter = sevenScenes::getT(listImage[match.first]);
//
//            distX = queryCenter(0) - imageCenter(0);
//            distY = queryCenter(1) - imageCenter(1);
//            distZ = queryCenter(2) - imageCenter(2);
//            double dist = sqrt(pow(distX, 2.0) + pow(distY, 2.0) + pow(distZ, 2.0));
//
//            if (dist > MIN) {
//                ++min;
//            }
//            if (dist > MED) {
//                ++med;
//            }
//            if (dist > MAX) {
//                ++max;
//            }
//        }
    }
//    cout << "Done!" << endl;
//
//    cout << "Average Error [m]: " << totalError / listQuery.size() << endl;
//    cout << "Average Number of Outliers (> 0.5): " << min / listQuery.size() << ", (> 2.0): " << med / listQuery.size() << ", (> 5.0): " << max / listQuery.size() << endl;
//    cout << "Percentage of best matches from outside the correct scene: " << (outOfBounds / listQuery.size()) * 100 << endl;
    cout << "Average error: " << totalError / listQuery.size() << endl;

}

// DATABASE PREPARATION FUNCTIONS
void createImageVector(vector<string> &listImage, vector<tuple<string, string, vector<string>, vector<string>>> &info, int scene)
{
    cout << "Creating image vector..." << endl;
    int begin;
    int end;
    if (scene == -1)
    {
        begin = 0;
        end = info.size();
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
                    string file = "seq-" + seq + "/" + image;
                    listImage.push_back(folder + file);
                }
                im_list.close();
            }
        }
    }
    cout << "done" << endl;
}

void createQueryVector(vector<string> &listQuery, vector<tuple<string, string, vector<string>, vector<string>>> &info, int scene)
{
    cout << "Creating query vector..." << endl;
    int begin;
    int end;
    if (scene == -1)
    {
        begin = 0;
        end = info.size();
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
                    string file = "seq-" + seq + "/" + image;
                    listQuery.push_back(folder + file);
                }
                im_list.close();
            }
        }
    }
    cout << "done" << endl;
}



//SURF functions
void loadFeaturesSURF(const vector<string> &listImage, vector<vector<vector<float>>> &features)
{
    features.clear();
    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(400, 4, 2, false);

    for (const string& im : listImage)
    {
        cv::Mat image = cv::imread(im + EXT, cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            cout << "The image " << im << "is empty" << endl;
            exit(1);
        }
        cv::Mat mask;
        vector<cv::KeyPoint> keyPoints;
        vector<float> descriptors;

        surf->detectAndCompute(image, mask, keyPoints, descriptors);
        features.push_back(vector<vector<float>>());
        changeStructureSURF(descriptors, features.back(), surf->descriptorSize());

        cout << im << endl;
    }
    cout << "Done!" << endl;
}

void changeStructureSURF(const vector<float> &plain, vector<vector<float>> &out, int L)
{
    out.resize(plain.size() / L);
    unsigned int j = 0;

    for(unsigned int i = 0; i < plain.size(); i += L, ++j)
    {
        out[j].resize(L);
        std::copy(plain.begin() + i, plain.begin() + i + L, out[j].begin());
    }
}

void DatabaseSaveSURF(const vector<vector<vector<float>>> &features)
{
    Surf64Vocabulary voc;

    cout << "Creating vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl << voc << endl;

    cout << "Saving vocabulary..." << endl;
    string vocFile = "voc.yml.gz";
    voc.save(FOLDER + vocFile);
    cout << "Done" << endl;
    cout << "Creating database..." << endl;

    Surf64Database db(voc, false, 0);

    for(const auto & feature : features)
    {
        db.add(feature);
    }

    cout << "... done!" << endl;

    cout << "Database information: " << endl << db << endl;

    cout << "Saving database..." << endl;
    string dbFile = "db.yml.gz";
    db.save(FOLDER + dbFile);
    cout << "... done!" << endl;

}


// ORB functions
void loadFeaturesORB(const vector<string> &listImage, vector<vector<cv::Mat>> &features)
{
    features.clear();
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    for (const string& im : listImage)
    {
        cv::Mat image = cv::imread(im + EXT, cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            cout << "The image " << im << "is empty" << endl;
            exit(1);
        }
        cv::Mat mask;
        vector<cv::KeyPoint> keyPoints;
        cv::Mat descriptors;

        orb->detectAndCompute(image, mask, keyPoints, descriptors);
        features.push_back(vector<cv::Mat>());
        changeStructureORB(descriptors, features.back());

        cout << im << endl;
    }
    cout << "Done!" << endl;
}

void changeStructureORB(const cv::Mat &plain, vector<cv::Mat> &out)
{
    out.resize(plain.rows);

    for(int i = 0; i < plain.rows; ++i)
    {
        out[i] = plain.row(i);
    }
}

void DatabaseSaveORB(const vector<vector<cv::Mat>> &features)
{
    OrbVocabulary voc;

    cout << "Creating vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl << voc << endl;

    cout << "Saving vocabulary..." << endl;
    string vocFile = "voc_orb.yml.gz";
    voc.save(FOLDER + vocFile);
    cout << "Done" << endl;
    cout << "Creating database..." << endl;

    OrbDatabase db(voc, false, 0);

    for(const auto & feature : features)
    {
        db.add(feature);
    }

    cout << "... done!" << endl;

    cout << "Database information: " << endl << db << endl;

    cout << "Saving database..." << endl;
    string dbFile = "db_orb.yml.gz";
    db.save(FOLDER + dbFile);
    cout << "... done!" << endl;

}


void saveResultVector(const vector<pair<int, float>>& result, const string& query_image, const string& method)
{
    ofstream file (query_image + "." + method + ".top50.txt");
    if(file.is_open())
    {
        for (const pair<int, float>& r : result)
        {
            file << to_string(r.first) + "  " + to_string(r.second) + "\n";
        }
        file.close();
    }
}

vector<pair<int, float>> getResultVector(const string& query_image, const string& method)
{
    vector<pair<int, float>> result;
    ifstream file (query_image + "." + method + ".top50.txt");
    if(file.is_open())
    {
        string line;
        while (getline(file, line))
        {
            stringstream ss (line);
            string num;
            vector<string> temp;
            while(ss >> num)
            {
                temp.push_back(num);
            }
            result.push_back(pair<int, float> (stoi(temp[0]), stof(temp[1])));
        }
        file.close();
    }
    return result;
}
