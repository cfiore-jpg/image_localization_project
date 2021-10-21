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
#define SCENE 0
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

    createImageVector(listImage, info, SCENE);
    createQueryVector(listQuery, info, SCENE);


    // Loading specific to SURF
    if (GENERATE_DATABASE) {
        vector<vector<vector<float>>> features;
        loadFeaturesSURF(listImage, features);
        DatabaseSaveSURF(features);
    }
    string dbFile = "db.yml.gz";
    imageMatcher_Orb im(IMAGE_LIST, FOLDER + dbFile, "SURF", "vocabularyTree");
    cout << "Loading done!" << endl;

    // Loading Specific to ORB
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
    int startIdx = 0;
    for (int i = startIdx; i < listQuery.size(); ++i)
    {
//        vector<pair<int, float>> result = im.matching_one_image(listQuery[i] + EXT);
//        saveResultVector(result, listQuery[i], "surf_jts");
          vector<pair<int, float>> result = getResultVector(listQuery[i], "surf");

//        cout << ", Cur Error = " << toAdd << ", Avg Error = " << totalError / (i - startIdx + 1) << endl;

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

// surf
// orb
// optional _jts meaning "just this scene"
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
