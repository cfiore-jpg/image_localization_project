#include "../include/imageMatcher_Orb.h"

imageMatcher_Orb::imageMatcher_Orb(){};

//imageMatcher_Orb::imageMatcher_Orb(string query_image_path, string indexed_image_list_file,
//                                   string database_file_path, string feature_name,
//                                   string matching_method)
//{
//	query_Images_Path_ = query_image_path;
//	indexed_image_information_ = indexed_image_list_file;
//	feature_name_ = feature_name;
//	matching_method_ = matching_method;
//
//	surf = cv::xfeatures2d::SURF::create(400,4,2,false);
//	db_.load(database_file_path);
//
//}

imageMatcher_Orb::imageMatcher_Orb(string indexed_image_list_file, string database_file_path, string feature_name, string matching_method)
{
	cout << "Construct" << endl;
	indexed_image_information_ = indexed_image_list_file;
	db_orb.load(database_file_path);

	feature_name_ = feature_name;
	matching_method_ = matching_method;
	orb = cv::ORB::create();
}

imageMatcher_Orb::imageMatcher_Orb(string indexed_image_list_file, OrbDatabase &db, string feature_name, string matching_method)
{
    indexed_image_information_ = indexed_image_list_file;
    db_orb = db;

    feature_name_ = feature_name;
    matching_method_ = matching_method;
    orb = cv::ORB::create();
}

imageMatcher_Orb::~imageMatcher_Orb()
{
	//delete surf;
}
//
//vector<string> imageMatcher_Orb::load_image_list(string path)
//{
//	vector<string> filenames;
//	ifstream filename_file;
//	filename_file.open(path.c_str());
//	string temp;
//	while(getline(filename_file,temp))
//	{
//		filenames.push_back(temp);
//	}
//	return filenames;
//};
//
////List file names of one folder. This function used for load name of images
////target_path: the path constains images
//vector<string> imageMatcher_Orb::list_names(const char* target_path)
//{
//	DIR *dir;
//	struct dirent *ent;
//	std::vector<std::string> fileList;
//
//	/* Open directory stream */
//	dir = opendir (target_path);
//	if (dir != NULL) {
//		/* Add all images within the directory to a vector list */
//		while ((ent = readdir (dir)) != NULL) {
//			std::string s;
//			switch (ent->d_type) {
//				case DT_REG:{
//					//Get the name of the file and extract the characters to the right of the last period
//					s = (ent->d_name);
//					unsigned posOfPeriod = s.find_last_of(".");
//					s  = s.substr(posOfPeriod+1);
//					//Check file extension
//					if (s.compare("bmp") == 0 || s.compare("BMP") == 0 || s.compare("jpg") == 0 || s.compare("JPG") == 0 || s.compare("jpeg") == 0 || s.compare("JPEG") == 0 ) {
//						fileList.push_back(std::string(target_path) + (ent->d_name));
//					}
//					break; }
//				case DT_DIR:{
//					// Do Nothing. Not required.
//					break;
//				}
//				default:
//					// Do Nothin
//					break;
//			}
//		}
//		closedir (dir);
//	} else {
//		/* Could not open directory */
//		printf ("Cannot open images directory %s\n", target_path);
//		exit (EXIT_FAILURE);
//	}
//	//Check if there were any supported image formats found
//	if (fileList.empty()){
//		std::cout << "No supported image formats found" << std::endl;
//		//exit (EXIT_FAILURE);
//	}
//	sort(fileList.begin(), fileList.end());
//	return fileList;
//};

void imageMatcher_Orb::changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
	out.resize(plain.rows);

	for(int i = 0; i < plain.rows; ++i)
	{
		out[i] = plain.row(i);
	}
};

vector<pair<int, float> > imageMatcher_Orb::matching_one_image(string query_image)
{
	vector<pair<int, float> > result_index;
	vector<cv::Mat> feature;
	QueryResults results;

	cv::Mat image = cv::imread(query_image, cv::IMREAD_GRAYSCALE);

	if (image.empty())
	{
		cout << "The image is empty" << endl;
		exit(1);
	}
	cv::Mat mask;
	vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	orb->detectAndCompute(image, mask, keypoints, descriptors);
	changeStructure(descriptors, feature);
	db_orb.query(feature, results, 50);

	for (int j = 0; j < results.size(); j++)
	{
		pair<int, float> temp;
		temp.first = results[j].Id;
		temp.second = results[j].Score;
		result_index.push_back(temp);
	}
	return result_index;
}

//vector<pair<int, float> > imageMatcher_Orb::matching_one_image(cv::Mat image)
//{
//	vector<pair<int, float> > result_index;
//	vector<vector<float> > feature;
//	QueryResults results;
//
//	if (image.empty())
//	{
//		cout << "The image is empty" << endl;
//		exit(1);
//	}
//	cv::Mat mask;
//	vector<cv::KeyPoint> keypoints;
//	vector<float> descriptors;
//    surf->detectAndCompute(image, mask, keypoints, descriptors);
//	changeStructure(descriptors, feature);
//	db_.query(feature,results,20);
//
//	for (int j = 0; j < results.size(); j++)
//	{
//		pair<int, float> temp;
//		temp.first = results[j].Id;
//		temp.second = results[j].Score;
//		result_index.push_back(temp);
//	}
//	return result_index;
//}
//
//vector<pair<int, int> > imageMatcher_Orb::matching_images(string query1, string query2)
//{
//	vector< pair<int, int> > returnVector;
//
//	vector<pair<int,float> > full_list_1;
//	vector<pair<int,float> > full_list_2;
//
//	full_list_1 = matching_one_image(query1);
//	full_list_2 = matching_one_image(query2);
//
//	vector<pair<int, float> > shortList1 = thresholding(full_list_1,0.02f);
//	vector<pair<int, float> > shortList2 = thresholding(full_list_2, 0.02f);
//
//	//Refine
//	if (shortList1.size() != 0 && shortList2.size() != 0)
//	{
//		int size1 = shortList1.size();
//		int size2 = shortList2.size();
//
//		int sizeFinal;
//		if (size1 > size2)
//		{
//			sizeFinal = size2;
//		}
//		else
//		{
//			sizeFinal = size1;
//		}
//
//		for (int k = 0; k < sizeFinal; k++)
//		{
//			returnVector.push_back(pair<int,int>(shortList1[k].first, shortList2[k].first));
//		}
//	}
//	else
//	{
//		if (shortList1.size() == 0)
//		{
//			for (int k = 0; k < shortList2.size(); k++)
//			{
//				returnVector.push_back(pair<int,int>(shortList2[k].first, shortList2[k].first));
//			}
//		}
//
//		if (shortList2.size() == 0)
//		{
//			for (int k = 0; k < shortList1.size(); k++)
//			{
//				returnVector.push_back(pair<int,int>(shortList1[k].first, shortList1[k].first));
//			}
//		}
//	}
//	return returnVector;
//}
//
//
//vector<pair<int, int> > imageMatcher_Orb::matching_images(cv::Mat image1, cv::Mat image2)
//{
//	vector< pair<int, int> > returnVector;
//
//	vector<pair<int,float> > full_list_1;
//	vector<pair<int,float> > full_list_2;
//
//	full_list_1 = matching_one_image(image1);
//	full_list_2 = matching_one_image(image2);
//	cout << "FullList 1:" << full_list_1[0].first << " " << full_list_1[0].second << endl;
//	cout << "FullList 2:" << full_list_2[0].first << " " << full_list_2[0].second << endl;
//
//	vector<pair<int, float> > shortList1 = thresholding(full_list_1,0.02f);
//	vector<pair<int, float> > shortList2 = thresholding(full_list_2, 0.02f);
//
//	//Refine
//	if (shortList1.size() != 0 && shortList2.size() != 0)
//	{
//		int size1 = shortList1.size();
//		int size2 = shortList2.size();
//
//		int sizeFinal;
//		if (size1 > size2)
//		{
//			sizeFinal = size2;
//		}
//		else
//		{
//			sizeFinal = size1;
//		}
//
//		for (int k = 0; k < sizeFinal; k++)
//		{
//			returnVector.push_back(pair<int,int>(shortList1[k].first, shortList2[k].first));
//		}
//	}
//	else
//	{
//		if (shortList1.size() == 0)
//		{
//			for (int k = 0; k < shortList2.size(); k++)
//			{
//				returnVector.push_back(pair<int,int>(shortList2[k].first, shortList2[k].first));
//			}
//		}
//
//		if (shortList2.size() == 0)
//		{
//			for (int k = 0; k < shortList1.size(); k++)
//			{
//				returnVector.push_back(pair<int,int>(shortList1[k].first, shortList1[k].first));
//			}
//		}
//	}
//	return returnVector;
//
//}
//
//vector<pair<int, float> > imageMatcher_Orb::thresholding(vector<pair<int,float> > full_list, float threshold)
//{
//	vector<pair<int,float> > short_list;
//	for (int i = 0; i < full_list.size(); i++)
//	{
//		if(full_list[i].second >= threshold)
//		{
//			short_list.push_back(full_list[i]);
//		}
//		else
//		{
//			return short_list;
//		}
//	}
//};

