/* Applied Video Sequence Analysis - Escuela Politecnica Superior - Universidad Autonoma de Madrid
 *
 *	This source code belongs to the template code for
 *	the assignment LAB 4 "Histogram-based tracking"
 *
 *	Implementation of utilities for LAB4.
 *
 * Author: Juan C. SanMiguel (juancarlos.sanmiguel@uam.es)
 * Date: April 2020
 */
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include <vector>

using namespace cv;
using namespace std;

HOGDescriptor hog;

/**
 * Reads a text file where each row contains comma separated values of
 * corners of groundtruth bounding boxes.
 *
 * The annotations are stored in a text file with the format:
 * Row format is "X1, Y1, X2, Y2, X3, Y3, X4, Y4" where Xi and Yi are
 * the coordinates of corner i of the bounding box in frame N, which
 * corresponds to the N-th row in the text file.
 *
 * Returns a list of cv::Rect with the bounding boxes data.
 *
 * @param ground_truth_path: full path to ground truth text file
 * @return bbox_list: list of ground truth bounding boxes of class Rect
 */
std::vector<Rect> readGroundTruthFile(std::string groundtruth_path)
{
	// variables for reading text file
	ifstream inFile; //file stream
	string bbox_values; //line of file containing all bounding box data
	string bbox_value;  //a single value of bbox_values

	vector<Rect> bbox_list; //output with all read bounding boxes

	// open text file
	inFile.open(groundtruth_path.c_str(),ifstream::in);
	if(!inFile)
		throw runtime_error("Could not open groundtrutfile " + groundtruth_path); //throw error if not possible to read file

	// Read each line of groundtruth file
	while(getline(inFile, bbox_values)){

		stringstream linestream(bbox_values); //convert read line to linestream
		//cout << "-->lineread=" << linestream.str() << endl;

		// Read comma separated values of groundtruth.txt
		vector<int> x_values,y_values; 	//values to be read from line
		int line_ctr = 0;						//control variable to read alternate Xi,Yi
		while(getline(linestream, bbox_value, ',')){

			//read alternate Xi,Yi coordinates
			if(line_ctr%2 == 0)
				x_values.push_back(stoi(bbox_value));
			else
				y_values.push_back(stoi(bbox_value));
			line_ctr++;
		}

		// Get width and height; and minimum X,Y coordinates
		double xmin = *min_element(x_values.begin(), x_values.end()); //x coordinate of the top-left corner
		double ymin = *min_element(y_values.begin(), y_values.end()); //y coordinate of the top-left corner

		if (xmin < 0) xmin=0;
		if (ymin < 0) ymin=0;

		double width = *max_element(x_values.begin(), x_values.end()) - xmin; //width
		double height = *max_element(y_values.begin(), y_values.end()) - ymin;//height

		// Initialize a cv::Rect for a bounding box and store it in a std<vector> list
		bbox_list.push_back(Rect(xmin, ymin, width, height));
		//std::cout << "-->Bbox=" << bbox_list[bbox_list.size()-1] << std::endl;
	}
	inFile.close();

	return bbox_list;
}

/**
 * Compare two lists of bounding boxes to estimate their overlap
 * using the criterion IOU (Intersection Over Union), which ranges
 * from 0 (worst) to 1(best) as described in the following paper:
 * Čehovin, L., Leonardis, A., & Kristan, M. (2016).
 * Visual object tracking performance measures revisited.
 * IEEE Transactions on Image Processing, 25(3), 1261-1274.
 *
 * Returns a list of floats with the IOU for each frame.
 *
 * @param Bbox_GT: list of elements of type cv::Rect describing
 * 				   the groundtruth bounding box of the object for each frame.
 * @param Bbox_est: list of elements of type cv::Rect describing
 * 				   the estimated bounding box of the object for each frame.
 * @return score: list of float values (IOU values) for each frame
 *
 * Comments:
 * 		- The two lists of bounding boxes must be aligned, meaning that
 * 		position 'i' for both lists corresponds to frame 'i'.
 * 		- Only estimated Bboxes are compared, so groundtruth Bbox can be
 * 		a list larger than the list of estimated Bboxes.
 */
std::vector<float> estimateTrackingPerformance(std::vector<cv::Rect> Bbox_GT, std::vector<cv::Rect> Bbox_est)
{
	vector<float> score;

	//For each data, we compute the IOU criteria for all estimations
	for(int f=0;f<(int)Bbox_est.size();f++)
	{
		Rect m_inter = Bbox_GT[f] & Bbox_est[f];//Intersection
		Rect m_union = Bbox_GT[f] | Bbox_est[f];//Union

		score.push_back((float)m_inter.area()/(float)m_union.area());
	}

	return score;
}

//Calculates the histogram
cv::Mat gradient_Histogram(Mat cropped_frame, int hist_bins)

{

        vector< float > descriptors;
        Mat gradientHistogram;

	    Size pimg_size = cropped_frame.size();
        pimg_size = pimg_size / 8 * 8;

	    hog.winSize = pimg_size;
	    hog.nbins = hist_bins;

	    hog.compute( cropped_frame, descriptors, Size(8, 8), Size(0, 0));
	    gradientHistogram =  Mat(descriptors).clone();

	    return gradientHistogram;
}

//Convert image into selected color channel
//H: H channel HSV
//S: S channel HSV
//R: R channel RGB
//G: G channel RGB
//B: B channel RGB
//default: gray level
cv::Mat select_channel(Mat frame, char channel){

	vector<Mat> split_img;
	Mat one_channelFrame;
	Mat grayImage;

	switch (channel)
	{

	case 'H':
		cv::cvtColor(frame, grayImage, COLOR_BGR2HSV);
		split(grayImage,split_img);
		one_channelFrame = split_img[0];
		break;

	case 'S':
			cv::cvtColor(frame, grayImage, COLOR_BGR2HSV);
			split(grayImage,split_img);
			one_channelFrame = split_img[1];
			break;

	case 'R':
			split(frame,split_img);
			one_channelFrame = split_img[2];
			break;

	case 'G':
			split(frame,split_img);
			one_channelFrame = split_img[1];
			break;

	case 'B':
			split(frame,split_img);
			one_channelFrame = split_img[0];
			break;

	default:
		cv::cvtColor(frame, grayImage, COLOR_BGR2GRAY);
		split(grayImage,split_img);
		one_channelFrame = split_img[0];
	}

	return one_channelFrame;
}


//candidates - to save the histogram of candidates
//candidate_rectangle to save the location of candidates
//currentState - (x,y) of the detected object
//size_nghbd - neighbors each direction
//step - number of pixels
void get_candidates(const Mat one_channelFrame,
		std::vector<Mat> &candidates,
		std::vector<Rect> &candidate_rectangle,
		Rect currentState,
		int size_nghbd,
		int step,
		int hist_bins
		)
{

	int endx, endy = 0;
	int startx = currentState.x;
	int starty = currentState.y;

	//check if our window is out of the frame
	if(currentState.x-size_nghbd > 0){
		startx = currentState.x-size_nghbd;
	}else{
		startx = 0;
	}
	if(currentState.y-size_nghbd > 0){
		starty = currentState.y-size_nghbd;
	}else{
		starty = 0;
	}
	if(currentState.x+size_nghbd < one_channelFrame.cols){
		endx = currentState.x+size_nghbd;
	}else{
		endx = one_channelFrame.cols;
	}
	if(currentState.y+size_nghbd < one_channelFrame.rows){
		endy = currentState.y+size_nghbd;
	}else{
		endy = one_channelFrame.rows;
	}

	for(int i = startx; i < endx+1; i+=step ){
		for(int j = starty; j < endy+1; j+=step){

			if( i > 0 && (i + currentState.width) <  one_channelFrame.cols && j > 0 && (j + currentState.height) <  one_channelFrame.rows  ){

				Mat singleCandidate;
				Mat cropped;

				cropped = one_channelFrame(Rect(i,j, currentState.width, currentState.height ));

			    singleCandidate =gradient_Histogram(cropped, hist_bins);

				candidates.push_back(singleCandidate);
				candidate_rectangle.push_back(Rect(i,j,currentState.width,currentState.height));
			}
		}
	}
}

//Plot histogram, and histogram of the best candidate
//model_histogram is the model (red)
//currentHistogram is best candidate histogram (green)
//hist_bins number of bins

cv::Mat plot_histogram(Mat  model_histogram,
		Mat currentHistogram,
		int hist_bins)
{

	Mat bestHistogram;
	Mat normHistogram;

	int weight_hist = 512;
	int height_hist = 400;
	int bin_w = cvRound((double) weight_hist/hist_bins);
	Mat histImage(height_hist, weight_hist, CV_8UC3, Scalar(0,0,0));

	// Normalize [(0, histImage.rows)
	normalize(model_histogram, normHistogram, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(currentHistogram, bestHistogram, 0, histImage.rows, NORM_MINMAX, -1, Mat() );


	for( int i = 1; i < hist_bins; i++ )
	{
	line( histImage, Point( bin_w*(i-1), height_hist - cvRound(bestHistogram.at<float>(i-1)) ) ,
			   Point( bin_w*(i), height_hist - cvRound(bestHistogram.at<float>(i)) ),
			   Scalar( 0, 255, 0), 2, 8, 0  );

	line( histImage, Point( bin_w*(i-1), height_hist - cvRound(normHistogram.at<float>(i-1)) ) ,
						   Point( bin_w*(i), height_hist - cvRound(normHistogram.at<float>(i)) ),
						   Scalar( 0, 0, 255), 2, 8, 0  );
	}

	return histImage;

}

//Battacharyya distance
int Battacharyya_distance(Mat model_histogram, std::vector<Mat> candidates){

	int index_best_distance = 0;
	double current_distance = 0;
    double smallest_distance  = norm(model_histogram, candidates[0],NORM_L2);

	for(int i = 1; i < candidates.size(); i++)
	{
		current_distance  = norm(model_histogram, candidates[i],NORM_L2);

		if(current_distance < smallest_distance)
		{
			smallest_distance = current_distance;
			index_best_distance = i;
		}
	}
		cout<< "L2 distance of HOG:  "<< smallest_distance <<endl;

	return index_best_distance;
}
