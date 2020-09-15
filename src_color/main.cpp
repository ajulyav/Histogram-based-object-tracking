/* Applied Video Sequence Analysis - Escuela Politecnica Superior - Universidad Autonoma de Madrid
 *
 *	This source code belongs to the template (sample program)
 *	provided for the assignment LAB 4 "Histogram-based tracking"
 *
 *	This code has been tested using:
 *	- Operative System: Ubuntu 18.04
 *	- OpenCV version: 3.4.4
 *	- Eclipse version: 2019-12
 *
 * Author: Juan C. SanMiguel (juancarlos.sanmiguel@uam.es)
 * Date: April 2020
 */
//includes
#include <stdio.h> 								//Standard I/O library
#include <numeric>								//For std::accumulate function
#include <string> 								//For std::to_string function
#include <opencv2/opencv.hpp>					//opencv libraries
#include "utils.hpp" 							//for functions readGroundTruthFile & estimateTrackingPerformance
#include <math.h>
#include "ShowManyImages.hpp"

//namespaces
using namespace cv;
using namespace std;

//main function
int main(int argc, char ** argv)
{
	//PLEASE CHANGE 'dataset_path' & 'output_path' ACCORDING TO YOUR PROJECT
	std::string dataset_path = "/home/avsa/AVSA2020datasets/datasets";									//dataset location.
	std::string output_path = "/home/avsa/AVSA2020results/Ex1";									//location to save output videos

	// dataset paths
	/*std::string sequences[] = {"bolt1",										//test data for lab4.1, 4.3 & 4.5
							   "sphere","car1",								//test data for lab4.2
							   "ball2","basketball",						//test data for lab4.4
							   "bag","ball","road",};*/						//test data for lab4.6

	std::string sequences[] = {"car1"};
	std::string image_path = "%08d.jpg"; 									//format of frames. DO NOT CHANGE
	std::string groundtruth_file = "groundtruth.txt"; 						//file for ground truth data. DO NOT CHANGE
	int NumSeq = sizeof(sequences)/sizeof(sequences[0]);					//number of sequences

	//Loop for all sequence of each category
	for (int s=0; s<NumSeq; s++ )
	{
		Mat frame;										//current Frame
		int frame_idx=0;								//index of current Frame
		std::vector<Rect> list_bbox_est, list_bbox_gt;	//estimated & groundtruth bounding boxes
		std::vector<double> procTimes;					//vector to accumulate processing times

		std::string inputvideo = dataset_path + "/" + sequences[s] + "/img/" + image_path; //path of videofile. DO NOT CHANGE
		VideoCapture cap(inputvideo);	// reader to grab frames from videofile

		//check if videofile exists
		if (!cap.isOpened())
			throw std::runtime_error("Could not open video file " + inputvideo); //error if not possible to read videofile

		// Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
		cv::Size frame_size(cap.get(cv::CAP_PROP_FRAME_WIDTH),cap.get(cv::CAP_PROP_FRAME_HEIGHT));//cv::Size frame_size(700,460);
		VideoWriter outputvideo(output_path+"outvid_" + sequences[s]+".avi",CV_FOURCC('X','V','I','D'),10, frame_size);	//xvid compression (cannot be changed in OpenCV)

		//Read ground truth file and store bounding boxes
		std::string inputGroundtruth = dataset_path + "/" + sequences[s] + "/" + groundtruth_file;//path of groundtruth file. DO NOT CHANGE
		list_bbox_gt = readGroundTruthFile(inputGroundtruth); //read groundtruth bounding boxes

		//main loop for the sequence
		std::cout << "Displaying sequence at " << inputvideo << std::endl;
		std::cout << "  with groundtruth at " << inputGroundtruth << std::endl;

        //INIT VARIABLES
		Rect objectModel = Rect(-1, -1, 0, 0);
		Rect currentState;
		Mat model_histogram;
        Mat model_img;

		//Histogram bins
		int hist_bins = 16;

		//Grid parameters effect the candidates:
		int size_nghbd = 4;
		int step = 2;

		char selected_channel = 'H';

		for (;;) {
			//get frame & check if we achieved the end of the videofile (e.g. frame.data is empty)
			cap >> frame;

			if (!frame.data)
				break;

			//Time measurement
			double t = (double)getTickCount();
			frame_idx=cap.get(cv::CAP_PROP_POS_FRAMES);			//get the current frame

			////////////////////////////////////////////////////////////////////////////////////////////
			//DO TRACKING
			//Change the following line with your own code
			//list_bbox_est.push_back(Rect(20,20,40,50));//we use a fixed value only for this demo program. Remove this line when you use your code
			//...

			Mat one_channelFrame = select_channel(frame, selected_channel);

			if(frame_idx == 1)
			{
				//If frame is first, then chose the ground truth to get x,y of model and initialize the model

				objectModel.x = list_bbox_gt[frame_idx-1].x; //top left x
				objectModel.y = list_bbox_gt[frame_idx-1].y; //top left y
				objectModel.width = list_bbox_gt[frame_idx-1].width; //the size will stay the same
				objectModel.height = list_bbox_gt[frame_idx-1].height;

				model_img = one_channelFrame(objectModel);

				//the model (histogram)
				model_histogram =calculate_RegionHist(model_img, hist_bins);
                currentState = objectModel;

			}else
			{
				//previous result center
				currentState.x = list_bbox_est[frame_idx-2].x;
				currentState.y = list_bbox_est[frame_idx-2].y;
			}

			std::vector<Mat> candidates; //histogram of the candidates
			std::vector<Rect> candidate_rectangle; //candidate rectangles corresponding to the position of candidates

			//Computing the histogram and location
			get_candidates(one_channelFrame, candidates, candidate_rectangle, currentState, size_nghbd, step, hist_bins);

			cout << "Number Of Candidates: " << candidates.size() << endl;

			//the best candidate
			int index_best_distance = Battacharyya_distance(model_histogram, candidates);
			list_bbox_est.push_back(candidate_rectangle[index_best_distance]);//HERE IF I COMMENT IT DOESN"T SEEM IT WILL CHANGE SMTH


			//plot histograms
			cv::Mat plotHistograms = plot_histogram(model_histogram, candidates[index_best_distance], hist_bins);

			//Time measurement
			procTimes.push_back(((double)getTickCount() - t)*1000. / cv::getTickFrequency());
			//std::cout << " processing time=" << procTimes[procTimes.size()-1] << " ms" << std::endl;

			// plot frame number & groundtruth bounding box for each frame
			putText(frame, std::to_string(frame_idx), cv::Point(10,15),FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255)); //text in red
			rectangle(frame, list_bbox_gt[frame_idx-1], Scalar(0, 255, 0));		//draw bounding box for groundtruth
			rectangle(frame, list_bbox_est[frame_idx-1], Scalar(0, 0, 255));	//draw bounding box (estimation)

			//show & save data
			//imshow("Tracking for "+sequences[s]+" (Green=GT, Red=Estimation)", frame);
			outputvideo.write(frame);//save frame to output video
			//and finally show our results: frame, candidate box (in RGB), model box (RGB) and both of the histograms
			ShowManyImages("Tracking", 4, frame, frame(list_bbox_est[frame_idx-1]),model_img, plotHistograms);

			//exit if ESC key is pressed
			if(waitKey(30) == 27) break;
			//waitKey(); //This is the line
		}

		//comparison groundtruth & estimation
		vector<float> trackPerf = estimateTrackingPerformance(list_bbox_gt, list_bbox_est);

		//print stats about processing time and tracking performance
		std::cout << "  Average processing time = " << std::accumulate( procTimes.begin(), procTimes.end(), 0.0) / procTimes.size() << " ms/frame" << std::endl;
		std::cout << "  Average tracking performance = " << std::accumulate( trackPerf.begin(), trackPerf.end(), 0.0) / trackPerf.size() << std::endl;
		waitKey(0);

		//release all resources
		cap.release();			// close inputvideo
		outputvideo.release(); 	// close outputvideo
		destroyAllWindows(); 	// close all the windows
	}
	printf("Finished program.");
	return 0;
}
