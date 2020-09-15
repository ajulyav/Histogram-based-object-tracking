/* Applied Video Sequence Analysis - Escuela Politecnica Superior - Universidad Autonoma de Madrid
 *
 *	This source code belongs to the template code for
 *	the assignment LAB 4 "Histogram-based tracking"
 *
 *	Header of utilities for LAB4.
 *	Some of these functions are adapted from OpenSource
 *
 * Author: Juan C. SanMiguel (juancarlos.sanmiguel@uam.es)
 * Date: April 2020
 */
#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <string> 		// for string class
#include <opencv2/opencv.hpp>

using namespace cv;

cv::Mat plot_histogram(Mat  basicModelHistogram,
		Mat currentHistogram,
		int hist_bins);

std::vector<cv::Rect> readGroundTruthFile(std::string groundtruth_path);
std::vector<float> estimateTrackingPerformance(std::vector<cv::Rect> Bbox_GT, std::vector<cv::Rect> Bbox_est);

Mat get_ColorHistogram(Mat cropped_frame, int hist_bins);
Mat get_GradientHistogram(Mat cropped_frame, int hist_bins);

cv::Mat select_channel(Mat frame, char selected_channel);
void get_candidates(const Mat one_channelFrame, std::vector<Mat> &candidates, std::vector<Rect> &candidate_rectangle, const Rect currentState, int size_nghbd, int step, int hist_bins, bool color_based);
int Battacharyya_distance(const Mat basicModelHistogram, const std::vector<Mat> candidates, std::vector<float> &distances, bool color_based);
int bestCombinedCandidate(const std::vector<float> candidates_color,const std::vector<float> candidates_gradient);

#endif /* UTILS_HPP_ */
