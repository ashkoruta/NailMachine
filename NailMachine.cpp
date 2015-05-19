// NailMachine.cpp

#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>

// global constants
const int DotSize = 1; // 1 mm is minimal diameter possible
const float DotSizeMetric = 0.001; // size in meters
const int MaxNailWidth = 20; // no nails wider than 20mm exist
const int hMeshCount = MaxNailWidth / DotSize; // number of meshes (they should be square so no concerning of longitudinal dimension)
const int ChannelNumber = 3;

std::stringstream devnull;
//#define DBG
#ifdef DBG
#define TRACE std::cout << __FUNCTION__ << ":" << __LINE__ << ":"
#else
#define TRACE devnull
#endif
/*
	General idea of a program
	1) Take an image and split it into squares corresponding to 1mm x 1mm squares (our tool size)
	2) Each square obviously will consist of many pixels (if not, the image is less than 20x40 - wtf?)
	3) Based on these pixels, decide what single color should this square be
	4) Split file into single-color pieces
	5) Each single-color piece is translated to list of point coordinates
*/

template<typename T> std::string channeledOutput(const T *avg)
{
	std::stringstream ret;
	for (int i = 0; i < ChannelNumber; i++)
		ret << " ch#" << i << " " << (int)avg[i];
	return ret.str();
}
void averageColor(cv::Mat &im, int v, int h, int vsz, int hsz)
{
	const int starthInd = h*hsz; // index of first column corresponding to start of horizontal mesh
	const int startvInd = v*vsz;  // index of first row corresponding to start of vertical mesh
	const int maxhInd = std::min(im.cols, (h + 1)*hsz); // last mesh is not necessarily full-sized
	const int maxvInd = std::min(im.rows, (v + 1)*vsz); // same thing
	// construct the submatrix for pixels from (v,h)
	cv::Mat tmp = im.rowRange(cv::Range(startvInd, maxvInd)).colRange(cv::Range(starthInd, maxhInd));
	cv::Scalar mean = cv::mean(tmp); // calculate mean value
	cv::Mat(tmp.rows, tmp.cols, tmp.type(), mean).copyTo(tmp); // assign
}
cv::Mat simplifyImageThreshold(const cv::Mat &in)
{
	std::vector<cv::Mat> spl;
	cv::split(in, spl); // split into three single-channel images
	CV_Assert(spl.size() == ChannelNumber);
	for (size_t i = 0; i < spl.size(); ++i) {
		cv::threshold(spl[i], spl[i], UCHAR_MAX / 2, UCHAR_MAX, cv::THRESH_BINARY); // every pixel is either full-color or no-color
	}
	cv::Mat out;
	cv::merge(spl, out); // merge
	// so colors in output image are simple: red, blue, green or sum of 2-3 of them (purple, white, yellow etc)
	return out;
}
cv::Mat simplifyImageLUT(const cv::Mat &in, const int colorCount)
{
	cv::Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.data;
	const unsigned int divider = 256 / colorCount;
	for (int i = 0; i < 256; ++i) {
		p[i] = (uchar)(divider * (i / divider));
	}
	cv::Mat out;
	cv::LUT(in, lookUpTable, out); // it simplifies channels, so we get not colorCount, but colorCount^3 colors (!)
	return out;
}
void scaleAndFileOutput(const cv::Mat &im, const std::string &filename)
{
	// open output file
	std::ofstream out(filename);
	// coordinates are counted from the middle of the nail and thus lie in
	// X : (-width/2*Dotsize, width/2*Dotsize)
	// Y : (-height/2*Dotsize, height/2*Dotsize)
	// go column by column because it's simpler for physical tool
	for (int h = 0; h < im.cols; ++h) {
		for (int v = 0; v < im.rows; ++v) {
			if (!im.at<short>(v, h)) // this point has 0, not present in picture
				continue;
			out << (0.5 + h - im.cols / 2.0)*DotSize << "\t" << (0.5 + v - im.rows / 2.0)*DotSize << std::endl;
		}
	}
}
void outputToolPath(const cv::Mat &im, int vMeshCount, int vMeshSz, int hMeshCount, int hMeshSz)
{
	/* general outline
		1) reduce block of the same colors to matrix vMeshSz x hMeshSz
		2) same color == all three channels are the same. split into three channels. for each block
		   1) for each channel: subtract value of this block from others. apply NOT. get 1 there value was the same
		   2) AND three channel results. resulting matrix is same-color with this block
		3) this matrix can be considered a bitmap for path points
	*/
	// reduced matrix
	cv::Mat convert;
	im.assignTo(convert, CV_16SC3); // store data in 16-bit signed shorts
	cv::Mat small(vMeshCount, hMeshCount, CV_16SC3);
	for (int v = 0; v < vMeshCount; ++v) {
		for (int h = 0; h < hMeshCount; ++h) {
			small.at<cv::Vec3s>(v, h) = convert.at<cv::Vec3s>(v*vMeshSz, h*hMeshSz); // Vec3s = vector of 3 shorts
		}
	}
	// split into 3
	std::vector<cv::Mat> spl;
	cv::split(small, spl);
	cv::Mat alreadyProcessed = cv::Mat::zeros(small.rows, small.cols, CV_16SC1); // boolean matrix: no need to check block if it was already matched with something
	// for each block of reduced matrix
	CV_Assert(small.rows == vMeshCount);
	CV_Assert(small.cols == hMeshCount);
	
	for (int i = 0; i < small.rows; ++i) {
		for (int j = 0; j < small.cols; ++j) {
			if (alreadyProcessed.at<short>(i, j))
				continue;
			// if match for it wasn't yet found, construct maps for channels
			const cv::Vec3s val = small.at<cv::Vec3s>(i, j);
			cv::Mat diff = cv::Mat::ones(small.rows, small.cols, CV_16SC1); // final result of 3 ANDs will be here
			for (int ch = 0; ch < ChannelNumber; ++ch) {
				// make matrix of val[ch] and subtract it from original channel
				TRACE << "cur val " << val[ch] << std::endl;
				TRACE << "channel " << spl[ch] << std::endl;
				cv::Mat tmp = cv::Mat::ones(small.rows, small.cols, CV_16SC1);
				tmp = spl[ch] - tmp*val[ch];
				TRACE << "SUBTRACT " << tmp << std::endl;
				TRACE << "NOT " << ~tmp << std::endl;
				diff = diff & (~tmp); // 0 where color channel is the same, not zero otherwise => logical not and then AND with previous result
				TRACE << "BITMAP " << diff << std::endl;
			}
			// ok, we have a matrix of 1's where all channels are the same as in our current block
			// these blocks should not be tested in future
			alreadyProcessed = alreadyProcessed | diff;
			std::stringstream filename;
			filename << "toolpath." << i << "." << j << ".txt";
			scaleAndFileOutput(diff, filename.str());
		}
	}
}
int main(int argc, char** argv)
{
	// open the file
	if (argc < 3) {
		std::cout << " Usage: NailMachine.exe fileToHandle colorCount" << std::endl;
		return -1;
	}
	std::stringstream ss;
	ss << argv[2];
	int colorCount = 0;
	ss >> colorCount;

	cv::Mat input, squared, simplified;
	input = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR); // Read the file
	
	if (!input.data || !colorCount) { // Check for invalid input
		std::cout << "Invalid input" << std::endl;
		return -1;
	}
	std::cout << "input channels " << input.channels() << " vertical size = " << input.rows << " horizontal size = " << input.cols << std::endl;
	CV_Assert(input.channels() == ChannelNumber);

	const int hMeshSize = input.cols / hMeshCount; // number of pixels in one mesh
	const int vMeshSize = hMeshSize; // mesh is square so vertical size is the same
	const int vMeshCount = input.rows / vMeshSize + (input.rows % vMeshSize ? 1 : 0); // how many square meshes needed to fill full length
	std::cout << "pixels per mesh: vMeshSize = " << vMeshSize << " hMeshSize = " << hMeshSize << std::endl;
	std::cout << hMeshCount << " horizontal and " << vMeshCount << " vertical meshes" << std::endl;
	//TODO pack all mesh sizes-counts in one structure
	squared = input.clone();
	for (int v = 0; v < vMeshCount; ++v) {
		for (int h = 0; h < hMeshCount; ++h) {
			// inside a mesh referenced by (v, h) we need to find average color and update every pixel
			// no reference to channels here! suppose it's just a normal matrix
			averageColor(squared, v, h, vMeshSize, hMeshSize);
		}
	}
	cv::imshow("Sampled picture", squared);
	cv::waitKey(0);
	// now lets lower number of different colors in the image
	// thresholding for binary (red-not red) channels
	simplified = simplifyImageThreshold(squared);
	// simplified = simplifyImageLUT(squared, colorCount);
	cv::imshow("Output", simplified);
	cv::waitKey(0);
	// now create txt file with point coordinates
	outputToolPath(simplified, vMeshCount, vMeshSize, hMeshCount, hMeshSize);
	return 0;
}
