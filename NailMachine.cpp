// NailMachine.cpp

#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

// global constants
const int DotSize = 1; // 1 mm is minimal diameter possible
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
	return 0;
}