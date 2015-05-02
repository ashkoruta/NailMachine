// NailMachine.cpp

#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
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
void averageColor(cv::Mat &im, int v, const int h, const int vsz, const int hsz)
{
	//TODO use mean / sum for God's sake
	TRACE << " (" << v << "," << h << ") " << std::endl;
	unsigned long avg[ChannelNumber] = {};
	const int starthInd = h*hsz*ChannelNumber; // index of first column corresponding to start of horizontal mesh
	const int startvInd = v*vsz;  // index of first row corresponding to start of vertical mesh
	const int maxhInd = std::min(im.cols*ChannelNumber, (h + 1)*hsz*ChannelNumber); // last mesh is not necessarily full-sized
	const int maxvInd = std::min(im.rows, (v + 1)*vsz); // same thing
	TRACE << startvInd << " " << starthInd << "/" << maxvInd << " " << maxhInd << std::endl;
	// get aggregated sum of pixels by channels
	for (int i = startvInd; i < maxvInd; ++i) { //for every row
		const uchar *row = im.ptr<uchar>(i);
		for (int j = starthInd; j < maxhInd; j += ChannelNumber) { // for every other column
			TRACE << channeledOutput(row + j) << std::endl;
			// here we explicitly use channeled representation. thus j points to first color of pixel. to access others: use ch (+0, +1, +2)
			// j=0 j=1 j=2   j=3  
			//  R   G   B   next pixel
			for (int ch = 0; ch < ChannelNumber; ++ch) { // for every channel
				avg[ch] += row[j + ch];
			}
		}
	}
	// now divide
	for (int ch = 0; ch < ChannelNumber; ++ch) {
		// need to divide by number of pixels. generally it's vsz*hsz but not on the edge (last meshes)
		// so use indexes instead
		avg[ch] = avg[ch]*ChannelNumber / ((maxvInd - startvInd)*(maxhInd - starthInd));
	}
	TRACE << channeledOutput(avg) << std::endl;
	// and set this color everywhere inside this block
	// FIXME surely there must be some easier way to iterate and not copy-paste
	for (int i = startvInd; i < maxvInd; ++i) { //for every row
		uchar *row = im.ptr<uchar>(i);
		for (int j = starthInd; j < maxhInd; j += ChannelNumber) { // for every other column, same as above
			for (int ch = 0; ch < ChannelNumber; ++ch) { // for every channel
				CV_Assert(avg[ch] <= UCHAR_MAX); // for cast below
				row[j + ch] = static_cast<uchar>(avg[ch]);
			}
		}
	}
	TRACE << "(" << v << "," << h << ") done" << std::endl;
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
	cv::Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.data;
	const unsigned int divider = 256 / colorCount;
	for (int i = 0; i < 256; ++i) {
		p[i] = (uchar)(divider * (i / divider));
	}
	cv::LUT(squared, lookUpTable, simplified);
	cv::imshow("Output", simplified);
	cv::waitKey(0);
	return 0;
}