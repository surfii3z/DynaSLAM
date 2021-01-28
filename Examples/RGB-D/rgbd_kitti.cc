/**
* This file is a modified version of ORB-SLAM2.<https://github.com/raulmur/ORB_SLAM2>
*
* This file is part of DynaSLAM.
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
*
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>
#include<unistd.h>
#include<opencv2/core/core.hpp>

#include"Geometry.h"
#include"MaskNet.h"
#include"System.h"

using namespace std;

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<string> &vstrDepthImageFilenames, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 4 && argc != 5)
    {
        cerr << endl << "Usage: ./rgbd_kitti path_to_vocabulary path_to_settings path_to_sequence (path_to_masks)" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<string> vstrDepthImageFilenames;

    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageFilenames, vstrDepthImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();
    std::cout << "nImages: " << nImages << std::endl;

    if(vstrImageFilenames.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrDepthImageFilenames.size()!=vstrImageFilenames.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Initialize Mask R-CNN
    DynaSLAM::SegmentDynObject *MaskNet;
    if (argc==5)
    {
        cout << "Loading Mask R-CNN. This could take a while..." << endl;
        MaskNet = new DynaSLAM::SegmentDynObject();
        cout << "Mask R-CNN loaded!" << endl;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im_rgb;
    cv::Mat im_depth;

    // Dilation settings
    int dilation_size = 15;
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        cv::Point( dilation_size, dilation_size ) );

    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im_rgb = cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        im_depth = cv::imread(vstrDepthImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

#ifdef DEBUG
        std::cout << " " << std::endl;
        std::cout << "Frame: " << ni << std::endl;
#endif
        
        // std::cout << "RGB size: " << im_rgb.rows << " " << im_rgb.cols << std::endl;
        // std::cout << "Depth size: " << im_depth.rows << " " << im_depth.cols << std::endl;

        if(im_rgb.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

        if(im_depth.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrDepthImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Segment out the images
        int h = im_rgb.rows;
        int w = im_rgb.cols;
        cv::Mat mask = cv::Mat::ones(h,w,CV_8U);
        if(argc == 5)
        {
            cv::Mat maskRCNN;
            maskRCNN = MaskNet->GetSegmentation(im_rgb,string(argv[4]),vstrImageFilenames[ni].replace(
                        vstrImageFilenames[ni].begin(), vstrImageFilenames[ni].end()-10,""));
            cv::Mat maskRCNNdil = maskRCNN.clone();
            cv::dilate(maskRCNN, maskRCNNdil, kernel);
            mask = mask - maskRCNN;
        }

        // Pass the image to the SLAM system
#ifdef DEBUG
        std::cout << "rgbd_kitti.cc Before track RGBD" << std::endl;
#endif
        SLAM.TrackRGBD(im_rgb, im_depth, mask, tframe);
#ifdef DEBUG
        std::cout << "rgbd_kitti.cc After track RGBD" << std::endl;
#endif

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryKITTI("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, 
                vector<string> &vstrDepthImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixDepth = strPathToSequence + "/image_depth/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);
    vstrDepthImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
        vstrDepthImageFilenames[i] = strPrefixDepth + ss.str() + ".png";
    }
}
