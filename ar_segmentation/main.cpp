/*
   Copyright DMLAB at Georgia State University

   This file is part of An Extended Noise-Aware PIL Dataset creation project.

   PIL dataset creation is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   PIL dataset creation is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with [PROJECT NAME]. If not, see <https://www.gnu.org/licenses/>.
*/
#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

#include <bits/stdc++.h>
#include <cstdlib>
#include <filesystem>

#include <sys/stat.h>
#include <sys/types.h>
#include <chrono>
#include <stdlib.h>
#include <string>
#include <vector>

#include <istream>
#include <fstream>

#include "fitsio.h"

#include <cstring>

#include <string.h>
#include <vector>
#include <set>

#include <omp.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>

#include "opencv2/cudawarping.hpp"
#include <opencv2/cudalegacy.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudafilters.hpp>
#include <cuda_runtime_api.h>

#include <dirent.h>
#include <H5Cpp.h>

// Local custom modules
#include "io.hpp"
#include "utils.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>

// Configuration struct to store YAML parameters
struct Config
{
    double pos_gauss;
    double neg_gauss;
    double low_threshold;
    double high_threshold;
    int dilation_size;
    int gap_size;
    int size_threshold;
    int pil_threshold;
    double strength_threshold;
    std::string data_dir;
    std::string output_dir;
    std::string log_dir;
};

// Function to load configuration from YAML
Config loadConfig(const std::string &config_path)
{
    Config config;
    YAML::Node config_yaml = YAML::LoadFile(config_path);

    config.pos_gauss = config_yaml["parameters"]["pos_gauss"].as<double>();
    config.neg_gauss = config_yaml["parameters"]["neg_gauss"].as<double>();
    config.low_threshold = config_yaml["parameters"]["low_threshold"].as<double>();
    config.high_threshold = config_yaml["parameters"]["high_threshold"].as<double>();
    config.dilation_size = config_yaml["parameters"]["dilation_size"].as<int>();
    config.gap_size = config_yaml["parameters"]["gap_size"].as<int>();
    config.size_threshold = config_yaml["parameters"]["size_threshold"].as<int>();
    config.strength_threshold = config_yaml["parameters"]["strength_threshold"].as<double>();
    config.pil_threshold = config_yaml["parameters"]["pil_threshold"].as<double>();
    config.data_dir = config_yaml["directories"]["data_dir"].as<std::string>();
    config.output_dir = config_yaml["directories"]["output_dir"].as<std::string>();
    config.log_dir = config_yaml["directories"]["log_dir"].as<std::string>();

    return config;
};

std::unordered_map<std::string, std::string> configToParameterMap(const Config& config)
{
    std::unordered_map<std::string, std::string> parameters;

    parameters["pos_gauss"] = std::to_string(config.pos_gauss);
    parameters["neg_gauss"] = std::to_string(config.neg_gauss);
    parameters["low_threshold"] = std::to_string(config.low_threshold);
    parameters["high_threshold"] = std::to_string(config.high_threshold);
    parameters["dilation_size"] = std::to_string(config.dilation_size);
    parameters["gap_size"] = std::to_string(config.gap_size);
    parameters["size_threshold"] = std::to_string(config.size_threshold);
    parameters["strength_threshold"] = std::to_string(config.strength_threshold);
    parameters["pil_threshold"] = std::to_string(config.pil_threshold);

    return parameters;
};

namespace fs = std::filesystem;

std::mutex queueMutex;
std::queue<int> tasks;
class pos_neg_detection
{
    /*
    This class is responsible for detecting region of opposite polarity inversions based on the Gauss thresholds
    */
private:
public:
    struct posneg_map
    {
        cv::Mat pos_map, neg_map;
    } posneg_struct;
    /**
     * Identifies the positive and negative polarity regions in a given image map
     * @param fits_map A vector of doubles representing the HMI image map
     * @param pos_gauss The Gaussian threshold for positive polarity regions (default: 100)
     * @param neg_gauss The Gaussian threshold for negative polarity regions (default: -100)
     */
    void identify_pos_neg_region(cv::Mat fits_image, int nRows, int nColumns, const Config &config)
    {
        // Initialize output mats
        cv::Mat pos_map, neg_map;
        fits_image.copyTo(pos_map);
        fits_image.copyTo(neg_map);
        // Create a mask for values larger than -100
        cv::Mat pos_mask = (fits_image < config.pos_gauss);
        cv::Mat neg_mask = (fits_image > config.neg_gauss);

        // set values: 0 for (values less than pos thres), 1 for (values greater than pos thres)
        // Set values: 0 for (values greater than neg thres), 1 for (values less than neg thres)
        // define positive area with 0 (<thres) and 1 (>thres)
        pos_map.setTo(0, pos_mask);
        cv::Mat negatedMask_pos;
        cv::bitwise_not(pos_mask, negatedMask_pos);
        pos_map.setTo(1, negatedMask_pos);
        cv::Mat pos_8bit;
        pos_map.convertTo(pos_8bit, CV_8U);

        // define negative are with 0 (<thres) and 1 (>thres)
        neg_map.setTo(0, neg_mask);
        cv::Mat negatedMask_neg;
        cv::bitwise_not(neg_mask, negatedMask_neg);
        neg_map.setTo(1, negatedMask_neg);
        cv::Mat neg_8bit;
        neg_map.convertTo(neg_8bit, CV_8U);

        // Assign output vectors to class member variables
        this->posneg_struct.neg_map = neg_8bit;
        this->posneg_struct.pos_map = pos_8bit;
    }
    /**
     * Detect edges in an input image using the Canny algorithm.
     *
     * @param input: Input image to detect edges on.
     * @return Output image with detected edges.
     */
    cv::Mat edge_detection(cv::Mat input, const Config &config)
    {
        // Low and high thresholds for the Canny algorithm are in config as parameters
        // Initialize GPU input and output mats
        // Create a CUDA stream for asynchronous execution
        cv::cuda::Stream stream;
        cv::cuda::GpuMat gpu_input, gpu_output;
        cv::Mat output;
        // Convert input image to 8-bit for compatibility with GPU
        cv::Mat input_8bit;
        
        if (input.type() != CV_8UC1)
        {
            input.convertTo(input_8bit, CV_8U);
        }
        else
        {
            input_8bit = input;
        }

        // Upload input image to GPU
        gpu_input.upload(input, stream);
        // Run the Canny edge detection on the GPU
        cv::Ptr<cv::cuda::CannyEdgeDetector> cannyFilter = cv::cuda::createCannyEdgeDetector(config.low_threshold, config.high_threshold);
        cannyFilter->detect(gpu_input, gpu_output, stream);
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
        {
            std::cerr << "CUDA error: " << cudaGetErrorString(cudaError) << std::endl;
        }
        // Download output image from GPU to CPU
        gpu_output.download(output, stream);

        // Wait for all asynchronous operations to complete
        stream.waitForCompletion();

        return output;
    }
    /**
     * Applies a morphological dilation filter to the input edges using CUDA.
     *
     * @param edges: The input binary image with edges to be dilated.
     * @param dilation_size: The size of the dilation kernel in pixels. Default is 10.
     * @return: The dilated edges as a cv::Mat.
     */
    cv::Mat buff_edge(cv::Mat edges, const Config &config)
    {
        // Create a CUDA stream for asynchronous execution
        cv::cuda::Stream stream;
        // Declare variables for the dilated edges and the GPU memory.
        cv::Mat dilated_edges;
        cv::cuda::GpuMat edges_gpu, dilated_edges_gpu;
        // Convert the input edges to 32-bit floating point and upload to GPU memory.
        cv::Mat input_8bit;
        // edges.convertTo(input_8bit, CV_32FC1);
        edges_gpu.upload(edges, stream);

        // Create a structuring element for morphological dilation
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(config.dilation_size, config.dilation_size)
        );

        // Create a morphological dilation filter on GPU
        cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createMorphologyFilter(
            cv::MORPH_DILATE, edges_gpu.type(), kernel
        );
        
        filter->apply(edges_gpu, dilated_edges_gpu, stream);
        filter->apply(dilated_edges_gpu, dilated_edges_gpu, stream);

        // Download the dilated edges from GPU memory and return.
        dilated_edges_gpu.download(dilated_edges, stream);

        // Wait for all asynchronous operations to complete
        stream.waitForCompletion();

        return dilated_edges;
    }
    cv::Mat PIL_extraction(const cv::Mat &pos_dil_edge, const cv::Mat &neg_dil_edge)
    {   
        // Create a CUDA stream for asynchronous execution
        cv::cuda::Stream stream;

        // Upload the images to GPU memory asynchronously
        cv::cuda::GpuMat gpuPosDilEdge, gpuNegDilEdge, gpuIntersection;
        gpuPosDilEdge.upload(pos_dil_edge, stream);
        gpuNegDilEdge.upload(neg_dil_edge, stream);

        // Perform bitwise AND operation asynchronously on GPU
        cv::cuda::bitwise_and(gpuPosDilEdge, gpuNegDilEdge, gpuIntersection, cv::noArray(), stream);

        // Download the result from GPU to CPU memory asynchronously
        cv::Mat intersection;
        gpuIntersection.download(intersection, stream);

        // Wait for all operations to complete before returning
        stream.waitForCompletion();
            return intersection;
        }

    cv::Mat overlapUnion(const cv::Mat &pos_dil, const cv::Mat &neg_dil)
    {
        // Create a CUDA stream for asynchronous execution
        cv::cuda::Stream stream;

        // Step 1: Upload input images to GPU
        cv::cuda::GpuMat gpu_pos_dil, gpu_neg_dil, gpu_union_mask, gpu_overlap_mask;
        gpu_pos_dil.upload(pos_dil, stream);
        gpu_neg_dil.upload(neg_dil, stream);

        // Step 2: Create the union mask on GPU
        cv::cuda::bitwise_or(gpu_pos_dil, gpu_neg_dil, gpu_union_mask, cv::noArray(), stream);
        cv::cuda::bitwise_and(gpu_pos_dil, gpu_neg_dil, gpu_overlap_mask, cv::noArray(), stream);

        // Download union and overlap mask asynchronously
        cv::Mat union_mask, overlap_mask;
        gpu_union_mask.download(union_mask, stream);
        gpu_overlap_mask.download(overlap_mask, stream);

        // Wait for all asynchronous operations to complete
        stream.waitForCompletion();

        // Convert union mask to 8U
        union_mask.convertTo(union_mask, CV_8U);
        overlap_mask.convertTo(overlap_mask, CV_8U);

        std::cout << "Union mask non-zero pixels: " << cv::countNonZero(union_mask) << std::endl;
        std::cout << "Overlap mask non-zero pixels: " << cv::countNonZero(overlap_mask) << std::endl;

        // Step 3: Find connected components on the union mask
        cv::Mat labelsThin, statsThin, centroidsThin;
        int ncompThin = cv::connectedComponentsWithStats(union_mask, labelsThin, statsThin, centroidsThin, 8, CV_32S);
        std::cout << "Number of components found: " << ncompThin - 1 << std::endl; // -1 for background

        // Create output mask for regions with intersections
        cv::Mat regionsWithIntersec = cv::Mat::zeros(labelsThin.size(), CV_8U);

        // Step 4: Process each component to check overlap
        for (int idx = 1; idx < ncompThin; ++idx)
        {
            // Create mask for the current region
            cv::Mat selectRegion = (labelsThin == idx);

            // Check overlap with overlap_mask (faster using inRange)
            cv::Mat overlap_bool;
            cv::bitwise_and(selectRegion, overlap_mask, overlap_bool);

            int overlapPixels = cv::countNonZero(overlap_bool);
            if (overlapPixels > 0)
            {
                // Add this region to the result
                cv::bitwise_or(regionsWithIntersec, selectRegion, regionsWithIntersec);
            }
        }

        return regionsWithIntersec;
    };
};

cv::Mat removeOutside(const cv::Mat &X, const std::unordered_map<std::string, double> &header, double set_value = NAN)
{
    // Create CUDA stream for asynchronous processing
    cv::cuda::Stream stream;

    // Extract header values
    double cx = header.at("CRPIX1");
    double cy = header.at("CRPIX2");
    double r = (header.at("RSUN_OBS") + 900) / header.at("CDELT1");

    // Upload input matrix to GPU
    cv::cuda::GpuMat gpu_X, gpu_mask, gpu_result;
    gpu_X.upload(X, stream);

    // Create and upload mask to GPU
    cv::Mat mask = cv::Mat::zeros(X.rows, X.cols, CV_8U);
    cv::circle(mask, cv::Point(static_cast<int>(cx), static_cast<int>(cy)), static_cast<int>(r), cv::Scalar(1), -1);
    gpu_mask.upload(mask, stream);

    // Apply mask asynchronously
    gpu_result = gpu_X.clone();
    cv::cuda::compare(gpu_mask, 0, gpu_mask, cv::CMP_EQ, stream);
    gpu_result.setTo(set_value, gpu_mask, stream);

    // Download result back to CPU
    cv::Mat X_return;
    gpu_result.download(X_return, stream);

    // Synchronize to ensure all operations are completed
    stream.waitForCompletion();

    return X_return;
}

/**
 * Check if a value is in an array.
 *
 * Args:
 *   my_array (int[]): The array to search.
 *   size (int): The size of the array.
 *   value (int): The value to search for.
 *
 * Returns:
 *   bool: True if the value is in the array, False otherwise.
 */
bool value_is_in(int my_array[], int size, int value)
{
    // Loop through the array.
    for (int i = 0; i < size; i++)
    {
        // Check if the current element is equal to the value.
        if (my_array[i] == value)
        {
            // If it is, return true.
            return true;
        }
    }
    // If the value is not found, return false.
    return false;
}

/**
 * Fills gaps in a gray scale image with different values using inpainting.
 * @param grayImage: Input image to be processed.
 * @param maxGap: Maximum gap size to be filled.
 * @return: Filled image.
 */

cv::Mat connectedComp_with_sizeFilter(cv::Mat thinComponents, const std::string &type, const Config &config)
{
    int threshold_size;

    // Set threshold depending on the object type
    if (type == "pil")
    {
        threshold_size = config.pil_threshold;
    }
    else if (type == "ar")
    {
        threshold_size = config.size_threshold;
    }
    else
    {
        throw std::invalid_argument("Invalid type provided. Expected 'pil' or 'ar'.");
    }

    // Run connected components and get stats
    cv::Mat labelsThin, statsThin, centroidsThin;
    int ncompThin = cv::connectedComponentsWithStats(thinComponents, labelsThin, statsThin, centroidsThin, 8, CV_32S);

    // Create mask for valid components
    cv::Mat mask = cv::Mat::zeros(labelsThin.size(), CV_8UC1);

    // Process component areas efficiently
    #pragma omp parallel for
    for (int idx = 1; idx < ncompThin; idx++)
    {
        int area = statsThin.at<int>(idx, 4);
        if (area >= threshold_size)
        {
            cv::Mat componentMask;
            cv::compare(labelsThin, idx, componentMask, cv::CMP_EQ);
            mask.setTo(1, componentMask);  // Mark valid components
        }
    }

    // Convert mask to binary image
    cv::Mat binaryImage;
    mask.convertTo(binaryImage, CV_8UC1);

    return binaryImage;
}

std::vector<std::string> list_directory_magnetogram(const std::string &directory_path)
{
    std::vector<std::string> file_list;
    DIR *dir;
    struct dirent *entry;

    dir = opendir(directory_path.c_str());
    if (dir == nullptr)
    {
        std::cerr << "Error: Failed to open directory '" << directory_path << "'." << std::endl;
        return file_list;
    }

    while ((entry = readdir(dir)) != nullptr)
    {
        std::string filename = entry->d_name;
        if (filename != "." && filename != "..")
        {
            // Check if the filename ends with ".magnetogram.fits"
            if (filename.rfind("magnetogram.fits") != std::string::npos)
            {
                file_list.push_back(filename);
            }
        }
    }
    std::sort(file_list.begin(), file_list.end());
    closedir(dir);
    return file_list;
}

std::string get_file_basename(std::string file_path)
{
    return file_path.substr(file_path.find_last_of("/") + 1);
}
std::string get_output_file_name(const std::string file_path, const std::string output_dir)
{
    std::string base_name = get_file_basename(file_path);
    return output_dir + base_name.substr(0, base_name.find_last_of(".")) + ".h5";
}

// function for writing the log file
void writeToLog(std::ofstream &log, const std::string &message)
{
    log << message << std::endl;
}

// void compute_main(std::string filename, std::string out_dir, std::ofstream &logfile)
void compute_main(const std::string &filename, const std::string &out_dir, std::ofstream &logfile, const Config &config)
{   

    std::string output_fn_path = get_output_file_name(filename, out_dir);

    if (std::filesystem::exists(output_fn_path)) {
        std::cout << "File exists.\n";
        return ;
    } 

    // Read Image
    int nColumns, nRows;
    int nRows_psf, nColumns_psf;
    double *pixelVector;
    utils::GetImageSize(filename, &nColumns, &nRows, false);
    pixelVector = io::ReadImageAsVector(filename, &nColumns_psf, &nRows_psf, false);

    std::vector<double> fits_map(pixelVector, pixelVector + nColumns * nRows);
    cv::Mat fits_image(nRows, nColumns, CV_64FC1, fits_map.data());

    // ######################## Initialize and open the HDF5 file to be used for writing
    H5::H5File h5_fid = io::create_h5_file(output_fn_path);
    std::unordered_map<std::string, std::string> metadata = io::read_header(filename);
    std::unordered_map<std::string, std::string> params = configToParameterMap(config);
    io::write_table(metadata, h5_fid, "lineage_metadata");
    io::write_table(params, h5_fid, "parameters");
    
    //transform meta data to header
    std::unordered_map<std::string, double> header;
    for (const auto& kv : metadata) {
        try {
            // Convert only numeric fields to double
            if (kv.first == "CRPIX1" || kv.first == "CRPIX2" || kv.first == "RSUN_OBS" || kv.first == "CDELT1") {
                header[kv.first] = stod(kv.second);
            }
        } catch (const std::invalid_argument& e) {
            std::cout << "Non-numeric value found for key: " << kv.first << std::endl;
        }
    }

    // >>>>>>>>>>>>>>>>>>>>>>>> 0- Write the original image to a HDF5 file
    // io::write_hdf5(h5_fid, fits_image, "00_mag", "Original Data", "Original Magnetogram Data.");
    // ########################

    // 1.
    // DETECT POLARITY REGIONS
    pos_neg_detection pos_neg_instance; // class object
    pos_neg_instance.identify_pos_neg_region(fits_image, nRows, nColumns, config);
    cv::Mat neg_map = pos_neg_instance.posneg_struct.neg_map;
    cv::Mat pos_map = pos_neg_instance.posneg_struct.pos_map;

    // ########################
    // 1.1.
    // APPLY SIZE THRESHOLD ON CONNECTED COMPONENTS
    cv::Mat pos_v1 = connectedComp_with_sizeFilter(pos_map, "ar", config);
    cv::Mat neg_v1 = connectedComp_with_sizeFilter(neg_map, "ar", config);
    
    // ########################
    // 1.2.
    // CREATE UNION MAP
    cv::Mat pos_neg_union;
    neg_v1.convertTo(pos_neg_union, CV_8S);
    pos_neg_union *= -1;
    pos_neg_union += pos_v1;

    // ########################
    // 2.
    // DILATION & INTERSECTION:
    // Compute Region of Polarity inversion (intersection of Neg and Pos regions)
    cv::Mat neg_dil = pos_neg_instance.buff_edge(neg_v1, config);
    cv::Mat pos_dil = pos_neg_instance.buff_edge(pos_v1, config);

    // Dilated union of positive and negative regions.
    cv::Mat pos_neg_union_dialated;
    neg_dil.convertTo(pos_neg_union_dialated, CV_8S);
    cv::bitwise_or(pos_dil, neg_dil, pos_neg_union_dialated);
    pos_neg_union_dialated += pos_dil; //there are non-overlapping regions between pos and neg.

    // std::cout << "fits image size: " << fits_image.size() << std::endl;
    // cv::Mat fits_image_nozero = removeOutside(fits_image, header, NAN); // remove values outside disk 
    // std::cout << "After outsize disk, fits image size: " << fits_image_nozero.size() << std::endl;
    cv::Mat pos_nolim = removeOutside(pos_dil, header, NAN);
    cv::Mat neg_nolim = removeOutside(neg_dil, header, NAN);

    cv::Mat union_with_intersect = pos_neg_instance.overlapUnion(pos_nolim, neg_nolim); 
    cv::Mat intersection = pos_neg_instance.PIL_extraction(pos_nolim, neg_nolim);
    
    // >>>>>>>>>>>>>>>>>>>>>>>> 1- Write the Negative regions image to a HDF5 file
    // io::write_hdf5(h5_fid, neg_map, "neg", true, "Negative Polarity Regions", "Negative Polarity Regions.");

    // >>>>>>>>>>>>>>>>>>>>>>>> 2- Write the Positive regions image to a HDF5 file
    // io::write_hdf5(h5_fid, pos_map, "pos", true, "Positive Polarity Regions", "Positive Polarity Regions.");

    // >>>>>>>>>>>>>>>>>>>>>>>> 3- Write Both (neg+pos as one matrix) to a HDF5 file
    // io::write_hdf5(h5_fid, pos_neg_union, "pos_neg", true, "Positive and Negative Polarity Regions", "Union of the Positive and Negative Polarity Regions.");

    // >>>>>>>>>>>>>>>>>>>>>>>> 4- Write Both (neg+pos after dilation as one matrix) to a HDF5 file
    // io::write_hdf5(h5_fid, pos_neg_union_dialated, "pos_neg_dilated", true, "Positive and Negative Polarity Regions after dilation", "Union of the Positive and Negative Polarity Regions after dilation.");

    // // >>>>>>>>>>>>>>>>>>>>>>>> 5- Write intersection regions to a HDF5 file
    io::write_hdf5(h5_fid, intersection, "intersection", true, "intersection of pos and neg regions", "intersection of pos and neg.");

    // >>>>>>>>>>>>>>>>>>>>>>>> 6- Write regions (union) which only have intersection to a HDF5 file
    io::write_hdf5(h5_fid, union_with_intersect, "union_with_intersect", true, "Union of pos and neg regions with overlapping", "Union of pos and neg regions with overlapping.");

    io::close_hdf5(h5_fid);
    
}
int create_directory(std::string out_dir)
{

    // Check if the directory exists
    if (!fs::exists(out_dir))
    {
        // The directory does not exist, so try to create it
        try
        {
            if (fs::create_directories(out_dir))
            {
                return 0;
            }
            else
            {
                std::cerr << "Failed to create directory." << std::endl;
            }
        }
        catch (const fs::filesystem_error &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
    else
    {
        // std::cout << "Directory already exists." << std::endl;
    }
    return 0;
}

int main(int argc, char *argv[])
{   
    if (!cv::cuda::getCudaEnabledDeviceCount()) {
        std::cout << "No CUDA-capable device found" << std::endl;
        // Fallback to CPU processing
    }
    else{
        std::cout << "CUDA-capable device found" << std::endl;
    }
    
    // Load config
    Config config = loadConfig("config.yaml");

    std::cout << "Size Threshold: " << config.size_threshold << std::endl;
    std::cout << "Positive Gauss Threshold: " << config.pos_gauss << std::endl;
    std::cout << "Negative Gauss Threshold: " << config.neg_gauss << std::endl;

    // Get HARP number from command-line arguments
    if (argc < 2)
    {
        std::cerr << "Please provide HARP number as a command-line argument." << std::endl;
        return 1;
    }
    std::string harp_no = argv[1];
    std::cout << "Processing full-disk(year/month/day): " << harp_no << std::endl;

    // Create directory paths using config
    std::string data_dir = config.data_dir + harp_no + "/";
    std::string out_dir = config.output_dir + harp_no + "/";
    std::string log_fn = config.log_dir + harp_no + "/";

    // Create output directory if it doesn~@~Yt exist
    int dir_created_out = create_directory(out_dir);
    int dir_created_log = create_directory(log_fn);

    // Get the list of files in the data directory
    std::vector<std::string> file_names = list_directory_magnetogram(data_dir);
    const int total_work = file_names.size();

    // Open log file for writing
    std::ofstream logfile(log_fn + "12.log");
    if (!logfile.is_open())
    {
        std::cerr << "Failed to open log file at: " << log_fn << std::endl;
        return 1;
    }

    // Start the timer for tracking execution time
    auto startTime = std::chrono::high_resolution_clock::now();

    // Processing each file
    int i = 0;
    for (const auto &file : file_names)
    {
        std::string in_file_path = data_dir + file;
        std::cout << "Processing file: " << in_file_path << std::endl;
        compute_main(in_file_path, out_dir, logfile, config); // Pass config to compute_main if it uses config values
        i++;
    }
    logfile.close();

    // Stop the timer and print the execution duration
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Full-disk: " << harp_no << " Execution time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
