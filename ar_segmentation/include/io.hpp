// #pragma once

#include <H5Cpp.h>
#include <string>
#include <vector>
#include "utils.hpp"
#include "model_parameters.hpp"

namespace io
{

    double *ReadImageAsVector(std::string filename, int *nColumns, int *nRows, bool verbose = false);

    H5::DataType get_hdf5_data_type(const cv::Mat &data, bool use_int);
    H5::H5File create_h5_file(const std::string &filename);
    void add_attribute_to_dataset(H5::DataSet &dataset, const std::string &attr_name, const std::string &attr_value);
    void write_hdf5(H5::H5File &file, cv::Mat data, const std::string &var_name, bool use_int,const std::string &long_name, const std::string &dataset_description);
    void close_hdf5(H5::H5File &file);
    std::unordered_map<std::string, std::string> read_header(std::string filename);
    bool containsAny(const std::string &target, const std::vector<std::string> &strings);
    void write_metadata(std::string filename, H5::H5File &file);
    std::unordered_map<std::string, std::string> get_params(ModelParameters params);
    void writeKeyValuePairsToCSV(const std::string &csvFilePath, const std::unordered_map<std::string, std::string> &keyValuePairs);
    void write_properties(const std::string &csvFilePath, const std::map<std::string, std::string> &keyValuePairs);
    void write_table(std::unordered_map<std::string, std::string> properties, H5::H5File &file, std::string datasetName);
    int readHeader(std::string fn);
}
