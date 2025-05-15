/*
   Copyright [2023] [Ziba Khani]

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
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <opencv2/opencv.hpp>
#include <H5Cpp.h>

#include <stdexcept>

#include "fitsio.h"

#include <mutex>
#include "io.hpp"
#include "utils.hpp"
#include "model_parameters.hpp"

namespace io
{
    /* ---------------- FUNCTION: ReadImageAsVector ------------------------ */
    /*    Given a filename, it opens the file, reads the size of the image and
     * stores that size in *nRows and *nColumns, then allocates memory for a 1-D
     * array to hold the image and reads the image from the file into the
     * array.  Finally, it returns the image array -- or, more precisely, it
     * returns a pointer to the array; it also stores the image dimensions
     * in the pointer-parameters nRows and nColumns.
     *    Note that this function does *not* use Numerical Recipes functions; instead
     * it allocates a standard 1-D C vector [this means that the first index will
     * be 0, not 1].
     *
     *    Returns 0 for successful operation, -1 if a CFITSIO-related error occurred.
     *
     */
    double *ReadImageAsVector(std::string filename, int *nColumns, int *nRows,
                              bool verbose)
    {
        fitsfile *imfile_ptr;
        double *imageVector;
        int status, nfound;
        int problems;
        long naxes[2];
        int nPixelsTot;
        long firstPixel[2] = {1, 1};
        int n_rows, n_columns;

        status = problems = 0;
        std::string fn = filename + "[1]";
        /* Open the FITS file: */
        problems = fits_open_file(&imfile_ptr, fn.c_str(), READONLY, &status);
        if (problems)
        {
            fprintf(stderr, "\n*** WARNING: Problems opening FITS file \"%s\"!\n    FITSIO error messages follow:", filename.c_str());
            utils::PrintError(status);
            return NULL;
        }

        /* read the NAXIS1 and NAXIS2 keyword to get image size */
        problems = fits_read_keys_lng(imfile_ptr, "ZNAXIS", 1, 2, naxes, &nfound,
                                      &status);
        if (problems)
        {
            fprintf(stderr, "\n*** WARNING: Problems reading FITS keywords from file \"%s\"!\n    FITSIO error messages follow:", filename.c_str());
            utils::PrintError(status);
            fits_close_file(imfile_ptr, &status); // Close file before returning
            return NULL;
        }
        if (verbose)
            printf("ReadImageAsVector: Image keywords: NAXIS1 = %ld, NAXIS2 = %ld\n", naxes[0], naxes[1]);

        n_columns = naxes[0]; // FITS keyword NAXIS1 = # columns
        *nColumns = n_columns;
        n_rows = naxes[1]; // FITS keyword NAXIS2 = # rows
        *nRows = n_rows;
        nPixelsTot = n_columns * n_rows; // number of pixels in the image

        // Allocate memory for the image-data vector:
        imageVector = (double *)malloc(nPixelsTot * sizeof(double));
        // Read in the image data
        problems = fits_read_pix(imfile_ptr, TDOUBLE, firstPixel, nPixelsTot, NULL, imageVector,
                                 NULL, &status);
        if (problems)
        {
            fprintf(stderr, "\n*** WARNING: Problems reading pixel data from FITS file \"%s\"!\n    FITSIO error messages follow:", filename.c_str());
            utils::PrintError(status);
            free(imageVector);                    // Free allocated memory
            fits_close_file(imfile_ptr, &status); // Close file before returning
            return NULL;
        }

        if (verbose)
            printf("\nReadImageAsVector: Image read.\n");

        problems = fits_close_file(imfile_ptr, &status);
        if (problems)
        {
            fprintf(stderr, "\n*** WARNING: Problems closing FITS file \"%s\"!\n    FITSIO error messages follow:", filename.c_str());
            utils::PrintError(status);
            return NULL;
        }
        // Print(imageVector);
        return imageVector;
    }

    H5::DataType get_hdf5_data_type(const cv::Mat &data, bool use_int)
    {
        // if (use_int)
        // {
        //     return H5::PredType::NATIVE_INT;
        // }
        // else
        // {
        switch (data.type())
        {
        case CV_8UC1:
            return H5::PredType::NATIVE_UINT8;
        case CV_8SC1:
            return H5::PredType::NATIVE_INT8;
        case CV_16UC1:
            return H5::PredType::NATIVE_UINT16;
        case CV_16SC1:
            return H5::PredType::NATIVE_INT16;
        case CV_32SC1:
            return H5::PredType::NATIVE_INT32;
        case CV_32FC1:
            return H5::PredType::NATIVE_FLOAT;
        case CV_64FC1:
            return H5::PredType::NATIVE_DOUBLE;
        default:
            throw std::runtime_error("Unsupported cv::Mat data type for HDF5.");
        }
        // }
    }

    H5::H5File create_h5_file(const std::string &filename)
    {
        try
        {
            // Create a new HDF5 file
            H5::H5File file(filename, H5F_ACC_TRUNC);

            // Create a group to store the spatial reference information
            // H5::Group spatialRefGroup = file.createGroup("/Carrington");

            // // Add the spatial reference attributes
            // H5::DataSpace attrDataspace = H5::DataSpace(H5S_SCALAR);
            // // Add projection attribute
            // std::string projection = "Carrington";
            // H5::StrType strType(H5::PredType::C_S1, projection.length());
            // H5::Attribute projectionAttr = spatialRefGroup.createAttribute("Projection", strType, attrDataspace);
            // projectionAttr.write(strType, projection);
            return file;
        }
        catch (const H5::FileIException &e)
        {
            std::cerr << "Error: Could not create HDF5 file: " << e.getCDetailMsg() << std::endl;
            throw;
        }
    }

    void add_attribute_to_dataset(H5::DataSet &dataset, const std::string &attr_name, const std::string &attr_value)
    {
        H5::DataSpace attr_dataspace = H5::DataSpace(H5S_SCALAR);
        H5::StrType attr_strtype(H5::PredType::C_S1, attr_value.size());
        H5::Attribute attribute = dataset.createAttribute(attr_name, attr_strtype, attr_dataspace);
        attribute.write(attr_strtype, attr_value);
    }

    void write_hdf5(H5::H5File &file, cv::Mat data, const std::string &var_name, bool use_int_type, const std::string &dataset_long_name, const std::string &dataset_description)
    {
        try
        {
            // write OpenCV matrix to HDF5 file

            cv::Mat continuous_data;
            continuous_data = data.clone();
            // if (use_int_type){
            //     data.convertTo(continuous_data, CV_8U);
            // }
            if (!continuous_data.isContinuous())
            {
                continuous_data = continuous_data.clone();
            }
            int width = continuous_data.rows;
            int height = continuous_data.cols;

            // Define chunk dimensions (2D: chunk_width, chunk_height)
            hsize_t chunk_dims[2] = {
                static_cast<hsize_t>(width), // Chunk width (adjust as needed)
                static_cast<hsize_t>(height) // Chunk height (adjust as needed)
            };

            // Create a dataspace for the dataset
            hsize_t dims[2] = {static_cast<hsize_t>(continuous_data.rows),
                               static_cast<hsize_t>(continuous_data.cols)};
            H5::DataSpace dataspace(2, dims);

            // Create a dataset creation property list
            H5::DSetCreatPropList plist;

            // Set compression level (0-9)
            plist.setDeflate(6); // Adjust compression level as needed

            // Set chunk size
            plist.setChunk(2, chunk_dims);

            // Create a dataset in the file
            H5::DataSet dataset = file.createDataSet(var_name, get_hdf5_data_type(data, use_int_type), dataspace, plist);

            // Add a description to the dataset
            add_attribute_to_dataset(dataset, "Description", dataset_description);

            // Add a long name to the dataset
            add_attribute_to_dataset(dataset, "LongName", dataset_long_name);

            // Write data to the dataset
            dataset.write(continuous_data.data, get_hdf5_data_type(data, use_int_type));

            dataset.close();
        }
        catch (const H5::DataSetIException &e)
        {
            std::cerr << "Error: Could not create dataset: " << e.getCDetailMsg() << std::endl;
            throw;
        }
        catch (const H5::DataSpaceIException &e)
        {
            std::cerr << "Error: Could not create dataspace: " << e.getCDetailMsg() << std::endl;
            throw;
        }
    }
    // void write_hdf5(H5::H5File &file, cv::Mat data, const std::string &var_name, bool use_int_type, const std::string &dataset_long_name, const std::string &dataset_description)
    // {
    //     try
    //     {
    //         // Convert data to the smallest appropriate data type
    //         cv::Mat continuous_data;
    //         if (use_int_type){
    //             // Change this type if there is a smaller data type that is sufficient
    //             data.convertTo(continuous_data, CV_8U);
    //         } else {
    //             continuous_data = data.clone();
    //         }
    //         if (!continuous_data.isContinuous())
    //         {
    //             continuous_data = continuous_data.clone();
    //         }
    //         int width = data.rows;
    //         int height = data.cols;

    //         // Define chunk dimensions (2D: chunk_width, chunk_height)
    //         // Adjust these values to find the best size for your dataset
    //         hsize_t chunk_dims[2] = {static_cast<hsize_t>(width),
    //                                  static_cast<hsize_t>(height)};

    //         // Create a dataspace for the dataset
    //         hsize_t dims[2] = {static_cast<hsize_t>(width),
    //                            static_cast<hsize_t>(height)};
    //         H5::DataSpace dataspace(2, dims);

    //         // Create a dataset creation property list
    //         H5::DSetCreatPropList plist;

    //         // Apply the gzip compression filter
    //         plist.setDeflate(9); // A moderate level of compression

    //         // Set the chunking property
    //         plist.setChunk(2, chunk_dims);

    //         // Create a dataset in the file
    //         H5::DataSet dataset = file.createDataSet(var_name, get_hdf5_data_type(continuous_data, use_int_type), dataspace, plist);

    //         // Add a description to the dataset
    //         add_attribute_to_dataset(dataset, "Description", dataset_description);

    //         // Add a long name to the dataset
    //         add_attribute_to_dataset(dataset, "LongName", dataset_long_name);

    //         // Write data to the dataset
    //         dataset.write(continuous_data.data, get_hdf5_data_type(continuous_data, use_int_type));

    //         dataset.close();
    //     }
    //     catch (const H5::Exception &e) // Catching all H5 exceptions
    //     {
    //         std::cerr << "HDF5 Exception: " << e.getCDetailMsg() << std::endl;
    //         throw;
    //     }
    // }
    // closes H5 file once the data are written to it
    void close_hdf5(H5::H5File &file)
    {
        try
        {
            file.close();
        }
        catch (const H5::FileIException &e)
        {
            std::cerr << "Error: Could not close HDF5 file: " << e.getCDetailMsg() << std::endl;
            throw;
        }
    }

    bool containsAny(const std::string &target, const std::vector<std::string> &strings)
    {
        for (const std::string &str : strings)
        {
            if (target.find(str) != std::string::npos)
            {
                return true;
            }
        }
        return false;
    }

    std::unordered_map<std::string, std::string> read_header(std::string filename)
    {
        fitsfile *fitsFilePtr;
        std::string fn = filename;
        int status = 0; // Status variable to check for errors

        fits_open_file(&fitsFilePtr, fn.c_str(), READONLY, &status);
        if (status != 0)
        {
            // Handle the error (e.g., print an error message and return)
        }

        int hdutype;
        fits_movabs_hdu(fitsFilePtr, 2, &hdutype, &status);
        if (status != 0)
        {
            // Handle the error (e.g., print an error message and return)
        }

        int nkeys;
        fits_get_hdrspace(fitsFilePtr, &nkeys, NULL, &status);

        char card[FLEN_CARD]; // Buffer to store each header card

        // create key-value pairs vector to save the metadata
        std::vector<std::string> lines;
        std::string line;

        for (int i = 1; i <= nkeys; i++)
        {
            fits_read_record(fitsFilePtr, i, card, &status);
            if (status != 0)
            {
                // Handle the error (e.g., print an error message and return)
            }

            // Process the header card (e.g., print it)
            // std::cout << card << std::endl;

            lines.push_back(card);
        }

        // read key-value pairs from lines to an array:
        // Separate key-value pairs and store them in a vector of pairs
        std::unordered_map<std::string, std::string> key_value_pairs;
        std::string eq_sign = "=";
        std::string single_quote = "'";
        char equal_sign = '=';
        std::vector<std::string> keys_to_extract = {"QUALITY",
                                                    "ORIGIN",
                                                    "CONTENT",
                                                    "BUNIT",
                                                    "HARPNUM",
                                                    "TELESCOP",
                                                    "WCSNAME",
                                                    "LAT_MIN",
                                                    "LAT_MAX",
                                                    "LON_MIN",
                                                    "LON_MAX",
                                                    "T_REC",
                                                    "SIZE_ACR",
                                                    "AREA_ACR",
                                                    "CRPIX1",
                                                    "CRPIX2",
                                                    "RSUN_OBS",
                                                    "CDELT1"};

        for (const auto &line : lines)
        {
            std::istringstream line_stream(line);
            std::string key;
            std::string value;

            line_stream >> key >> std::ws >> equal_sign >> value;
            size_t pos = key.find(eq_sign);
            if (pos != std::string::npos)
            {
                // remove equal sign from key string
                key.replace(pos, eq_sign.length(), "");
            }

            size_t pos_quote;
            while ((pos_quote = value.find('\'')) != std::string::npos)
            {
                value.erase(pos_quote, 1);
            }

            if (containsAny(key, keys_to_extract))
            {
                key_value_pairs.insert(std::make_pair(key, value));
            }
        }
        // for (const auto &pair : key_value_pairs)
        // {
        //     std::cout << "Key: " << pair.first << " | Value: " << pair.second << std::endl;
        // }
        fits_close_file(fitsFilePtr, &status);
        if (status != 0)
        {
            // Handle the error (e.g., print an error message)
        }
        return key_value_pairs;
    }

    std::unordered_map<std::string, std::string> get_params(ModelParameters params)
    {
        try
        {
            struct ParamInfo
            {
                std::string name;
                std::string value;
            };

            std::vector<ParamInfo> paramInfos = {
                {"pos_gauss", std::to_string(params.pos_gauss)},
                {"neg_gauss", std::to_string(params.neg_gauss)},
                {"dilation_size", std::to_string(params.dilation_size)},
                {"strength_threshold", std::to_string(params.strength_threshold)},
                {"size_threshold", std::to_string(params.size_threshold)},
                {"gap_size", std::to_string(params.gap_size)}};

            std::unordered_map<std::string, std::string> key_value_pairs;
            for (const ParamInfo &paramInfo : paramInfos)
            {
                key_value_pairs.insert(std::make_pair(paramInfo.name, paramInfo.value));
            }

            return key_value_pairs;
        }
        catch (const H5::Exception &e)
        {
            std::cerr << "Error: " << e.getCDetailMsg() << std::endl;
            throw;
        }
    }

    void writeKeyValuePairsToCSV(const std::string &csvFilePath, const std::unordered_map<std::string, std::string> &keyValuePairs)
    {
        std::ofstream csvFile(csvFilePath);
        if (!csvFile)
        {
            std::cerr << "Error creating CSV file: " << csvFilePath << std::endl;
            return;
        }

        // Write the header row
        csvFile << "Key,Value\n";

        // Write each key-value pair
        for (const auto &kvp : keyValuePairs)
        {
            csvFile << kvp.first << "," << kvp.second << "\n";
        }

        csvFile.close();
    }

    void write_properties(const std::string &csvFilePath, const std::map<std::string, std::string> &keyValuePairs)
    {
        std::ofstream csvFile(csvFilePath);
        if (!csvFile)
        {
            std::cerr << "Error creating CSV file: " << csvFilePath << std::endl;
            return;
        }

        // Write the header row
        csvFile << "Key,Value\n";

        // Write each key-value pair
        for (const auto &kvp : keyValuePairs)
        {
            csvFile << kvp.first << "," << kvp.second << "\n";
        }

        csvFile.close();
    }
    void write_table(std::unordered_map<std::string, std::string> properties, H5::H5File &file, std::string datasetName)
    {
        struct TableRow
        {
            std::string key;
            std::string value;
        };
        // Define the compound datatype for TableRow
        H5::CompType rowType(sizeof(TableRow));
        rowType.insertMember("key", HOFFSET(TableRow, key), H5::StrType(H5::PredType::C_S1, H5T_VARIABLE));
        rowType.insertMember("value", HOFFSET(TableRow, value), H5::StrType(H5::PredType::C_S1, H5T_VARIABLE));

        // Create the dataspace
        hsize_t dims[1] = {properties.size()}; // Number of rows
        H5::DataSpace dataSpace(1, dims);
        H5::DataSet dataSet = file.createDataSet(datasetName, rowType, dataSpace);

        // Prepare data
        TableRow tableData[properties.size()]; // Example: 10 rows of data

        for (int i = 0; i < properties.size(); ++i)
        {
            auto it = properties.begin();
            std::advance(it, i);
            tableData[i].key = it->first;
            tableData[i].value = it->second;
        }

        // Write the data to the dataset
        dataSet.write(tableData, rowType);
    }
}
