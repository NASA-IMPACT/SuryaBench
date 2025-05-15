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
#include <fitsio.h>
#include <string>
#include <mutex>
#include <iostream>

#include "utils.hpp"
// #include "fitsio.h"



// Global mutex for FITSIO operations
std::mutex fitsioMutex;




namespace utils
{
    /* ---------------- FUNCTION: GetImageSize ----------------------------- */
    /*    Given a filename, it opens the file, reads the size of the image and
     * stores that size in *nRows and *nColumns.
     *
     *    Returns 0 for successful operation, -1 if a CFITSIO-related error occurred.
     */
    // int GetImageSize(std::string filename, int *nColumns, int *nRows, bool verbose)
    // {
    //     fitsfile *imfile_ptr;
    //     int status, nfound;
    //     int problems;
    //     long naxes[2];
    //     int n_rows, n_columns;

    //     status = problems = 0;
    //     std::string fn = filename + "[1]";
    //     /* Open the FITS file: */
    //     problems = fits_open_file(&imfile_ptr, fn.c_str(), READONLY, &status);
    //     if (problems)
    //     {
    //         fprintf(stderr, "\n*** WARNING: Problems opening FITS file \"%s\"!\n    FITSIO error messages follow:", filename.c_str());
    //         PrintError(status);
    //         return -1;
    //     }

    //     /* read the NAXIS1 and NAXIS2 keyword to get image size */
    //     problems = fits_read_keys_lng(imfile_ptr, "ZNAXIS", 1, 2, naxes, &nfound,
    //                                   &status);
    //     if (problems)
    //     {
    //         fprintf(stderr, "\n*** WARNING: Problems reading FITS keywords from file \"%s\"!\n    FITSIO error messages follow:", filename.c_str());
    //         PrintError(status);
    //         return -1;
    //     }
    //     if (verbose)
    //         printf("GetImageSize: Image keywords: NAXIS1 = %ld, NAXIS2 = %ld\n", naxes[0], naxes[1]);

    //     n_columns = naxes[0]; // FITS keyword NAXIS1 = # columns
    //     *nColumns = n_columns;
    //     n_rows = naxes[1]; // FITS keyword NAXIS2 = # rows
    //     *nRows = n_rows;

    //     if (problems)
    //     {
    //         fprintf(stderr, "\n*** WARNING: Problems closing FITS file \"%s\"!\n    FITSIO error messages follow:", filename.c_str());
    //         PrintError(status);
    //         return -1;
    //     }

    //     return 0;
    // }

int GetImageSize(std::string filename, int *nColumns, int *nRows, bool verbose)
{
    fitsfile *imfile_ptr;
    int status, nfound;
    long naxes[2];
    int problems;

    status = 0;
    std::string fn = filename + "[1]";

    {
        std::lock_guard<std::mutex> lock(fitsioMutex); // Lock for FITSIO operations
        if (fits_open_file(&imfile_ptr, fn.c_str(), READONLY, &status))
        {
            fprintf(stderr, "\n*** HEADER WARNING: Problems opening FITS file \"%s\"!\n    FITSIO error messages follow:", filename.c_str());
            PrintError(status);
            return -1;
        }

        if (fits_read_keys_lng(imfile_ptr, "ZNAXIS", 1, 2, naxes, &nfound, &status))
        {
            fprintf(stderr, "\n*** HEADER WARNING: Problems reading FITS keywords from file \"%s\"!\n    FITSIO error messages follow:", filename.c_str());
            PrintError(status);
            fits_close_file(imfile_ptr, &status); // Close file before returning
            return -1;
        }
    }

    if (verbose)
        printf("GetImageSize: Image keywords: NAXIS1 = %ld, NAXIS2 = %ld\n", naxes[0], naxes[1]);

    *nColumns = naxes[0]; // FITS keyword NAXIS1 = # columns
    *nRows = naxes[1];    // FITS keyword NAXIS2 = # rows

    {
        std::lock_guard<std::mutex> lock(fitsioMutex); // Lock for FITSIO operations
        if (fits_close_file(imfile_ptr, &status))
        {
            fprintf(stderr, "\n*** HEADER WARNING: Problems closing FITS file \"%s\"!\n    FITSIO error messages follow:", filename.c_str());
            PrintError(status);
            return -1;
        }
    }

    return 0;
}
    void PrintError(int status)
    {
        if (status)
        {
            fits_report_error(stderr, status);
            fprintf(stderr, "\n");
            //    exit(status);
        }
    }

    void display_progress_bar(int current, int total, int bar_width = 50)
    {
        float progress = static_cast<float>(current) / static_cast<float>(total);
        int pos = static_cast<int>(bar_width * progress);

        std::cout << "Computing PILS: [";
        for (int i = 0; i < bar_width; ++i)
        {
            if (i < pos)
            {
                std::cout << "=";
            }
            else if (i == pos)
            {
                std::cout << ">";
            }
            else
            {
                std::cout << " ";
            }
        }
        std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0) << "%\r";
        std::cout.flush();
    }
}
