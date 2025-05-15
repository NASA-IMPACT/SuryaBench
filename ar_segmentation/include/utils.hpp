#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>


namespace utils
{

	int GetImageSize(std::string filename, int *nColumns, int *nRows, bool verbose);
	void PrintError(int status);
	void display_progress_bar(int current, int total, int bar_width);
}
