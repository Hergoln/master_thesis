#include "global.h"
#include <fstream>

int main(int argc, char* argv[]) {
  std::string outfile, infile;
  std::fstream ofs, ifs;

  std::string option = find_arg(argv, argc, "outfile");
  if (option.empty()) {
    std::cerr << "Didn't receive output file" << std::endl;
    outfile = "test_out_decoded.parsed";
  } else {
    outfile += option;
  }

  option = find_arg(argv, argc, "inputfile");
  if (option.empty()) {
    std::cerr << "Didn't receive input file" << std::endl;
    infile = "test_out.parsed";
  } else {
    infile += option;
  }

  fill_vocabulary();
  ifs.open(infile, std::ofstream::in);
  ofs.open(outfile, std::ofstream::out);
  std::string s;
  while(!ifs.eof()) {
    ifs >> s;
    ofs << decode(s) << " ";
  }
  ofs.close();
  ifs.close();
  std::cout << "Write to file: " << outfile << std::endl;
  return 0;
}