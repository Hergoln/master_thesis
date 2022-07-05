#include "global.h"
#include <fstream>

int main(int argc, char* argv[]) {
  std::string outfile, infile;
  std::fstream ofs, ifs;

  if (argc <= 2) {
    std::cerr << "Didn't receive output and input files" << std::endl;
    outfile = "test_out_decoded.parsed";
    infile = "test_out.parsed";
  } else {
    outfile += argv[1];
    infile += argv[2];
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