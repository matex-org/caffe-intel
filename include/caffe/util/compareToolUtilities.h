/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef INCLUDE_CAFFE_UTIL_COMPARETOOLUTILITIES_H_
#define INCLUDE_CAFFE_UTIL_COMPARETOOLUTILITIES_H_

#include <dirent.h>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

template <typename DataType>
class Data {
    int dataSize;
    DataType *dataPointer;

    Data(const Data<DataType> &data);
    Data<DataType> &operator =(const Data<DataType> &data);

 public:
        Data() :
            dataSize(0),
            dataPointer(NULL) {
        }

        ~Data() {
            clear();
        }

        int getDataSize() const {
            return dataSize;
        }

        const DataType *getDataPointer() const {
            return dataPointer;
        }

        void clear() {
            delete [] dataPointer;
            dataPointer = NULL;
            dataSize = 0;
        }

        bool loadFromFile(const char *fileName) {
            boost::filesystem::path filePath(fileName);
            if (!boost::filesystem::exists(filePath)) {
                return false;
            }

            if (boost::filesystem::is_empty(filePath)) {
                return false;
            }

            FILE *file = fopen(fileName, "rb");
            if (!file)
                return false;

            int64_t fileSize = boost::filesystem::file_size(filePath);
            DataType *fileDataPointer = new DataType[fileSize];
            size_t bytesRead = fread(fileDataPointer, 1, fileSize, file);
            fclose(file);
            if (bytesRead != fileSize) {
                delete [] fileDataPointer;
                return false;
            }

            clear();

            dataPointer = fileDataPointer;
            dataSize = fileSize / sizeof(DataType);
            return true;
        }
};

class FileList {
    std::vector<std::string> fileList;

    FileList(const FileList &fileList);
    FileList &operator =(const FileList &fileList);

 public:
        FileList() {}

        int getNumberOfFiles() const {
            return static_cast<int>(fileList.size());
        }

        const char *getFileName(int fileIndex) const {
            return fileList[fileIndex].c_str();
        }

        void clear() {
            fileList.clear();
        }

        void findFiles() {
            fileList.clear();
            fileList.reserve(1024);

            DIR *dir = opendir(FLAGS_collect_dir.c_str());
            if (dir) {
                struct dirent *dirEntry = readdir(dir);
                while (dirEntry) {
                    if (!strncmp(dirEntry->d_name, "REF", 3))
                        fileList.push_back(&dirEntry->d_name[3]);
                    dirEntry = readdir(dir);
                }
                closedir(dir);
            }

            std::sort(fileList.begin(), fileList.end());
            fileList.shrink_to_fit();
        }
};

class Log {
    FILE *logFile;

    Log() {
        logFile = fopen("log.txt", "w+b");
        CHECK(logFile != NULL) << "Could not open log.txt file";
    }

 public:
        ~Log() {
            if (logFile)
                fclose(logFile);
        }

        static void log(const char *format, ...) {
            // #pragma omp critical
            {
                va_list args;

                static Log log;

                va_start(args, format);
                vfprintf(log.logFile, format, args);

                va_start(args, format);
                vprintf(format, args);

                va_end(args);
            }
        }
};

double compareFiles(const char *diffFileName, const char *referenceFileName,
  const char *targetFileName, double *maxDiff, unsigned *diffCounter) {
    typedef float DataType;
    typedef uint32_t CastType;
    const char *format = "%i;%08X;%08X;%g;%g;%g\n";
    const DataType epsilon = (DataType) FLAGS_epsilon;

    Data<DataType> referenceData;
    if (!referenceData.loadFromFile(referenceFileName)) {
        Log::log("Failed to load reference data file '%s'.\n",
          referenceFileName);
        return false;
    }

    Data<DataType> targetData;
    if (!targetData.loadFromFile(targetFileName)) {
        Log::log("Failed to load target data file '%s'.\n", targetFileName);
        return false;
    }

    if (targetData.getDataSize() != referenceData.getDataSize()) {
        Log::log("Data length is not equal.\n");
        return false;
    }

    FILE *file = NULL;
    if (diffFileName)
        file = fopen(diffFileName, "w+t");

    *maxDiff = -1;
    *diffCounter = 0;

    int dataSize = targetData.getDataSize();
    const DataType *referenceDataPointer = referenceData.getDataPointer();
    const DataType *targetDataPointer = targetData.getDataPointer();
    for (int i = 0; i < dataSize; i++) {
        DataType a = referenceDataPointer[i];
        DataType b = targetDataPointer[i];

        DataType aAbs = (DataType) fabs(a), bAbs = (DataType) fabs(b);
        DataType diff =
            ((a * b) < 0) ? 1 :
            (aAbs && (aAbs < bAbs)) ? bAbs / aAbs - 1 :
            (bAbs && (bAbs < aAbs)) ? aAbs / bAbs - 1 :
            (aAbs == bAbs) ? 0 : 1;

        if (file && (diff >= epsilon)) {
            fprintf(file, format, i,
              *reinterpret_cast<CastType *> (&a),
              *reinterpret_cast<CastType *> (&b), diff, a, b);
            (*diffCounter)++;
        }

        if (*maxDiff < diff)
            *maxDiff = diff;
    }

    if (file)
        fclose(file);

    return true;
}

void processFile(const char *fileName, const string& layerType,
  std::unordered_map<string, int> *errorsDictionary) {
    char referenceFileName[FILENAME_MAX];
    char targetFileName[FILENAME_MAX];
    char diffFileName[FILENAME_MAX];
    snprintf(referenceFileName, sizeof(referenceFileName), "./%s/REF%s",
      FLAGS_collect_dir.c_str(), fileName);
    snprintf(targetFileName, sizeof(targetFileName), "./%s/TGT%s",
      FLAGS_compare_output_dir.c_str(), fileName);
    snprintf(diffFileName, sizeof(diffFileName), "./%s/OUT%s",
      FLAGS_compare_output_dir.c_str(), fileName);
    double maxDiff;
    unsigned diffCounter;
    bool success = compareFiles(diffFileName, referenceFileName,
      targetFileName, &maxDiff, &diffCounter);
    if (!success) {
      Log::log("%-16s %-20s : failed\n", fileName, layerType.c_str());
    } else if (!diffCounter) {
      Log::log("%-16s %-20s : success\n", fileName, layerType.c_str());
    } else {
      Log::log("%-16s %-20s : %g %u\n", fileName, layerType.c_str(),
        maxDiff, diffCounter);
      (*errorsDictionary)[layerType]++;
    }
}

void proceedWithCompare(const string& infoPath,
  std::unordered_map<string, int> *errorsDictionary) {
  FileList fileList;
  fileList.findFiles();

  std::unordered_map<string, string> layersInfo;
  std::ifstream layersInfoFile;
  layersInfoFile.open(infoPath);
  string key, name;
  while (layersInfoFile >> key >> name) {
    layersInfo[key + ".bin"] = name;
  }

  layersInfoFile.close();

  int numberOfFiles = fileList.getNumberOfFiles();

  LOG(INFO) << "Comparing layers data";
  // #pragma omp parallel for
  for (int fileIndex = 0; fileIndex < numberOfFiles; fileIndex++) {
    const char* binFileName = fileList.getFileName(fileIndex);
    processFile(binFileName, layersInfo[binFileName], errorsDictionary);
  }
}

#endif  // INCLUDE_CAFFE_UTIL_COMPARETOOLUTILITIES_H_
