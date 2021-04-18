#pragma once
#include "opencv2/opencv.hpp"

using namespace cv;
#include "config.h"


#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace  nvinfer1;
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


class Logger : public nvinfer1::ILogger
{
public:
	Logger(Severity severity = Severity::kWARNING)
	{

	}

	~Logger()
	{

	}
	nvinfer1::ILogger& getTRTLogger()
	{
		return *this;
	}

	void log(nvinfer1::ILogger::Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity == Severity::kINFO) return;

		switch (severity)
		{
		case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: " << msg << std::endl; break;
		case Severity::kERROR: std::cerr << "ERROR: " << msg << std::endl; break;
		case Severity::kWARNING: std::cerr << "WARNING: " << msg << std::endl; break;
		case Severity::kINFO: std::cerr << "INFO: " << msg << std::endl; break;
		case Severity::kVERBOSE: break;
			//  default: std::cerr <<"UNKNOW:"<< msg << std::endl;break;
		}
	}
};
static Logger gLogger;


//template <typename T>
class Detection //should use boost::noncopyable //interface 
{
public:
	virtual ~Detection() {};

	virtual bool init() = 0;
	virtual void get_detection(cv::Mat& frame, std::vector<DetectionBox>& vec_db, std::vector<cv::Mat>& vec_features) = 0;
};

class FileDetector :public Detection
{
public:
	FileDetector() = delete;
	explicit FileDetector(FileDetectorConfig& config);
	virtual ~FileDetector();
	FileDetector(const FileDetector&) = delete;
	FileDetector& operator=(const FileDetector&) = delete;

	bool init();
	void get_detection(cv::Mat& frame, std::vector<DetectionBox>& vec_db, std::vector<cv::Mat>& vec_features);
private:
	bool read_from_list();
	bool read_from_file();
	std::vector<DetectionBox> read_one_file(int index);
private:
	FileDetectorConfig& params_;
	int frame_idx_;
	std::vector<std::string> vec_det_filename_;
	std::vector<std::vector<DetectionBox>> vec_vec_det_;
};


class FairMOTDetector :public Detection
{
public:
	FairMOTDetector() = delete;
	explicit FairMOTDetector(FairMOTDetectorConfig& config);
	virtual ~FairMOTDetector();
	FairMOTDetector(const FairMOTDetector&) = delete;
	FairMOTDetector& operator=(const FairMOTDetector&) = delete;

	bool init();
	void get_detection(cv::Mat& frame, std::vector<DetectionBox>& vec_db, std::vector<cv::Mat>& vec_features);
private:
	cv::Rect restoreCenterNetBox(float dx, float dy, float l, float t, float r, float b, float cellx, float celly, int stride, cv::Size netSize, cv::Size imageSize);
	bool generate_engine(std::string& onnx_model_name, std::string& engine_model_name);
private:
	FairMOTDetectorConfig& params_;
	int frame_idx_;

	const static int output_tensor_num = 5;
	float* data_;
	int input_index;
	std::string input_name= "input.1";
	int output_idx[5];
	int output_size[5];
	float *output_data[5];
	std::string output_names[5] = { "hm", "wh","reg","hm_pool", "id" };

	nvinfer1::IRuntime* runtime_;
	nvinfer1::ICudaEngine* engine_;
	nvinfer1::IExecutionContext* context_;
	int input_h;
	int input_w;
};

//    -m 
//template <typename T>
class DetectorFactory
{
public:
	// 	FaceDetectionFactory();
	// 	virtual ~FaceDetectionFactory();
	static Detection* create_object(/*const*/ DetectorConfig& config)
	{
		switch (config.method)
		{
		case DetectorMethod::FromFile:
			return new FileDetector(config.fd);
		case DetectorMethod::FromFairMOT:
			return new FairMOTDetector(config.fairmot);
			
		}
	}
};

