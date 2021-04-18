#include <fstream>
#include "utils.h"
#include "detection.h"

FileDetector::FileDetector(FileDetectorConfig& config)
	:params_(config)
	, frame_idx_(-1)
{
	clear_2d_vector(vec_vec_det_);
}

FileDetector::~FileDetector()
{
}

std::vector<DetectionBox> FileDetector::read_one_file(int index)
{
	std::vector<DetectionBox > vec_det;
	float score;

	std::ifstream infile(vec_det_filename_[index]);
	if (infile.fail())
	{
		std::cout << "read file fails: " << vec_det_filename_[index] << ", cannot read annotation file." << std::endl;
		return vec_det;
	}

	int fileindex, id;
	std::string detLine;
	std::istringstream ss;
	char ch;
	float tpx, tpy, tpw, tph;

	while (std::getline(infile, detLine))
	{
		ss.str(detLine);
		ss >> fileindex >> ch >> id >> ch;
		ss >> tpx >> ch >> tpy >> ch >> tpw >> ch >> tph >> ch >> score;
		ss.str("");
		if (score < params_.threshold)
		{
			continue;
		}
		int size = tph * tpw;
		if (params_.min_size != -1)
		{
			if (size < params_.min_size) continue;
		}
		if (params_.max_size != -1)
		{
			if (size > params_.max_size) continue;
		}
		vec_det.push_back(DetectionBox(fileindex, cv::Rect_<float>(cv::Point_<float>(tpx, tpy), cv::Point_<float>(tpx + tpw, tpy + tph)), score));
	}
	infile.close();
	//std::cout << "aaaaaaaa" << std::endl;
	return vec_det;
}

bool FileDetector::read_from_list()
{
	if (params_.det_list_name == "")
		return false;
	read_filelist(params_.det_list_name, vec_det_filename_);
	return true;
}

bool FileDetector::read_from_file()
{
	if (params_.det_file_name == "")
		return false;
	std::ifstream detectionFile;
	detectionFile.open(params_.det_file_name);

	if (!detectionFile.is_open())
	{
		std::cerr << "Error: can not find file " << params_.det_file_name << std::endl;
		return false;
	}
	float score;
	int fileindex, id;
	std::string detLine;
	std::istringstream ss;
	std::vector<DetectionBox> detbbx;
	char ch;
	float tpx, tpy, tpw, tph;

	while (std::getline(detectionFile, detLine))
	{
		ss.str(detLine);
		ss >> fileindex >> ch >> id >> ch;
		ss >> tpx >> ch >> tpy >> ch >> tpw >> ch >> tph >> ch >> score;
		ss.str("");
		if (score < params_.threshold)
		{
			continue;
		}
		detbbx.push_back(DetectionBox(fileindex, cv::Rect_<float>(cv::Point_<float>(tpx, tpy), cv::Point_<float>(tpx + tpw, tpy + tph)), score));
	}
	detectionFile.close();

	// 2. group detData by frame
	int maxFrame = 0;
	for (auto tb : detbbx) // find max frame number
	{
		if (maxFrame < tb.frame)
			maxFrame = tb.frame;
	}

	std::vector<DetectionBox> tempVec;
	for (int fi = 0; fi < maxFrame; fi++)
	{
		for (auto tb : detbbx)
			if (tb.frame == fi) // frame num starts from 1
				tempVec.push_back(tb);
		vec_vec_det_.push_back(tempVec);
		tempVec.clear();
	}
	return true;
}

bool FileDetector::init()
{
	if (params_.det_file_name != "" && params_.det_list_name != "")
	{
		std::cout << "please check two files exists, only one should be non-empty..." << std::endl;
		return false;
	}
	if (params_.det_file_name != "")
		return read_from_file();
	else if (params_.det_list_name != "")
		return read_from_list();
}

void FileDetector::get_detection(cv::Mat& frame, std::vector<DetectionBox>& vec_db, std::vector<cv::Mat>& vec_features)
{
	int im_w = frame.cols;
	int im_h = frame.rows;
	vec_db.clear();
	vec_features.clear();
	frame_idx_++;
// 	if (params_.det_file_name != "")
// 		vec_db = vec_vec_det_[frame_idx_];
// 	else if (params_.det_list_name != "")
// 		vec_db = read_one_file(frame_idx_);

	std::ifstream infile(vec_det_filename_[frame_idx_]+".size");
	if (infile.fail())
	{
		std::cout << "read file fails: " << vec_det_filename_[frame_idx_] << ", cannot read annotation file." << std::endl;
		return;
	}
	int det_cnt, det_dim, fea_cnt, fea_dim;
	infile >> det_cnt >> det_dim >> fea_cnt >> fea_dim;
	infile.close();
	std::vector<float> dets(det_cnt*det_dim);
	std::ifstream in_det(vec_det_filename_[frame_idx_] + ".det", ios::in | ios::binary);
	if (det_cnt>0)
		in_det.read((char *)&dets[0], sizeof(float)*dets.size());
	in_det.close();
	for (int i=0;i<det_cnt;i++)
	{
		float x1 = dets[i*det_dim]; if (x1 < 0)x1 = 0; if (x1 > (im_w - 1)) x1 = im_w - 1;
		float y1 = dets[i*det_dim + 1]; if (y1 < 0)y1 = 0; if (y1 > (im_h - 1)) y1 = im_h - 1;
		float x2 = dets[i*det_dim + 2]; if (x2 < 0)x2 = 0; if (x2 > (im_w - 1)) x2 = im_w - 1;
		float y2 = dets[i*det_dim + 3]; if (y2 < 0)y2 = 0; if (y2 > (im_h - 1)) y2 = im_h - 1;
		vec_db.push_back(DetectionBox(frame_idx_,
			cv::Rect_<float>(cv::Point_<float>(x1, y1),
				cv::Point_<float>(x2, y2)), dets[i*det_dim + 4]));
	}
	std::vector<float> features(fea_cnt*fea_dim);
	std::ifstream in_fea(vec_det_filename_[frame_idx_] + ".feature", ios::in | ios::binary);
	if (fea_cnt > 0)
		in_fea.read((char *)&features[0], sizeof(float)*features.size());
	in_fea.close();
	for (int i = 0; i < fea_cnt; i++)
	{
		cv::Mat im(1, fea_dim, CV_32FC1, &features[0] + i * fea_dim);
		float*data = (float*)im.data;
		vec_features.push_back(im.clone());
	}
}





















FairMOTDetector::FairMOTDetector(FairMOTDetectorConfig& config)
	:params_(config)
	, frame_idx_(-1)
{
}

FairMOTDetector::~FairMOTDetector()
{
	delete[] data_; data_ = NULL;
	for (int i = 0; i < output_tensor_num; i++)
	{
		delete[] output_data[i];
		output_data[i] = NULL;
	}
	context_->destroy();
	engine_->destroy();
	runtime_->destroy();
}


bool FairMOTDetector::generate_engine(std::string& onnx_model_name, std::string& engine_model_name)
{
	auto builder = nvinfer1::createInferBuilder(gLogger);
	if (!builder)
	{
		return false;
	}

	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);

	if (!network)
	{
		return false;
	}

	auto config = builder->createBuilderConfig();
	if (!config)
	{
		return false;
	}

	auto parser = nvonnxparser::createParser(*network, gLogger);
	if (!parser)
	{
		return false;
	}

	parser->parseFromFile(onnx_model_name.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
	config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
	config->setFlag(nvinfer1::BuilderFlag::kTF32);

	std::cout << "Building engine, please wait for a while..." << std::endl;
	nvinfer1::ICudaEngine* mEngine = builder->buildEngineWithConfig(*network, *config);
	if (!mEngine)
	{
		return false;
	}
	std::cout << "Build engine successfully!" << std::endl;

	auto mInputDims = network->getInput(0)->getDimensions();
	auto mOutputDims = network->getOutput(0)->getDimensions();
	std::cout << "network->getNbOutputs(): " << network->getNbOutputs() << std::endl;
	std::cout << "mOutputDims.nbDims: " << mOutputDims.nbDims << std::endl;

	network->destroy();
	// Serialize the engine
	nvinfer1::IHostMemory* modelStream = mEngine->serialize();

	std::ofstream p(engine_model_name, std::ios::binary);
	if (!p) {
		std::cerr << "could not open plan output file" << std::endl;
		return -1;
	}
	p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	modelStream->destroy();

	// Close everything down
	mEngine->destroy();
	builder->destroy();
	return true;
}

bool FairMOTDetector::init()
{
	auto engine_model_name = params_.model_file + ".fp32.trtmodel";
	if (!exists(engine_model_name))
	{
		auto onnx_model_name = params_.model_file + ".onnx";
		if (!exists(onnx_model_name))
		{
			LOG("onnx models not found");
			return false;
		}

		if (!generate_engine(onnx_model_name, engine_model_name))
		{
			LOG("convert onnx models failed");
			return false;
		}
	}

	char*trtModelStream; int size;
	std::ifstream file(engine_model_name, std::ios::binary);
	if (file.good())
	{
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream = new char[size];
		file.read(trtModelStream, size);
		file.close();
	}
	else return false;

	runtime_ = nvinfer1::createInferRuntime(gLogger);
	assert(runtime_ != nullptr);
	engine_ = runtime_->deserializeCudaEngine(trtModelStream, size);
	assert(engine_ != nullptr);
	context_ = engine_->createExecutionContext();
	assert(context_ != nullptr);
	delete[] trtModelStream;
	const nvinfer1::ICudaEngine& engine = context_->getEngine();
	input_index = engine.getBindingIndex(input_name.c_str());
	auto input_dims = engine.getBindingDimensions(input_index);
	input_h = input_dims.d[2];
	input_w = input_dims.d[3];
	data_ = new float[3 * input_h * input_w];

	for (int i = 0; i < output_tensor_num; i++)
	{
		output_idx[i] = engine.getBindingIndex(output_names[i].c_str());
		auto output_dims = engine.getBindingDimensions(output_idx[i]);
		int size = 1;
		for (int j = 0; j < output_dims.nbDims; j++)
		{
			size *= output_dims.d[j];
		}
		output_data[i] = new float[size];
		output_size[i] = size;
	}
	return true;
}

cv::Rect FairMOTDetector::restoreCenterNetBox(float dx, float dy, float l, float t, float r, float b, float cellx, float celly, int stride, cv::Size netSize, cv::Size imageSize)
{
	float scale = 0;
	if (imageSize.width >= imageSize.height)
		scale = netSize.width / (float)imageSize.width;
	else
		scale = netSize.height / (float)imageSize.height;

	float xx = ((cellx + dx - l) * stride - netSize.width * 0.5) / scale + imageSize.width * 0.5;
	float yy = ((celly + dy - t) * stride - netSize.height * 0.5) / scale + imageSize.height * 0.5;
	float rr = ((cellx + dx + r) * stride - netSize.width * 0.5) / scale + imageSize.width * 0.5;
	float bb = ((celly + dy + b) * stride - netSize.height * 0.5) / scale + imageSize.height * 0.5;
	return cv::Rect(cv::Point(xx, yy), cv::Point(rr + 1, bb + 1));
}



void FairMOTDetector::get_detection(cv::Mat& frame, std::vector<DetectionBox>& vec_db, std::vector<cv::Mat>& vec_features)
{
	vec_db.clear();
	vec_features.clear();
	frame_idx_++;

	int frame_width = frame.cols;
	int frame_height = frame.rows;
	cv::Mat sized; int padding = 0;
	std::vector<cv::Mat> ms(frame.channels());

	float ratio = float(input_w) / float(frame_width) < float(input_h) / float(frame_height) ? float(input_w) / float(frame_width) : float(input_h) / float(frame_height);
	cv::Mat flt_img = cv::Mat::zeros(cv::Size(input_w, input_h), CV_8UC3);
	cv::Mat rsz_img;
	cv::resize(frame, rsz_img, cv::Size(), ratio, ratio);
	rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
	flt_img.convertTo(flt_img, CV_32FC3, 1.0 / 255);
	for (int i = 0; i < ms.size(); ++i)
		ms[i] = cv::Mat(input_h, input_w, CV_32F, &data_[i*input_h * input_w]);
	cv::split(flt_img, ms);
	const nvinfer1::ICudaEngine& engine = context_->getEngine();
	void*buffers[output_tensor_num + 1];
	int inputIndex = 0;
	auto input_dims = engine.getBindingDimensions(inputIndex);
	int input_size = 1;
	for (int j = 0; j < input_dims.nbDims; j++)
	{
		input_size *= input_dims.d[j];
	}
	CHECK(cudaMalloc(&buffers[inputIndex], input_size * sizeof(float)));

	for (int i = 0; i < output_tensor_num; i++)
	{
		CHECK(cudaMalloc(&buffers[output_idx[i]], output_size[i] * sizeof(float)));
	}
	// Create stream
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
	CHECK(cudaMemcpyAsync(buffers[inputIndex], data_, input_size * sizeof(float), cudaMemcpyHostToDevice, stream));
	context_->enqueue(1, buffers, stream, nullptr);
	for (int i = 0; i < output_tensor_num; i++)
	{
		CHECK(cudaMemcpyAsync(output_data[i], buffers[output_idx[i]], output_size[i] * sizeof(float), cudaMemcpyDeviceToHost, stream));
	}
	cudaStreamSynchronize(stream);

	// Release stream and buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	for (int i = 0; i < output_tensor_num; i++)
	{
		CHECK(cudaFree(buffers[output_idx[i]]));
	}

	float *hm_ptr = output_data[0];
	float *wh_ptr = output_data[1];
	float *reg_ptr = output_data[2];
	float *hmpool_ptr = output_data[3];
	float *id_ptr = output_data[4];
	int size = output_size[0];
	int x, y;
	int down_ratio = 4;
	int hm_width = engine.getBindingDimensions(output_idx[0]).d[3];
	int hm_height = engine.getBindingDimensions(output_idx[0]).d[2];
	int id_dims[4];
	for (int i = 0; i < 4; i++) id_dims[i] = engine.getBindingDimensions(output_idx[4]).d[i];
	std::vector<cv::RotatedRect> vec_rrect;
	std::vector<float> vec_score;
	ratio = 1.0f / ratio;

	int outsize = hm_width * hm_height;
	for (int idx = 0; idx < outsize; idx++)
	{
		if (*hm_ptr == *hmpool_ptr && *hmpool_ptr > params_.threshold)
		{
			x = idx % hm_width;
			y = idx / hm_width;

			float dx = *(reg_ptr + idx);
			float dy = *(reg_ptr + outsize + idx);
			float l = *(wh_ptr + idx);
			float t = *(wh_ptr + outsize + idx);
			float r = *(wh_ptr + outsize * 2 + idx);
			float b = *(wh_ptr + outsize * 3 + idx);

			cv::Rect box = restoreCenterNetBox(dx, dy, l, t, r, b, x, y, down_ratio, cv::Size(input_w, input_h), frame.size());
			box = box & cv::Rect(0, 0, frame.cols, frame.rows);
			if (box.area() > 0)
			{
				vec_db.push_back(DetectionBox(frame_idx_,
					cv::Rect_<float>(box.x, box.y, box.width, box.height), *hmpool_ptr));
				float*dim_fea = id_ptr + y * (id_dims[2]* id_dims[3]) + x * id_dims[3];
				cv::Mat fea(1, id_dims[3], CV_32FC1, dim_fea);
				float*tmpdata2 = (float*)fea.data;
				vec_features.emplace_back(fea.clone());
			}
		}
		hm_ptr++;
		hmpool_ptr++;
	}
}
