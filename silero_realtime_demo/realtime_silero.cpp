#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <limits>
#include <chrono>
#include <memory>
#include <string>
#include <stdexcept>
#include <iostream>
#include <string>
#include "onnxruntime_cxx_api.h"
#include "wav.h"
#include "getopt.h"
#include <cstdio>
#include <cstdarg>
#if __cplusplus < 201703L
#include <memory>
#endif

//#define __DEBUG_SPEECH_PROB___

class timestamp_t
{
public:
    int start;
    int end;

    // default + parameterized constructor
    timestamp_t(int start = -1, int end = -1)
        : start(start), end(end)
    {
    };

    // assignment operator modifies object, therefore non-const
    timestamp_t& operator=(const timestamp_t& a)
    {
        start = a.start;
        end = a.end;
        return *this;
    };

    // equality comparison. doesn't modify object. therefore const.
    bool operator==(const timestamp_t& a) const
    {
        return (start == a.start && end == a.end);
    };
    std::string c_str()
    {
        //return std::format("timestamp {:08d}, {:08d}", start, end);
        return format("{start:%08d,end:%08d}", start, end);
    };
private:

    std::string format(const char* fmt, ...)
    {
        char buf[256];

        va_list args;
        va_start(args, fmt);
        const auto r = std::vsnprintf(buf, sizeof buf, fmt, args);
        va_end(args);

        if (r < 0)
            // conversion failed
            return {};

        const size_t len = r;
        if (len < sizeof buf)
            // we fit in the buffer
            return { buf, len };

#if __cplusplus >= 201703L
        // C++17: Create a string and write to its underlying array
        std::string s(len, '\0');
        va_start(args, fmt);
        std::vsnprintf(s.data(), len + 1, fmt, args);
        va_end(args);

        return s;
#else
        // C++11 or C++14: We need to allocate scratch memory
        auto vbuf = std::unique_ptr<char[]>(new char[len + 1]);
        va_start(args, fmt);
        std::vsnprintf(vbuf.get(), len + 1, fmt, args);
        va_end(args);

        return { vbuf.get(), len };
#endif
    };
};


class VadIterator
{
private:
    // OnnxRuntime resources
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> session = nullptr;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

private:
    void init_engine_threads(int inter_threads, int intra_threads)
    {
        // The method should be called in each thread/proc in multi-thread/proc work
        session_options.SetIntraOpNumThreads(intra_threads);
        session_options.SetInterOpNumThreads(inter_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    };

    void init_onnx_model(const std::string& model_path)
    {
        // Init threads = 1 for 
        init_engine_threads(1, 1);
        // Load model
        session = std::make_shared<Ort::Session>(env, model_path.c_str(), session_options);
    };

    void reset_states()
    {
        // Call reset before each audio start
        std::memset(_state.data(), 0.0f, _state.size() * sizeof(float));
        triggered = false;
        temp_end = 0;
        current_sample = 0;

        prev_end = next_start = 0;

        speeches.clear();
        current_speech = timestamp_t();
    };

    
public:
    bool predict(const std::vector<float> &data)
    {
    // Prepare input tensor from the data
    Ort::Value input_ort = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(data.data()), data.size(), input_node_dims, 2);

    // Prepare state tensor from the internal state
    Ort::Value state_ort = Ort::Value::CreateTensor<float>(
        memory_info, _state.data(), _state.size(), state_node_dims, 3);

    // Sample rate tensor
    Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(
        memory_info, sr.data(), sr.size(), sr_node_dims, 1);

    // Clear and setup inputs
    ort_inputs.clear();
    ort_inputs.emplace_back(std::move(input_ort));
    ort_inputs.emplace_back(std::move(state_ort));
    ort_inputs.emplace_back(std::move(sr_ort));

    // Execute model
    ort_outputs = session->Run(
        Ort::RunOptions{nullptr},
        input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
        output_node_names.data(), output_node_names.size());

    // Get speech probability from the output
    float speech_prob = ort_outputs[0].GetTensorMutableData<float>()[0];

    // Update the internal state with the new state from the model
    float *stateN = ort_outputs[1].GetTensorMutableData<float>();
    std::memcpy(_state.data(), stateN, size_state * sizeof(float));

    // Determine if the current window contains speech
    return speech_prob>threshold ;
    }

    void process(const std::vector<float>& input_wav)
    {
        reset_states();

        audio_length_samples = input_wav.size();

        for (int j = 0; j < audio_length_samples; j += window_size_samples)
        {
            if (j + window_size_samples > audio_length_samples)
                break;
            std::vector<float> r{ &input_wav[0] + j, &input_wav[0] + j + window_size_samples };
            predict(r);
        }

        if (current_speech.start >= 0) {
            current_speech.end = audio_length_samples;
            speeches.push_back(current_speech);
            current_speech = timestamp_t();
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            triggered = false;
        }
    };

    void process(const std::vector<float>& input_wav, std::vector<float>& output_wav)
    {
        process(input_wav);
        collect_chunks(input_wav, output_wav);
    }

    void collect_chunks(const std::vector<float>& input_wav, std::vector<float>& output_wav)
    {
        output_wav.clear();
        for (int i = 0; i < speeches.size(); i++) {
#ifdef __DEBUG_SPEECH_PROB___
            std::cout << speeches[i].c_str() << std::endl;
#endif //#ifdef __DEBUG_SPEECH_PROB___
            std::vector<float> slice(&input_wav[speeches[i].start], &input_wav[speeches[i].end]);
            output_wav.insert(output_wav.end(),slice.begin(),slice.end());
        }
    };

    const std::vector<timestamp_t> get_speech_timestamps() const
    {
        return speeches;
    }

    void drop_chunks(const std::vector<float>& input_wav, std::vector<float>& output_wav)
    {
        output_wav.clear();
        int current_start = 0;
        for (int i = 0; i < speeches.size(); i++) {

            std::vector<float> slice(&input_wav[current_start],&input_wav[speeches[i].start]);
            output_wav.insert(output_wav.end(), slice.begin(), slice.end());
            current_start = speeches[i].end;
        }

        std::vector<float> slice(&input_wav[current_start], &input_wav[input_wav.size()]);
        output_wav.insert(output_wav.end(), slice.begin(), slice.end());
    };

private:
    // model config
    int64_t window_size_samples;  // Assign when init, support 256 512 768 for 8k; 512 1024 1536 for 16k.
    int sample_rate;  //Assign when init support 16000 or 8000      
    int sr_per_ms;   // Assign when init, support 8 or 16
    float threshold; 
    int min_silence_samples; // sr_per_ms * #ms
    int min_silence_samples_at_max_speech; // sr_per_ms * #98
    int min_speech_samples; // sr_per_ms * #ms
    float max_speech_samples;
    int speech_pad_samples; // usually a 
    int audio_length_samples;

    // model states
    bool triggered = false;
    unsigned int temp_end = 0;
    unsigned int current_sample = 0;    
    // MAX 4294967295 samples / 8sample per ms / 1000 / 60 = 8947 minutes  
    int prev_end;
    int next_start = 0;

    //Output timestamp
    std::vector<timestamp_t> speeches;
    timestamp_t current_speech;


    // Onnx model
    // Inputs
    std::vector<Ort::Value> ort_inputs;
    
    std::vector<const char *> input_node_names = {"input", "state", "sr"};
    std::vector<float> input;
    unsigned int size_state = 2 * 1 * 128; // It's FIXED.
    std::vector<float> _state;
    std::vector<int64_t> sr;

    int64_t input_node_dims[2] = {};
    const int64_t state_node_dims[3] = {2, 1, 128}; 
    const int64_t sr_node_dims[1] = {1};

    // Outputs
    std::vector<Ort::Value> ort_outputs;
    std::vector<const char *> output_node_names = {"output", "stateN"};

public:
    // Construction
    VadIterator(const std::string ModelPath,
        int Sample_rate = 16000, int windows_frame_size = 32,
        float Threshold = 0.5, int min_silence_duration_ms = 0,
        int speech_pad_ms = 32, int min_speech_duration_ms = 32,
        float max_speech_duration_s = std::numeric_limits<float>::infinity())
    {
        init_onnx_model(ModelPath);
        threshold = Threshold;
        sample_rate = Sample_rate;
        sr_per_ms = sample_rate / 1000;

        window_size_samples = windows_frame_size * sr_per_ms;

        min_speech_samples = sr_per_ms * min_speech_duration_ms;
        speech_pad_samples = sr_per_ms * speech_pad_ms;

        max_speech_samples = (
            sample_rate * max_speech_duration_s
            - window_size_samples
            - 2 * speech_pad_samples
            );

        min_silence_samples = sr_per_ms * min_silence_duration_ms;
        min_silence_samples_at_max_speech = sr_per_ms * 98;

        input.resize(window_size_samples);
        input_node_dims[0] = 1;
        input_node_dims[1] = window_size_samples;

        _state.resize(size_state);
        sr.resize(1);
        sr[0] = sample_rate;
    };
};

int main(int argn, char** argv)
{
    //初始化模型
    std::string path = "silero_vad.onnx";
    VadIterator vad(path);
    int opt = 0;
    std::string pcm_file;
    int rate = 16000;
    int deepth = 1;//声道数
    while ((opt = getopt(argn, argv, "hi:r:d:")) != -1)
    {
      switch (opt) {
        case 'i':
            pcm_file = optarg;
            break;
        case 'r':
            rate = atoi(optarg);
            break;
        case 'd':
            deepth = atoi(optarg);
            break;
        default:
          std::cerr << "Usage: " << argv[0] << "\r\n"
               << " [-i pcm file] \r\n"
               << " [-r sample rate] \r\n"
               << " [-d deepth] \r\n"
               << std::endl;
          return -1;
      }
    }
    if (pcm_file.empty()) {
        std::cout << "pcm file is empty. \r\n";
        return -1;
    }
    FILE* file_p = fopen(pcm_file.c_str(), "r");
    if (!file_p) {
        std::cout << "fail to open:" << pcm_file << ".\r\n";
        return -2;
    }
    if (deepth ==2)
    {
        int frameLength = (32 * rate) / 1000; // 32 ms 的帧长
        std::vector<int16_t> buffer(frameLength*2);
        //std::vector<float> mono_samples(frameLength);
        std::vector<float> left_samples(frameLength);
        std::vector<float> right_samples(frameLength);
        int indexs = 0;
        while (true) 
        {
            size_t n = fread(buffer.data(), sizeof(int16_t), 2*frameLength, file_p);
            if (n < frameLength) 
            {
                if (feof(file_p)) 
                {
                    std::cout << "End of file reached." << std::endl;
                } else 
                {
                    std::cout << "Read error occurred." << std::endl;
                }
            break;
            }
            left_samples.clear();
            for (int i = 0; i < 2 * frameLength; i += 2) 
            {
            left_samples.push_back(static_cast<float>(buffer[i]) / 32768.0f);  // 将16位整数PCM转换为浮点数并规范化
            }
            indexs ++;
            bool voice_present = vad.predict(left_samples);
            std::cout << "Index :" <<  indexs << " return: " << voice_present << "\r\n";
            left_samples.clear(); // Clear previous samples
    }
        fclose(file_p);
        return 0;
    }
    if(deepth == 1)
    {
        int frameLength = (32 * rate) / 1000; // 32 ms 的帧长
        std::vector<int16_t> buffer(frameLength);
        std::vector<float> data(frameLength);
        int indexs = 0;
        while (true) 
        {
            size_t n = fread(buffer.data(), sizeof(int16_t), frameLength, file_p);
            if (n < frameLength) 
            {
                if (feof(file_p)) 
                {
                    std::cout << "End of file reached." << std::endl;
                } else 
                {
                    std::cout << "Read error occurred." << std::endl;
                }
            break;
            }

            for (int i = 0; i < frameLength; i += 1) 
            {
            data.push_back(static_cast<float>(buffer[i]) / 32768.0f);  // 将16位整数PCM转换为浮点数并规范化
            }
            indexs ++;
            bool voice_present = vad.predict(data);
            std::cout << "Index :" <<  indexs << " return: " << voice_present << "\r\n";
            data.clear(); // Clear previous samples
    }
        fclose(file_p);
        return 0;
    }
    
}
