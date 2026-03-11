#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h> // 必须声明设备上下文
#include <cuda_runtime.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

using namespace torch::indexing;

struct PrefetchTask {
    torch::Tensor token_ids;
    int layer_idx;
    torch::Tensor cpu_idx, cpu_mean, cpu_std;
    torch::Tensor gpu_idx, gpu_mean, gpu_std;
    int buffer_idx; // 0 或 1，用于乒乓缓冲
};

class AsyncPrefetcher {
private:
    std::thread worker;
    std::queue<PrefetchTask> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop = false;

    cudaStream_t stream;
    cudaEvent_t events[2];
    bool initialized = false;

public:
    AsyncPrefetcher() { worker = std::thread(&AsyncPrefetcher::process, this); }

    ~AsyncPrefetcher() {
        { std::lock_guard<std::mutex> lock(mtx); stop = true; }
        cv.notify_all();
        if (worker.joinable()) worker.join();
        if (initialized) {
            cudaStreamDestroy(stream);
            cudaEventDestroy(events[0]);
            cudaEventDestroy(events[1]);
        }
    }

    void init_cuda(int device_index) {
        if (!initialized) {
            cudaSetDevice(device_index);
            cudaStreamCreate(&stream);
            cudaEventCreate(&events[0]);
            cudaEventCreate(&events[1]);
            initialized = true;
        }
    }

    void submit(PrefetchTask task) {
        std::lock_guard<std::mutex> lock(mtx);
        tasks.push(task);
        cv.notify_one();
    }

    void wait_buffer(int buffer_idx) {
        if (initialized) {
            cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream().stream();
            cudaStreamWaitEvent(current_stream, events[buffer_idx], 0);
        }
    }

    void process() {
        while (true) {
            PrefetchTask task;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this] { return !tasks.empty() || stop; });
                if (stop && tasks.empty()) return;
                task = tasks.front();
                tasks.pop();
            }

            c10::cuda::CUDAGuard device_guard(task.gpu_idx.device());
            init_cuda(task.gpu_idx.device().index());

            // 提取切片
            torch::Tensor slice_idx = task.cpu_idx.index({task.token_ids, task.layer_idx}).contiguous();
            torch::Tensor slice_mean = task.cpu_mean.index({task.token_ids, task.layer_idx}).contiguous();
            torch::Tensor slice_std = task.cpu_std.index({task.token_ids, task.layer_idx}).contiguous();

            // 真正的 C++ 底层 DMA 裸指针异步拷贝 (绕过一切限制，速度最快)
            cudaMemcpyAsync(task.gpu_idx.data_ptr(), slice_idx.data_ptr(), slice_idx.nbytes(), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(task.gpu_mean.data_ptr(), slice_mean.data_ptr(), slice_mean.nbytes(), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(task.gpu_std.data_ptr(), slice_std.data_ptr(), slice_std.nbytes(), cudaMemcpyHostToDevice, stream);

            cudaEventRecord(events[task.buffer_idx], stream);
        }
    }
};

static AsyncPrefetcher global_prefetcher;

void launch_prefetch(
    torch::Tensor token_ids, int layer_idx,
    torch::Tensor cpu_idx, torch::Tensor cpu_mean, torch::Tensor cpu_std,
    torch::Tensor gpu_idx, torch::Tensor gpu_mean, torch::Tensor gpu_std,
    int buffer_idx) 
{
    global_prefetcher.submit({token_ids, layer_idx, cpu_idx, cpu_mean, cpu_std, gpu_idx, gpu_mean, gpu_std, buffer_idx});
}

void wait_buffer(int buffer_idx) { global_prefetcher.wait_buffer(buffer_idx); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_prefetch", &launch_prefetch, "Launch C++ Async Prefetch task");
    m.def("wait_buffer", &wait_buffer, "Wait for buffer prefetch to complete");
}