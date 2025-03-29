#pragma once
#include <queue>
#include <future>
#include <thread>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>


template <typename T>
class Server {
private:
    std::queue<std::pair<size_t, std::future<int>>> queue_;
public:
    Server();
    ~Server();
    
    void start();
    void stop();
    
    size_t add_task(std::string task);
    T request_result(size_t id_res);
};
