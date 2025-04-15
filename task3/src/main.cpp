#include <iostream>
#include <vector>
#include <queue>
#include <future>
#include <thread>
#include <chrono>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <random>
#include <fstream>

#define N 10000

template <typename T>
class Thread {
private:
    std::queue<std::pair<size_t, std::future<T>>> tasks;
    std::mutex mux;
    std::unordered_map<size_t, T> results;
    std::atomic<bool> stop_thread;
    std::thread thread;
    std::condition_variable new_task;
    std::condition_variable ready_to_new_task;
    size_t next_id;

    void run()
    {
        std::cout << "Started.\n";
        while (!this->stop_thread.load())
        {
            {
                std::unique_lock<std::mutex> lock(this->mux);

                this->new_task.wait(lock, [this]
                                    { return !this->tasks.empty() || this->stop_thread.load(); });
                
                if (this->tasks.empty() && this->stop_thread.load())
                {
                    break;
                }
                
                if (!this->tasks.empty())
                {
                    auto task = std::move(this->tasks.front());
                    this->tasks.pop();
                    this->results[task.first] = task.second.get();
                }
            }
        }
        std::cout << "Stoped.\n";
    }

public:
    Thread()
    {
        this->stop_thread.store(false);
        this->next_id = 0;
    }
    ~Thread()
    {
        this->stop();
    }
    
    void start()
    {
        this->thread = std::thread(&Thread::run, this);
    }

    void stop()
    {
        this->stop_thread.store(true);
        new_task.notify_all();
        if (this->thread.joinable())
        {
            this->thread.join();
        }
    }
    
    template<typename Func, typename... Args>
    size_t add_task(Func func, Args&&... args)
    {
        this->ready_to_new_task.wait();

        size_t task_id = this->next_id++;
        std::future<T> result = std::async(std::launch::async, func, std::forward<Args>(args)...);
        {
            std::lock_guard<std::mutex> lock(this->mux);
            this->tasks.push({ task_id, std::move(result) });
        }
        this->new_task.notify_one();
        return task_id;
    }

    T request_result(size_t id_res)
    {
        std::unique_lock<std::mutex> lock(this->mux);
        this->new_task.wait(lock, [this, id_res] { return this->results.find(id_res) != this->results.end() || this->stop_thread.load(); });
        
        if (this->results.find(id_res) == this->results.end()) {
            throw std::runtime_error("Result not available");
        }

        T res = this->results[id_res];
        this->results.erase(id_res);
        return res;
    }
};

template<typename T>
T fun_sin(T arg)
{
    return std::sin(arg);
}
template<typename T>
T fun_sqrt(T arg)
{
    return std::sqrt(arg);
}
template<typename T>
T fun_pow(T x, T y)
{
    return std::pow(x, y);
}

template<typename T>
void client_sin(Thread<T> &thread)
{
    std::ofstream file("results/sin.txt");
    
    // start random
    std::random_device rd;
    std::mt19937 gen(rd());

    double min = -10.0;
    double max = 10.0;

    std::uniform_real_distribution<T> dis(min, max);

    // start client
    auto sin_func = static_cast<T(*)(T)>(fun_sin<T>);
    T arg = 0.0;
    std::vector<std::tuple<size_t, size_t, T>> ids;
    size_t task_id = 0;
    for (size_t i = 0; i < N; i++) {
        arg = dis(gen);
        task_id = thread.add_task(sin_func, arg);
        ids.push_back({i, task_id, arg});
        std::cout << "added sin: " << i << "\n";
    }

    file << "Task number | id | input | output\n";
    T res = 0.0;
    for (size_t i = 0; i < N; i++) {
        std::cout << "weiting sin " << i << "...\n";
        res = thread.request_result(std::get<1>(ids[i]));
        file << std::get<0>(ids[i]) << " " << std::get<1>(ids[i]) << " " << std::get<2>(ids[i]) << " " << res << "\n";
    }
    
    file.close();
}

template<typename T>
void client_sqrt(Thread<T> &thread)
{
    auto file_deleter = [](std::fstream* fp) {
        if (fp && fp->is_open()) fp->close();
        delete fp;
    };

    std::shared_ptr<std::fstream> file_ptr(new std::fstream("results/sqrt.txt", std::ios::out), file_deleter);
    
    // start random
    std::random_device rd;
    std::mt19937 gen(rd());

    double min = 0.0;
    double max = 100.0;

    std::uniform_real_distribution<T> dis(min, max);

    // start client
    auto sin_func = static_cast<T(*)(T)>(fun_sqrt<T>);
    T arg = 0.0;
    std::vector<std::tuple<size_t, size_t, T>> ids;
    size_t task_id = 0;
    for (size_t i = 0; i < N; i++) {
        arg = dis(gen);
        task_id = thread.add_task(sin_func, arg);
        ids.push_back({i, task_id, arg});
        std::cout << "added sqrt: " << i << "\n";
    }

    (*file_ptr) << "Task number | id | input | output\n";
    T res = 0.0;
    for (size_t i = 0; i < N; i++) {
        std::cout << "weiting sqrt " << i << "...\n";
        res = thread.request_result(std::get<1>(ids[i]));
        (*file_ptr) << std::get<0>(ids[i]) << " " << std::get<1>(ids[i]) << " " << std::get<2>(ids[i]) << " " << res << "\n";
    }
}

template<typename T>
void client_pow(Thread<T> &thread)
{
    std::ofstream file("results/pow.txt");
    
    // start random
    std::random_device rd;
    std::mt19937 gen(rd());

    double min_a = 0.0;
    double max_a = 10.0;

    double min_b = -2.0;
    double max_b = 2.0;

    std::uniform_real_distribution<T> dis_a(min_a, max_a);
    std::uniform_real_distribution<T> dis_b(min_b, max_b);

    // start client
    auto sin_func = static_cast<T(*)(T, T)>(fun_pow<T>);
    T arg1 = 0.0;
    T arg2 = 0.0;
    std::vector<std::tuple<size_t, size_t, T, T>> ids;
    size_t task_id = 0;
    for (size_t i = 0; i < N; i++) {
        arg1 = dis_a(gen);
        arg2 = dis_b(gen);
        task_id = thread.add_task(sin_func, arg1, arg2);
        ids.push_back({i, task_id, arg1, arg2});
        std::cout << "added pow: " << i << "\n";
    }

    file << "Task number | id | input | output\n";
    T res = 0.0;
    for (size_t i = 0; i < N; i++) {
        std::cout << "weiting pow " << i << "...\n";
        res = thread.request_result(std::get<1>(ids[i]));
        file << std::get<0>(ids[i]) << " " << std::get<1>(ids[i]) << " (" << std::get<2>(ids[i]) << " " << std::get<3>(ids[i]) << ") " << res << "\n";
    }
    
    file.close();
}

int main()
{
    Thread<double> server;
    
    server.start();
    
    std::thread sin_client = std::thread(client_sin<double>, std::ref(server));
    std::thread sqrt_client = std::thread(client_sqrt<double>, std::ref(server));
    std::thread pow_client = std::thread(client_pow<double>, std::ref(server));
    
    sin_client.join();
    sqrt_client.join();
    pow_client.join();

    server.stop();

    return 0;
}
