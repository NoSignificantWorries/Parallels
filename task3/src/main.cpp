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
    std::thread thread;
    std::queue<std::pair<size_t, std::future<T>>> tasks;
    std::unordered_map<size_t, T> results;
    std::mutex queue_mutex;
    std::mutex results_mutex;
    std::atomic<bool> running;
    std::condition_variable task_waiter;
    size_t next_id;

    void run()
    {
        std::cout << "Started.\n";
        while (this->running.load())
        {
            std::unique_lock<std::mutex> lock(this->queue_mutex);

            this->task_waiter.wait(lock, [this] { return !this->tasks.empty() || !this->running.load(); });
            
            if (!this->running.load()) break;

            auto task = std::move(this->tasks.front());
            this->tasks.pop();
            lock.unlock();
            
            size_t task_id = task.first;
            std::future<T> future = std::move(task.second);
            
            {
                T result = future.get();

                std::lock_guard<std::mutex> lock_results(this->results_mutex);
                this->results[task_id] = result;
            }
        }
        std::cout << "Stoped.\n";
    }

public:
    Thread()
    {
        this->running.store(false);
        this->next_id = 0;
    }
    ~Thread()
    {
        this->stop();
    }
    
    void start()
    {
        if (!this->running.load())
        {
            this->running.store(true);
            this->thread = std::thread(&Thread::run, this);
        }
    }

    void stop()
    {
        if (this->running.load())
        {
            this->running.store(false);
            this->task_waiter.notify_all();
            if (this->thread.joinable())
            {
                this->thread.join();
            }
        }
    }

    template<typename Func, typename... Args>
    size_t add_task(Func&& func, Args&&... args)
    {
        std::lock_guard<std::mutex> lock(this->queue_mutex);
        size_t task_id = ++this->next_id;

        std::future<T> future = std::async(std::launch::async, std::forward<Func>(func), std::forward<Args>(args)...);
        this->tasks.push({ task_id, std::move(future) });
        
        this->task_waiter.notify_one();

        return task_id;
    }

    T request_result(size_t id_res)
    {
        std::lock_guard<std::mutex> lock(this->results_mutex);
        if (this->results.count(id_res) > 0)
        {
            T res = std::move(results[id_res]);
            this->results.erase(id_res);
            return res;
        }
        else
        {
            throw std::runtime_error("Task not completed.");
        }
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
    auto file_deleter = [](std::fstream* fp) {
        if (fp && fp->is_open()) fp->close();
        delete fp;
    };

    std::shared_ptr<std::fstream> file_ptr(new std::fstream("results/sin.txt", std::ios::out), file_deleter);
    
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
        // std::cout << "added sin: " << i << "\n";
    }

    (*file_ptr) << "Task number | id | input | output\n";
    T res = 0.0;
    size_t i = 0;
    while (!ids.empty())
    {
        // std::cout << "weiting sin " << i << "...\n";
        try
        {
            res = thread.request_result(std::get<1>(ids[i]));
            (*file_ptr) << std::get<0>(ids[i]) << " " << std::get<1>(ids[i]) << " " << std::get<2>(ids[i]) << " " << res << "\n";
            ids.erase(ids.begin() + i);
            if (ids.size() <= 0) break;
            i = (i + 1) % ids.size();
        }
        catch (const std::exception& error)
        {
            i = (i + 1) % ids.size();
        }
    }
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
        // std::cout << "added sqrt: " << i << "\n";
    }

    (*file_ptr) << "Task number | id | input | output\n";
    T res = 0.0;
    size_t i = 0;
    while (!ids.empty())
    {
        // std::cout << "weiting sqrt " << i << "...\n";
        try
        {
            res = thread.request_result(std::get<1>(ids[i]));
            (*file_ptr) << std::get<0>(ids[i]) << " " << std::get<1>(ids[i]) << " " << std::get<2>(ids[i]) << " " << res << "\n";
            ids.erase(ids.begin() + i);
            if (ids.size() <= 0) break;
            i = (i + 1) % ids.size();
        }
        catch (const std::exception& error)
        {
            i = (i + 1) % ids.size();
        }
    }
}

template<typename T>
void client_pow(Thread<T> &thread)
{
    auto file_deleter = [](std::fstream* fp) {
        if (fp && fp->is_open()) fp->close();
        delete fp;
    };

    std::shared_ptr<std::fstream> file_ptr(new std::fstream("results/pow.txt", std::ios::out), file_deleter);
    
    // start random
    std::random_device rd;
    std::mt19937 gen(rd());

    double min_a = 1.0;
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
        // std::cout << "added pow: " << i << "\n";
    }

    (*file_ptr) << "Task number | id | input | output\n";
    T res = 0.0;
    size_t i = 0;
    while (!ids.empty())
    {
        // std::cout << "weiting pow " << i << "...\n";
        try
        {
            res = thread.request_result(std::get<1>(ids[i]));
            (*file_ptr) << std::get<0>(ids[i]) << " " << std::get<1>(ids[i]) << " (" << std::get<2>(ids[i]) << " " << std::get<3>(ids[i]) << ") " << res << "\n";
            ids.erase(ids.begin() + i);
            if (ids.size() <= 0) break;
            i = (i + 1) % ids.size();
        }
        catch (const std::exception& error)
        {
            i = (i + 1) % ids.size();
        }
    }
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
