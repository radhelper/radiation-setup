//
// Created by fernando on 14/06/2021.
//

#ifndef LOG_HELPER_BASE_H
#define LOG_HELPER_BASE_H

#include <string>
#include <iomanip>
#include <ctime>
#include <utility>

class log_helper_base {
public:
    virtual uint8_t start_iteration() = 0;

    virtual uint8_t end_iteration() = 0;

    virtual uint8_t log_error_count(size_t kernel_errors) = 0;

    virtual uint8_t log_info_count(size_t info_count) = 0;

    virtual uint8_t log_error_detail(const std::string &string) = 0;

    virtual uint8_t log_info_detail(const std::string &string) = 0;

    size_t set_max_errors_iter(size_t max_errors) {
        this->max_errors_per_iter = max_errors;
        return this->max_errors_per_iter;
    }

    size_t set_max_infos_iter(size_t max_infos) {
        this->max_infos_per_iter = max_infos;
        return this->max_infos_per_iter;
    }

    size_t set_iter_interval_print(size_t interval) {
        if (interval < 1) {
            this->iter_interval_print = 1;
        } else {
            this->iter_interval_print = interval;
        }
        return this->iter_interval_print;
    }

    void disable_double_error_kill() {
        this->double_error_kill = false;
    }

    std::string get_log_file_name() {
        return this->log_file_name;
    }

    size_t get_iteration_number() const {
        return this->iteration_number;
    }

    virtual ~log_helper_base() = default;
protected:
    log_helper_base(std::string  benchmark_name, std::string  test_info)
            : benchmark_name(std::move(benchmark_name)), header(std::move(test_info)) {
    }

    // does not change over the time
    static constexpr char config_file[] = "/etc/radiation-benchmarks.conf";
    static constexpr char var_dir_key[] = "vardir";

    std::string log_file_name;
    std::string header;
    std::string benchmark_name;

    // Max errors that can be found for a single iteration
    // If more than max errors is found, exit the program
    size_t max_errors_per_iter = 500;
    size_t max_infos_per_iter = 500;

    // Used to print the log only for some iterations, equal 1 means print every iteration
    size_t iter_interval_print = 1;

    // Saves the last amount of error found for a specific iteration
    size_t last_iter_errors = 0;
    // Saves the last iteration index that had an error
    size_t last_iter_with_errors = 0;

    size_t kernels_total_errors = 0;
    size_t iteration_number = 0;
    double kernel_time_acc = 0;
    double kernel_time = 0;
    size_t it_time_start = 0;

    bool double_error_kill = true;
};


#endif //LOG_HELPER_BASE_H
