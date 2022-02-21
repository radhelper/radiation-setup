#include "log_helper_base.h"
#include "log_helper_udp.h"
#include "log_helper.h"

#include <cstring>
#include <memory>

std::unique_ptr<log_helper::log_helper_base> log_helper_ptr;

size_t set_max_errors_iter(size_t max_errors) {
    //check if it is empty
    if (log_helper_ptr) {
        return log_helper_ptr->set_max_errors_iter(max_errors);
    }
    return 0;
}

size_t set_max_infos_iter(size_t max_infos) {
    //check if it is empty
    if (log_helper_ptr) {
        return log_helper_ptr->set_max_infos_iter(max_infos);
    }
    return 0;
}

size_t set_iter_interval_print(size_t interval) {
    //check if it is empty
    if (log_helper_ptr) {
        return log_helper_ptr->set_iter_interval_print(interval);
    }
    return 0;
}

void disable_double_error_kill() {
    //check if it is empty
    if (log_helper_ptr) {
        return log_helper_ptr->disable_double_error_kill();
    }
}

void get_log_file_name(char *log_file_name) {
    //check if it is empty
    if (log_helper_ptr) {
        auto log_file_name_str = log_helper_ptr->get_log_file_name();
        if (std::strlen(log_file_name) < log_file_name_str.size()) {
            throw std::out_of_range(
                    log_helper::EXCEPTION_LINE("String passed as parameter has smaller size than the logfilename ")
            );
        }
        std::copy(log_file_name_str.begin(), log_file_name_str.end(), log_file_name);
    }
}

uint8_t start_log_file(const char *benchmark_name, const char *test_info) {
    //TODO: do for local log file
    log_helper_ptr = std::make_unique<log_helper::log_helper_tcp>(benchmark_name, test_info);
    return bool(log_helper_ptr) == 0;
}

uint8_t end_log_file() {
    auto mem = log_helper_ptr.release();
    delete mem;
    return uint8_t(bool(mem));
}

uint8_t start_iteration() {
    if (log_helper_ptr) {
        return log_helper_ptr->start_iteration();
    }
    return 0;
}

uint8_t end_iteration() {
    if (log_helper_ptr) {
        return log_helper_ptr->end_iteration();
    }
    return 0;
}

uint8_t log_error_count(size_t kernel_errors) {
    if (log_helper_ptr) {
        return log_helper_ptr->log_error_count(kernel_errors);
    }
    return 0;
}

uint8_t log_info_count(size_t info_count) {
    if (log_helper_ptr) {
        return log_helper_ptr->log_info_count(info_count);
    }
    return 0;
}

uint8_t log_error_detail(char *string) {
    std::string error_detail(string);
    if (log_helper_ptr) {
        return log_helper_ptr->log_error_detail(error_detail);
    }
    return 0;
}

uint8_t log_info_detail(char *string) {
    std::string info_detail(string);
    if (log_helper_ptr) {
        return log_helper_ptr->log_info_detail(info_detail);
    }
    return 0;
}

size_t get_iteration_number() {
    if (log_helper_ptr) {
        return log_helper_ptr->get_iteration_number();
    }
    return 0;
}



