#pragma once

#include <string>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace cuda {

template <typename Exception>
class Error
{
  public:
    Error() = default;

    template <typename String>
    Error(String message) : message_{message} { ; }

    template <typename String>
    Error<Exception>& operator()(String message)
    {
        message_ = message;
        return *this;
    }

    Error<Exception>& operator=(cudaError_t err)
    {
        if (err != cudaSuccess)
        {
            std::string msg{};
            if (!message_.empty())
            {
                msg = message_ + ". ";
            }
            throw Exception{msg + cudaGetErrorString(err)};
        }
        return *this;
    }

  private:
    std::string message_;
};

using Err = Error<std::runtime_error>;

template <typename Exception>
void operator||(cudaError err, Error<Exception> e)
{
    e = err;
}

}