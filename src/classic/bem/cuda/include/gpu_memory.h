#pragma once
#include <common.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

NWOB_NAMESPACE_BEGIN
template <typename T>
struct PitchedPtr
{
    HOST_DEVICE PitchedPtr() : ptr{nullptr}, stride_in_bytes{sizeof(T)} {}
    HOST_DEVICE PitchedPtr(T *ptr, size_t stride_in_elements, size_t offset = 0, size_t extra_stride_bytes = 0)
        : ptr{ptr + offset}, stride_in_bytes{(uint32_t)(stride_in_elements * sizeof(T) + extra_stride_bytes)}
    {
    }

    template <typename U>
    HOST_DEVICE explicit PitchedPtr(PitchedPtr<U> other)
        : ptr{(T *)other.ptr}, stride_in_bytes{other.stride_in_bytes}
    {
    }

    HOST_DEVICE T *operator[](uint32_t y) const { return (T *)((const char *)ptr + y * stride_in_bytes); }

    HOST_DEVICE void operator+=(uint32_t y) { ptr = (T *)((const char *)ptr + y * stride_in_bytes); }

    HOST_DEVICE void operator-=(uint32_t y) { ptr = (T *)((const char *)ptr - y * stride_in_bytes); }

    HOST_DEVICE explicit operator bool() const { return ptr; }

    HOST_DEVICE uint32_t stride() const { return stride_in_bytes / sizeof(T); }

    T *ptr;
    uint32_t stride_in_bytes;
};

#define DEBUG_GUARD_SIZE 0

inline std::atomic<size_t> &total_n_bytes_allocated()
{
    static std::atomic<size_t> s_total_n_bytes_allocated{0};
    return s_total_n_bytes_allocated;
}

template <class T>
class GPUMemory
{
private:
    T *m_data = nullptr;
    size_t m_size = 0; // Number of elements

public:
    GPUMemory() {}
    GPUMemory(size_t size) { resize(size); }
    // Don't permit copy assignment to prevent performance accidents.
    // Copy is permitted through an explicit copy constructor.
    GPUMemory<T> &operator=(const GPUMemory<T> &other) = delete;
    explicit GPUMemory(const GPUMemory<T> &other) { copy_from_device(other); }

    void check_guards() const
    {
#if DEBUG_GUARD_SIZE > 0
        if (!m_data)
            return;
        uint8_t buf[DEBUG_GUARD_SIZE];
        const uint8_t *rawptr = (const uint8_t *)m_data;
        cudaMemcpy(buf, rawptr - DEBUG_GUARD_SIZE, DEBUG_GUARD_SIZE, cudaMemcpyDeviceToHost);
        for (int i = 0; i < DEBUG_GUARD_SIZE; ++i)
            if (buf[i] != 0xff)
            {
                printf("TRASH BEFORE BLOCK offset %d data %p, read 0x%02x expected 0xff!\n", i, m_data, buf[i]);
                break;
            }
        cudaMemcpy(buf, rawptr + m_size * sizeof(T), DEBUG_GUARD_SIZE, cudaMemcpyDeviceToHost);
        for (int i = 0; i < DEBUG_GUARD_SIZE; ++i)
            if (buf[i] != 0xfe)
            {
                printf("TRASH AFTER BLOCK offset %d data %p, read 0x%02x expected 0xfe!\n", i, m_data, buf[i]);
                break;
            }
#endif
    }

    void allocate_memory(size_t n_bytes)
    {
        if (n_bytes == 0)
        {
            return;
        }
        uint8_t *rawptr = nullptr;
        CUDA_CHECK_THROW(cudaMalloc(&rawptr, n_bytes + DEBUG_GUARD_SIZE * 2));

#if DEBUG_GUARD_SIZE > 0
        CUDA_CHECK_THROW(cudaMemset(rawptr, 0xff, DEBUG_GUARD_SIZE));
        CUDA_CHECK_THROW(cudaMemset(rawptr + n_bytes + DEBUG_GUARD_SIZE, 0xfe, DEBUG_GUARD_SIZE));
#endif
        if (rawptr)
            rawptr += DEBUG_GUARD_SIZE;
        m_data = (T *)(rawptr);
        total_n_bytes_allocated() += n_bytes;
    }

    void free_memory()
    {
        if (!m_data)
        {
            return;
        }

        uint8_t *rawptr = (uint8_t *)m_data;
        if (rawptr)
            rawptr -= DEBUG_GUARD_SIZE;
        CUDA_CHECK_THROW(cudaFree(rawptr));

        total_n_bytes_allocated() -= get_bytes();

        m_data = nullptr;
        m_size = 0;
    }

    /// Frees memory again
    HOST_DEVICE ~GPUMemory()
    {
#ifndef __CUDA_ARCH__
        try
        {
            if (m_data)
            {
                free_memory();
            }
        }
        catch (std::runtime_error error)
        {
            // Don't need to report on memory-free problems when the driver is shutting down.
            if (std::string{error.what()}.find("driver shutting down") == std::string::npos)
            {
                std::cerr << "Could not free memory: " << error.what() << std::endl;
            }
        }
#endif
    }

    /** @name Resizing/enlargement
     *  @{
     */
    /// Resizes the array to the exact new size, even if it is already larger
    void resize(const size_t size)
    {
        if (m_size != size)
        {
            if (m_size)
            {
                try
                {
                    free_memory();
                }
                catch (std::runtime_error error)
                {
                    throw std::runtime_error{std::string{"Could not free memory: "} + error.what()};
                }
            }

            if (size > 0)
            {
                try
                {
                    allocate_memory(size * sizeof(T));
                }
                catch (std::runtime_error error)
                {
                    throw std::runtime_error{std::string{"Could not allocate memory: "} + error.what()};
                }
            }

            m_size = size;
        }
    }

    /// Enlarges the array if its size is smaller
    void enlarge(const size_t size)
    {
        if (size > m_size)
        {
            resize(size);
        }
    }
    /** @} */

    /** @name Memset
     *  @{
     */
    /// Sets the memory of the first num_elements to value
    void memset(const int value, const size_t num_elements, const size_t offset = 0)
    {
        if (num_elements + offset > m_size)
        {
            throw std::runtime_error{std::string{"Trying to memset "} + std::to_string(num_elements) +
                                     " elements, but memory size is " + std::to_string(m_size)};
        }

        CUDA_CHECK_THROW(cudaMemset(m_data + offset, value, num_elements * sizeof(T)));
    }

    /// Sets the memory of the all elements to value
    void memset(const int value) { memset(value, m_size); }
    /** @} */

    /** @name Copy operations
     *  @{
     */
    /// Copy data of num_elements from the raw pointer on the host
    void copy_from_host(const T *host_data, const size_t num_elements)
    {
        CUDA_CHECK_THROW(cudaMemcpy(data(), host_data, num_elements * sizeof(T), cudaMemcpyHostToDevice));
    }

    /// Copy num_elements from the host vector
    void copy_from_host(const std::vector<T> &data, const size_t num_elements)
    {
        if (data.size() < num_elements)
        {
            throw std::runtime_error{std::string{"Trying to copy "} + std::to_string(num_elements) +
                                     " elements, but vector size is " + std::to_string(data.size())};
        }
        copy_from_host(data.data(), num_elements);
    }

    /// Copies data from the raw host pointer to fill the entire array
    void copy_from_host(const T *data) { copy_from_host(data, m_size); }

    /// Copies num_elements of data from the raw host pointer after enlarging the array so that everything fits in
    void enlarge_and_copy_from_host(const T *data, const size_t num_elements)
    {
        enlarge(num_elements);
        copy_from_host(data, num_elements);
    }

    /// Copies num_elements from the host vector after enlarging the array so that everything fits in
    void enlarge_and_copy_from_host(const std::vector<T> &data, const size_t num_elements)
    {
        enlarge_and_copy_from_host(data.data(), num_elements);
    }

    /// Copies the entire host vector after enlarging the array so that everything fits in
    void enlarge_and_copy_from_host(const std::vector<T> &data)
    {
        enlarge_and_copy_from_host(data.data(), data.size());
    }

    /// Copies num_elements of data from the raw host pointer after resizing the array
    void resize_and_copy_from_host(const T *data, const size_t num_elements)
    {
        resize(num_elements);
        copy_from_host(data, num_elements);
    }

    /// Copies num_elements from the host vector after resizing the array
    void resize_and_copy_from_host(const std::vector<T> &data, const size_t num_elements)
    {
        resize_and_copy_from_host(data.data(), num_elements);
    }

    /// Copies the entire host vector after resizing the array
    void resize_and_copy_from_host(const std::vector<T> &data)
    {
        resize_and_copy_from_host(data.data(), data.size());
    }

    /// Copies the entire host vector to the device. Fails if there is not enough space available.
    void copy_from_host(const std::vector<T> &data)
    {
        if (data.size() < m_size)
        {
            throw std::runtime_error{std::string{"Trying to copy "} + std::to_string(m_size) +
                                     " elements, but vector size is " + std::to_string(data.size())};
        }
        copy_from_host(data.data(), m_size);
    }

    /// Copies num_elements of data from the raw host pointer to the device. Fails if there is not enough space
    /// available.
    void copy_to_host(T *host_data, const size_t num_elements) const
    {
        if (num_elements > m_size)
        {
            throw std::runtime_error{std::string{"Trying to copy "} + std::to_string(num_elements) +
                                     " elements, but memory size is " + std::to_string(m_size)};
        }

        CUDA_CHECK_THROW(cudaMemcpy(host_data, data(), num_elements * sizeof(T), cudaMemcpyDeviceToHost));
    }

    /// Copies num_elements from the device to a vector on the host
    void copy_to_host(std::vector<T> &data, const size_t num_elements) const
    {
        if (data.size() < num_elements)
        {
            throw std::runtime_error{std::string{"Trying to copy "} + std::to_string(num_elements) +
                                     " elements, but vector size is " + std::to_string(data.size())};
        }

        copy_to_host(data.data(), num_elements);
    }

    /// Copies num_elements from the device to a raw pointer on the host
    void copy_to_host(T *data) const { copy_to_host(data, m_size); }

    /// Copies all elements from the device to a vector on the host
    void copy_to_host(std::vector<T> &data) const
    {
        if (data.size() < m_size)
        {
            throw std::runtime_error{std::string{"Trying to copy "} + std::to_string(m_size) +
                                     " elements, but vector size is " + std::to_string(data.size())};
        }

        copy_to_host(data.data(), m_size);
    }

    /// Copies size elements from another device array to this one, automatically resizing it
    void copy_from_device(const GPUMemory<T> &other, const size_t size)
    {
        if (size == 0)
        {
            return;
        }

        if (m_size < size)
        {
            resize(size);
        }

        CUDA_CHECK_THROW(cudaMemcpy(m_data, other.m_data, size * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    /// Copies data from another device array to this one, automatically resizing it
    void copy_from_device(const GPUMemory<T> &other) { copy_from_device(other, other.m_size); }

    // Created an (owned) copy of the data
    GPUMemory<T> copy(size_t size) const
    {
        GPUMemory<T> result{size};
        result.copy_from_device(*this);
        return result;
    }

    GPUMemory<T> copy() const { return copy(m_size); }

    T *data() const
    {
        check_guards();
        return m_data;
    }

    T *begin() const { return data(); }
    T *end() const { return data() + m_size; }

    size_t get_num_elements() const { return m_size; }

    size_t size() const { return get_num_elements(); }

    size_t get_bytes() const { return m_size * sizeof(T); }

    size_t bytes() const { return get_bytes(); }

    T *device_ptr() const { return data(); }
};

template <class T>
class GPUMatrix
{
private:
    size_t m_width = 0;
    size_t m_height = 0;

public:
    GPUMemory<T> memory;
    GPUMatrix() {}
    GPUMatrix(size_t width, size_t height) { resize(width, height); }
    // Don't permit copy assignment to prevent performance accidents.
    // Copy is permitted through an explicit copy constructor.
    GPUMatrix<T> &operator=(const GPUMatrix<T> &other) = delete;
    explicit GPUMatrix(const GPUMatrix<T> &other)
    {
        memory.copy_from_device(other.memory);
        m_width = other.m_width;
        m_height = other.m_height;
    }
    size_t width() const { return m_width; }
    size_t height() const { return m_height; }
    void resize(size_t width, size_t height)
    {
        memory.resize(width * height);
        m_width = width;
        m_height = height;
    }
    void free_memory()
    {
        memory.free_memory();
        m_width = 0;
        m_height = 0;
    }
    T *data() const { return memory.data(); }
    size_t size() const { return memory.size(); }
    PitchedPtr<T> device_ptr() const { return {memory.data(), m_width}; }
};

NWOB_NAMESPACE_END