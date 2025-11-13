#include <vector>
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cstring>
#include <omp.h>

namespace py = pybind11;


class TCMSCpp {
public:
    // w_bytes: 总字节数，必须是4的倍数
    TCMSCpp(int d, int w_bytes)
        : d_(d), w_(w_bytes / 4),
          counters8_(d, std::vector<uint8_t>(w_ * 4, 0)),
          counters16_(d, std::vector<uint16_t>(w_ * 2, 0)),
          counters32_(d, std::vector<uint32_t>(w_, 0)),
          locks8_(d, std::vector<omp_lock_t>(w_ * 4)),
          locks16_(d, std::vector<omp_lock_t>(w_ * 2)),
          locks32_(d, std::vector<omp_lock_t>(w_)) {
        for (int i = 0; i < d_; ++i) {
            for (int j = 0; j < w_ * 4; ++j) omp_init_lock(&locks8_[i][j]);
            for (int j = 0; j < w_ * 2; ++j) omp_init_lock(&locks16_[i][j]);
            for (int j = 0; j < w_; ++j) omp_init_lock(&locks32_[i][j]);
        }
    }

    void update_batch(py::array x_batch, py::array c_batch) {
        auto x = x_batch.unchecked<int64_t, 1>();
        auto c = c_batch.unchecked<int32_t, 1>();
        uint32_t batch_size = static_cast<uint32_t>(x.shape(0));
#pragma omp parallel for
        for (int i = 0; i < d_; ++i) {
            uint64_t seed = hash_seeds_[i % 16];
            for (uint32_t idx = 0; idx < batch_size; ++idx) {
                int64_t key = x(idx);
                uint32_t pos8 = std::hash<int64_t>{}(key ^ seed) % (w_ * 4);
                uint32_t freq = static_cast<uint32_t>(c(idx));
                // 8位
                omp_set_lock(&locks8_[i][pos8]);
                if (counters8_[i][pos8] + freq <= 0xFF) {
                    counters8_[i][pos8] += freq;
                    omp_unset_lock(&locks8_[i][pos8]);
                } else {
                    uint32_t remain = freq - (0xFF - counters8_[i][pos8]);
                    counters8_[i][pos8] = 0xFF;
                    omp_unset_lock(&locks8_[i][pos8]);
                    // 16位
                    uint32_t pos16 = pos8 / 2;
                    omp_set_lock(&locks16_[i][pos16]);
                    if (counters16_[i][pos16] + remain <= 0xFFFF) {
                        counters16_[i][pos16] += remain;
                        omp_unset_lock(&locks16_[i][pos16]);
                    } else {
                        uint32_t remain2 = remain - (0xFFFF - counters16_[i][pos16]);
                        counters16_[i][pos16] = 0xFFFF;
                        omp_unset_lock(&locks16_[i][pos16]);
                        // 32位
                        uint32_t pos32 = pos8 / 4;
                        omp_set_lock(&locks32_[i][pos32]);
                        if (counters32_[i][pos32] + remain2 <= 0xFFFFFFFF) {
                            counters32_[i][pos32] += remain2;
                        } else {
                            counters32_[i][pos32] = 0xFFFFFFFF;
                        }
                        omp_unset_lock(&locks32_[i][pos32]);
                    }
                }
            }
        }
    }

    py::array_t<float> query_all_hashes(py::array x_batch) {
        auto x = x_batch.unchecked<int64_t, 1>();
        uint32_t batch_size = static_cast<uint32_t>(x.shape(0));
        py::array_t<float> est({batch_size, static_cast<uint32_t>(d_)});
        auto est_mut = est.mutable_unchecked<2>();
#pragma omp parallel for
        for (int i = 0; i < d_; ++i) {
            uint64_t seed = hash_seeds_[i % 16];
            for (uint32_t j = 0; j < batch_size; ++j) {
                int64_t key = x(j);
                uint32_t pos8 = std::hash<int64_t>{}(key ^ seed) % (w_ * 4);
                float v = INFINITY;
                if (counters8_[i][pos8] < 0xFF) {
                    v = std::min(v, static_cast<float>(counters8_[i][pos8]));
                } else {
                    uint32_t pos16 = pos8 / 2;
                    if (counters16_[i][pos16] < 0xFFFF) {
                        v = std::min(v, static_cast<float>(counters16_[i][pos16]) + 0xFF);
                    } else {
                        uint32_t pos32 = pos8 / 4;
                        v = std::min(v, static_cast<float>(counters32_[i][pos32]) + 0xFFFF + 0xFF);
                    }
                }
                est_mut(j, i) = v;
            }
        }
        return est;
    }

    py::array_t<float> create_sketch() {
        int d = d_;
        int w8 = w_ * 4, w16 = w_ * 2, w32 = w_;
        // 组的顺序: w8, w8, w16, w8, w8, w16, w32
        std::vector<int> block_sizes = {w8, w8, w16, w8, w8, w16, w32};
        int block_sum = 0;
        for (int sz : block_sizes) block_sum += sz;
        // 计算需要多少组才能 >=64
        int group_num = (64 + 6) / 7; // 至少7个区块一组
        // 但实际上我们只拼一组即可（每个d一组），因为你只需要 >=64
        // 这里按原始需求实现一组
        std::vector<float> buf(d * block_sum);
#pragma omp parallel for
        for (int i = 0; i < d; ++i) {
            // create_sketch不涉及哈希扰动，保持原有顺序即可
            int offset = 0;
            std::copy(counters8_[i].begin(), counters8_[i].begin() + w8, buf.begin() + i * block_sum + offset);
            offset += w8;
            std::copy(counters8_[i].begin(), counters8_[i].begin() + w8, buf.begin() + i * block_sum + offset);
            offset += w8;
            std::copy(counters16_[i].begin(), counters16_[i].begin() + w16, buf.begin() + i * block_sum + offset);
            offset += w16;
            std::copy(counters8_[i].begin(), counters8_[i].begin() + w8, buf.begin() + i * block_sum + offset);
            offset += w8;
            std::copy(counters8_[i].begin(), counters8_[i].begin() + w8, buf.begin() + i * block_sum + offset);
            offset += w8;
            std::copy(counters16_[i].begin(), counters16_[i].begin() + w16, buf.begin() + i * block_sum + offset);
            offset += w16;
            std::copy(counters32_[i].begin(), counters32_[i].begin() + w32, buf.begin() + i * block_sum + offset);
        }
        py::array_t<float> arr(buf.size());
        float* ptr = arr.mutable_data();
        std::memcpy(ptr, buf.data(), sizeof(float) * buf.size());
        return arr;
    }
private:
    // 推荐16个独立扰动种子，d_<=16时足够，若更大可扩展
    static constexpr uint64_t hash_seeds_[16] = {
        0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL, 0xA4093822299F31D0ULL, 0x082EFA98EC4E6C89ULL,
        0x452821E638D01377ULL, 0xBE5466CF34E90C6CULL, 0xC0AC29B7C97C50DDULL, 0x3F84D5B5B5470917ULL,
        0x9216D5D98979FB1BULL, 0xD1310BA698DFB5ACULL, 0x2FFD72DBD01ADFB7ULL, 0xB8E1AFED6A267E96ULL,
        0xBA7C9045F12C7F99ULL, 0x24A19947B3916CF7ULL, 0x0801F2E2858EFC16ULL, 0x636920D871574E69ULL
    };
    int d_, w_;
    std::vector<std::vector<uint8_t>> counters8_;
    std::vector<std::vector<uint16_t>> counters16_;
    std::vector<std::vector<uint32_t>> counters32_;
    std::vector<std::vector<omp_lock_t>> locks8_;
    std::vector<std::vector<omp_lock_t>> locks16_;
    std::vector<std::vector<omp_lock_t>> locks32_;
};

// 兼容部分编译器和pybind11，添加类外初始化定义，确保符号导出
constexpr uint64_t TCMSCpp::hash_seeds_[16];


PYBIND11_MODULE(tcms_cpp, m) {
    py::class_<TCMSCpp>(m, "TCMSCpp")
        .def(py::init<int, int>())
        .def("update_batch", &TCMSCpp::update_batch)
        .def("query_all_hashes", &TCMSCpp::query_all_hashes)
        .def("create_sketch", &TCMSCpp::create_sketch);
}
