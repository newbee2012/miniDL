#ifndef MATH_UTILS_HPP_
#define MATH_UTILS_HPP_

#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include <boost/shared_ptr.hpp>
#include <cblas.h>

namespace dong
{
typedef boost::mt19937 rng_t;

class RNG
{
public:
    RNG();
    explicit RNG(unsigned int seed);
    explicit RNG(const RNG&);
    RNG& operator=(const RNG&);
    void* generator();
private:
    class Generator;
    boost::shared_ptr<Generator> generator_;
};


class RandomGenerator
{
public:
    static boost::shared_ptr<RNG> random_generator_;
    static boost::shared_ptr<rng_t> engine;
    static void init_engine(int seed)
    {
        engine.reset(new rng_t(seed));
    }

    static RNG& rng_stream()
    {
        if (!random_generator_)
        {
            random_generator_.reset(new RNG());
        }

        return *(random_generator_);
    }

    static float nextafter(const float b)
    {
        return boost::math::nextafter<float>(b, std::numeric_limits<float>::max());
    }


    static void rng_uniform(const int n, const float a, const float b, float* r)
    {
        boost::uniform_real<float> random_distribution(a, nextafter(b));
        boost::variate_generator<dong::rng_t*, boost::uniform_real<float> > my_generator(engine.get(), random_distribution);
        for (int i = 0; i < n; ++i)
        {
            r[i] = my_generator();
        }
    }

    static void rng_gaussian(const int n, const float a,const float sigma, float* r)
    {
        boost::normal_distribution<float> random_distribution(a, sigma);
        boost::variate_generator<dong::rng_t*, boost::normal_distribution<float> > my_generator(engine.get(), random_distribution);
        for (int i = 0; i < n; ++i)
        {
            r[i] = my_generator();
        }
    }
};

class Blas
{
public:
    static void caffe_cpu_gemm (const CBLAS_TRANSPOSE TransA,
                                const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                                const float alpha, const float* A, const float* B, const float beta,
                                float* C)
    {
        int lda = (TransA == CblasNoTrans) ? K : M;
        int ldb = (TransB == CblasNoTrans) ? N : K;
        cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
                    ldb, beta, C, N);
    }
};

}


#endif  // DONG_GEN_BMP_HPP_
