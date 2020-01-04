#pragma once

#include <goofit/PDFs/GooPdf.h>
#include <goofit/detail/Complex.h>


namespace GooFit {

#define MAXNKNOBS 1000


typedef fpcomplex (*resonance_function_ptr)(fptype, fptype, fptype, unsigned int *);

__device__ fptype Momentum( const fptype &m,
                            const  fptype &m1,
                            const fptype &m2 
                           );

//for a0_f0 mixing
__device__ fptype kallenFunction( const fptype &m,
    const fptype &m1,
    const fptype &m2
    );

//for a0_f0 mixing
__device__ fptype rhoBC( const fptype &m,
    const fptype &m1,
    const fptype &m2
    );

__device__ fptype BWFactors(const fptype &q,
                            const fptype &q0, 
                            unsigned int &spin , 
                            const fptype &meson_radius);

__device__ fptype Gamma(const fptype &m,
                        const fptype &m0,
                        const fptype &width, 
                        const fptype &q,    
                        const fptype &q0,
                        const fptype &BWFactor,
                        const unsigned int &spin);



//gounaris

__device__ fptype h(const fptype &m,const fptype &q);
__device__ fptype h_prime(const fptype &m0,const fptype &q0);
__device__ fptype d(const fptype &m0,const fptype &q0);
__device__ fptype f(const fptype &m, const fptype &m0,const fptype &width , const fptype &q, const fptype &q0);



__device__ fptype spinFactor(unsigned int spin,
                             fptype motherMass,
                             fptype daug1Mass,
                             fptype daug2Mass,
                             fptype daug3Mass,
                             fptype m12,
                             fptype m13,
                             fptype m23,
                             unsigned int cyclic_index);

class ResonancePdf : public GooPdf {
    friend class TddpPdf;
    friend class DalitzPlotPdf;
    friend class IncoherentSumPdf;

  public:
    ~ResonancePdf() override = default;

    __host__ virtual void recalculateCache() const {}

    __host__ Variable get_amp_real() const { return amp_real; }
    __host__ Variable get_amp_img() const { return amp_imag; }

  protected:
    /// Special constructor that subclasses use
    ResonancePdf(std::string name, Variable ar, Variable ai)
        : GooPdf(name)
        , amp_real(ar)
        , amp_imag(ai) {
        // Dummy index for constants - won't use it, but calling
        // functions can't know that and will call setConstantIndex anyway.
        pindices.push_back(0);
    }

    void setConstantIndex(unsigned int idx) { host_indices[parameters + 1] = idx; }

    Variable amp_real;
    Variable amp_imag;

    std::vector<unsigned int> pindices;

    std::vector<fptype> host_constants;
};

namespace Resonances {
/// Relativistic Breit-Wigner
class RBW : public ResonancePdf {
  public:
    RBW(std::string name,
        Variable ar,
        Variable ai,
        Variable mass,
        Variable width,
        unsigned int sp,
        unsigned int cyc,
        bool symmDP = false);
    ~RBW() override = default;
};

/// POLE
class POLE : public ResonancePdf {
  public:
    POLE(std::string name,
        Variable ar,
        Variable ai,
Variable real,
        Variable img,
        unsigned int sp,
        unsigned int cyc,
        bool symmDP = false);
    ~POLE() override = default;
};

/// Rho-Omega Mixing Amplitude
class RHOOMEGAMIX : public ResonancePdf {
  public:
    RHOOMEGAMIX(std::string name,
        Variable ar,
        Variable ai,
        Variable real,
        Variable img,
        Variable delta,
        unsigned int sp,
        unsigned int cyc,
        bool symmDP = false);
    ~RHOOMEGAMIX() override = default;
};

/// LASS
class LASS : public ResonancePdf {
  public:
    LASS(std::string name, Variable ar, Variable ai, Variable mass, Variable width, unsigned int sp, unsigned int cyc);
    ~LASS() override = default;
};

/// Gounaris-Sakurai
class GS : public ResonancePdf {
  public:
    GS(std::string name, Variable ar, Variable ai, Variable mass, Variable width, unsigned int sp, unsigned int cyc,bool symDP = false);
    ~GS() override = default;
};

/// FLATTE constructor
class FLATTE : public ResonancePdf {
  public:
    FLATTE(std::string name,
           Variable ar,
           Variable ai,
           Variable mean,
           Variable g1,
           Variable rg2og1,
           unsigned int cyc,
           bool symmDP);
    ~FLATTE() override = default;
};

/// a0_f0_mixing
class f0_MIXING : public ResonancePdf {
  public:
    f0_MIXING(std::string name,
           Variable ar,
           Variable ai,
           Variable g1, //ga_kk coupling in isospin basis
           Variable g2, //gf_kk coupling in isospin basis
           Variable g3, //ga_eta_pi coupling in isospin basis
           Variable g4, //gf_pi_pi coupling in isospin basis
           unsigned int cyc,
           bool symmDP);
    ~f0_MIXING() override = default;
};


/// Gaussian constructor
class Gauss : public ResonancePdf {
  public:
    Gauss(std::string name, Variable ar, Variable ai, Variable mean, Variable sigma, unsigned int cyc);
    ~Gauss() override = default;
};

/// Nonresonant constructor
class NonRes : public ResonancePdf {
  public:
    NonRes(std::string name, Variable ar, Variable ai);
    ~NonRes() override = default;
};

/// Cubic spline constructor
class Spline : public ResonancePdf {
  public:
    Spline(std::string name,
           Variable ar,
           Variable ai,
           std::vector<fptype> &HH_bin_limits,
           std::vector<Variable> &pwa_coefs_reals,
           std::vector<Variable> &pwa_coefs_imags,
           unsigned int cyc,
           bool symmDP = false);
    ~Spline() override = default;

 
};

class SplinePolar : public ResonancePdf {
  public:
    SplinePolar(std::string name,
           Variable ar,
           Variable ai,
           std::vector<fptype> &HH_bin_limits,
           std::vector<Variable> &pwa_coefs_reals,
           std::vector<Variable> &pwa_coefs_imags,
           unsigned int cyc,
           bool symmDP = false);
    ~SplinePolar() override = default;

    
};

class BoseEinstein : public ResonancePdf {
  public:
   BoseEinstein(std::string name, Variable ar, Variable ai, Variable coef );
    ~BoseEinstein() override = default;
};


} // namespace Resonances

} // namespace GooFit
