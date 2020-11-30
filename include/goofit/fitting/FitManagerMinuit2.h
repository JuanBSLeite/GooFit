#pragma once

#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnStrategy.h>
#include <memory>

#include <goofit/GlobalCudaDefines.h>
#include <goofit/fitting/FCN.h>
#include <TRandom3.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <TMath.h>


using namespace Eigen;

namespace GooFit {

class PdfBase;

enum class FitErrors : int { Valid = 0, NotRun = 50, InValid };

class FitManagerMinuit2 {
  public:
    FitManagerMinuit2(PdfBase *dat);

    /// This runs the fit
    ROOT::Minuit2::FunctionMinimum fit();

    /// Set the maximum number of calls. 0 for Minuit2 default.
    void setMaxCalls(unsigned int max_calls = 0) { maxfcn_ = max_calls; }

    /// Get a pointer to the params
    Params *getParams() { return &upar_; }

    /// Get a pointer to the fcn
    FCN *getFCN() { return &fcn_; }

    /// Check to see if fit is valid
    operator bool() const { return retval_ == FitErrors::Valid; }

    /// Return value for program
    operator int() const { return static_cast<int>(retval_); }

    /// Set the fitting verbosity
    void setVerbosity(int value) { verbosity = value; }

    /// Get the fitting verbosity
    int getVerbosity() const { return verbosity; }

    void setTolerance(double tolerance){ tolerance_=tolerance;}

    double getTolerance(){return tolerance_;}

    void printCovMat();
    double dpda(double, double);
    double dpdb(double, double);
    double dmda(double, double);
    double dmdb(double, double);

    std::vector <std::vector<double>> printParams();
    void printOriginalParams();
    void setRandMinuitValues (const int nSamples);
    void loadSample(const int iSample);

  private:
    Params upar_;
    FCN fcn_;
    unsigned int maxfcn_{0};
    FitErrors retval_{FitErrors::NotRun};
    int verbosity{3};
    double tolerance_{0.1};
    TRandom3 rnd;
    Minuit2::MnUserCovariance matCov;
    MatrixXd* sqrtCov;
    std::vector<VectorXd> samples;
    PdfBase *pdfPointer;
};
} // namespace GooFit
