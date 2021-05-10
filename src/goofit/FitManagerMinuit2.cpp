#include <goofit/fitting/FitManagerMinuit2.h>

#include <goofit/Color.h>

#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnMigrad.h>
#include <Minuit2/MnPrint.h>
#include <Minuit2/MnUserParameterState.h>
#include <Minuit2/MnUserParameters.h>
#include <Minuit2/MnUserCovariance.h>
#include <CLI/Timer.hpp>
#include <goofit/PdfBase.h>

#include <cuda_runtime.h>

#include <fstream>

namespace GooFit {

FitManagerMinuit2::FitManagerMinuit2(PdfBase *dat)
    : upar_(*dat)
    , fcn_(upar_) {pdfPointer = dat;}

Minuit2::FunctionMinimum FitManagerMinuit2::fit() {
    auto val = Minuit2::MnPrint::Level();
    Minuit2::MnPrint::SetLevel(verbosity);
    
    // Setting global call number to 0
    host_callnumber = 0;

    CLI::Timer timer{"The minimization took"};

    Minuit2::MnMigrad migrad{fcn_, upar_};
    migrad.SetPrecision(1.e-18); 
    // Do the minimization
    if(verbosity > 0)
        std::cout << GooFit::gray << GooFit::bold;

    CLI::Timer avetimer{"Average time per call"};
    Minuit2::FunctionMinimum min = migrad(maxfcn_,tolerance_);
    
    //Cov Matrix
    matCov = migrad.Covariance();

    // Print nice output
    if(verbosity > 0) {
        std::cout << GooFit::reset << (min.IsValid() ? GooFit::green : GooFit::red);
        std::cout << min << GooFit::reset;
        std::cout << GooFit::magenta << timer.to_string() << GooFit::reset << std::endl;
        std::cout << (avetimer / min.NFcn()).to_string() << std::endl;
    }

    if(min.IsValid()) {
        retval_ = FitErrors::Valid;
    } else {
        if(verbosity > 0) {
            std::cout << GooFit::red;
            std::cout << "HesseFailed: " << min.HesseFailed() << std::endl;
            std::cout << "HasCovariance: " << min.HasCovariance() << std::endl;
            std::cout << "HasValidCovariance: " << min.HasValidCovariance() << std::endl;
            std::cout << "HasValidParameters: " << min.HasValidParameters() << std::endl;
            std::cout << "IsAboveMaxEdm: " << min.IsAboveMaxEdm() << std::endl;
            std::cout << "HasReachedCallLimit: " << min.HasReachedCallLimit() << std::endl;
            std::cout << "HasAccurateCovar: " << min.HasAccurateCovar() << std::endl;
            std::cout << "HasPosDefCovar : " << min.HasPosDefCovar() << std::endl;
            std::cout << "HasMadePosDefCovar : " << min.HasMadePosDefCovar() << std::endl;
            std::cout << GooFit::reset;
        }

        retval_ = FitErrors::InValid;
    }

    // Set the parameters in GooFit to the new values
    upar_.SetGooFitParams(min.UserState());

    Minuit2::MnPrint::SetLevel(val);
    return min;
}

void FitManagerMinuit2::printCovMat()
{
	std::cout << std::endl << matCov << std::endl;
}

double FitManagerMinuit2::dmda(double a, double b)
{
	double ret = a/sqrt(a*a+b*b);
	return ret;
}

double FitManagerMinuit2::dmdb(double a, double b)
{
	double ret = b/sqrt(a*a+b*b);
	return ret;
}

double FitManagerMinuit2::dpda(double a, double b)
{
	double ret = (-b/(a*a + b*b));
	return ret;
}

double FitManagerMinuit2::dpdb(double a, double b)
{
	double ret = (a/(a*a+b*b));
	return ret;
}

void FitManagerMinuit2::printOriginalParams()
{
	std::vector<Variable> vec_vars = pdfPointer->getParameters();
	std::vector<double> floatVarVal, floatVarErr;
		for(Variable &var : vec_vars) {
			if (var.IsFixed()) continue;
			std::cout << var.getValue() << std::endl;
			std::cout << var.getError() << std::endl;
		}
}

std::vector <std::vector<double>> FitManagerMinuit2::printParams(std::string path)
{

	std::ofstream wt(path);

	std::vector<Variable> vec_vars = pdfPointer->getParameters();
	std::vector<double> floatVarVal;
	floatVarVal.clear();

	for(Variable &var : vec_vars) {
		if (var.IsFixed()) continue;
		int counter = var.getFitterIndex();
		floatVarVal.push_back(var.getValue());

	}

	std::vector<double> vec_mag, vec_mag_err;
	vec_mag.clear(); vec_mag_err.clear();
	std::vector<double> vec_phi, vec_phi_err;
	vec_phi.clear(); vec_phi_err.clear();
	std::cout << "free parameter resonance: " << floatVarVal.size()/2 << std::endl;

	std::cout << std::fixed << std::setprecision(8);
	std::cout << "                      Magnitude            Phase   " << std::endl;

	for(int i = 0; i < floatVarVal.size(); i+=2){
		double a = floatVarVal[i];
		double b = floatVarVal[i+1];
		double mag = sqrt(a*a + b*b);
		double phi = atan(b/a)*180./TMath::Pi();
		//std::cout << matCov(i,i) << '\t' << matCov(i+1,i+1) << '\t' << matCov(i,i+1) << '\n'; 
		double mag_err = dmda(a,b)*dmda(a,b)*matCov(i,i)
					 +dmdb(a,b)*dmdb(a,b)*matCov(i+1,i+1)
					 +2*dmda(a,b)*dmdb(a,b)*matCov(i,i+1);
		if(mag_err<0) mag_err=0;
		mag_err = sqrt(mag_err);

		double phi_err = dpda(a,b)*dpda(a,b)*matCov(i,i)
					 +dpdb(a,b)*dpdb(a,b)*matCov(i+1,i+1)
					 +2*dpda(a,b)*dpdb(a,b)*matCov(i,i+1);
		if(phi_err<0) phi_err=0;
		phi_err = sqrt(phi_err)*180./TMath::Pi();

		if(a<0&&b<0) phi-=180;
		if(a<0&&b>0) phi+=180;
		vec_mag.push_back(mag);
		vec_phi.push_back(phi);
		vec_mag_err.push_back(mag_err);
		vec_phi_err.push_back(phi_err);
		std::cout << "coefficient Res #" << (i+2)/2 << ":  " << mag << " +- " << mag_err << "       " << phi << " +- " << phi_err << std::endl;
		wt << "coefficient Res #" << (i+2)/2 << ":  " << mag << " +- " << mag_err << "       " << phi << " +- " << phi_err << std::endl;
	}

	std::vector <std::vector<double>> ret; ret.clear();
	ret.push_back(vec_phi);
	ret.push_back(vec_phi_err);
	return ret;
}

__global__ void AdjointEigenSolver(MatrixXd *d_A,MatrixXd *d_out){

	SelfAdjointEigenSolver<MatrixXd> se(*d_A);
	d_out = new MatrixXd(se.operatorSqrt());
}

void FitManagerMinuit2::setRandMinuitValues (const int nSamples){
	rnd.SetSeed(nSamples+388);
	std::vector<double> floatVarVal;
	floatVarVal.clear();
	std::vector<double> floatVarErr;
	floatVarErr.clear();
	//std::vector<Variable> vec_vars = upar_.get_vars();
	std::vector<Variable> vec_vars = pdfPointer->getParameters();
//	fvarIndex.clear();
	for(Variable &var : vec_vars) {
		if (var.IsFixed()) continue;
		int counter = var.getFitterIndex();
//		fvarIndex.push_back(var.getFitterIndex());
		floatVarVal.push_back(var.getValue());
		floatVarErr.push_back(var.getError());
//		std::cout << "check for Index " << counter << ": value = " << var.getValue() << "  ;  error = " << var.getError() << std::endl;
	}
	const int nFPars = floatVarVal.size();
	VectorXd vmean(nFPars);
	for (int i = 0; i < nFPars; i++)
		vmean(i) = floatVarVal[i];

	MatrixXd A(nFPars,nFPars);
        //MatrixXd out(nFPars,nFPars);
	const int n = matCov.Nrow();
	if(nFPars != n) std::cout <<"Error!!!!!!!!" << std::endl;
	for (int ii = 0; ii < n; ii++)
		for (int jj = 0; jj < n; jj++)
			A(ii,jj) = matCov(ii,jj);

	//MatrixXd *d_A;
	//MatrixXd *d_out;
	
	//cudaMalloc((void **)&d_A, sizeof(Eigen::MatrixXd)*nFPars);
	//cudaMalloc((void **)&d_out, sizeof(Eigen::MatrixXd)*nFPars);

	//cudaMemcpy(d_A, A.data(), sizeof(Eigen::MatrixXd)*nFPars, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_out, out.data(), sizeof(Eigen::MatrixXd)*nFPars, cudaMemcpyHostToDevice);
	SelfAdjointEigenSolver<MatrixXd> es(A);
	//AdjointEigenSolver(d_A,d_out);

	//cudaMemcpy(out, d_out, sizeof(Eigen::MatrixXd)*nFPars, cudaMemcpyDeviceToHost);
	
	sqrtCov = new MatrixXd(es.operatorSqrt());


	VectorXd vy(nFPars);
	samples.clear();
	for (int ii=0;ii<nSamples;ii++){
		for (int i=0;i<nFPars;i++) 
			vy(i) = rnd.Gaus(0,1);
		vy = vmean + (*sqrtCov) * vy;
//		std::cout<<"Mean and random: "<<std::endl<<vmean<<std::endl<<vy<<std::endl;
		samples.push_back(vy);
	}
}


void FitManagerMinuit2::loadSample (const int iSample){
//	std::vector<Variable> vec_vars = upar_.get_vars();
	std::vector<Variable> vec_vars = pdfPointer->getParameters();
	int counter = 0;
//	for(int i = 0; i < vec_vars.size(); ++i){
	for(Variable &var : vec_vars){
		if (var.IsFixed()) continue;
//		std::cout << "load value Index " << var.getFitterIndex() << " : " << samples[iSample][counter] << std::endl;
		pdfPointer->updateVariable(var,(fptype)samples[iSample][counter]);
		counter++;
	}
}

} // namespace GooFit
