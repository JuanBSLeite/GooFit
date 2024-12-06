// ROOT stuff
#include <TCanvas.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TLegend.h>
#include <TLine.h>
#include <TRandom.h>
#include <TRandom3.h>

// System stuff
#include <fstream>
#include <sys/time.h>
#include <sys/times.h>

// GooFit stuff
#include <goofit/Application.h>
#include <goofit/FitManager.h>
#include <goofit/PDFs/GooPdf.h>
#include <goofit/PDFs/basic/PolynomialPdf.h>
#include <goofit/PDFs/combine/AddPdf.h>
#include <goofit/PDFs/combine/ProdPdf.h>
#include <goofit/PDFs/physics/DalitzPlotPdf.h>
#include <goofit/PDFs/physics/DalitzPlotter.h>
#include <goofit/PDFs/physics/DalitzVetoPdf.h>
#include <goofit/PDFs/physics/ResonancePdf.h>
#include <goofit/UnbinnedDataSet.h>
#include <goofit/Variable.h>
#include <goofit/detail/Style.h>

//root stuff
#include "TF1.h"
#include "TCanvas.h"
#include "goofit/FitControl.h"

using namespace std;
using namespace GooFit;

Variable fixedRhoMass("rho_mass", 0.7758);
Variable fixedRhoWidth("rho_width", 0.1503);

const fptype _mD0       = 1.86484;
const fptype _mD02      = _mD0 * _mD0;
const fptype _mD02inv   = 1. / _mD02;
const fptype piPlusMass = 0.13957018;
const fptype piZeroMass = 0.1349766;

Observable m12("m12", 4*piPlusMass*piPlusMass, pow(_mD0-piPlusMass,2));
Observable m13("m13", 4*piPlusMass*piPlusMass, pow(_mD0-piPlusMass,2));
// Observables setup

EventNumber eventNumber("eventNumber");



// Constants used in more than one PDF component.
Variable motherM("motherM", _mD0);
Variable chargeM("chargeM", piPlusMass);
Variable neutrlM("neutrlM", piZeroMass);
Variable massSum("massSum", _mD0 *_mD0 + 2 * piPlusMass * piPlusMass + piZeroMass * piZeroMass); // = 3.53481
Variable constantOne("constantOne", 1);
Variable constantZero("constantZero", 0);


DalitzPlotPdf *makeSignalPdf(Observable m12, Observable m13, EventNumber eventNumber, GooPdf *eff = 0) {
    DecayInfo3 dtop0pp;
    dtop0pp.motherMass   = _mD0;
    dtop0pp.daug1Mass    = piPlusMass;
    dtop0pp.daug2Mass    = piPlusMass;
    dtop0pp.daug3Mass    = piPlusMass;
    dtop0pp.daug_meson_radius = 1.5;
    dtop0pp.mother_meson_radius = 5.0;

    ResonancePdf *rhop = new Resonances::RBW(
        "rhop", Variable("rhop_amp_real", 1), Variable("rhop_amp_imag", 0), fixedRhoMass, fixedRhoWidth, 1, PAIR_13,false);

    ResonancePdf *f0980 = new Resonances::RBW(
       "f0980", Variable("f0980_amp_real", 0), Variable("f0980_amp_imag", 1), Variable("f0_mass",0.990), Variable("f0_width",0.05), 0, PAIR_13,false);

    ResonancePdf *f01370 = new Resonances::RBW(
      "f01370", Variable("f01370_amp_real", 1), Variable("f01370_amp_imag", 0), Variable("f0_mass",1.370), Variable("f0_width",0.1), 0, PAIR_13,false);


    dtop0pp.resonances.push_back(rhop);
    dtop0pp.resonances.push_back(f0980);
    dtop0pp.resonances.push_back(f01370);


    if(!eff) {
        // By default create a constant efficiency.
        vector<Variable> offsets       = {constantZero, constantZero};
        vector<Observable> observables = {m12, m13};
        vector<Variable> coefficients  = {constantOne};

        eff = new PolynomialPdf("constantEff", observables, coefficients, offsets, 0);
    }

    return new DalitzPlotPdf("signalPDF", m12, m13, eventNumber, dtop0pp, eff);
}


class  PDF_Plotter {
public:
    PDF_Plotter(fptype m12, fptype m13, fptype m23, unsigned int pair, bool ismag):
    _m12(m12),
    _m13(m13),
    _m23(m23),
    _pair(pair),
    _ismag(ismag){};

    void setDalitz(DalitzPlotPdf *signal) {
        _signal=signal;
    }

    double operator() (double *x, double *p) {

        auto dtoppp = _signal->getDecayInfo();
        auto resonances = dtoppp.resonances;
        _signal->copyParams();
        fpcomplex  evalR{0.,0.};

        for(auto r: resonances) {
            auto pars = r->getParameters();
            fpcomplex coef(r->get_amp_real(),r->get_amp_img());
            switch(_pair) {
                case PAIR_12:
                    evalR += getResonanceAmplitude(x[0],_m13,_m23, r->getFunctionIndex(), r->getParameterIndex())*coef;
                    break;
                case PAIR_13:
                    evalR += getResonanceAmplitude(_m12,x[0],_m23, r->getFunctionIndex(), r->getParameterIndex())*coef;
                    break;
                case PAIR_23:
                    evalR += getResonanceAmplitude(_m12,_m13,x[0], r->getFunctionIndex(), r->getParameterIndex())*coef;
                    break;
            }
        }

        if(_ismag) {
            return static_cast<double>(thrust::abs(evalR));
        }else {
            return static_cast<double>(thrust::arg(evalR)*180./M_PI);
        }
    }

private:
    fptype _m12;
    fptype _m13;
    fptype _m23;
    unsigned int _pair;
    bool _ismag;
    DalitzPlotPdf *_signal;

};


int main(int argc, char **argv) {

    GooFit::setROOTStyle();
    auto signal = makeSignalPdf(m12,m13,eventNumber,0);
    m12.setNumBins(100);
    m13.setNumBins(100);
    PDF_Plotter abspdf(0.7758*0.7758,0,0,PAIR_13,true);
    abspdf.setDalitz(signal);

    PDF_Plotter phspdf(0.7758*0.7758,0,0,PAIR_13,false);
    phspdf.setDalitz(signal);
    TCanvas c("c","",1000,500);
    c.Divide(2,0);

    c.cd(1);
    TF1 fmag("PDF Mag",abspdf,4*piPlusMass*piPlusMass, pow(_mD0-piPlusMass,2),0);
    fmag.GetXaxis()->SetTitle("s(#pi#pi)");
    fmag.GetYaxis()->SetTitle("PDF Mag");
    fmag.Draw("L");
    c.cd(2);
    TF1 fphs("PDF Phs",phspdf,4*piPlusMass*piPlusMass, pow(_mD0-piPlusMass,2)+1,0);
    fphs.GetXaxis()->SetTitle("s(#pi#pi)");
    fphs.GetYaxis()->SetTitle("PDF Phs (deg)");
    fphs.Draw("L");
    c.SaveAs("test.png");

    return 0;
}
