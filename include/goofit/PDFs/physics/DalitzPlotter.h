#pragma once

#include <goofit/PDFs/GooPdf.h>
#include <goofit/PDFs/physics/DalitzPlotHelpers.h>
#include <goofit/PDFs/physics/DalitzPlotPdf.h>
#include <goofit/Version.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <fstream>

#if GOOFIT_ROOT_FOUND
#include <TH2.h>
#include <TH1.h>
#include <TStyle.h>
#include <TColor.h>
#include <TH2Poly.h>
#include <TMath.h>
#include <TCanvas.h>
#include <TRandom3.h>
#endif

namespace GooFit {

/// This class makes it easy to make plots over 3 body Dalitz PDFs. You can use ROOT style value access or bin numbers.

class DalitzPlotter {
    std::vector<size_t> xbins;
    std::vector<size_t> ybins;
    std::vector<std::vector<fptype>> pdfValues;
    Observable m12;
    Observable m13;
    EventNumber eventNumber;
    UnbinnedDataSet data;
    fptype mother;
    fptype daug1Mass;
    fptype daug2Mass;
    fptype daug3Mass;
    GooPdf * overallSignal = nullptr;
    DalitzPlotPdf * signalDalitz = nullptr;
    
  private:
    
    double cpuGetM23(Variable massSum, double sij, double sik) { return (massSum.getValue() - sij - sik); }

    #if GOOFIT_ROOT_FOUND
    void drawFitPlotsWithPulls(TH1 *hd, TH1 *ht, std::string plotdir) {
                
                const char *hname = hd->GetName();
                char obsname[10];
                for(int i = 0;; i++) {
                    if(hname[i] == '_')
                        obsname[i] = '\0';
                    else
                        obsname[i] = hname[i];
                    if(obsname[i] == '\0')
                        break;
                }
                ht->Scale(hd->Integral() / ht->Integral());
                ht->SetLineColor(kRed);
                ht->SetLineWidth(2);
                

                hd->SetMarkerColor(kBlue);
                //hd->SetFillColor(kBlue);
                //hd->Rebin(20);


                TCanvas foo("foo","",1020,720);
               
                hd->Draw("E");
                ht->Draw("HISTsame");


                foo.SaveAs( (plotdir+"/"+obsname+"_fit.png").c_str()    );


            }
        #endif

  public:
    DalitzPlotter( GooPdf * overallSignal , DalitzPlotPdf * signalDalitz )
        : m12(signalDalitz->_m12)
        , m13(signalDalitz->_m13)
        , eventNumber(signalDalitz->_eventNumber)
        , data({m12, m13, eventNumber})
        , mother(signalDalitz->decayInfo.motherMass)
        , daug1Mass(signalDalitz->decayInfo.daug1Mass)
        , daug2Mass(signalDalitz->decayInfo.daug2Mass)
        , daug3Mass(signalDalitz->decayInfo.daug3Mass)
        , overallSignal(overallSignal)
        , signalDalitz(signalDalitz)
        {
        eventNumber.setValue(0);

        for(size_t i = 0; i < m12.getNumBins(); ++i) {
            m12.setValue(m12.getLowerLimit() + m12.getBinSize() * (i + 0.5));
            for(size_t j = 0; j < m13.getNumBins(); ++j) {
                m13.setValue(m13.getLowerLimit() + m13.getBinSize() * (j + 0.5));
                if(inDalitz(m12.getValue(),
                            m13.getValue(),
                            signalDalitz->decayInfo.motherMass,
                            signalDalitz->decayInfo.daug1Mass,
                            signalDalitz->decayInfo.daug2Mass,
                            signalDalitz->decayInfo.daug3Mass)) {
                    xbins.push_back(i);
                    ybins.push_back(j);
                    data.addEvent();
                    eventNumber.setValue(eventNumber.getValue() + 1);
                }
            }
        }

        auto old = overallSignal->getData();
        overallSignal->setData(&data);
        signalDalitz->setDataSize(data.getNumEvents());
        pdfValues = overallSignal->getCompProbsAtDataPoints();
        overallSignal->setData(old);
    }

    /// Fill a dataset with MC events
    void fillDataSetMC(UnbinnedDataSet &dataset, size_t nTotal) {
        // Setup random numbers
        std::random_device rd;
        std::mt19937 gen(rd());

        // Uniform distribution
        std::uniform_real_distribution<> unihalf(-.5, .5);
        std::uniform_real_distribution<> uniwhole(0.0, 1.0);

        // CumSum in other languages
        std::vector<double> integral(pdfValues[0].size());
        std::partial_sum(pdfValues[0].begin(), pdfValues[0].end(), integral.begin());

        // Make this a 0-1 fraction by dividing by the end value
        std::for_each(integral.begin(), integral.end(), [&integral](double &val) { val /= integral.back(); });

        for(size_t i = 0; i < nTotal; i++) {
            double r = uniwhole(gen);

            // Binary search for integral[cell-1] < r < integral[cell]
            size_t j = std::lower_bound(integral.begin(), integral.end(), r) - integral.begin();

            // Fill in the grid square randomly
            double currm12 = data.getValue(m12, j) + m12.getBinSize() * unihalf(gen);
            double currm13 = data.getValue(m13, j) + m13.getBinSize() * unihalf(gen);

            m12.setValue(currm12);
            m13.setValue(currm13);
            eventNumber.setValue(i);
            dataset.addEvent();
        }
    }

    
    

    size_t getNumEvents() const { return data.getNumEvents(); }

    size_t getX(size_t event) const { return xbins.at(event); }

    size_t getY(size_t event) const { return ybins.at(event); }

    fptype getXval(size_t event) const { return data.getValue(m12, event); }

    fptype getYval(size_t event) const { return data.getValue(m13, event); }

    fptype getZval(size_t event) const { return POW2(mother) - POW2(getXval(event)) - POW2(getYval(event)); }

    fptype getVal(size_t event, size_t num = 0) const { return pdfValues.at(num).at(event); }

    UnbinnedDataSet *getDataSet() { return &data; }

    const Observable &getM12() const { return m12; }
    const Observable &getM13() const { return m13; }

#if GOOFIT_ROOT_FOUND
    /// Produce a TH2F over the contained evaluation
    TH2F *make2D(std::string name = "dalitzplot", std::string title = "") {
        auto *dalitzplot = new TH2F(name.c_str(),
                                    title.c_str(),
                                    m12.getNumBins(),
                                    m12.getLowerLimit(),
                                    m12.getUpperLimit(),
                                    m13.getNumBins(),
                                    m13.getLowerLimit(),
                                    m13.getUpperLimit());

        for(unsigned int j = 0; j < getNumEvents(); ++j) {
            size_t currm12 = getX(j);
            size_t currm13 = getY(j);
            double val     = getVal(j);

            dalitzplot->SetBinContent(1 + currm12, 1 + currm13, val);
        }

        return dalitzplot;
    }

    void Plot( std::string sij, std::string sik, std::string sjk  , std::string plotdir ,UnbinnedDataSet data) {

        fptype s12_min = m12.getLowerLimit();
        fptype s12_max = m12.getUpperLimit();
        fptype s13_min = m13.getLowerLimit();
        fptype s13_max = m13.getUpperLimit();
        fptype s23_min = (daug2Mass + daug3Mass)*(daug2Mass + daug3Mass);
        fptype s23_max = (mother - daug1Mass)*(mother - daug1Mass);

        TH1F m12_dat_hist("s12_dat_hist", "", 200, s12_min, s12_max);
        m12_dat_hist.GetXaxis()->SetTitle( (sij+"[GeV]^{2}").c_str());
        m12_dat_hist.GetYaxis()->SetTitle("Events");

        TH1F m12_pdf_hist("s12_pdf_hist", "", 200, s12_min, s12_max);

        TH1F m13_dat_hist("s13_dat_hist", "", 200,s13_min, s13_max);
        m13_dat_hist.GetXaxis()->SetTitle( (sik+"[GeV]^{2}").c_str());
        m13_dat_hist.GetYaxis()->SetTitle("Events");

        TH1F m13_pdf_hist("s13_pdf_hist", "", 200, s13_min, s13_max);

        TH1F m23_dat_hist("s23_dat_hist", "",200, s23_min, s23_max);
        m23_dat_hist.GetXaxis()->SetTitle( (sjk+"[GeV]^{2}").c_str());
        m23_dat_hist.GetYaxis()->SetTitle("Events");

        TH1F m23_pdf_hist("s23_pdf_hist", "", 200, s23_min, s23_max);

        double totalPdf = 0;
        double totalDat = 0;
        TH2F dalitz_dat_hist("dalitz_data_hist",
                                "",
                                200,
                                s12_min,
                                s12_max,
                                200,
                                s13_min,
                                s13_max);
        dalitz_dat_hist.SetStats(false);
        dalitz_dat_hist.GetXaxis()->SetTitle((sij+"[GeV]^{2}").c_str());
        dalitz_dat_hist.GetYaxis()->SetTitle((sik+"[GeV]^{2}").c_str());
        TH2F dalitz_pdf_hist("dalitz_pdf_hist",
                                "",
                               200,
                                s12_min,
                                s12_max,
                                200,
                                s13_min,
                                s13_max);

        dalitz_pdf_hist.GetXaxis()->SetTitle((sij+"[GeV]^{2}").c_str());
        dalitz_pdf_hist.GetYaxis()->SetTitle((sik+"[GeV]^{2}").c_str());
        dalitz_pdf_hist.SetStats(false);

        int NevG = 1e7;

        std::vector<Observable> vars;
        vars.push_back(m12);
        vars.push_back(m13);
        
        vars.push_back(eventNumber);
        UnbinnedDataSet currData(vars);
        int evtCounter = 0;

        TRandom3 donram(50);
        for(int i = 0; i < NevG; i++) {
            do {
                m12.setValue(donram.Uniform(s12_min, s12_max));
                m13.setValue(donram.Uniform(s13_min,s13_max));
            } while(!inDalitz(m12.getValue(), m13.getValue(), mother, daug1Mass, daug2Mass, daug3Mass));

                eventNumber.setValue(evtCounter);
                evtCounter++;
                currData.addEvent();
        }

        overallSignal->setData(&currData);
        signalDalitz->setDataSize(currData.getNumEvents());
        std::vector<std::vector<double>> pdfValues = overallSignal->getCompProbsAtDataPoints();

        Variable massSum("massSum", POW2(mother) + POW2(daug1Mass) + POW2(daug2Mass) + POW2(daug3Mass));


        for(unsigned int j = 0; j < pdfValues[0].size(); ++j) {
            
            double currm12 = currData.getValue(m12, j);
            double currm13 = currData.getValue(m13, j);

            dalitz_pdf_hist.Fill(currm12, currm13, pdfValues[0][j]);
            m12_pdf_hist.Fill(currm12, pdfValues[0][j]);
            m13_pdf_hist.Fill(currm13, pdfValues[0][j]);
            m23_pdf_hist.Fill( cpuGetM23(massSum,currm12, currm13) , pdfValues[0][j]);

            totalPdf += pdfValues[0][j];
        
        }


        TCanvas foo("foo","",1020,720);
        foo.SetLogz(true);
        dalitz_pdf_hist.Draw("colz");

        foo.SaveAs( (plotdir+"/dalitz_pdf.png").c_str() );
       
        for(unsigned int evt = 0; evt < data.getNumEvents(); ++evt) {
            double data_m12 = data.getValue(m12, evt);
            m12_dat_hist.Fill(data_m12);
            double data_m13 = data.getValue(m13, evt);
            m13_dat_hist.Fill(data_m13);
            m23_dat_hist.Fill(cpuGetM23(massSum,data_m12, data_m13));
            dalitz_dat_hist.Fill(data_m12, data_m13);
            
            totalDat++;
        }
        dalitz_dat_hist.Draw("colz");
        foo.SaveAs( (plotdir+"/dalitz_dat.png").c_str());

        drawFitPlotsWithPulls(&m12_dat_hist, &m12_pdf_hist, plotdir);
        drawFitPlotsWithPulls(&m13_dat_hist, &m13_pdf_hist, plotdir);
        drawFitPlotsWithPulls(&m23_dat_hist, &m23_pdf_hist, plotdir);
    }

    void chi2(size_t npar, std::string bins_file, float min_x, float max_x,float min_y, float max_y, UnbinnedDataSet data,std::string plotdir){

        TH2Poly* dp_data = new TH2Poly("dp_data","",min_x,max_x,min_y,max_y);
	    TH2Poly*  dp_toy = new TH2Poly("dp_toy","",min_x,max_x,min_y,max_y);
	    TH2Poly*  dp_pdf = new TH2Poly("dp_pdf","",min_x,max_x,min_y,max_y);
        TH2Poly*  residuals = new TH2Poly("dp_pdf","",min_x,max_x,min_y,max_y);
	    TH1F* Proj = new TH1F("projection","",50,-5.,+5.);

        fptype s12_min = m12.getLowerLimit();
        fptype s12_max = m12.getUpperLimit();
        fptype s13_min = m13.getLowerLimit();
        fptype s13_max = m13.getUpperLimit();

        std::ifstream w(bins_file.c_str());
        double min1,max1,min2,max2;
        
        while(w>>min1>>min2>>max1>>max2){
            dp_data->AddBin(min1,max1,min2,max2);
            dp_toy->AddBin(min1,max1,min2,max2);
            dp_pdf->AddBin(min1,max1,min2,max2);
            residuals->AddBin(min1,max1,min2,max2);
        }

        w.close();

         //fill dp_data
        for(size_t i = 0; i < data.getNumEvents(); i++){
            data.loadEvent(i);
            if(m12.getValue()<m13.getValue()){
                dp_data->Fill(m12.getValue(),m13.getValue());
            }
            if(m13.getValue()>m12.getValue()){
                dp_data->Fill(m13.getValue(),m12.getValue());
            }
        }

        
        int NevG = 10000000;
        int evtCounter = 0;
        TRandom3 donram(50);

        UnbinnedDataSet toyMC({m12,m13,eventNumber});
        fillDataSetMC(toyMC,NevG);

        for(size_t i = 0; i < toyMC.getNumEvents(); i++){
            toyMC.loadEvent(i);
            if(m12.getValue()<m13.getValue()){
                dp_toy->Fill(m12.getValue(),m13.getValue());
                dp_pdf->Fill(m12.getValue(),m13.getValue());
            }
            if(m13.getValue()>m12.getValue()){
                dp_toy->Fill(m13.getValue(),m12.getValue());
                dp_pdf->Fill(m13.getValue(),m12.getValue());
            }
        }

        double scale = double(data.getNumEvents())/double(toyMC.getNumEvents());
        dp_pdf->Scale(scale);

        //number of adaptative bins
        size_t nbins = dp_toy->GetNumberOfBins();
        fptype chi2 = 0;
        for(size_t i = 1; i <= nbins ; ++i){
            auto diff = dp_pdf->GetBinContent(i) - dp_data->GetBinContent(i);
            auto errSq  = pow( (dp_pdf->GetBinContent(i)/dp_toy->GetBinContent(i))*dp_toy->GetBinError(i),2) + dp_data->GetBinContent(i);
            chi2 += diff*diff/errSq;
            residuals->SetBinContent(i,diff/sqrt(errSq));
            Proj->Fill(diff/sqrt(dp_pdf->GetBinContent(i)));
        
        }

        TCanvas foo("foo","",1020,720);
        residuals->Draw("colz");
        gStyle->SetOptStat(0);
	auto output = fmt::format("{0}/residuals.png",plotdir);
        foo.SaveAs(output.c_str());

        gStyle->SetOptFit(1111);
        Proj->Fit("gaus");
        Proj->Draw("E");
	output = fmt::format("{0}/Residuals_proj.png",plotdir);
        foo.SaveAs(output.c_str());

        fptype ndof = nbins - npar -1;

        std::cout << "chi2/ndof is within the range [" << (chi2/ndof) << " - " << (chi2/(nbins-1)) << "] and the p-value is [" << TMath::Prob(chi2,ndof) << " - " << TMath::Prob(chi2,nbins-1)<< "]" << std::endl;	
        
	
}

#endif
};

} // namespace GooFit
