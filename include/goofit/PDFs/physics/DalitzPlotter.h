#pragma once

#include <goofit/PDFs/GooPdf.h>
#include <goofit/PDFs/physics/DalitzPlotHelpers.h>
#include <goofit/PDFs/physics/DalitzPlotPdf.h>
#include <goofit/Version.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <string>

#if GOOFIT_ROOT_FOUND
#include <TH2.h>
#include <TH1.h>
#include <TStyle.h>
#include <TColor.h>

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
    GooPdf * overallSignal = nullptr;
    DalitzPlotPdf * signalDalitz = nullptr;
    
  private:
    
    double cpuGetM23(Variable massSum, double sij, double sik) { return (massSum.getValue() - sij - sik); }

    #if GOOFIT_ROOT_FOUND
        void style(){
            TStyle *myStyle= new TStyle( "myStyle", "Josue (LHCb) official plots style" );
            Double_t lhcbWidth = 3;
            myStyle->SetPadColor(0);
            myStyle->SetCanvasColor(0);
            myStyle->SetStatColor(0); 
            myStyle->SetLineWidth(lhcbWidth);
            myStyle->SetFrameLineWidth(lhcbWidth);
            myStyle->SetHistLineWidth(lhcbWidth);
            myStyle->SetFuncWidth(lhcbWidth);
            myStyle->SetGridWidth(lhcbWidth);
            myStyle->SetMarkerStyle(8);
            myStyle->SetMarkerSize(1.5);
            myStyle->SetPadTickX(1);            
            myStyle->SetPadTickY(1);   
            myStyle->SetOptStat(0); 
            myStyle->SetOptFit(1111111);
            const Int_t NRGBs = 5;
            const Int_t NCont = 255;
            Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
            Double_t red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
            Double_t green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
            Double_t blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
            TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
            gStyle->SetNumberContours(NCont);
            gROOT->SetStyle("myStyle");
        }

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
                ht->SetLineWidth(3);
                //hd->SetMarkerStyle(15);

                hd->SetMarkerColor(kBlue);
                hd->SetFillColor(kBlue);
                //hd->Rebin(20);


                TCanvas foo;

                hd->Draw("HIST");
                ht->Draw("HIST  same");


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

    void Plot(double Mother_mass,double d1_mass,double d2_mass,double d3_mass, std::string sij, std::string sik , std::string sjk , std::string plotdir,UnbinnedDataSet data) {
        

        m12.setNumBins(120);
        m13.setNumBins(120);
        TH1F m12_dat_hist("m12_dat_hist", "", m12.getNumBins(), 0.01*m12.getLowerLimit(), m12.getUpperLimit());
        m12_dat_hist.GetXaxis()->SetTitle( (sij+"[GeV]^{2}").c_str());
        m12_dat_hist.GetYaxis()->SetTitle("Events");

        TH1F m12_pdf_hist("m12_pdf_hist", "", m12.getNumBins(), 0.01*m12.getLowerLimit(), m12.getUpperLimit());

        TH1F m13_dat_hist("m13_dat_hist", "", m13.getNumBins(), 0.01*m13.getLowerLimit(), m13.getUpperLimit());
        m13_dat_hist.GetXaxis()->SetTitle( (sik+"[GeV]^{2}").c_str());
        m13_dat_hist.GetYaxis()->SetTitle("Events");

        TH1F m13_pdf_hist("m13_pdf_hist", "", m13.getNumBins(), m13.getLowerLimit(), m13.getUpperLimit());

        TH1F m23_dat_hist("m23_dat_hist", "", m13.getNumBins(), 0.01*m13.getLowerLimit(), m13.getUpperLimit());
        m23_dat_hist.GetXaxis()->SetTitle( (sjk+"[GeV]^{2}").c_str());
        m23_dat_hist.GetYaxis()->SetTitle("Events");

        TH1F m23_pdf_hist("m23_pdf_hist", "", m13.getNumBins(), 0.01*m13.getLowerLimit(), m13.getUpperLimit());

        double totalPdf = 0;
        double totalDat = 0;
        TH2F dalitz_dat_hist("dalitz_data_hist",
                                "",
                                m12.getNumBins(),
                                m12.getLowerLimit(),
                                m12.getUpperLimit(),
                                m13.getNumBins(),
                                m13.getLowerLimit(),
                                m13.getUpperLimit());
        dalitz_dat_hist.SetStats(false);
        dalitz_dat_hist.GetXaxis()->SetTitle((sij+"[GeV]^{2}").c_str());
        dalitz_dat_hist.GetYaxis()->SetTitle((sik+"[GeV]^{2}").c_str());
        TH2F dalitz_pdf_hist("dalitz_pdf_hist",
                                "",
                                m12.getNumBins(),
                                m12.getLowerLimit(),
                                m12.getUpperLimit(),
                                m13.getNumBins(),
                                m13.getLowerLimit(),
                                m13.getUpperLimit());

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
                m12.setValue(donram.Uniform(m12.getLowerLimit(), m12.getUpperLimit()));
                m13.setValue(donram.Uniform(m13.getLowerLimit(), m13.getUpperLimit()));
            } while(!inDalitz(m12.getValue(), m13.getValue(), Mother_mass, d1_mass, d2_mass, d3_mass));

                eventNumber.setValue(evtCounter);
                evtCounter++;
                currData.addEvent();
        }

        overallSignal->setData(&currData);
        signalDalitz->setDataSize(currData.getNumEvents());
        std::vector<std::vector<double>> pdfValues = overallSignal->getCompProbsAtDataPoints();

        Variable massSum("massSum", POW2(Mother_mass) + POW2(d1_mass) + POW2(d2_mass) + POW2(d3_mass));


        for(unsigned int j = 0; j < pdfValues[0].size(); ++j) {
            
            double currm12 = currData.getValue(m12, j);
            double currm13 = currData.getValue(m13, j);

            dalitz_pdf_hist.Fill(currm12, currm13, pdfValues[0][j]);
            m12_pdf_hist.Fill(currm12, pdfValues[0][j]);
            m13_pdf_hist.Fill(currm13, pdfValues[0][j]);
            m23_pdf_hist.Fill( cpuGetM23(massSum,currm12, currm13) , pdfValues[0][j]);

            totalPdf += pdfValues[0][j];
        
        }


        TCanvas foo;
        foo.SetLogz(false);
        dalitz_pdf_hist.Draw("colz");

        foo.SaveAs("plots/dalitz_pdf.png");

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
#endif
};

} // namespace GooFit
