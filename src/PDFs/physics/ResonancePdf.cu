#include <goofit/PDFs/detail/ComplexUtils.h>
#include <goofit/PDFs/physics/DalitzPlotHelpers.h>
#include <goofit/PDFs/physics/ResonancePdf.h>

namespace GooFit {

__device__ fptype cDeriatives[2 * MAXNKNOBS];

__device__ fptype twoBodyCMmom(double rMassSq, fptype d1m, fptype d2m, fptype mR) {
    fptype x = rMassSq;
    fptype y = d1m * d1m;
    fptype z = d2m * d2m;
    double l = POW2(x - y - z) - 4 * y * z;

    return sqrt(l) / (2 * mR);
}

__device__ fptype twoBodyCMmom(fptype rMassSq, fptype d1m, fptype d2m) {
    // For A -> B + C, calculate momentum of B and C in rest frame of A.
    // PDG 38.16.

    fptype kin1 = 1 - POW2(d1m + d2m) / rMassSq;

    kin1 = kin1 >= 0 ? sqrt(kin1) : 1;

    fptype kin2 = 1 - POW2(d1m - d2m) / rMassSq;
    kin2        = kin2 >= 0 ? sqrt(kin2) : 1;

    return 0.5 * sqrt(rMassSq) * kin1 * kin2;
}



__device__ fptype dampingFactorSquare(const fptype &cmmom, const int &spin, const fptype &mRadius) {
    fptype square = mRadius * mRadius * cmmom * cmmom;
    fptype dfsq   = 1 + square; // This accounts for spin 1
    // if (2 == spin) dfsq += 8 + 2*square + square*square; // Coefficients are 9, 3, 1.
    fptype dfsqres = 9 + 3 * square + square * square;

    // Spin 3 and up not accounted for.
    // return dfsq;
    return (spin == 2) ? 1./dfsqres : 1./dfsq;
}

/*__device__ fptype dampingFactorSquare(const fptype &cmmom, const int &spin, const fptype &mRadius) {
    fptype z = mRadius * mRadius * cmmom * cmmom;
    fptype dfsq   =sqrt(2.0*z/(z + 1.0)); // This accounts for spin 1
    // if (2 == spin) dfsq += 8 + 2*square + square*square; // Coefficients are 9, 3, 1.
    fptype dfsqres = sqrt(13.0*z*z/(z*z + 3.0*z + 9.0));

    // Spin 3 and up not accounted for.
    // return dfsq;
    return (spin == 2) ? dfsqres : dfsq;
}*/

__device__ fptype spinFactor(unsigned int spin,
                             fptype motherMass,
                             fptype daug1Mass,
                             fptype daug2Mass,
                             fptype daug3Mass,
                             fptype m12,
                             fptype m13,
                             fptype m23,
                             unsigned int cyclic_index) {
    if(0 == spin)
        return 1; // Should not cause branching since every thread evaluates the same resonance at the same time.

     // Copied from EvtDalitzReso, with assumption that pairAng convention matches pipipi0 from EvtD0mixDalitz.
    // Again, all threads should get the same branch.
    fptype _mA  = (PAIR_12 == cyclic_index ? daug1Mass : (PAIR_13 == cyclic_index ? daug3Mass : daug2Mass));
    fptype _mB  = (PAIR_12 == cyclic_index ? daug2Mass : (PAIR_13 == cyclic_index ? daug1Mass : daug3Mass));
    fptype _mC  = (PAIR_12 == cyclic_index ? daug3Mass : (PAIR_13 == cyclic_index ? daug2Mass : daug1Mass));
    fptype _mAC = (PAIR_12 == cyclic_index ? m13 : (PAIR_13 == cyclic_index ? m23 : m12));
    fptype _mBC = (PAIR_12 == cyclic_index ? m23 : (PAIR_13 == cyclic_index ? m12 : m13));
    fptype _mAB = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));

    fptype massFactor = 1.0 / _mAB;
    fptype sFactor    = 1;
    sFactor *= ((_mBC - _mAC) + (massFactor * (motherMass * motherMass - _mC * _mC) * (_mA * _mA - _mB * _mB)));

    if(2 == spin) {
        sFactor *= sFactor;
        fptype extraterm = ((_mAB - (2 * motherMass * motherMass) - (2 * _mC * _mC))
                            + massFactor * POW2(motherMass * motherMass - _mC * _mC));
        extraterm *= ((_mAB - (2 * _mA * _mA) - (2 * _mB * _mB)) + massFactor * POW2(_mA * _mA - _mB * _mB));
        extraterm /= 3;
        sFactor -= extraterm;
    }

    return sFactor;
}

template <int I>
__device__ fpcomplex plainBW(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
    fptype c_motherMass   = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 0]);
    fptype c_daug1Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 1]);
    fptype c_daug2Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 2]);
    fptype c_daug3Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 3]);
    fptype c_meson_radius = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 4]);

    fptype resmass            = RO_CACHE(cudaArray[RO_CACHE(indices[2])]);
    fptype reswidth           = RO_CACHE(cudaArray[RO_CACHE(indices[3])]);
    unsigned int spin         = RO_CACHE(indices[4]);
    unsigned int cyclic_index = RO_CACHE(indices[5]);
    unsigned int symmDP       = RO_CACHE(indices[6]);

    fpcomplex result(0., 0.);
    fptype resmass2 = POW2(resmass);

#pragma unroll
    for(int i = 0; i < I; i++) {
        fptype rMassSq    = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
        fptype rMass = sqrt(rMassSq);
        fptype mass_daug1 = PAIR_23 == cyclic_index ? c_daug2Mass : c_daug1Mass;
        fptype mass_daug2 = PAIR_12 == cyclic_index ? c_daug2Mass : c_daug3Mass;
        fptype mass_daug3 = PAIR_23 == cyclic_index ? c_daug1Mass : (PAIR_13 == cyclic_index?c_daug2Mass:c_daug3Mass);

        fptype frFactor = 1;
        fptype fdFactor = 1;

        // Calculate momentum of the two daughters in the resonance rest frame
        // Note symmetry under interchange (dm1 <-> dm2)

        fptype q0 = twoBodyCMmom(resmass2, mass_daug1, mass_daug2, resmass);
        fptype q = twoBodyCMmom(rMassSq, mass_daug1, mass_daug2, rMass);

        if(0 != spin) {
            frFactor = dampingFactorSquare(q, spin, c_meson_radius)
                       / dampingFactorSquare(q0, spin, c_meson_radius);


            fptype q_2 = twoBodyCMmom(c_motherMass*c_motherMass, rMass, mass_daug3,c_motherMass);
            fptype q0_2 = twoBodyCMmom(c_motherMass*c_motherMass, resmass, mass_daug3,c_motherMass);
            fdFactor =  dampingFactorSquare(q_2, spin, 5.)
                           / dampingFactorSquare(q0_2, spin, 5.);
        }

	fptype gamma = resmass*reswidth*pow(q/q0, 2*spin + 1)*frFactor/sqrt(rMassSq);
        // RBW evaluation
        fptype A = (resmass2 - rMassSq);
        fptype B = resmass*gamma;
        fptype C = 1.0 / (POW2(A) + POW2(B));

        fpcomplex ret(A * C, B * C); // Dropping F_D=1

        ret *= sqrt(frFactor*fdFactor);
        ret *= spinFactor(spin, c_motherMass, c_daug1Mass, c_daug2Mass, c_daug3Mass, m12, m13, m23, cyclic_index);

        result += ret;
                                          
        if(I != 0) {
            fptype swpmass = m12;
            m12            = m13;
            m13            = swpmass;
        }
    }

    return result;
}

__device__ fpcomplex gaussian(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
    // indices[1] is unused constant index, for consistency with other function types.
    fptype resmass            = RO_CACHE(cudaArray[RO_CACHE(indices[2])]);
    fptype reswidth           = RO_CACHE(cudaArray[RO_CACHE(indices[3])]);
    unsigned int cyclic_index = indices[4];

    // Notice sqrt - this function uses mass, not mass-squared like the other resonance types.
    fptype massToUse = sqrt(PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
    massToUse -= resmass;
    massToUse /= reswidth;
    massToUse *= massToUse;
    fptype ret = exp(-0.5 * massToUse);

    // Ignore factor 1/sqrt(2pi).
    ret /= reswidth;

    return {ret, 0};
}

__device__ fptype hFun(double s, double daug2Mass, double daug3Mass) {
    // Last helper function
    const fptype _pi = 3.14159265359;
    double sm        = daug2Mass + daug3Mass;
    double sqrt_s    = sqrt(s);
    double k_s       = twoBodyCMmom(s, daug2Mass, daug3Mass);

    return ((2 / _pi) * (k_s / sqrt_s) * log((sqrt_s + 2 * k_s) / (sm)));
}

__device__ fptype dh_dsFun(double s, double daug2Mass, double daug3Mass) {
    // Yet another helper function
    const fptype _pi = 3.14159265359;
    double k_s       = twoBodyCMmom(s, daug2Mass, daug3Mass);

    return hFun(s, daug2Mass, daug3Mass) * (1.0 / (8.0 * POW2(k_s)) - 1.0 / (2.0 * s)) + 1.0 / (2.0 * _pi * s);
}

__device__ fptype dFun(double s, double daug2Mass, double daug3Mass) {
    // Helper function used in Gronau-Sakurai
    const fptype _pi = 3.14159265359;
    double sm        = daug2Mass + daug3Mass;
    double sm24      = sm * sm / 4.0;
    double m         = sqrt(s);
    double k_m2      = twoBodyCMmom(s, daug2Mass, daug3Mass);

    return 3.0 / _pi * sm24 / POW2(k_m2) * log((m + 2 * k_m2) / sm) + m / (2 * _pi * k_m2)
           - sm24 * m / (_pi * POW3(k_m2));
}

__device__ fptype fsFun(double s, double m2, double gam, double daug2Mass, double daug3Mass) {
    // Another G-S helper function

    double k_s   = twoBodyCMmom(s, daug2Mass, daug3Mass);
    double k_Am2 = twoBodyCMmom(m2, daug2Mass, daug3Mass);

    double f = gam * m2 / POW3(k_Am2);
    f *= (POW2(k_s) * (hFun(s, daug2Mass, daug3Mass) - hFun(m2, daug2Mass, daug3Mass))
          + (m2 - s) * POW2(k_Am2) * dh_dsFun(m2, daug2Mass, daug3Mass));

    return f;
}

template<int I>
__device__ fpcomplex gouSak(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
    fptype motherMass   = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 0]);
    fptype daug1Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 1]);
    fptype daug2Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 2]);
    fptype daug3Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 3]);
    fptype meson_radius = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 4]);

    fptype resmass            = RO_CACHE(cudaArray[RO_CACHE(indices[2])]);
    fptype reswidth           = RO_CACHE(cudaArray[RO_CACHE(indices[3])]);
    unsigned int spin         = RO_CACHE(indices[4]);
    unsigned int cyclic_index = RO_CACHE(indices[5]);
    unsigned int symmDP       = RO_CACHE(indices[6]);

    fpcomplex ret(0.0,0.0);

    for(int i = 0; i < I; i++) {
    	fptype rMassSq  = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
    	fptype frFactor = 1;

        fptype resmassSq = resmass*resmass;
    // Calculate momentum of the two daughters in the resonance rest frame; note symmetry under interchange (dm1 <->
    // dm2).
    	fptype measureDaughterMoms = twoBodyCMmom(
        	rMassSq, (PAIR_23 == cyclic_index ? daug2Mass : daug1Mass), (PAIR_12 == cyclic_index ? daug2Mass : daug3Mass));
    	fptype nominalDaughterMoms = twoBodyCMmom(
        	resmassSq, (PAIR_23 == cyclic_index ? daug2Mass : daug1Mass), (PAIR_12 == cyclic_index ? daug2Mass : daug3Mass));

    	if(0 != spin) {
        	frFactor = dampingFactorSquare(nominalDaughterMoms, spin, meson_radius);
        	frFactor /= dampingFactorSquare(measureDaughterMoms, spin, meson_radius);
    	}

    // Implement Gou-Sak:

    	fptype D = (1.0 + dFun(resmassSq, daug2Mass, daug3Mass) * reswidth / sqrt(resmassSq));
    	fptype E = resmassSq - rMassSq + fsFun(rMassSq, resmassSq, reswidth, daug2Mass, daug3Mass);
    	fptype F = sqrt(resmassSq) * reswidth * pow(measureDaughterMoms / nominalDaughterMoms, 2.0 * spin + 1) * frFactor;

    	D /= (E * E + F * F);
    	fpcomplex retur(D * E, D * F); // Dropping F_D=1
   	retur *= sqrt(frFactor);
   	retur *= spinFactor(spin, motherMass, daug1Mass, daug2Mass, daug3Mass, m12, m13, m23, cyclic_index);

	ret += retur;
        if(I != 0) {
            fptype swpmass = m12;
            m12            = m13;
            m13            = swpmass;
        }

   }

    return ret;
}

__device__ fpcomplex lass(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
    fptype motherMass   = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 0]);
    fptype daug1Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 1]);
    fptype daug2Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 2]);
    fptype daug3Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 3]);
    fptype meson_radius = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 4]);

    fptype resmass            = RO_CACHE(cudaArray[RO_CACHE(indices[2])]);
    fptype reswidth           = RO_CACHE(cudaArray[RO_CACHE(indices[3])]);
    unsigned int spin         = RO_CACHE(indices[4]);
    unsigned int cyclic_index = RO_CACHE(indices[5]);

    fptype rMassSq  = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
    fptype frFactor = 1;

    resmass *= resmass;
    // Calculate momentum of the two daughters in the resonance rest frame; note symmetry under interchange (dm1 <->
    // dm2).

    fptype measureDaughterMoms = twoBodyCMmom(
        rMassSq, (PAIR_23 == cyclic_index ? daug2Mass : daug1Mass), (PAIR_23 == cyclic_index ? daug3Mass : daug2Mass));
    fptype nominalDaughterMoms = twoBodyCMmom(
        resmass, (PAIR_23 == cyclic_index ? daug2Mass : daug1Mass), (PAIR_23 == cyclic_index ? daug3Mass : daug2Mass));

    if(0 != spin) {
        frFactor = dampingFactorSquare(nominalDaughterMoms, spin, meson_radius);
        frFactor /= dampingFactorSquare(measureDaughterMoms, spin, meson_radius);
    }

    // Implement LASS:
    /*
    fptype s = kinematics(m12, m13, _trackinfo[i]);
    fptype q = twoBodyCMmom(s, _trackinfo[i]);
    fptype m0  = _massRes[i]->getValFast();
    fptype _g0 = _gammaRes[i]->getValFast();
    int spin   = _spinRes[i];
    fptype g = runningWidthFast(s, m0, _g0, spin, _trackinfo[i], FrEval(s, m0, _trackinfo[i], spin));
    */

    fptype q = measureDaughterMoms;
    fptype g = reswidth * pow(measureDaughterMoms / nominalDaughterMoms, 2.0 * spin + 1) * frFactor / sqrt(rMassSq);

    fptype _a    = 0.22357;
    fptype _r    = -15.042;
    fptype _R    = 1; // ?
    fptype _phiR = 1.10644;
    fptype _B    = 0.614463;
    fptype _phiB = -0.0981907;

    // background phase motion
    fptype cot_deltaB  = (1.0 / (_a * q)) + 0.5 * _r * q;
    fptype qcot_deltaB = (1.0 / _a) + 0.5 * _r * q * q;

    // calculate resonant part
    fpcomplex expi2deltaB = fpcomplex(qcot_deltaB, q) / fpcomplex(qcot_deltaB, -q);
    fpcomplex resT        = fpcomplex(cos(_phiR + 2 * _phiB), sin(_phiR + 2 * _phiB)) * _R;

    fpcomplex prop = fpcomplex(1, 0) / fpcomplex(resmass - rMassSq, sqrt(resmass) * g);
    // resT *= prop*m0*_g0*m0/twoBodyCMmom(m0*m0, _trackinfo[i])*expi2deltaB;
    resT *= prop * (resmass * reswidth / nominalDaughterMoms) * expi2deltaB;

    // calculate bkg part
    resT += fpcomplex(cos(_phiB), sin(_phiB)) * _B * (cos(_phiB) + cot_deltaB * sin(_phiB)) * sqrt(rMassSq)
            / fpcomplex(qcot_deltaB, -q);

    resT *= sqrt(frFactor);
    resT *= spinFactor(spin, motherMass, daug1Mass, daug2Mass, daug3Mass, m12, m13, m23, cyclic_index);

    return resT;
}

__device__ fpcomplex nonres(fptype m12, fptype m13, fptype m23, unsigned int *indices) { return {1., 0.}; }

__device__ void
getAmplitudeCoefficients(fpcomplex a1, fpcomplex a2, fptype &a1sq, fptype &a2sq, fptype &a1a2real, fptype &a1a2imag) {
    // Returns A_1^2, A_2^2, real and imaginary parts of A_1A_2^*
    a1sq = thrust::norm(a1);
    a2sq = thrust::norm(a2);
    a1 *= conj(a2);
    a1a2real = a1.real();
    a1a2imag = a1.imag();
}
/*
__device__ fpcomplex flatte(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
    // indices[1] is unused constant index, for consistency with other function types.
    fptype resmass            = cudaArray[indices[2]];
    fptype g1                 = cudaArray[indices[3]];
    fptype g2                 = cudaArray[indices[4]]*g1;
    unsigned int cyclic_index = indices[5];
    unsigned int doSwap       = indices[6];

    fptype pipmass = 0.13957018;
    fptype pi0mass = 0.1349766;
    fptype kpmass  = 0.493677;
    fptype k0mass  = 0.497614;

    fptype twopimasssq  = 4 * pipmass * pipmass;
    fptype twopi0masssq = 4 * pi0mass * pi0mass;
    fptype twokmasssq   = 4 * kpmass * kpmass;
    fptype twok0masssq  = 4 * k0mass * k0mass;

    fpcomplex ret(0., 0.);
    for(int i = 0; i < 1 + doSwap; i++) {
        fptype rhopipi_real = 0, rhopipi_imag = 0;
        fptype rhokk_real = 0, rhokk_imag = 0;

        fptype s = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));

        if(s >= twopimasssq)
            rhopipi_real += (2. / 3) * sqrt(1 - twopimasssq / s); // Above pi+pi- threshold
        else
            rhopipi_imag += (2. / 3) * sqrt(-1 + twopimasssq / s);
        if(s >= twopi0masssq)
            rhopipi_real += (1. / 3) * sqrt(1 - twopi0masssq / s); // Above pi0pi0 threshold
        else
            rhopipi_imag += (1. / 3) * sqrt(-1 + twopi0masssq / s);
        if(s >= twokmasssq)
            rhokk_real += 0.5 * sqrt(1 - twokmasssq / s); // Above K+K- threshold
        else
            rhokk_imag += 0.5 * sqrt(-1 + twokmasssq / s);
        if(s >= twok0masssq)
            rhokk_real += 0.5 * sqrt(1 - twok0masssq / s); // Above K0K0 threshold
        else
            rhokk_imag += 0.5 * sqrt(-1 + twok0masssq / s);
		
        fptype A = (resmass * resmass - s) + resmass * (rhopipi_imag * g1 + rhokk_imag * g2);
        fptype B = resmass * (rhopipi_real * g1 + rhokk_real * g2);
        fptype C = 1.0 / (A * A + B * B);
        fpcomplex retur(A * C, B * C);
        ret += retur;
        if(doSwap) {
            fptype swpmass = m12;
            m12            = m13;
            m13            = swpmass;
        }
    }

    return ret;
}*/

//From Laura++
__device__ fpcomplex flatte(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
    // indices[1] is unused constant index, for consistency with other function types.
    fptype resmass            = cudaArray[indices[2]];
    fptype g1                 = cudaArray[indices[3]];
    fptype g2                 = cudaArray[indices[4]]*g1;
    unsigned int cyclic_index = indices[5];
    unsigned int doSwap       = indices[6];

    fptype pipmass = 0.13957018;
    fptype pi0mass = 0.1349766;
    fptype kpmass  = 0.493677;
    fptype k0mass  = 0.497614;

    fptype twopimasssq  = 4 * pipmass * pipmass;
    fptype twopi0masssq = 4 * pi0mass * pi0mass;
    fptype twokmasssq   = 4 * kpmass * kpmass;
    fptype twok0masssq  = 4 * k0mass * k0mass;

    fptype resmass2 = POW2(resmass);

    fpcomplex ret(0., 0.);
    
    fptype rho1(0.0), rho2(0.0);
    
    for(int i = 0; i < 1 + doSwap; i++) {
        fptype s = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
        fptype dMSq = resmass2 - s;

        if (s > twopi0masssq) {
            rho1 = sqrt(1.0 - twopi0masssq/s)/3.0;
            if (s > twopimasssq) {
                rho1 += 2.0*sqrt(1.0 - twopimasssq/s)/3.0;
                if (s > twokmasssq) {
                    rho2 = 0.5*sqrt(1.0 - twokmasssq/s);
                    if (s > twok0masssq) {
                        rho2 += 0.5*sqrt(1.0 - twok0masssq/s);
                    } else {
                        // Continue analytically below higher channel thresholds
                        // This contributes to the real part of the amplitude denominator
                        dMSq += g2*resmass*0.5*sqrt(twok0masssq/s - 1.0);
                    }
                } else {
                    // Continue analytically below higher channel thresholds
                    // This contributes to the real part of the amplitude denominator
                    rho2 = 0.0;
                    dMSq += g2*resmass*(0.5*sqrt(twokmasssq/s - 1.0) + 0.5*sqrt(twok0masssq/s - 1.0));
                }
            } else {
                // Continue analytically below higher channel thresholds
                // This contributes to the real part of the amplitude denominator
                dMSq += g1*resmass*2.0*sqrt(twopimasssq/s - 1.0)/3.0;
            }
        }
    
        //the Adler-zero term fA = (m2 − sA)/(m20 − sA) can be used to suppress false 
        //kinematic singularities when m goes below threshold. For f(0)(980), sA = 0.
        
        fptype massFactor = s/resmass2;
        
        fptype width1 = g1*rho1*massFactor;
        fptype width2 = g2*rho2*massFactor;
        fptype widthTerm = width1 + width2;
    
        fpcomplex resAmplitude(dMSq, widthTerm);
    
        fptype denomFactor = dMSq*dMSq + widthTerm*widthTerm;
    
        fptype invDenomFactor = 0.0;
        if (denomFactor > 1e-10) {invDenomFactor = 1.0/denomFactor;}
    
        resAmplitude *= invDenomFactor;

        ret += resAmplitude;
       
        if(doSwap) {
            fptype swpmass = m12;
            m12            = m13;
            m13            = swpmass;
        }
    }

    return ret;
}

__device__ fpcomplex cubicspline(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
    fpcomplex ret(0, 0);
    unsigned int cyclic_index        = indices[2];
    unsigned int doSwap              = indices[3];
    const unsigned int nKnobs        = indices[4];
    unsigned int idx                 = 5; // Next index
    unsigned int i                   = 0;
    const unsigned int pwa_coefs_idx = idx;
    idx += 2 * nKnobs;
    const fptype *mKKlimits = &(functorConstants[indices[idx]]);
    fptype mAB = m12, mAC = m13;
    switch(cyclic_index) {
    case PAIR_13:
        mAB = m13;
        mAC = m12;
        break;
    case PAIR_23:
        mAB = m23;
        mAC = m12;
        break;
    }

    int khiAB = 0, khiAC = 0;
    fptype dmKK, aa, bb, aa3, bb3;
    unsigned int timestorun = 1 + doSwap;
    while(khiAB < nKnobs) {
        if(mAB < mKKlimits[khiAB])
            break;
        khiAB++;
    }

    if(khiAB <= 0 || khiAB == nKnobs)
        timestorun = 0;
    while(khiAC < nKnobs) {
        if(mAC < mKKlimits[khiAC])
            break;
        khiAC++;
    }

    if(khiAC <= 0 || khiAC == nKnobs)
        timestorun = 0;

    for(i = 0; i < timestorun; i++) {
        unsigned int kloAB                = khiAB - 1; //, kloAC = khiAC -1;
        unsigned int twokloAB             = kloAB + kloAB;
        unsigned int twokhiAB             = khiAB + khiAB;
        fptype pwa_coefs_real_kloAB       = cudaArray[indices[pwa_coefs_idx + twokloAB]];
        fptype pwa_coefs_real_khiAB       = cudaArray[indices[pwa_coefs_idx + twokhiAB]];
        fptype pwa_coefs_imag_kloAB       = cudaArray[indices[pwa_coefs_idx + twokloAB + 1]];
        fptype pwa_coefs_imag_khiAB       = cudaArray[indices[pwa_coefs_idx + twokhiAB + 1]];
        fptype pwa_coefs_prime_real_kloAB = cDeriatives[twokloAB];
        fptype pwa_coefs_prime_real_khiAB = cDeriatives[twokhiAB];
        fptype pwa_coefs_prime_imag_kloAB = cDeriatives[twokloAB + 1];
        fptype pwa_coefs_prime_imag_khiAB = cDeriatives[twokhiAB + 1];
        //  printf("m12: %f: %f %f %f %f %f %f %d %d %d\n", mAB, mKKlimits[0], mKKlimits[nKnobs-1],
        //  pwa_coefs_real_khiAB, pwa_coefs_imag_khiAB, pwa_coefs_prime_real_khiAB, pwa_coefs_prime_imag_khiAB, khiAB,
        //  khiAC, timestorun );

        dmKK = mKKlimits[khiAB] - mKKlimits[kloAB];
        aa   = (mKKlimits[khiAB] - mAB) / dmKK;
        bb   = 1 - aa;
        aa3  = aa * aa * aa;
        bb3  = bb * bb * bb;

        ret.real(ret.real() + aa * pwa_coefs_real_kloAB + bb * pwa_coefs_real_khiAB
                 + ((aa3 - aa) * pwa_coefs_prime_real_kloAB + (bb3 - bb) * pwa_coefs_prime_real_khiAB) * (dmKK * dmKK)
                       / 6.0);
        ret.imag(ret.imag() + aa * pwa_coefs_imag_kloAB + bb * pwa_coefs_imag_khiAB
                 + ((aa3 - aa) * pwa_coefs_prime_imag_kloAB + (bb3 - bb) * pwa_coefs_prime_imag_khiAB) * (dmKK * dmKK)
                       / 6.0);
        khiAB = khiAC;
        mAB   = mAC;
    }
    return ret;
}

/*__device__ fpcomplex cubicspline(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
    fpcomplex ret(0, 0);
    unsigned int cyclic_index        = indices[2];
    unsigned int doSwap              = indices[3];
    const unsigned int nKnobs        = indices[4];
    unsigned int idx                 = 5; // Next index
    unsigned int i                   = 0;
    const unsigned int pwa_coefs_idx = idx;
    idx += 2 * nKnobs;
    const fptype *mKKlimits = &(functorConstants[indices[idx]]);
    fptype mAB = m12, mAC = m13;
    switch(cyclic_index) {
    case PAIR_13:
        mAB = m13;
        mAC = m12;
        break;
    case PAIR_23:
        mAB = m23;
        mAC = m12;
        break;
    }
    mAB = mAB; mAC = mAC; 
    int khiAB = 0, khiAC = 0;
    fptype dmKK, aa, bb, aa3, bb3;
    unsigned int timestorun = 1 + doSwap;
    while(khiAB < nKnobs) {
        if(mAB < mKKlimits[khiAB])
            break;
        khiAB++;
    }

    if(khiAB <= 0 || khiAB == nKnobs)
        timestorun = 0;
    while(khiAC < nKnobs) {
        if(mAC < mKKlimits[khiAC])
            break;
        khiAC++;
    }

    if(khiAC <= 0 || khiAC == nKnobs)
        timestorun = 0;

    for(i = 0; i < timestorun; i++) {
        unsigned int kloAB                = khiAB - 1; //, kloAC = khiAC -1;
        unsigned int twokloAB             = kloAB + kloAB;
        unsigned int twokhiAB             = khiAB + khiAB;
        fptype pwa_coefs_mag_kloAB       = cudaArray[indices[pwa_coefs_idx + twokloAB]];
        fptype pwa_coefs_mag_khiAB       = cudaArray[indices[pwa_coefs_idx + twokhiAB]];
        fptype pwa_coefs_phase_kloAB       = cudaArray[indices[pwa_coefs_idx + twokloAB + 1]];
        fptype pwa_coefs_phase_khiAB       = cudaArray[indices[pwa_coefs_idx + twokhiAB + 1]];
        fptype pwa_coefs_real_kloAB = pwa_coefs_mag_kloAB*cos(pwa_coefs_phase_kloAB);
        fptype pwa_coefs_real_khiAB = pwa_coefs_mag_khiAB*cos(pwa_coefs_phase_khiAB);
        fptype pwa_coefs_imag_kloAB = pwa_coefs_mag_kloAB*sin(pwa_coefs_phase_kloAB);
        fptype pwa_coefs_imag_khiAB = pwa_coefs_mag_khiAB*sin(pwa_coefs_phase_khiAB);
        
        fptype pwa_coefs_prime_real_kloAB = cDeriatives[twokloAB];
        fptype pwa_coefs_prime_real_khiAB = cDeriatives[twokhiAB];
        fptype pwa_coefs_prime_imag_kloAB = cDeriatives[twokloAB + 1];
        fptype pwa_coefs_prime_imag_khiAB = cDeriatives[twokhiAB + 1];
        //  printf("m12: %f: %f %f %f %f %f %f %d %d %d\n", mAB, mKKlimits[0], mKKlimits[nKnobs-1],
        //  pwa_coefs_real_khiAB, pwa_coefs_imag_khiAB, pwa_coefs_prime_real_khiAB, pwa_coefs_prime_imag_khiAB, khiAB,
        //  khiAC, timestorun );

        dmKK = mKKlimits[khiAB] - mKKlimits[kloAB];
        aa   = (mKKlimits[khiAB] - mAB) / dmKK;
        bb   = 1 - aa;
        aa3  = aa * aa * aa;
        bb3  = bb * bb * bb;

        ret.real(ret.real() + aa * pwa_coefs_real_kloAB + bb * pwa_coefs_real_khiAB
                 + ((aa3 - aa) * pwa_coefs_prime_real_kloAB + (bb3 - bb) * pwa_coefs_prime_real_khiAB) * (dmKK * dmKK)
                       / 6.0);
        ret.imag(ret.imag() + aa * pwa_coefs_imag_kloAB + bb * pwa_coefs_imag_khiAB
                 + ((aa3 - aa) * pwa_coefs_prime_imag_kloAB + (bb3 - bb) * pwa_coefs_prime_imag_khiAB) * (dmKK * dmKK)
                       / 6.0);
        khiAB = khiAC;
        mAB   = mAC;
    }
    return ret;
}*/

__device__ fpcomplex BE(fptype m12, fptype m13, fptype m23,unsigned int *indices){

     
    fptype coef            = RO_CACHE(cudaArray[RO_CACHE(indices[2])]);
    const fptype mpisq = 0.01947977;
    fptype Q = sqrt(m23 - 4*mpisq);
    fptype Omega = exp(-Q*coef);

    return fpcomplex(Omega,0.);

}

__device__ resonance_function_ptr ptr_to_RBW      = plainBW<1>;
__device__ resonance_function_ptr ptr_to_RBW_SYM  = plainBW<2>;
__device__ resonance_function_ptr ptr_to_GOUSAK   = gouSak<1>;
__device__ resonance_function_ptr ptr_to_GOUSAK_SYM = gouSak<2>;
__device__ resonance_function_ptr ptr_to_GAUSSIAN = gaussian;
__device__ resonance_function_ptr ptr_to_NONRES   = nonres;
__device__ resonance_function_ptr ptr_to_LASS     = lass;
__device__ resonance_function_ptr ptr_to_FLATTE   = flatte;
__device__ resonance_function_ptr ptr_to_SPLINE   = cubicspline;
__device__ resonance_function_ptr ptr_to_BoseEinstein = BE;

namespace Resonances {

RBW::RBW(std::string name,
         Variable ar,
         Variable ai,
         Variable mass,
         Variable width,
         unsigned int sp,
         unsigned int cyc,
         bool symmDP)
    : ResonancePdf(name, ar, ai) {
    pindices.push_back(registerParameter(mass));
    pindices.push_back(registerParameter(width));
    pindices.push_back(sp);
    pindices.push_back(cyc);
    pindices.push_back(symmDP);

    if(symmDP) {
        GET_FUNCTION_ADDR(ptr_to_RBW_SYM);
    } else {
        GET_FUNCTION_ADDR(ptr_to_RBW);
    }

    initialize(pindices);
}

GS::GS(std::string name, Variable ar, Variable ai, Variable mass, Variable width, unsigned int sp, unsigned int cyc,bool symmDP)
    : ResonancePdf(name, ar, ai) {
    pindices.push_back(registerParameter(mass));
    pindices.push_back(registerParameter(width));
    pindices.push_back(sp);
    pindices.push_back(cyc);
    pindices.push_back(symmDP);
    
    if(symmDP) {
        GET_FUNCTION_ADDR(ptr_to_GOUSAK_SYM);
    } else {
        GET_FUNCTION_ADDR(ptr_to_GOUSAK);
    }

    initialize(pindices);
}

LASS::LASS(std::string name, Variable ar, Variable ai, Variable mass, Variable width, unsigned int sp, unsigned int cyc)
    : ResonancePdf(name, ar, ai) {
    pindices.push_back(registerParameter(mass));
    pindices.push_back(registerParameter(width));
    pindices.push_back(sp);
    pindices.push_back(cyc);

    GET_FUNCTION_ADDR(ptr_to_LASS);

    initialize(pindices);
}

// Constructor for regular BW,Gounaris-Sakurai,LASS
Gauss::Gauss(std::string name, Variable ar, Variable ai, Variable mass, Variable width, unsigned int cyc)
    : ResonancePdf(name, ar, ai) {
    // Making room for index of decay-related constants. Assumption:
    // These are mother mass and three daughter masses in that order.
    // They will be registered by the object that uses this resonance,
    // which will tell this object where to find them by calling setConstantIndex.

    std::vector<unsigned int> pindices;
    pindices.push_back(0);
    pindices.push_back(registerParameter(mass));
    pindices.push_back(registerParameter(width));
    pindices.push_back(cyc);

    GET_FUNCTION_ADDR(ptr_to_GAUSSIAN);

    initialize(pindices);
}

NonRes::NonRes(std::string name, Variable ar, Variable ai)
    : ResonancePdf(name, ar, ai) {
    GET_FUNCTION_ADDR(ptr_to_NONRES);

    initialize(pindices);
}

FLATTE::FLATTE(std::string name,
               Variable ar,
               Variable ai,
               Variable mean,
               Variable g1,
               Variable rg2og1,
               unsigned int cyc,
               bool symmDP)
    : ResonancePdf(name, ar, ai) {
    pindices.push_back(registerParameter(mean));
    pindices.push_back(registerParameter(g1));
    pindices.push_back(registerParameter(rg2og1));
    pindices.push_back(cyc);
    pindices.push_back((unsigned int)symmDP);

    GET_FUNCTION_ADDR(ptr_to_FLATTE);

    initialize(pindices);
}

Spline::Spline(std::string name,
               Variable ar,
               Variable ai,
               std::vector<fptype> &HH_bin_limits,
               std::vector<Variable> &pwa_coefs_reals,
               std::vector<Variable> &pwa_coefs_imags,
               unsigned int cyc,
               bool symmDP)
    : ResonancePdf(name, ar, ai) {
    std::vector<unsigned int> pindices;
    const unsigned int nKnobs = HH_bin_limits.size();
    host_constants.resize(nKnobs);

    pindices.push_back(0);
    pindices.push_back(cyc);
    pindices.push_back((unsigned int)symmDP);
    pindices.push_back(nKnobs);

    for(int i = 0; i < pwa_coefs_reals.size(); i++) {
        host_constants[i] = HH_bin_limits[i];
        pindices.push_back(registerParameter(pwa_coefs_reals[i]));
        pindices.push_back(registerParameter(pwa_coefs_imags[i]));
    }
    pindices.push_back(registerConstants(nKnobs));

    MEMCPY_TO_SYMBOL(functorConstants,
                     host_constants.data(),
                     nKnobs * sizeof(fptype),
                     cIndex * sizeof(fptype),
                     cudaMemcpyHostToDevice);

    GET_FUNCTION_ADDR(ptr_to_SPLINE);

    initialize(pindices);
}

/*__host__ void Spline::recalculateCache() const {
    auto params           = getParameters();
    const unsigned nKnobs = params.size() / 2;
    std::vector<fpcomplex> y(nKnobs);
    std::vector<fptype> x(nKnobs);
    unsigned int i = 0;
    fptype prevvalue = 0;
    fptype prevang = 0;
    for(auto v = params.begin(); v != params.end(); ++v, ++i) {
        unsigned int idx = i / 2;
        fptype value     = host_params[v->getIndex()];
        if(i % 2 != 0){
            prevang = value;
            y[idx].real((prevvalue)*cos(prevang));
            y[idx].imag((prevvalue)*sin(prevang));
        }
        prevvalue = value;
    }
    std::vector<fptype> y2_flat = flatten(complex_derivative(host_constants, y));

    MEMCPY_TO_SYMBOL(cDeriatives, y2_flat.data(), 2 * nKnobs * sizeof(fptype), 0, cudaMemcpyHostToDevice);
}*/

__host__ void Spline::recalculateCache() const {
    auto params           = getParameters();
    const unsigned nKnobs = params.size() / 2;
    std::vector<fpcomplex> y(nKnobs);
    unsigned int i = 0;
    for(auto v = params.begin(); v != params.end(); ++v, ++i) {
        unsigned int idx = i / 2;
        fptype value     = host_params[v->getIndex()];
        if(i % 2 == 0){
            y[idx].real(value);
           
        }else{
            y[idx].imag(value);
        }
    }
    std::vector<fptype> y2_flat = flatten(complex_derivative(host_constants, y));

    MEMCPY_TO_SYMBOL(cDeriatives, y2_flat.data(), 2 * nKnobs * sizeof(fptype), 0, cudaMemcpyHostToDevice);
}

BoseEinstein::BoseEinstein(std::string name,Variable ar, Variable ai, Variable coef)
    : ResonancePdf(name, ar, ai) {
    pindices.push_back(registerParameter(coef));
    GET_FUNCTION_ADDR(ptr_to_BoseEinstein);

    initialize(pindices);

    
}

} // namespace Resonances

} // namespace GooFit
