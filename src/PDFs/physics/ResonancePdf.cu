#include <goofit/PDFs/detail/ComplexUtils.h>
#include <goofit/PDFs/physics/DalitzPlotHelpers.h>
#include <goofit/PDFs/physics/ResonancePdf.h>

#include <utility>
#include <iterator>

#include <Eigen/Core>
#include <Eigen/LU>

#include <goofit/detail/Macros.h>


//s = Mass Squared
//m = Sqrt(s)
//m0  = Resonance Mass

namespace GooFit {

__device__ fptype cDeriatives[2 * MAXNKNOBS];


//from Grace Young - New Package for RooFit Supporting Dalitz Analysis
__device__ fptype Momentum( const fptype &m,
                            const fptype &m1,
                            const fptype &m2
                            ) {
  
    fptype k1 = m*m - POW2(m1+m2);
    fptype k2 = m*m - POW2(m1-m2);
    fptype q = 0.5*sqrt(k1*k2)/m;

    return k1*k2>0 ? q : 0.;
}

//for a0_f0 mixing
__device__ fptype kallenFunction( const fptype &m,
    const fptype &m1,
    const fptype &m2
    ) {

    fptype q = m*m + m1*m1 + m2*m2 - 2*m*m1 - 2*m*m2 - 2*m1*m2;
    q = q>0.?q:0.;

    return sqrt(q)/m;
}
//for a0_f0 mixing
__device__ fptype rhoBC( const fptype &m,
    const fptype &m1,
    const fptype &m2
    ) {

    fptype q = m*m + m1*m1 + m2*m2 - 2*m*m1 - 2*m*m2 - 2*m1*m2;
    q = q>0.?q:0.;

    return 0.5*sqrt(q)/sqrt(m);
}

//from Grace Young - New Package for RooFit Supporting Dalitz Analysis
__device__ fptype BWFactors(const fptype &q,const fptype &q0, unsigned int &spin , const fptype &meson_radius){

    fptype R = meson_radius;
    fptype B = 1.;

    if(spin==1){
        B*= sqrt((1+POW2(q0*R))/(1+POW2(q*R)));
    }

    if(spin==2){
        B*= POW2(POW2(R*q0)-3) + 9*POW2(R*q0);
        B/= POW2(POW2(R*q)-3) + 9*POW2(R*q);
        B = sqrt(B);
    }
    

    return B;

}


//from Grace Young - New Package for RooFit Supporting Dalitz Analysis
__device__ fptype Gamma(const fptype &m,
                        const fptype &m0,
                        const fptype &width, 
                        const fptype &q,    
                        const fptype &q0,
                        const fptype &BWFactor,
                        const unsigned int &spin){

                        fptype g = 1.;

                        if(spin==0){
                            g*= width*(q/q0)*(m0/m)*POW2(BWFactor);
                        }

                        if(spin==1){
                            g*= width*pow(q/q0 , 2*1 + 1)*(m0/m)*POW2(BWFactor);
                        }

                        if(spin==2){
                            g*= width*pow(q/q0 , 2*2 + 1)*(m0/m)*POW2(BWFactor);
                        }

                        return g;
}



__device__ fptype h(const fptype &m,const fptype &q){
    const fptype mpi = 0.13957018;
    return (2*q/M_PI*m)*log( (m+2*q)/2*mpi );
}

__device__ fptype h_prime(const fptype &m0,const fptype &q0){
    return (h(m0,q0)*( (1./8*q0*q0) - (1./2*m0*m0) ) + (1./2*M_PI*m0*m0));
}

__device__ fptype d(const fptype &m0,const fptype &q0){
    const fptype mpi = 0.13957018;
    return ((3.*POW2(mpi)/M_PI*POW2(q0))*log( (m0+2*q0)/2*mpi) + (m0/2.*M_PI*q0) - (POW2(mpi)*m0/M_PI*POW2(q0)*q0));
}

__device__ fptype f(const fptype &m, const fptype &m0,const fptype &width , const fptype &q, const fptype &q0){
    return width*(POW2(m0)/POW2(q0)*q0)*( POW2(q)*(h(m,q)-h(m0,q0)) + (POW2(m0)-POW2(m))*q0*q0*h_prime(m0,q0));
}

//from Grace Young - New Package for RooFit Supporting Dalitz Analysis
__device__ fptype spinFactor(unsigned int spin,
                             fptype motherMass,
                             fptype daug1Mass,
                             fptype daug2Mass,
                             fptype daug3Mass,
                             fptype m12,
                             fptype m13,
                             fptype m23,
                             unsigned int cyclic_index) {

    fptype ret;

    fptype _mA  = (PAIR_12 == cyclic_index ? daug1Mass : (PAIR_13 == cyclic_index ? daug3Mass : daug2Mass));
    fptype _mB  = (PAIR_12 == cyclic_index ? daug2Mass : (PAIR_13 == cyclic_index ? daug1Mass : daug3Mass));
    fptype _mC  = (PAIR_12 == cyclic_index ? daug3Mass : (PAIR_13 == cyclic_index ? daug2Mass : daug1Mass));
    fptype _mAC = (PAIR_12 == cyclic_index ? m13 : (PAIR_13 == cyclic_index ? m23 : m12));
    fptype _mBC = (PAIR_12 == cyclic_index ? m23 : (PAIR_13 == cyclic_index ? m12 : m13));
    fptype _mAB = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));

    if(0 == spin){
        ret = 1.;
    }

    if(1 == spin){

        fptype massFactor = 1.0 / _mAB;
        ret = ((_mBC - _mAC) + (massFactor * (motherMass * motherMass - _mC * _mC) * (_mA * _mA - _mB * _mB)));
        
    }
    
    
    if(2 == spin) {
        fptype massFactor = 1.0 / _mAB;
        fptype a1 = ((_mBC - _mAC) + (massFactor * (motherMass * motherMass - _mC * _mC) * (_mA * _mA - _mB * _mB)));
        fptype a2 = ((_mAB - (2 * motherMass * motherMass) - (2 * _mC * _mC))
        + massFactor * POW2(motherMass * motherMass - _mC * _mC));
        fptype a3 = ((_mAB - (2 * _mA * _mA) - (2 * _mB * _mB)) + massFactor * POW2(_mA * _mA - _mB * _mB));
        
        ret = POW2(a1) - a2*a3/3;
        
    }

    return ret;
}

//from Grace Young - New Package for RooFit Supporting Dalitz Analysis
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

    fpcomplex result(0., 0.);

#pragma unroll
    for(int i = 0; i < I; i++) {
        fptype s    = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
        fptype m = sqrt(s);
        fptype m1 = PAIR_23 == cyclic_index ? c_daug2Mass : c_daug1Mass;
        fptype m2 = PAIR_12 == cyclic_index ? c_daug2Mass : c_daug3Mass;
        fptype m3 = PAIR_23 == cyclic_index ? c_daug1Mass : (PAIR_13 == cyclic_index?c_daug2Mass:c_daug3Mass);

        fptype resmass2 = POW2(resmass);
        fptype q  = Momentum(m,m1,m2);
        fptype q0 = Momentum(resmass,m1,m2);
        fptype BWFactors_Res = BWFactors(q,q0,spin,c_meson_radius);

        fptype qD = Momentum(c_motherMass,m,m3);
        fptype qD0 = Momentum(c_motherMass,resmass,m3);
        fptype BWFactors_D = BWFactors(qD,qD0,spin,5.);

        fptype gamma = Gamma(m,resmass,reswidth,q,q0,BWFactors_Res,spin);


        // RBW evaluation
        fptype A = (resmass2 - s);
        fptype B = resmass*gamma;
        fptype C = 1.0 / (POW2(A) + POW2(B));

        fpcomplex ret(A * C, B * C); // Dropping F_D=1

        ret *= BWFactors_Res*BWFactors_D;
        ret *= spinFactor(spin, c_motherMass, c_daug1Mass, c_daug2Mass, c_daug3Mass, m12, m13, m23, cyclic_index);

        result += ret;
                                          
        if(I != 0) {
            fptype swpmass = m12;
            m12            = m13;
            m13            = swpmass;
        }
    }

    return reswidth>0 ? result : 0;
}

//from Laura++
template <int I>
__device__ fpcomplex Pole(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
    fptype c_motherMass   = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 0]);
    fptype c_daug1Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 1]);
    fptype c_daug2Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 2]);
    fptype c_daug3Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 3]);
    
    fptype real            = RO_CACHE(cudaArray[RO_CACHE(indices[2])]);
    fptype img           = RO_CACHE(cudaArray[RO_CACHE(indices[3])]);
    unsigned int spin         = RO_CACHE(indices[4]);
    unsigned int cyclic_index = RO_CACHE(indices[5]);
    
    fpcomplex result(0., 0.);
    

#pragma unroll
    for(int i = 0; i < I; i++) {
        fptype s    = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
      
        
        fptype reTerm = (real*real - img*img) - s;
        fptype imTerm = 2.0*real*img;

	fptype scale = 1./(reTerm*reTerm + imTerm*imTerm);
	fpcomplex ret(reTerm*scale,imTerm*scale);
	

	fptype angular = spinFactor(spin, c_motherMass, c_daug1Mass, c_daug2Mass, c_daug3Mass, m12, m13, m23, cyclic_index);
    ret *= angular;
	
        result += ret;

                                          
        if(I != 0) {
            fptype swpmass = m12;
            m12            = m13;
            m13            = swpmass;
        }
    }

    return result;
}

//from GooFit
__device__ fpcomplex gaussian(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
    // indices[1] is unused constant index, for consistency with other function types.
    fptype resmass            = RO_CACHE(cudaArray[RO_CACHE(indices[2])]);
    fptype reswidth           = RO_CACHE(cudaArray[RO_CACHE(indices[3])]);
    unsigned int cyclic_index = indices[4];

    // Notice sqrt - this function uses mass, not mass-squared like the other resonance types.
    fptype s = sqrt(PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
    s -= resmass;
    s /= reswidth;
    s *= s;
    fptype ret = exp(-0.5 * s);

    ret /= reswidth;

    return {ret, 0};
}

//from B ± → π ± π +π  amplitude analysis on run1 data paper
template<int I>
__device__ fpcomplex gouSak(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
    fptype c_motherMass   = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 0]);
    fptype c_daug1Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 1]);
    fptype c_daug2Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 2]);
    fptype c_daug3Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 3]);
    fptype c_meson_radius = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 4]);

    fptype resmass            = RO_CACHE(cudaArray[RO_CACHE(indices[2])]);
    fptype reswidth           = RO_CACHE(cudaArray[RO_CACHE(indices[3])]);
    unsigned int spin         = RO_CACHE(indices[4]);
    unsigned int cyclic_index = RO_CACHE(indices[5]);

    fpcomplex result(0., 0.);
    

    #pragma unroll
    for(int i = 0; i < I; i++) {
        fptype s    = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
        fptype m = sqrt(s);
        fptype m1 = PAIR_23 == cyclic_index ? c_daug2Mass : c_daug1Mass;
        fptype m2 = PAIR_12 == cyclic_index ? c_daug2Mass : c_daug3Mass;
        fptype m3 = PAIR_23 == cyclic_index ? c_daug1Mass : (PAIR_13 == cyclic_index?c_daug2Mass:c_daug3Mass);
       
        fptype resmass2 = POW2(resmass);
        fptype q  = Momentum(m,m1,m2);
        fptype q0 = Momentum(resmass,m1,m2);
        fptype BWFactors_Res = BWFactors(q,q0,spin,c_meson_radius);

        fptype qD = Momentum(c_motherMass,m,m3);
        fptype qD0 = Momentum(c_motherMass,resmass,m3);
        fptype BWFactors_D = BWFactors(qD,qD0,spin,5.);

        fptype gamma = Gamma(m,resmass,reswidth,q,q0,BWFactors_Res,spin);

        
        fptype d_ = d(resmass,q0);
        fptype f_ = f(m,resmass,reswidth,q,q0);

        fptype A = (resmass2-s) + f_;
        fptype B = resmass*gamma;
        fptype C = 1./(A*A + B*B);
        fptype D = 1+(reswidth*d_/resmass);

        fpcomplex retur(A*C,B*C);
        fptype angular = spinFactor(spin, c_motherMass, c_daug1Mass, c_daug2Mass, c_daug3Mass, m12, m13, m23, cyclic_index);
        retur*= D*BWFactors_Res*BWFactors_D*angular;


        result += retur;

        if(I != 0) {
            fptype swpmass = m12;
            m12            = m13;
            m13            = swpmass;
        }

   }

    return reswidth>0  ? result : 0;
}

//from B ± → π ± π +π  amplitude analysis on run1 data paper and Laura++
template<int I>
__device__ fpcomplex RhoOmegaMix(fptype m12, fptype m13, fptype m23, unsigned int *indices){
    
    fptype c_motherMass   = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 0]);
    fptype c_daug1Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 1]);
    fptype c_daug2Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 2]);
    fptype c_daug3Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 3]);
    fptype c_meson_radius = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 4]);

    fptype real            = RO_CACHE(cudaArray[RO_CACHE(indices[2])]);
    fptype img           = RO_CACHE(cudaArray[RO_CACHE(indices[3])]);
    fptype delta           = RO_CACHE(cudaArray[RO_CACHE(indices[4])]);

    unsigned int spin         = RO_CACHE(indices[5]);
    unsigned int cyclic_index = RO_CACHE(indices[6]);
    
  
    fpcomplex result(0., 0.);
    
    const fptype rho_mass = 0.77526;
    const fptype rho_width = 0.1478;
    const fptype omega_mass = 0.78265;
    const fptype omega_width = 0.00849; 

    fptype Delta_= delta*(rho_mass + omega_mass);
    fpcomplex Bterm(real,img);
    Bterm*=Delta_;
    fpcomplex unity(1.0,0.0);
    
#pragma unroll
    for(int i = 0; i < I; i++) {
        fptype s    = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
        fptype m = sqrt(s);
        fptype m1 = PAIR_23 == cyclic_index ? c_daug2Mass : c_daug1Mass;
        fptype m2 = PAIR_12 == cyclic_index ? c_daug2Mass : c_daug3Mass;
        fptype m3 = PAIR_23 == cyclic_index ? c_daug1Mass : (PAIR_13 == cyclic_index?c_daug2Mass:c_daug3Mass);

	    fptype angular = spinFactor(spin, c_motherMass, c_daug1Mass, c_daug2Mass, c_daug3Mass, m12, m13, m23, cyclic_index);

        // Omega RBW evaluation
        fptype A = (POW2(omega_mass) - s);
        fptype B = omega_mass*omega_width;
        fptype C = 1.0 / (POW2(A) + POW2(B));
        fpcomplex omega(A * C, B * C); 
        

        //Rho GS evaluation

        fptype q  = Momentum(m,m1,m2);
        fptype q0 = Momentum(rho_mass,m1,m2);
        fptype BWFactors_Res = BWFactors(q,q0,spin,c_meson_radius);

        fptype qD = Momentum(c_motherMass,m,m3);
        fptype qD0 = Momentum(c_motherMass,rho_mass,m3);
        fptype BWFactors_D = BWFactors(qD,qD0,spin,5.);

        fptype gamma = Gamma(m,rho_mass,rho_width,q,q0,BWFactors_Res,spin);

        fptype d_ = d(rho_mass,q0);
        fptype f_ = f(m,rho_mass,rho_width,q,q0);

        A = (POW2(rho_mass)-s) + f_;
        B = rho_mass*gamma;
        C = 1./(A*A + B*B);
        fptype D = 1+(rho_width*d_/rho_mass);

        fpcomplex rho(A*C,B*C);
        rho*= D*BWFactors_Res*BWFactors_D*angular;

         //end of Gousak

        //rho-omega mix
        fpcomplex mixingTerm = Bterm*omega + unity;
        result += rho*mixingTerm;     
                                          
        if(I != 0) {
            fptype swpmass = m12;
            m12            = m13;
            m13            = swpmass;
        }
    }

    return result;

}

//from GooFit
__device__ fpcomplex lass(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
    fptype motherMass   = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 0]);
    fptype m1    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 1]);
    fptype m2    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 2]);
    fptype m3    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 3]);
    fptype meson_radius = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 4]);

    fptype resmass            = RO_CACHE(cudaArray[RO_CACHE(indices[2])]);
    fptype reswidth           = RO_CACHE(cudaArray[RO_CACHE(indices[3])]);
    unsigned int spin         = RO_CACHE(indices[4]);
    unsigned int cyclic_index = RO_CACHE(indices[5]);

    fptype s  = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
    fptype m  = sqrt(s);
    
    fptype q  = Momentum(m,m1,m2);
    fptype q0 = Momentum(resmass,m1,m2);
    fptype BWFactors_Res = BWFactors(q,q0,spin,meson_radius);

    fptype qD = Momentum(motherMass,m,m3);
    fptype qD0 = Momentum(motherMass,resmass,m3);
    fptype BWFactors_D = BWFactors(qD,qD0,spin,5.);

    fptype g = Gamma(m,resmass,reswidth,q,q0,BWFactors_Res,spin);
    
    fptype _a    = 0.22357;
    fptype _r    = -15.042;
    fptype _R    = 1; 
    fptype _phiR = 1.10644;
    fptype _B    = 0.614463;
    fptype _phiB = -0.0981907;

    // background phase motion
    fptype cot_deltaB  = (1.0 / (_a * q)) + 0.5 * _r * q;
    fptype qcot_deltaB = (1.0 / _a) + 0.5 * _r * q * q;

    // calculate resonant part
    fpcomplex expi2deltaB = fpcomplex(qcot_deltaB, q) / fpcomplex(qcot_deltaB, -q);
    fpcomplex resT        = fpcomplex(cos(_phiR + 2 * _phiB), sin(_phiR + 2 * _phiB)) * _R;

    fpcomplex prop = fpcomplex(1, 0) / fpcomplex(resmass - s, sqrt(resmass) * g);
    // resT *= prop*m0*_g0*m0/twoBodyCMmom(m0*m0, _trackinfo[i])*expi2deltaB;
    resT *= prop * (resmass * reswidth / q0) * expi2deltaB;

    // calculate bkg part
    resT += fpcomplex(cos(_phiB), sin(_phiB)) * _B * (cos(_phiB) + cot_deltaB * sin(_phiB)) * sqrt(s)
            / fpcomplex(qcot_deltaB, -q);

    resT *= BWFactors_Res*BWFactors_D;
    resT *= spinFactor(spin, motherMass, m1, m2, m3, m12, m13, m23, cyclic_index);

    return reswidth>0? resT: 0.;
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


//From Laura++
__device__ fpcomplex flatte(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
    // indices[1] is unused constant index, for consistency with other function types.
    fptype resmass            = cudaArray[indices[2]];
    fptype g1                 = cudaArray[indices[3]];
    fptype g2                 = cudaArray[indices[4]];
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
    
    fptype rho1(0.0), rho2(0.0);
    
    for(int i = 0; i < 1 + doSwap; i++) {
        fptype s = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
        fptype resmass2 = POW2(resmass);
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

    return (g1>0 && g2>0) ? ret : 0.;
}

//from http://arxiv.org/abs/1409.2213v4 and http://arxiv.org/abs/0704.3652v3
template<int I>
__device__ fpcomplex a0_f0_Mixing(fptype m12, fptype m13, fptype m23, unsigned int *indices){

    fptype c_motherMass   = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 0]);
    fptype c_meson_radius = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 4]);
    unsigned int cyclic_index = indices[7];
  
    //parameters
    fptype g_a_kk                 = cudaArray[indices[2]];
    fptype g_f_kk                 = cudaArray[indices[3]];
    fptype g_a_eta_pi             = cudaArray[indices[5]];
    fptype g_f_pi_pi              = cudaArray[indices[6]];

    const fptype pipmass = 0.13957018;
    const fptype pimass  = 0.13804;
    const fptype kpmass  = 0.493677;
    const fptype k0mass  = 0.497614;
    const fptype kmass   = 0.49565;
    const fptype eta     = 0.54785;

    const fptype a0_resmass = 0.998; //CLEO
    const fptype f0_resmass = 0.977; //KLOE

    //couplings in isospin basis
    

    fpcomplex ret(0.,0.);

    #pragma unroll
    for(int i = 0; i < I; i++) {
        fptype s    = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
        fptype m = sqrt(s);

        fptype q_1 = kallenFunction(s,POW2(kpmass),POW2(kpmass));
        fptype q_2 = kallenFunction(s,POW2(k0mass),POW2(k0mass));

        fpcomplex lambda_1_p_2 = fpcomplex(0.,-1.)*(1./(16.*M_PI))*g_a_kk*g_f_kk*(q_1-q_2);

        //photon_exchange
        fptype q2 = ( POW2(2*kpmass) - s)/POW2(kpmass);
        q2 = q2>0.?q2:1.;
        fptype photon_ex = (-1./(137.*32.*M_PI))*( log(q2) + log(2.) + (21.*1.20205)/(2*POW2(M_PI)) );

        //lambda total
        fpcomplex lambda = lambda_1_p_2 + g_a_kk*photon_ex*g_f_kk;

        //flatte for a0
        fptype q_eta_pi = rhoBC(s,POW2(eta),POW2(pipmass));
        fptype gamma_eta_pi = POW2(g_a_eta_pi)*(q_eta_pi/(8.*M_PI*s));
        fptype q_k_k = rhoBC(s,POW2(kmass),POW2(kmass));
        fptype gamma_k_k = POW2(g_a_kk)*(q_k_k/(8.*M_PI*s));
        fptype A = s - POW2(a0_resmass);
        fptype B = m*(gamma_eta_pi+gamma_k_k);
        fpcomplex Da(A,B);

        //flatte for f0
        fptype q_pi_pi = rhoBC(s,POW2(pimass),POW2(pimass));
        fptype gamma_pi_pi = POW2(g_f_pi_pi)*q_pi_pi/(8.*M_PI*s);
        gamma_k_k = POW2(g_f_kk)*(q_k_k/(8.*M_PI*s));
        A = s - POW2(f0_resmass);
        B = m*(gamma_pi_pi+gamma_k_k);
        fpcomplex Df(A,B);
      
        fpcomplex mix = Da/(Df*Da - POW2(lambda)) ; //if you want a0-propagator with mixing just
                                                    // change Da by Df in numerator!
                                                    //if you want the mixing-propagtor change Da by lambda in numerator!
       
        ret += mix;
        
        //printf("m12= %f : Df = (%f,%f) Da=(%f,%f) ret=(%f,%f) \n",m12,Df.real(),Df.imag(),Da.real(),Da.imag(),ret.real(),ret.imag());
        if(I != 0) {
            fptype swpmass = m12;


            m12            = m13;
            m13            = swpmass;
        }

    }

    return ret;

}

//From GooFit
__device__ fpcomplex cubicspline(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
    fpcomplex ret(0, 0);
    unsigned int cyclic_index        = indices[2];
    unsigned int doSwap              = indices[3];
    const unsigned int nKnobs                   = indices[4];
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
     
      
          /*printf("m12: %f: %f %f %f %f %f %f %d %d %d\n", mAB, mKKlimits[twokloAB], mKKlimits[twokhiAB],
          pwa_coefs_real_khiAB, pwa_coefs_imag_khiAB, pwa_coefs_prime_real_khiAB, pwa_coefs_prime_imag_khiAB, khiAB,
          khiAC, timestorun );*/
       
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

//from GooFit
__device__ fpcomplex cubicsplinePolar(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
    fpcomplex ret(0., 0.);
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
    return ret ;
}

//From Rio Amp Analysis Package
__device__ fpcomplex BE(fptype m12, fptype m13, fptype m23,unsigned int *indices){

     
    fptype coef            = RO_CACHE(cudaArray[RO_CACHE(indices[2])]);
    const fptype mpisq = 0.01947977;
    fptype Q = sqrt(m23 - 4*mpisq);
    fptype Omega = exp(-Q*coef);

    return fpcomplex(Omega,0.);

}

__device__ resonance_function_ptr ptr_to_RBW      = plainBW<1>;
__device__ resonance_function_ptr ptr_to_RBW_SYM  = plainBW<2>;
__device__ resonance_function_ptr ptr_to_POLE      = Pole<1>;
__device__ resonance_function_ptr ptr_to_POLE_SYM  = Pole<2>;
__device__ resonance_function_ptr ptr_to_GOUSAK   = gouSak<1>;
__device__ resonance_function_ptr ptr_to_GOUSAK_SYM = gouSak<2>;
__device__ resonance_function_ptr ptr_to_GAUSSIAN = gaussian;
__device__ resonance_function_ptr ptr_to_NONRES   = nonres;
__device__ resonance_function_ptr ptr_to_RHOOMEGAMIX      = RhoOmegaMix<1>;
__device__ resonance_function_ptr ptr_to_RHOOMEGAMIX_SYM  = RhoOmegaMix<2>;
__device__ resonance_function_ptr ptr_to_LASS     = lass;
__device__ resonance_function_ptr ptr_to_FLATTE   = flatte;
__device__ resonance_function_ptr ptr_to_f0_MIXING = a0_f0_Mixing<1>;
__device__ resonance_function_ptr ptr_to_f0_MIXING_SYM = a0_f0_Mixing<2>;
__device__ resonance_function_ptr ptr_to_SPLINE   = cubicspline;
__device__ resonance_function_ptr ptr_to_SPLINE_POLAR   = cubicsplinePolar;
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


POLE::POLE(std::string name,
         Variable ar,
         Variable ai,
	     Variable real,
         Variable img,
         unsigned int sp,
         unsigned int cyc,
         bool symmDP)
    : ResonancePdf(name, ar, ai) {
pindices.push_back(registerParameter(real));
    pindices.push_back(registerParameter(img));
    pindices.push_back(sp);
    pindices.push_back(cyc);
    pindices.push_back(symmDP);

    if(symmDP) {
        GET_FUNCTION_ADDR(ptr_to_POLE_SYM);
    } else {
        GET_FUNCTION_ADDR(ptr_to_POLE);
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

RHOOMEGAMIX::RHOOMEGAMIX(std::string name,
         Variable ar,
         Variable ai,
	     Variable real,
         Variable img,
         Variable delta,
         unsigned int sp,
         unsigned int cyc,
         bool symmDP)
    : ResonancePdf(name, ar, ai) {
    pindices.push_back(registerParameter(real));
    pindices.push_back(registerParameter(img));
    pindices.push_back(registerParameter(delta));
    pindices.push_back(sp);
    pindices.push_back(cyc);
    pindices.push_back(symmDP);

    if(symmDP) {
        GET_FUNCTION_ADDR(ptr_to_RHOOMEGAMIX_SYM);
    } else {
        GET_FUNCTION_ADDR(ptr_to_RHOOMEGAMIX);
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

f0_MIXING::f0_MIXING(std::string name,
    Variable ar,
    Variable ai,
    Variable g1,
    Variable g2,
    Variable g3,
    Variable g4,
    unsigned int cyc,
    bool symmDP)
: ResonancePdf(name, ar, ai) {
pindices.push_back(registerParameter(g1));
pindices.push_back(registerParameter(g2));
pindices.push_back(registerParameter(g3));
pindices.push_back(registerParameter(g4));
pindices.push_back(cyc);
pindices.push_back((unsigned int)symmDP);

if(symmDP) {
    GET_FUNCTION_ADDR(ptr_to_f0_MIXING_SYM);
} else {
    GET_FUNCTION_ADDR(ptr_to_f0_MIXING);
}

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
    std::vector<fpcomplex> y(nKnobs);

    pindices.push_back(0);
    pindices.push_back(cyc);
    pindices.push_back((unsigned int)symmDP);
    pindices.push_back(nKnobs);

    for(int i = 0; i < pwa_coefs_reals.size(); i++) {
        host_constants[i] = HH_bin_limits[i];
        pindices.push_back(registerParameter(pwa_coefs_reals[i]));
        pindices.push_back(registerParameter(pwa_coefs_imags[i]));
        y[i].real(pwa_coefs_reals[i].getValue());
        y[i].imag(pwa_coefs_imags[i].getValue());
    }
    pindices.push_back(registerConstants(nKnobs));
    std::vector<fptype> y2_flat = flatten(complex_derivative(host_constants, y));
    MEMCPY_TO_SYMBOL(cDeriatives, y2_flat.data(), 2 * nKnobs * sizeof(fptype), 0, cudaMemcpyHostToDevice);
    
    MEMCPY_TO_SYMBOL(functorConstants,
                     host_constants.data(),
                     nKnobs * sizeof(fptype),
                     cIndex * sizeof(fptype),
                     cudaMemcpyHostToDevice);
    
    GET_FUNCTION_ADDR(ptr_to_SPLINE);

    initialize(pindices);

   
}

SplinePolar::SplinePolar(std::string name,
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
    std::vector<fpcomplex> y(nKnobs);

    pindices.push_back(0);
    pindices.push_back(cyc);
    pindices.push_back((unsigned int)symmDP);
    pindices.push_back(nKnobs);

    for(int i = 0; i < pwa_coefs_reals.size(); i++) {
        host_constants[i] = HH_bin_limits[i];
        pindices.push_back(registerParameter(pwa_coefs_reals[i]));
        pindices.push_back(registerParameter(pwa_coefs_imags[i]));
        y[i].real( (pwa_coefs_reals[i].getValue())*cos(pwa_coefs_imags[i].getValue()) );
        y[i].imag( (pwa_coefs_reals[i].getValue())*sin(pwa_coefs_imags[i].getValue()) );
    }
    pindices.push_back(registerConstants(nKnobs));
    std::vector<fptype> y2_flat = flatten(complex_derivative(host_constants, y));
    MEMCPY_TO_SYMBOL(cDeriatives, y2_flat.data(), 2 * nKnobs * sizeof(fptype), 0, cudaMemcpyHostToDevice);

    MEMCPY_TO_SYMBOL(functorConstants,
                     host_constants.data(),
                     nKnobs * sizeof(fptype),
                     cIndex * sizeof(fptype),
                     cudaMemcpyHostToDevice);

   
    GET_FUNCTION_ADDR(ptr_to_SPLINE_POLAR);

    initialize(pindices);
}

BoseEinstein::BoseEinstein(std::string name,Variable ar, Variable ai, Variable coef)
    : ResonancePdf(name, ar, ai) {
    pindices.push_back(registerParameter(coef));
    GET_FUNCTION_ADDR(ptr_to_BoseEinstein);

    initialize(pindices);

    
}





} // namespace Resonances

} // namespace GooFit
