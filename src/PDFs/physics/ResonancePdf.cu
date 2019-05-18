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


__device__ fptype  lambda (double x, double y, double z){

	double l;
	l = (x - y - z)*(x - y - z) - 4*y*z;

	return l;

}

__device__ fptype Form_Factor_Mother_Decay(int spin, double M, double sab, double mcsq, double mR){
    
    double s = M*M, mRsq = mR*mR;
    double fD, fD0, pstr, pstr0, q2 =0;
    double const rD2 = 25.0;
    double ret;
    
    if (spin == 0){
         ret = 1;
    }
    
    if (spin == 1) {
        pstr0 = sqrt(lambda(s,mRsq,mcsq))/(2*M);
        q2 = rD2*pstr0*pstr0;
        fD0 = sqrt(1 + q2);
        
        pstr = sqrt(lambda(s,sab,mcsq))/(2*M);
        q2 = rD2*pstr*pstr;
        fD = fD0/sqrt(1 + q2);
        ret = fD;
    }
   
    if(spin == 2){
        pstr0 = sqrt(lambda(s,mRsq,mcsq))/(2*M);
        q2 = rD2*pstr0*pstr0;
        fD0 = sqrt(9 + 3*q2 + q2*q2);
        
        pstr = sqrt(lambda(s,sab,mcsq))/(2*M);
        q2 = rD2*pstr*pstr;
        fD = fD0/sqrt(9 + 3*q2 + q2*q2);
        ret = fD;
    }

    return ret;
    
}

__device__ fptype Form_Factor_Resonance_Decay(int spin, double mR, double sab, double masq, double mbsq){

	double mRsq = mR*mR;
    double fR, fR0, pstr, pstr0, q2 = 0;
   
    double const rR2 = 2.25;
    double ret=-1;

	if (spin == 0){
        ret =1;
    }
    
    if (spin == 1) {

		pstr0 = sqrt(lambda(mRsq,masq,mbsq))/(2*mR);
		q2 = rR2*pstr0*pstr0;
		fR0 = sqrt(1 + q2);

		pstr = sqrt(lambda(sab,masq,mbsq))/(2*sqrt(sab));
		q2 = rR2*pstr*pstr;
		fR = fR0/sqrt(1 + q2);

		ret = fR;

	}
    
    if(spin == 2){

		pstr0 = sqrt((mRsq - masq - mbsq)*(mRsq - masq - mbsq) - 4*masq*mbsq)/(2*mR);
		q2 = rR2*pstr0*pstr0;
		fR0 = sqrt(9 + 3*q2 + q2*q2);

		//pstr = sqrt(lambda(sab,masq,mbsq))/(2*sqrt(sab));
		pstr = sqrt((sab - masq + mbsq)*(sab - masq + mbsq) - 4*sab*mbsq)/(2*sqrt(sab));
		q2 = rR2*pstr*pstr;
		fR = fR0/sqrt(9 + 3*q2 + q2*q2);

		ret = fR;
    }
    return ret;

}

__device__ fptype Angular_Distribution(int l, double M, double ma, double mb, double mc, double sab, double sbc, double sac){
    
    double spin, spin1, spin2, A, B, C, D = 0;
    
    spin1 = sbc - sac + (M*M-mc*mc)*(ma*ma-mb*mb)/sab;
    A = sab-2*M*M-2*mc*mc;
    B = ((M*M-mc*mc)*(M*M-mc*mc))/sab;
    C = sab-2*ma*ma-2*mb*mb;
    D = (ma*ma-mb*mb)*(ma*ma-mb*mb)/sab;
    spin2 = spin1*spin1-((A+B)*(C+D))/3.;
    if (l==0) spin=1.0;
    else if (l==1) spin = -0.5*spin1;
    else if (l==2) spin = 3./8*spin2;
 
	return spin;

}

__device__ fptype Gamma(int spin, double mR, double width, double mab, double masq, double mbsq){

    double pstr, pstr0,fR, mRsq = mR*mR, sab = mab;
    double ret=-1;

	pstr0 = sqrt(lambda(mRsq,masq,mbsq))/(2*mR);
	pstr = sqrt(lambda(sab,masq,mbsq))/(2*mab);
	if (spin == 0){
        ret =  width*(pstr/pstr0)*(mR/mab);
    }
    
    if (spin == 1){
		fR = Form_Factor_Resonance_Decay(spin, mR, sab, masq, mbsq);
		ret = width*pow((pstr/pstr0),3)*(mR/mab)*fR*fR;
    }
    
    if (spin == 2){
		fR = Form_Factor_Resonance_Decay(spin, mR, sab, masq, mbsq);
		ret = width*pow((pstr/pstr0),5)*(mR/mab)*fR*fR;
    }
    
    return ret;

}

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

   
    fptype _mA  = (PAIR_12 == cyclic_index ? daug1Mass : (PAIR_13 == cyclic_index ? daug3Mass : daug2Mass));
    fptype _mB  = (PAIR_12 == cyclic_index ? daug2Mass : (PAIR_13 == cyclic_index ? daug1Mass : daug3Mass));
    fptype _mC  = (PAIR_12 == cyclic_index ? daug3Mass : (PAIR_13 == cyclic_index ? daug2Mass : daug1Mass));
    fptype _mAC = (PAIR_12 == cyclic_index ? m13 : (PAIR_13 == cyclic_index ? m23 : m12));
    fptype _mBC = (PAIR_12 == cyclic_index ? m23 : (PAIR_13 == cyclic_index ? m12 : m13));
    fptype _mAB = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));

    fptype massFactor = 1.0 / _mAB;
    fptype sFactor    = -1;
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







//Lineshapes


template <int I>
__device__ fpcomplex plainBW(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
    fptype motherMass   = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 0]);
    fptype daug1Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 1]);
    fptype daug2Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 2]);
    fptype daug3Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 3]);
    //fptype meson_radius = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 4]);

    fptype resmass            = RO_CACHE(cudaArray[RO_CACHE(indices[2])]);
    fptype reswidth           = RO_CACHE(cudaArray[RO_CACHE(indices[3])]);
    unsigned int spin         = RO_CACHE(indices[4]);
    unsigned int cyclic_index = RO_CACHE(indices[5]);
    //unsigned int symmDP       = RO_CACHE(indices[6]);

    fpcomplex ret(0., 0.);
    
#pragma unroll
    for(int i = 0; i < I; i++) {
        fptype mass2  = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));      
        
        fptype FF_MD = Form_Factor_Mother_Decay(spin, motherMass, mass2, 
			PAIR_12==cyclic_index ? POW2(daug3Mass) : (PAIR_13==cyclic_index ? POW2(daug2Mass) : POW2(daug1Mass))
			,resmass);

        fptype FF_RD = Form_Factor_Resonance_Decay(spin, resmass, mass2,
		       	PAIR_12==cyclic_index ? POW2(daug1Mass) : (PAIR_23==cyclic_index ?  POW2(daug2Mass) :  POW2(daug2Mass)),
			PAIR_12==cyclic_index ? POW2(daug2Mass) : (PAIR_23==cyclic_index ?  POW2(daug3Mass) :  POW2(daug1Mass)));
        
	    fptype Angular = Angular_Distribution(spin, motherMass, daug1Mass, daug2Mass,daug3Mass, m12, m23, m13);
        
        fptype Width = Gamma(spin, resmass, reswidth, mass2,
            PAIR_12==cyclic_index ? POW2(daug1Mass) : (PAIR_23==cyclic_index ?  POW2(daug2Mass) :  POW2(daug3Mass)),
            PAIR_12==cyclic_index ? POW2(daug2Mass) : (PAIR_23==cyclic_index ?  POW2(daug3Mass) :  POW2(daug1Mass))
        );
   
        // RBW evaluation
        fptype A =  mass2 - POW2(resmass) ;
        fptype B = resmass * Width ;
        fptype C = 1.0 / (POW2(A) + POW2(B));
        fpcomplex _BW(A * C, B * C); 

        _BW *= Angular*FF_MD*FF_RD*reswidth*resmass;

        ret+=_BW;

        if(I ==2 ) {
            fptype swpmass = m12;
            m12            = m13;
            m13            = swpmass;
        }
    }

    return ret;
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

    double const mpi = 0.13956995;

    fpcomplex ret(0.0,0.0);

    for(int i = 0; i < I; i++) {
    	fptype mass2  = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
        fptype massSqTerm = POW2(resmass) - mass2;

        fptype FF_MD = Form_Factor_Mother_Decay(spin, motherMass, mass2, 
			PAIR_12==cyclic_index ? POW2(daug3Mass) : (PAIR_13==cyclic_index ? POW2(daug2Mass) : POW2(daug1Mass))
			,resmass);

        fptype FF_RD = Form_Factor_Resonance_Decay(spin, resmass, mass2,
		       	PAIR_12==cyclic_index ? POW2(daug1Mass) : (PAIR_23==cyclic_index ?  POW2(daug2Mass) :  POW2(daug2Mass)),
			PAIR_12==cyclic_index ? POW2(daug2Mass) : (PAIR_23==cyclic_index ?  POW2(daug3Mass) :  POW2(daug1Mass)));
        
	    fptype Angular = Angular_Distribution(spin, motherMass, daug1Mass, daug2Mass,daug3Mass, m12, m23, m13);

        fptype q0_ = sqrt( lambda( POW2(resmass), 
                PAIR_12==cyclic_index ? POW2(daug1Mass) : (PAIR_23==cyclic_index ?  POW2(daug2Mass) :  POW2(daug2Mass)),
                PAIR_12==cyclic_index ? POW2(daug2Mass) : (PAIR_23==cyclic_index ?  POW2(daug3Mass) :  POW2(daug1Mass))
                ) )/2.*resmass;
        
       fptype q = sqrt( lambda( mass2, 
                PAIR_12==cyclic_index ? POW2(daug1Mass) : (PAIR_23==cyclic_index ?  POW2(daug2Mass) :  POW2(daug2Mass)),
                PAIR_12==cyclic_index ? POW2(daug2Mass) : (PAIR_23==cyclic_index ?  POW2(daug3Mass) :  POW2(daug1Mass))
                ) )/2.*sqrt(mass2);

        fptype gamma = Gamma(spin, resmass, reswidth, mass2,
            PAIR_12==cyclic_index ? POW2(daug1Mass) : (PAIR_23==cyclic_index ?  POW2(daug2Mass) :  POW2(daug3Mass)),
            PAIR_12==cyclic_index ? POW2(daug2Mass) : (PAIR_23==cyclic_index ?  POW2(daug3Mass) :  POW2(daug1Mass))
        );

        fptype h0_ = ( 2.0/M_PI ) * q0_/resmass * log( ( resmass + 2.0*q0_ )/( 2.0*mpi ) );
	    fptype dhdm0_ = h0_ * ( 1.0/( 8.0*q0_*q0_ ) - 1.0/( 2.0*POW2(resmass) ) ) + 1.0/( 2*M_PI*POW2(resmass) );
	    fptype d_ = ( 3.0/M_PI ) * mpi*mpi/( q0_*q0_ ) * log( ( resmass + 2.0*q0_ )/( 2.0*mpi ) ) + resmass/( 2*M_PI*q0_ ) - mpi*mpi*resmass/( M_PI*q0_*q0_*q0_ );

	    fptype h = (2.0/M_PI) * q/( sqrt(mass2) )* log(( sqrt(mass2) + 2.0*q)/(2.0*mpi));
        fptype f = gamma * POW2(resmass)/(q0_*q0_*q0_) * (q*q * (h - h0_) + massSqTerm * q0_*q0_ * dhdm0_);
        
        fptype A = massSqTerm + f;
        fptype B = resmass*gamma;
        fptype C = 1./ (POW2(A) + POW2(B));
        fpcomplex retur = FF_MD*FF_RD*Angular*(1 + d_ * reswidth/reswidth )*(A*C , B*C);
	            
	    ret += retur;
        if(I == 2) {
            fptype swpmass = m12;
            m12            = m13;
            m13            = swpmass;
        }

   }

    return ret;
}

__device__ fpcomplex nonres(fptype m12, fptype m13, fptype m23, unsigned int *indices) { return {1., 0.}; }

__device__ fpcomplex BE(fptype m12, fptype m13, fptype m23,unsigned int *indices){

     
    fptype coef            = RO_CACHE(cudaArray[RO_CACHE(indices[2])]);
    const fptype mpisq = 0.01947977;
    fptype Q = sqrt(m23 - 4*mpisq);
    fptype Omega = exp(-Q*coef);

    return fpcomplex(Omega,0.);

}

__device__ void
getAmplitudeCoefficients(fpcomplex a1, fpcomplex a2, fptype &a1sq, fptype &a2sq, fptype &a1a2real, fptype &a1a2imag) {
    // Returns A_1^2, A_2^2, real and imaginary parts of A_1A_2^*
    a1sq = thrust::norm(a1);
    a2sq = thrust::norm(a2);
    a1 *= conj(a2);
    a1a2real = a1.real();
    a1a2imag = a1.imag();
}

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


    __device__ fpcomplex Bes(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
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
        fptype dmKK, aa, bb;
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


            dmKK = mKKlimits[khiAB] - mKKlimits[kloAB];
            aa   = (mKKlimits[khiAB] - mAB) / dmKK;
             bb   = 1 - aa;

            ret.real(ret.real() + aa * pwa_coefs_real_kloAB + bb * pwa_coefs_real_khiAB );
            ret.imag(ret.imag() + aa * pwa_coefs_imag_kloAB + bb * pwa_coefs_imag_khiAB );

            khiAB = khiAC;
            mAB   = mAC;
        }
        return ret;
    }

__device__ resonance_function_ptr ptr_to_RBW      = plainBW<1>;
__device__ resonance_function_ptr ptr_to_RBW_SYM  = plainBW<2>;
__device__ resonance_function_ptr ptr_to_GOUSAK   = gouSak<1>;
__device__ resonance_function_ptr ptr_to_GOUSAK_SYM = gouSak<2>;
__device__ resonance_function_ptr ptr_to_GAUSSIAN = gaussian;
__device__ resonance_function_ptr ptr_to_NONRES   = nonres;
__device__ resonance_function_ptr ptr_to_FLATTE   = flatte;
__device__ resonance_function_ptr ptr_to_SPLINE   = cubicspline;
__device__ resonance_function_ptr ptr_to_BES   = Bes;
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


BoseEinstein::BoseEinstein(std::string name,Variable ar, Variable ai, Variable coef)
    : ResonancePdf(name, ar, ai) {
    pindices.push_back(registerParameter(coef));
    GET_FUNCTION_ADDR(ptr_to_BoseEinstein);

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

__host__ void Spline::recalculateCache() const {
    auto params           = getParameters();
    const unsigned nKnobs = params.size() / 2;
    std::vector<fpcomplex> y(nKnobs);
    unsigned int i = 0;
    for(auto v = params.begin(); v != params.end(); ++v, ++i) {
        unsigned int idx = i / 2;
        fptype value     = host_params[v->getIndex()];
        if(i % 2 == 0)
            y[idx].real(value);
        else
            y[idx].imag(value);
    }
    std::vector<fptype> y2_flat = flatten(complex_derivative(host_constants, y));

    MEMCPY_TO_SYMBOL(cDeriatives, y2_flat.data(), 2 * nKnobs * sizeof(fptype), 0, cudaMemcpyHostToDevice);
}

Bes::Bes(std::string name,
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

        GET_FUNCTION_ADDR(ptr_to_BES);

        initialize(pindices);
    }

} // namespace Resonances

} // namespace GooFit

