#include <goofit/PDFs/detail/ComplexUtils.h>
#include <goofit/PDFs/physics/DalitzPlotHelpers.h>
#include <goofit/PDFs/physics/ResonancePdf.h>

#if GOOFIT_KMATRIX && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <goofit/detail/compute_inverse5.h>
#endif

#include <utility>

#include <Eigen/Core>
#include <Eigen/LU>

#include <goofit/detail/Macros.h>



namespace GooFit {


    const fptype mPiPlus =  0.139570;
    const fptype mKPlus  = 0.493677;
    const fptype mEta    = 0.547862;
    const fptype mEtap   = 0.96778;
    const size_t NPOLES = 5;
    const size_t NCHANNELS = 5;

__device__ fptype cDeriatives[2 * MAXNKNOBS];

__device__ fptype twoBodyCMmom(fptype rMassSq, fptype d1m, fptype d2m, fptype mR) {
    fptype x = rMassSq;
    fptype y = d1m * d1m;
    fptype z = d2m * d2m;
    fptype l = POW2(x - y - z) - 4 * y * z;

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


__device__ fptype  lambda (const fptype &x,const fptype &y,const fptype &z){

	fptype l;
	l = (x - y - z)*(x - y - z) - 4*y*z;
      	
	return l>=0? l : 1 ;

}

__device__ fptype Form_Factor_Mother_Decay(unsigned int &spin,const fptype &M,const fptype &sab,const fptype &mcsq,const fptype &mR){
    
    fptype s = M*M, mRsq = mR*mR;
    fptype fD, fD0, pstr, pstr0, q2;
    fptype rD2 = 25.0;
    fptype rR2 = 2.25;
    
    if (spin == 0){
	 fD = 1.;
    }

    if (spin == 1) {
        pstr0 = sqrt(lambda(s,mRsq,mcsq))/(2*M);
        q2 = rD2*pstr0*pstr0;
        fD0 = sqrt(1 + q2);
        
        pstr = sqrt(lambda(s,sab,mcsq))/(2*M);
        q2 = rD2*pstr*pstr;
        fD = fD0/sqrt(1 + q2);
        
    }
    
    if(spin == 2){
        pstr0 = sqrt(lambda(s,mRsq,mcsq))/(2*M);
        q2 = rD2*pstr0*pstr0;
        fD0 = sqrt(9 + 3*q2 + q2*q2);
        
        pstr = sqrt(lambda(s,sab,mcsq))/(2*M);
        q2 = rD2*pstr*pstr;
        fD = fD0/sqrt(9 + 3*q2 + q2*q2);
        
    }

    return fD;
}

__device__ fptype Form_Factor_Resonance_Decay(unsigned int &spin,const fptype &mR,const fptype &sab,const fptype &masq,const fptype &mbsq){

	fptype mRsq = mR*mR;
	fptype fR, fR0, pstr, pstr0, q2;
	const fptype rD2 = 25.0;
    	const fptype rR2 = 2.25;

    if (spin == 0) return 1;
    
    if (spin == 1) {

		pstr0 = sqrt(lambda(mRsq,masq,mbsq))/(2*mR);
		q2 = rR2*pstr0*pstr0;
		fR0 = sqrt(1 + q2);

		pstr = sqrt(lambda(sab,masq,mbsq))/(2*sqrt(sab));
		q2 = rR2*pstr*pstr;
		fR = fR0/sqrt(1 + q2);

		

	}
    
    if(spin == 2){

		pstr0 = sqrt(lambda(mRsq,masq,mbsq))/(2*mR);
//sqrt((mRsq - masq - mbsq)*(mRsq - masq - mbsq) - 4*masq*mbsq)/(2*mR);
		q2 = rR2*pstr0*pstr0;
		fR0 = sqrt(9 + 3*q2 + q2*q2);

		pstr = sqrt(lambda(sab,masq,mbsq))/(2*sqrt(sab));
		//pstr = sqrt((sab - masq + mbsq)*(sab - masq + mbsq) - 4*sab*mbsq)/(2*sqrt(sab));
		q2 = rR2*pstr*pstr;
		fR = fR0/sqrt(9 + 3*q2 + q2*q2);

		

    }
    
    return fR;
}

// Mass-dependent width
////////////////////////////////////////////////////////////////////////
__device__ fptype Gamma(unsigned int &spin,const fptype &mR,const fptype &width,const fptype &mab, const fptype &masq,const fptype &mbsq){

    fptype pstr, pstr0,fR, mRsq = mR*mR,sab = mab*mab;
    fptype gamma;

	pstr0 = sqrt(lambda(mRsq,masq,mbsq))/(2*mR);
	pstr = sqrt(lambda(sab,masq,mbsq))/(2*mab);
    
    if (spin == 0){
        gamma = width*(pstr/pstr0)*(mR/mab);
    }
    
    if (spin == 1){
		fR = Form_Factor_Resonance_Decay(spin, mR, sab, masq, mbsq);
		gamma = width*pow((pstr/pstr0),3)*(mR/mab)*fR*fR;
    }
    
    if (spin == 2){
		fR = Form_Factor_Resonance_Decay(spin, mR, sab, masq, mbsq);
		gamma = width*pow((pstr/pstr0),5)*(mR/mab)*fR*fR;
	}
    
    return gamma;

}




__device__ fptype dampingFactorSquare(const fptype &cmmom, const int &spin, const fptype &mRadius) {
    fptype square = mRadius * mRadius * cmmom * cmmom;
    fptype dfsq   = 1 + square; // This accounts for spin 1
    fptype dfsqres = 9 + 3 * square + square * square;

    return (spin == 2) ? 1./dfsqres : 1./dfsq;
}

//From Grace-Young RooAmplitudes Summer Student Report and Rio Dalitz Plot Analysis Package

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

    
     // Copied from EvtDalitzReso, with assumption that pairAng convention matches pipipi0 from EvtD0mixDalitz.
    // Again, all threads should get the same branch.
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

        fptype fD = Form_Factor_Mother_Decay(spin, c_motherMass, rMassSq, POW2(mass_daug3), resmass);
        fptype fR = Form_Factor_Resonance_Decay(spin, resmass, rMassSq, POW2(mass_daug1), POW2(mass_daug2));
	fptype gamma = Gamma(spin, resmass, reswidth, rMass,  POW2(mass_daug1), POW2(mass_daug2));
        // RBW evaluation
        fptype A = (resmass2 - rMassSq);
        fptype B = resmass*gamma;
        fptype C = 1.0 / (POW2(A) + POW2(B));

        fpcomplex ret(A * C, B * C); // Dropping F_D=1

        ret *= fD*fR;
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

template <int I>
__device__ fpcomplex Pole(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
    fptype c_motherMass   = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 0]);
    fptype c_daug1Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 1]);
    fptype c_daug2Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 2]);
    fptype c_daug3Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 3]);
    fptype c_meson_radius = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 4]);

    fptype real            = RO_CACHE(cudaArray[RO_CACHE(indices[2])]);
    fptype img           = RO_CACHE(cudaArray[RO_CACHE(indices[3])]);
    unsigned int spin         = RO_CACHE(indices[4]);
    unsigned int cyclic_index = RO_CACHE(indices[5]);
    unsigned int symmDP       = RO_CACHE(indices[6]);

    fpcomplex result(0., 0.);
    

#pragma unroll
    for(int i = 0; i < I; i++) {
        fptype rMassSq    = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
        fptype rMass = sqrt(rMassSq);
        fptype mass_daug1 = PAIR_23 == cyclic_index ? c_daug2Mass : c_daug1Mass;
        fptype mass_daug2 = PAIR_12 == cyclic_index ? c_daug2Mass : c_daug3Mass;
        fptype mass_daug3 = PAIR_23 == cyclic_index ? c_daug1Mass : (PAIR_13 == cyclic_index?c_daug2Mass:c_daug3Mass);

        fptype reTerm = real*real - img*img - rMassSq;
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
    const	fptype mpi = 0.13957018;

    #pragma unroll
    for(int i = 0; i < I; i++) {
        fptype rMassSq    = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
        fptype rMass = sqrt(rMassSq);
        fptype mass_daug1 = PAIR_23 == cyclic_index ? c_daug2Mass : c_daug1Mass;
        fptype mass_daug2 = PAIR_12 == cyclic_index ? c_daug2Mass : c_daug3Mass;
        fptype mass_daug3 = PAIR_23 == cyclic_index ? c_daug1Mass : (PAIR_13 == cyclic_index?c_daug2Mass:c_daug3Mass);
    // Calculate momentum of the two daughters in the resonance rest frame; note symmetry under interchange (dm1 <->
    // dm2).


        fptype fD = Form_Factor_Mother_Decay(spin, c_motherMass, rMassSq, POW2(mass_daug3), resmass);

        fptype fR = Form_Factor_Resonance_Decay(spin, resmass, rMassSq, POW2(mass_daug1), POW2(mass_daug2));

        fptype  mRSq_rho = POW2(resmass);

        fptype  massSqTerm = mRSq_rho - rMassSq;

        fptype q0_ = sqrt( twoBodyCMmom( mRSq_rho, c_daug1Mass, c_daug2Mass, resmass ) );

        fptype q   = sqrt( twoBodyCMmom( rMassSq, c_daug1Mass, c_daug2Mass,rMass ) );

        fptype h0_ = ( 2.0/M_PI ) * q0_/resmass * log( ( resmass + 2.0*q0_ )/( 2.0*mpi ) );

        fptype dhdm0_ = h0_ * ( 1.0/( 8.0*q0_*q0_ )
                        - 1.0/( 2.0*mRSq_rho ) ) + 1.0/( 2*M_PI*mRSq_rho );

        fptype d_ = ( 3.0/M_PI ) * POW2(mpi)/( q0_*q0_ ) * log( ( resmass + 2.0*q0_ )/( 2.0*mpi ) )
            + resmass/( 2*M_PI*q0_ ) - POW2(mpi)*resmass/( M_PI*q0_*q0_*q0_ );

        fptype h = (2.0/M_PI) * q/( rMass )* log(( rMass + 2.0*q)/(2.0*mpi));

        fptype ff = resmass * rMassSq/(q0_*q0_*q0_) * (q*q * (h - h0_) + massSqTerm * q0_*q0_ * dhdm0_);

        fptype D  = massSqTerm + ff;

        fptype gamma_rho = Gamma(spin, resmass, reswidth, rMass,  POW2(mass_daug1), POW2(mass_daug2));

        fptype E = resmass*gamma_rho;

        fptype F = 1./(D*D + E*E);

        fpcomplex retur(D*F,E*F);

        result += retur;

        if(I != 0) {
            fptype swpmass = m12;
            m12            = m13;
            m13            = swpmass;
        }

   }

    return result;
}

template<int I>
__device__ fpcomplex RhoOmegaMix(fptype m12, fptype m13, fptype m23, unsigned int *indices){
    
    fptype c_motherMass   = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 0]);
    fptype c_daug1Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 1]);
    fptype c_daug2Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 2]);
    fptype c_daug3Mass    = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 3]);
    fptype c_meson_radius = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 4]);

    fptype rho_resmass            = RO_CACHE(cudaArray[RO_CACHE(indices[2])]);
    fptype rho_reswidth           = RO_CACHE(cudaArray[RO_CACHE(indices[3])]);
    fptype omega_resmass            = RO_CACHE(cudaArray[RO_CACHE(indices[4])]);
    fptype omega_reswidth           = RO_CACHE(cudaArray[RO_CACHE(indices[5])]);
    fptype mgB            = RO_CACHE(cudaArray[RO_CACHE(indices[6])]);
    fptype phsB           = RO_CACHE(cudaArray[RO_CACHE(indices[7])]);

    unsigned int spin         = RO_CACHE(indices[8]);
    unsigned int cyclic_index = RO_CACHE(indices[9]);
    unsigned int symmDP       = RO_CACHE(indices[10]);
  
    fpcomplex result(0., 0.);
    const	fptype mpi = 0.13957018;
    const 	fptype delta = 0.00215;
    fptype Delta_= delta*(rho_resmass + omega_resmass);
    mgB *= Delta_;
    fpcomplex Bterm(mgB*cos(phsB),mgB*sin(phsB));
    fpcomplex unity(1.0,0.0);
    
#pragma unroll
    for(int i = 0; i < I; i++) {
        fptype rMassSq    = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));
        fptype rMass = sqrt(rMassSq);
        fptype mass_daug1 = PAIR_23 == cyclic_index ? c_daug2Mass : c_daug1Mass;
        fptype mass_daug2 = PAIR_12 == cyclic_index ? c_daug2Mass : c_daug3Mass;
        fptype mass_daug3 = PAIR_23 == cyclic_index ? c_daug1Mass : (PAIR_13 == cyclic_index?c_daug2Mass:c_daug3Mass);

	fptype angular = spinFactor(spin, c_motherMass, c_daug1Mass, c_daug2Mass, c_daug3Mass, m12, m13, m23, cyclic_index);

        // RBW evaluation
	//fptype gamma = Gamma(spin, omega_resmass, omega_reswidth, rMass,  POW2(mass_daug1), POW2(mass_daug2));
	fptype gamma = omega_reswidth;
        fptype A = (POW2(omega_resmass) - rMassSq);
        fptype B = omega_resmass*gamma;
        fptype C = 1.0 / (POW2(A) + POW2(B));
        

        fpcomplex RBW(A * C, B * C); 
	
	// End of RBW

	// GouSak evaluation for rho

       fptype fD = Form_Factor_Mother_Decay(spin, c_motherMass, rMassSq, POW2(mass_daug3), rho_resmass);
       
       fptype fR = Form_Factor_Resonance_Decay(spin, rho_resmass, rMassSq, POW2(mass_daug1), POW2(mass_daug2));

       fptype  mRSq_rho = POW2(rho_resmass);

       fptype  massSqTerm = mRSq_rho - rMassSq;

       fptype q0_ = sqrt( twoBodyCMmom( mRSq_rho, c_daug1Mass, c_daug2Mass, rho_resmass ) );

       fptype q   = sqrt( twoBodyCMmom( rMassSq, c_daug1Mass, c_daug2Mass,rMass ) );

       fptype h0_ = ( 2.0/M_PI ) * q0_/rho_resmass * log( ( rho_resmass + 2.0*q0_ )/( 2.0*mpi ) );
       
       fptype dhdm0_ = h0_ * ( 1.0/( 8.0*q0_*q0_ ) 
			- 1.0/( 2.0*mRSq_rho ) ) + 1.0/( 2*M_PI*mRSq_rho );
       
       fptype d_ = ( 3.0/M_PI ) * POW2(mpi)/( q0_*q0_ ) * log( ( rho_resmass + 2.0*q0_ )/( 2.0*mpi ) )
		 + rho_resmass/( 2*M_PI*q0_ ) - POW2(mpi)*rho_resmass/( M_PI*q0_*q0_*q0_ );

       fptype h = (2.0/M_PI) * q/( rMass )* log(( rMass + 2.0*q)/(2.0*mpi));

       fptype ff = rho_resmass * rMassSq/(q0_*q0_*q0_) * (q*q * (h - h0_) + massSqTerm * q0_*q0_ * dhdm0_);

       fptype D  = massSqTerm + ff;

       fptype gamma_rho = Gamma(spin, rho_resmass, rho_reswidth, rMass,  POW2(mass_daug1), POW2(mass_daug2));	

       fptype E = rho_resmass*gamma_rho;

       fptype F = 1./(D*D + E*E); 

       fpcomplex GouSak(D*F,E*F);

       GouSak *= (unity + d_*(rho_reswidth/rho_resmass) )*angular*fD*fR;

       //end of Gousak

       fpcomplex mixingTerm = Bterm*RBW + unity;
      
       
       result += GouSak*mixingTerm;
       
                                          
        if(I != 0) {
            fptype swpmass = m12;
            m12            = m13;
            m13            = swpmass;
        }
    }

    return result;

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

__device__ fpcomplex cubicsplinePolar(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
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
}

__device__ fpcomplex BE(fptype m12, fptype m13, fptype m23,unsigned int *indices){

     
    fptype coef            = RO_CACHE(cudaArray[RO_CACHE(indices[2])]);
    const fptype mpisq = 0.01947977;
    fptype Q = sqrt(m23 - 4*mpisq);
    fptype Omega = exp(-Q*coef);

    return fpcomplex(Omega,0.);

}



__device__ fptype phsp_twobody(fptype s, fptype m0, fptype m1) { return sqrt(1. - POW2(m0 + m1) / s); }


__device__ fptype phsp_fourpi(fptype s) {
if(s > 1)
return phsp_twobody(s, 2 * mPiPlus, 2 * mPiPlus);
else
return 0.00051 + -0.01933 * s + 0.13851 * s * s + -0.20840 * s * s * s + -0.29744 * s * s * s * s
+ 0.13655 * s * s * s * s * s + 1.07885 * s * s * s * s * s * s;
}

#if GOOFIT_KMATRIX


__device__ Eigen::Array<fpcomplex, NCHANNELS, NCHANNELS>
getPropagator(const Eigen::Array<fptype, NCHANNELS, NCHANNELS> &kMatrix,
              const Eigen::Matrix<fptype, 5, 1> &phaseSpace,
              fptype adlerTerm) {
    Eigen::Array<fpcomplex, NCHANNELS, NCHANNELS> tMatrix;

    for(unsigned int i = 0; i < NCHANNELS; ++i) {
        for(unsigned int j = 0; j < NCHANNELS; ++j) {
            tMatrix(i, j) = (i == j ? 1. : 0.) - fpcomplex(0, adlerTerm) * kMatrix(i, j) * phaseSpace(j);
        }
    }

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    // Here we assume that some values are 0
    return compute_inverse5<-1,
                            -1,
                            0,
                            -1,
                            -1,
                            -1,
                            -1,
                            0,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1>(tMatrix);
#else
    return Eigen::inverse(tMatrix);
#endif
}

template<int I>
__device__ fpcomplex kMatrixFunction(fptype m12, fptype m13, fptype m23, unsigned int *indices) {
    // const fptype mass  = GOOFIT_GET_PARAM(2);
    // const fptype width = GOOFIT_GET_PARAM(3);
    // const unsigned int L = GOOFIT_GET_INT(4);
    // const fptype radius = GOOFIT_GET_CONST(7);

    // const fptype pTerm = GOOFIT_GET_INT();

    fpcomplex ret(0.,0.);

    fptype sA0      = RO_CACHE(cudaArray[RO_CACHE(indices[2])]);
    fptype sA       = RO_CACHE(cudaArray[RO_CACHE(indices[3])]);
    fptype s0_prod  = RO_CACHE(cudaArray[RO_CACHE(indices[4])]);
    fptype s0_scatt = RO_CACHE(cudaArray[RO_CACHE(indices[5])]);


    Eigen::Array<fptype, NCHANNELS, 1> fscat;
    Eigen::Array<fptype, NPOLES, 1> pmasses;
    Eigen::Array<fptype, NPOLES, NPOLES> couplings;

    for(int i = 0; i < NCHANNELS; i++) {
        fscat(i) = RO_CACHE(cudaArray[RO_CACHE(indices[5+i])]);;
    }

    for(int i = 0; i < NPOLES; i++) {
        for(int j = 0; j < NPOLES; j++)
            couplings(i, j) = GOOFIT_GET_PARAM(5 + NCHANNELS + i * (NPOLES + 1) + j);
        pmasses(i) = GOOFIT_GET_PARAM(5 + NCHANNELS + i * (NPOLES + 1) + NPOLES);
    }

    unsigned int pterm = RO_CACHE(cudaArray[RO_CACHE(indices[5 + NCHANNELS + NPOLES* (NPOLES + 1) + NPOLES])]);
    bool is_pole       =  RO_CACHE(cudaArray[RO_CACHE(indices[5 + NCHANNELS + NPOLES* (NPOLES + 1) + NPOLES + 1])]);
    unsigned int cyclic_index = RO_CACHE(cudaArray[RO_CACHE(indices[5 + NCHANNELS + NPOLES* (NPOLES + 1) + NPOLES + 3])]);
    bool symmdp = RO_CACHE(cudaArray[RO_CACHE(indices[5 + NCHANNELS + NPOLES* (NPOLES + 1) + NPOLES + 4])]);

#pragma unroll
    for(int i = 0; i < I ; i++){

    fptype s = (PAIR_12 == cyclic_index ? m12 : (PAIR_13 == cyclic_index ? m13 : m23));

    // constructKMatrix

    Eigen::Array<fptype, NCHANNELS, NCHANNELS> kMatrix;
    kMatrix.setZero();

    // TODO: Make sure the order (k,i,j) is correct

    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 5; j++) {
            for(int k = 0; k < 5; k++)
                kMatrix(i, j) += couplings(k, i) * couplings(k, j) / (pmasses(k) - s);
            if(i == 0 || j == 0) // Scattering term
                kMatrix(i, j) += fscat(i + j) * (1 - s0_scatt) / (s - s0_scatt);
        }
    }

    fptype adlerTerm = (1. - sA0) * (s - sA * mPiPlus * mPiPlus / 2) / (s - sA0);

    Eigen::Matrix<fptype, 5, 1> phaseSpace;
    phaseSpace << phsp_twobody(s, mPiPlus, mPiPlus), phsp_twobody(s, mKPlus, mKPlus), phsp_fourpi(s),
        phsp_twobody(s, mEta, mEta), phsp_twobody(s, mEta, mEtap);

    Eigen::Array<fpcomplex, NCHANNELS, NCHANNELS> F = getPropagator(kMatrix, phaseSpace, adlerTerm);

    if(is_pole) { // pole
        fpcomplex M = 0;
        for(int i = 0; i < NCHANNELS; i++) {
            fptype pole = couplings(i, pterm);
            M += F(0, i) * pole;
        }
        ret+= M / (POW2(pmasses(pterm)) - s);
    } else { // prod
        ret += F(0, pterm) * (1 - s0_prod) / (s - s0_prod);
    }


        if(I != 0) {
            fptype swpmass = m12;
            m12            = m13;
            m13            = swpmass;
        }

    }

    return ret;
}
#endif


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
__device__ resonance_function_ptr ptr_to_SPLINE   = cubicspline;
__device__ resonance_function_ptr ptr_to_SPLINE_POLAR   = cubicsplinePolar;
__device__ resonance_function_ptr ptr_to_BoseEinstein = BE;
#if GOOFIT_KMATRIX
__device__ resonance_function_ptr ptr_to_kMatrix_SYM = kMatrixFunction<2>;
__device__ resonance_function_ptr ptr_to_kMatrix = kMatrixFunction<1>;
#endif

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
         Variable rho_mass,
         Variable rho_width,
	 Variable omega_mass,
         Variable omega_width,
	 Variable magB,
         Variable phsB,
         unsigned int sp,
         unsigned int cyc,
         bool symmDP)
    : ResonancePdf(name, ar, ai) {
    pindices.push_back(registerParameter(rho_mass));
    pindices.push_back(registerParameter(rho_width));
    pindices.push_back(registerParameter(omega_mass));
    pindices.push_back(registerParameter(omega_width));
    pindices.push_back(registerParameter(magB));
    pindices.push_back(registerParameter(phsB));
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

    GET_FUNCTION_ADDR(ptr_to_SPLINE_POLAR);

    initialize(pindices);
}

__host__ void SplinePolar::recalculateCache() const {
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
}

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



#if GOOFIT_KMATRIX
    kMatrix::kMatrix(std::string name,
                             Variable ar,
                             Variable ai,
                             Variable sA0,
                             Variable sA,
                             Variable s0_prod,
                             Variable s0_scatt,
                             std::array<Variable, NCHANNELS> fscat,
                             std::array<Variable, NPOLES *(NPOLES + 1)> poles,
                             unsigned int pterm,
                             bool is_pole,
                             unsigned int L,
                             unsigned int Mpair,
                             bool symmdp)
    : ResonancePdf(name,ar,ai) {

    pindices.push_back(registerParameter(sA0));
    pindices.push_back(registerParameter(sA));
    pindices.push_back(registerParameter(s0_prod));
    pindices.push_back(registerParameter(s0_scatt));

    for(int i = 0; i < NCHANNELS; i++) {
         pindices.push_back(registerParameter(fscat[i]));
    }

    for(int i = 0; i < NPOLES * (NPOLES + 1); i++) {
         pindices.push_back(registerParameter(poles[i]));
    }

    pindices.push_back(pterm);
    pindices.push_back(is_pole);
    pindices.push_back(L);
    pindices.push_back(Mpair);
    pindices.push_back(symmdp);

    if(symmdp){
        GET_FUNCTION_ADDR(ptr_to_kMatrix_SYM);
    }else{
        GET_FUNCTION_ADDR(ptr_to_kMatrix);
    }
      initialize(pindices);
}
#endif

} // namespace Resonances

} // namespace GooFit
