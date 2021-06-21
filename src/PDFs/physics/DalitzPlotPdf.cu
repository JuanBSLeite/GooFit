#include <goofit/Error.h>
#include <goofit/PDFs/physics/DalitzPlotPdf.h>

#include <goofit/detail/Complex.h>
#include <thrust/transform_reduce.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random.h>

namespace GooFit {

	// Functor used for fit fraction sum
	struct CoefSumFunctor {
		fpcomplex coef_i;
		fpcomplex coef_j;

		CoefSumFunctor(fpcomplex coef_i, fpcomplex coef_j)
			: coef_i(coef_i)
			  , coef_j(coef_j) {}

		__device__ fptype operator()(thrust::tuple<fpcomplex, fpcomplex> val) {
			return (coef_i * thrust::conj<fptype>(coef_j) * thrust::get<0>(val) * thrust::conj<fptype>(thrust::get<1>(val)))
				.real();
		}
	};

	constexpr int resonanceOffset_DP = 4; // Offset of the first resonance into the parameter index array
	// Offset is number of parameters, constant index, number of resonances (not calculable
	// from nP because we don't know what the efficiency might need), and cache index. Efficiency
	// parameters are after the resonance information.

	// The function of this array is to hold all the cached waves; specific
	// waves are recalculated when the corresponding resonance mass or width
	// changes. Note that in a multithread environment each thread needs its
	// own cache, hence the '10'. Ten threads should be enough for anyone!

	// NOTE: This is does not support ten instances (ten threads) of resoncances now, only one set of resonances.
	__device__ fpcomplex *cResonances[16];

	__device__ inline int parIndexFromResIndex_DP(int resIndex) { return resonanceOffset_DP + resIndex * resonanceSize; }

	__device__ fpcomplex
		device_DalitzPlot_calcIntegrals(fptype m12, fptype m13, int res_i, int res_j, fptype *p, unsigned int *indices) {
			// Calculates BW_i(m12, m13) * BW_j^*(m12, m13).
			// This calculation is in a separate function so
			// it can be cached. Note that this function expects
			// to be called on a normalisation grid, not on
			// observed points, that's why it doesn't use
			// cResonances. No need to cache the values at individual
			// grid points - we only care about totals.
			auto motherMass = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 0]);
			auto daug1Mass  = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 1]);
			auto daug2Mass  = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 2]);
			auto daug3Mass  = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 3]);

			fpcomplex ret{0., 0.};

			if(!inDalitz(m12, m13, motherMass, daug1Mass, daug2Mass, daug3Mass))
				return ret;

			auto m23 = motherMass * motherMass + daug1Mass * daug1Mass + daug2Mass * daug2Mass + daug3Mass * daug3Mass - m12 - m13;

			auto parameter_i       = parIndexFromResIndex_DP(res_i);
			auto functn_i = RO_CACHE(indices[parameter_i + 2]);
			auto params_i = RO_CACHE(indices[parameter_i + 3]);
			ret  = getResonanceAmplitude(m12, m13, m23, functn_i, params_i);

			auto parameter_j       = parIndexFromResIndex_DP(res_j); 
			auto functn_j = RO_CACHE(indices[parameter_j + 2]);
			auto params_j = RO_CACHE(indices[parameter_j + 3]);
			ret *= conj(getResonanceAmplitude(m12, m13, m23, functn_j, params_j));

			return ret;
		}

	__device__ fptype device_DalitzPlot(fptype *evt, fptype *p, unsigned int *indices) {
		auto motherMass = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 0]);
		auto daug1Mass  = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 1]);
		auto daug2Mass  = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 2]);
		auto daug3Mass  = RO_CACHE(functorConstants[RO_CACHE(indices[1]) + 3]);

		auto m12 = RO_CACHE(evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]);
		auto m13 = RO_CACHE(evt[RO_CACHE(indices[3 + RO_CACHE(indices[0])])]);

		if(!inDalitz(m12, m13, motherMass, daug1Mass, daug2Mass, daug3Mass))
			return 0;

		auto evtIndex = RO_CACHE(evt[RO_CACHE(indices[4 + RO_CACHE(indices[0])])]);
		auto evtNum = static_cast<int>(floor(0.5 + evtIndex));

		fpcomplex totalAmp(0, 0);
		auto numResonances = RO_CACHE(indices[2]);

		for(int i = 0; i < numResonances; ++i) {
			auto paramIndex = parIndexFromResIndex_DP(i);

			auto mag = RO_CACHE(p[RO_CACHE(indices[paramIndex + 0])]);
			auto  phase = RO_CACHE(p[RO_CACHE(indices[paramIndex + 1])]);
			auto amp = fpcomplex(mag,phase);
			auto me = RO_CACHE(cResonances[i][evtNum]);
			totalAmp += amp * me;
		}

		auto ret     = thrust::norm(totalAmp);
		auto effFunctionIdx = parIndexFromResIndex_DP(numResonances);
		auto eff  = callFunction(evt, RO_CACHE(indices[effFunctionIdx]), RO_CACHE(indices[effFunctionIdx + 1]));

		return ret*eff;
	}

	__device__ device_function_ptr ptr_to_DalitzPlot = device_DalitzPlot;

	__host__ DalitzPlotPdf::DalitzPlotPdf(
			std::string n, Observable m12, Observable m13, EventNumber eventNumber, DecayInfo3 decay, GooPdf *efficiency)
		: GooPdf(n, m12, m13, eventNumber)
		  , decayInfo(decay)
		  , _m12(m12)
		  , _m13(m13)
		  , _eventNumber(eventNumber)
		  , dalitzNormRange(nullptr)
		  //, cachedWaves(0)
		  , integrals(nullptr)
		  , integrals_ff(nullptr)
		  , integrators_ff(nullptr)
		  , forceRedoIntegrals(true)
		  , totalEventSize(3) // Default 3 = m12, m13, evtNum
		  , cacheToUse(0)
		  , integrators(nullptr)
		  , calculators(nullptr) {
			  fptype decayConstants[5];

			  for(auto &cachedWave : cachedWaves)
				  cachedWave = nullptr;

			  std::vector<unsigned int> pindices;
			  pindices.reserve(1000);

			  pindices.emplace_back(registerConstants(5));
			  decayConstants[0] = decayInfo.motherMass;
			  decayConstants[1] = decayInfo.daug1Mass;
			  decayConstants[2] = decayInfo.daug2Mass;
			  decayConstants[3] = decayInfo.daug3Mass;
			  decayConstants[4] = decayInfo.meson_radius;
			  MEMCPY_TO_SYMBOL(
					  functorConstants, decayConstants, 5 * sizeof(fptype), cIndex * sizeof(fptype), cudaMemcpyHostToDevice);

			  pindices.emplace_back(decayInfo.resonances.size());
			  static int cacheCount = 0;
			  cacheToUse            = cacheCount++;
			  pindices.emplace_back(cacheToUse);

			  for(auto &resonance : decayInfo.resonances) {
				  pindices.emplace_back(registerParameter(resonance->amp_real));
				  pindices.emplace_back(registerParameter(resonance->amp_imag));
				  pindices.emplace_back(resonance->getFunctionIndex());
				  pindices.emplace_back(resonance->getParameterIndex());
				  resonance->setConstantIndex(cIndex);
				  components.emplace_back(resonance);
			  }

			  pindices.emplace_back(efficiency->getFunctionIndex());
			  pindices.emplace_back(efficiency->getParameterIndex());
			  components.emplace_back(efficiency);

			  GET_FUNCTION_ADDR(ptr_to_DalitzPlot);
			  initialize(pindices);

			  redoIntegral = new bool[decayInfo.resonances.size()];
			  cachedMasses = new fptype[decayInfo.resonances.size()];
			  cachedWidths = new fptype[decayInfo.resonances.size()];
			  integrals    = new fpcomplex **[decayInfo.resonances.size()];
			  integrators  = new SpecialResonanceIntegrator **[decayInfo.resonances.size()];
			  integrals_ff = new fpcomplex **[decayInfo.resonances.size()];
			  integrators_ff = new SpecialResonanceIntegrator **[decayInfo.resonances.size()];

			  calculators  = new SpecialResonanceCalculator *[decayInfo.resonances.size()];

			  for(int i = 0; i < decayInfo.resonances.size(); ++i) {
				  redoIntegral[i] = true;
				  cachedMasses[i] = -1;
				  cachedWidths[i] = -1;
				  integrators[i]  = new SpecialResonanceIntegrator *[decayInfo.resonances.size()];
				  calculators[i]  = new SpecialResonanceCalculator(parameters, i);
				  integrals[i]    = new fpcomplex *[decayInfo.resonances.size()];

				  integrals_ff[i]=new fpcomplex *[decayInfo.resonances.size()];
				  integrators_ff[i] = new SpecialResonanceIntegrator *[decayInfo.resonances.size()];

				  for(int j = 0; j < decayInfo.resonances.size(); ++j) {
					  integrals[i][j]   = new fpcomplex(0, 0);
					  integrators[i][j] = new SpecialResonanceIntegrator(parameters, i, j);
					  integrals_ff[i][j]   = new fpcomplex(0, 0);
					  integrators_ff[i][j] = new SpecialResonanceIntegrator(parameters, i, j);	

				  }
			  }

			  addSpecialMask(PdfBase::ForceSeparateNorm);
		  }

	__host__ void DalitzPlotPdf::setDataSize(unsigned int dataSize, unsigned int evtSize) {
		// Default 3 is m12, m13, evtNum
		totalEventSize = evtSize;
		if(totalEventSize < 3)
			throw GooFit::GeneralError("totalEventSize {} must be 3 or more", totalEventSize);

		// if (cachedWaves) delete cachedWaves;
		if(cachedWaves[0]) {
			for(auto &cachedWave : cachedWaves) {
				delete cachedWave;
				cachedWave = nullptr;
			}
		}

		numEntries = dataSize;

		for(int i = 0; i < 16; i++) {
#ifdef GOOFIT_MPI
			cachedWaves[i] = new thrust::device_vector<fpcomplex>(m_iEventsPerTask);
#else
			cachedWaves[i] = new thrust::device_vector<fpcomplex>(dataSize);
#endif
			void *dummy = thrust::raw_pointer_cast(cachedWaves[i]->data());
			MEMCPY_TO_SYMBOL(cResonances, &dummy, sizeof(fpcomplex *), i * sizeof(fpcomplex *), cudaMemcpyHostToDevice);
		}

		setForceIntegrals();
	}

	__host__ fptype DalitzPlotPdf::normalize() const {
		recursiveSetNormalisation(1); // Not going to normalize efficiency,
		// so set normalisation factor to 1 so it doesn't get multiplied by zero.
		// Copy at this time to ensure that the SpecialResonanceCalculators, which need the efficiency,
		// don't get zeroes through multiplying by the normFactor.
		MEMCPY_TO_SYMBOL(normalisationFactors, host_normalisation, totalParams * sizeof(fptype), 0, cudaMemcpyHostToDevice);

		auto totalBins = _m12.getNumBins() * _m13.getNumBins();

		if(!dalitzNormRange) {
			gooMalloc((void **)&dalitzNormRange, 6 * sizeof(fptype));
		}

		// This line runs once
		static std::array<fptype, 6> host_norms{{0, 0, 0, 0, 0, 0}};

		std::array<fptype, 6> current_host_norms{{_m12.getLowerLimit(),
			_m12.getUpperLimit(),
			static_cast<fptype>(_m12.getNumBins()),
			_m13.getLowerLimit(),
			_m13.getUpperLimit(),
			static_cast<fptype>(_m13.getNumBins())}};

		if(host_norms != current_host_norms) {
			host_norms = current_host_norms;
			MEMCPY(dalitzNormRange, host_norms.data(), 6 * sizeof(fptype), cudaMemcpyHostToDevice);
		}

		for(unsigned int i = 0; i < decayInfo.resonances.size(); ++i) {
			redoIntegral[i] = forceRedoIntegrals;

			if(!(decayInfo.resonances[i]->parametersChanged()))
				continue;

			redoIntegral[i] = true;
		}

		forceRedoIntegrals = false;

		// Only do this bit if masses or widths have changed.
		thrust::constant_iterator<fptype *> arrayAddress(dalitzNormRange);
		thrust::counting_iterator<int> binIndex(0);

		// NB, SpecialResonanceCalculator assumes that fit is unbinned!
		// And it needs to know the total event size, not just observables
		// for this particular PDF component.
		thrust::constant_iterator<fptype *> dataArray(dev_event_array);
		thrust::constant_iterator<int> eventSize(totalEventSize);
		thrust::counting_iterator<int> eventIndex(0);

		for(int i = 0; i < decayInfo.resonances.size(); ++i) {
			if(redoIntegral[i]) {
#ifdef GOOFIT_MPI
				thrust::transform(
						thrust::make_zip_iterator(thrust::make_tuple(eventIndex, dataArray, eventSize)),
						thrust::make_zip_iterator(thrust::make_tuple(eventIndex + m_iEventsPerTask, arrayAddress, eventSize)),
						strided_range<thrust::device_vector<fpcomplex>::iterator>(
							cachedWaves[i]->begin(), cachedWaves[i]->end(), 1)
						.begin(),
						*(calculators[i]));
#else
				thrust::transform(
						thrust::make_zip_iterator(thrust::make_tuple(eventIndex, dataArray, eventSize)),
						thrust::make_zip_iterator(thrust::make_tuple(eventIndex + numEntries, arrayAddress, eventSize)),
						strided_range<thrust::device_vector<fpcomplex>::iterator>(
							cachedWaves[i]->begin(), cachedWaves[i]->end(), 1)
						.begin(),
						*(calculators[i]));
#endif
			}

			// Possibly this can be done more efficiently by exploiting symmetry?
			for(int j = 0; j < decayInfo.resonances.size(); ++j) {
				if((!redoIntegral[i]) && (!redoIntegral[j]))
					continue;

				fpcomplex dummy(0, 0);
				thrust::plus<fpcomplex> complexSum;
				(*(integrals[i][j])) = i==j ? thrust::transform_reduce(
						thrust::make_zip_iterator(thrust::make_tuple(binIndex, arrayAddress)),
						thrust::make_zip_iterator(thrust::make_tuple(binIndex + totalBins, arrayAddress)),
						*(integrators[i][j]),
						dummy,
						complexSum):(j<i ? thrust::transform_reduce(
                                                thrust::make_zip_iterator(thrust::make_tuple(binIndex, arrayAddress)),
                                                thrust::make_zip_iterator(thrust::make_tuple(binIndex + totalBins, arrayAddress)),
                                                *(integrators[i][j]),
                                                dummy,
                                                complexSum):fpcomplex(0.,0.));

			}
		}

		// End of time-consuming integrals.
		fpcomplex sumIntegral(0, 0);

		/*for(unsigned int i = 0; i < decayInfo.resonances.size(); ++i) {
			auto param_i = parameters + resonanceOffset_DP + resonanceSize * i;
			const fpcomplex amplitude_i(host_params[host_indices[param_i]], host_params[host_indices[param_i + 1]]);

			for(unsigned int j = 0; j < decayInfo.resonances.size(); ++j) {
				auto param_j = parameters + resonanceOffset_DP + resonanceSize * j;
				const fpcomplex amplitude_j(host_params[host_indices[param_j]], host_params[host_indices[param_j + 1]]);
				sumIntegral += amplitude_i * thrust::conj<fptype>(amplitude_j) * (*(integrals[i][j]));
			}
		}*/

		for(unsigned int i = 0; i < decayInfo.resonances.size(); ++i) {
                        auto param_i = parameters + resonanceOffset_DP + resonanceSize * i;
                        const fpcomplex amplitude_i(host_params[host_indices[param_i]], host_params[host_indices[param_i + 1]]);

                        for(unsigned int j = 0; j < decayInfo.resonances.size(); ++j) {
                                auto param_j = parameters + resonanceOffset_DP + resonanceSize * j;
                                const fpcomplex amplitude_j(host_params[host_indices[param_j]], -host_params[host_indices[param_j + 1]]);

                               	sumIntegral += i==j ? amplitude_i * amplitude_j * (*(integrals[i][j])) : ( j<i ? 2.*amplitude_i * amplitude_j * (*(integrals[i][j])): fpcomplex(0.,0.));						
                        }
                }


		fptype ret           = sumIntegral.real(); // That complex number is a square, so it's fully real
		double binSizeFactor = 1;
		binSizeFactor *= _m12.getBinSize();
		binSizeFactor *= _m13.getBinSize();
		ret *= binSizeFactor;

		host_normalisation[parameters] = 1.0 / ret;
		return ret;
	}

	__host__ std::vector<std::vector<fptype>> DalitzPlotPdf::fit_fractions(unsigned int nBins) {
		recursiveSetNormalisation(1); // Not going to normalize efficiency,
		// so set normalisation factor to 1 so it doesn't get multiplied by zero.
		// Copy at this time to ensure that the SpecialResonanceCalculators, which need the efficiency,
		// don't get zeroes through multiplying by the normFactor.
		_m12.setNumBins(nBins);
		_m13.setNumBins(nBins);
		MEMCPY_TO_SYMBOL(normalisationFactors, host_normalisation, totalParams * sizeof(fptype), 0, cudaMemcpyHostToDevice);

		auto totalBins = _m12.getNumBins() * _m13.getNumBins();
		auto nres = decayInfo.resonances.size();

		if(!dalitzNormRange) {
			gooMalloc((void **)&dalitzNormRange, 6 * sizeof(fptype));
		}

		// This line runs once
		static std::array<fptype, 6> host_norms{{0, 0, 0, 0, 0, 0}};

		std::array<fptype, 6> current_host_norms{{_m12.getLowerLimit(),
			_m12.getUpperLimit(),
			static_cast<fptype>(_m12.getNumBins()),
			_m13.getLowerLimit(),
			_m13.getUpperLimit(),
			static_cast<fptype>(_m13.getNumBins())}};

		if(host_norms != current_host_norms) {
			host_norms = current_host_norms;
			MEMCPY(dalitzNormRange, host_norms.data(), 6 * sizeof(fptype), cudaMemcpyHostToDevice);
		}

		forceRedoIntegrals = false;

		// Only do this bit if masses or widths have changed.
		thrust::constant_iterator<fptype *> arrayAddress(dalitzNormRange);
		thrust::counting_iterator<int> binIndex(0);

		for(int i = 0; i < nres; ++i) {
			//if((!redoIntegral[i]) && (!redoIntegral[j]))
			//	continue;
			for(int j =0; j<nres; j++){
				integrators_ff[i][j]->setNoEff();
				fpcomplex dummy_ff(0, 0);
				thrust::plus<fpcomplex> complexSum_ff;
				(*(integrals_ff[i][j])) = i==j ? thrust::transform_reduce(
						thrust::make_zip_iterator(thrust::make_tuple(binIndex, arrayAddress)),
						thrust::make_zip_iterator(thrust::make_tuple(binIndex + totalBins, arrayAddress)),
						*(integrators_ff[i][j]),
						dummy_ff,
						complexSum_ff):(j<i ? thrust::transform_reduce(
                                                thrust::make_zip_iterator(thrust::make_tuple(binIndex, arrayAddress)),
                                                thrust::make_zip_iterator(thrust::make_tuple(binIndex + totalBins, arrayAddress)),
                                                *(integrators_ff[i][j]),
                                                dummy_ff,
                                                complexSum_ff): fpcomplex(0.,0.));
			}
		}

		// End of time-consuming integrals.
		fpcomplex sumIntegral(0, 0);
		std::vector<std::vector<fptype>> AmpIntegral(nres,std::vector<fptype>(nres));

		for(unsigned int i = 0; i < nres; ++i) {
			auto param_i = parameters + resonanceOffset_DP + resonanceSize * i;
			const fpcomplex amplitude_i(host_params[host_indices[param_i]], host_params[host_indices[param_i + 1]]);

			for(unsigned int j = 0; j < nres; ++j) {
				auto param_j = parameters + resonanceOffset_DP + resonanceSize * j;
				const fpcomplex amplitude_j(host_params[host_indices[param_j]], -host_params[host_indices[param_j + 1]]);
				if(i==j){
					const fpcomplex buffer = amplitude_i * amplitude_j * (*(integrals_ff[i][j]));
					AmpIntegral[i][j] = buffer.real();
					sumIntegral += buffer;
				}else if(j<i){
					const fpcomplex buffer = 2.*amplitude_i * amplitude_j * (*(integrals_ff[i][j]));
                                        AmpIntegral[i][j] = buffer.real();
                                        sumIntegral += buffer;
				}else{sumIntegral+=fpcomplex(0.,0.);}

				
			}
		}

		const fptype totalIntegral      = sumIntegral.real();

		for(int i=0; i<nres; i++){
			for(int j=0; j<nres; j++){
				
				if(i==j)
					AmpIntegral[i][j] /= totalIntegral;
				else if(j<i)
					AmpIntegral[i][j] /= totalIntegral;
				else
					AmpIntegral[i][j]=0.;
			}
		}	

		return AmpIntegral;
	}

	__host__ fpcomplex DalitzPlotPdf::sumCachedWave(size_t i) const {
		const thrust::device_vector<fpcomplex> &vec = getCachedWave(i);

		fpcomplex ret = thrust::reduce(vec.begin(), vec.end(), fpcomplex(0, 0), thrust::plus<fpcomplex>());

		return ret;
	}


	SpecialResonanceIntegrator::SpecialResonanceIntegrator(int pIdx, unsigned int ri, unsigned int rj)
		: resonance_i(ri)
		  , resonance_j(rj)
		  , parameters(pIdx) {}

	__device__ fpcomplex SpecialResonanceIntegrator::operator()(thrust::tuple<int, fptype *> t) const {
		// Bin index, base address [lower, upper,getNumBins]
		// Notice that this is basically MetricTaker::operator (binned) with the special-case knowledge
		// that event size is two, and that the function to call is dev_DalitzPlot_calcIntegrals.

		auto globalBinNumber  = thrust::get<0>(t);
		auto lowerBoundM12 = thrust::get<1>(t)[0];
		auto upperBoundM12 = thrust::get<1>(t)[1];
		auto numBinsM12      = static_cast<int>(floor(thrust::get<1>(t)[2] + 0.5));
		auto binNumberM12     = globalBinNumber % numBinsM12;
		auto binCenterM12  = upperBoundM12 - lowerBoundM12;
		binCenterM12 /= numBinsM12;
		binCenterM12 *= (binNumberM12 + 0.5);
		binCenterM12 += lowerBoundM12;

		globalBinNumber /= numBinsM12;
		auto lowerBoundM13 = thrust::get<1>(t)[3];
		auto  upperBoundM13 = thrust::get<1>(t)[4];
		auto numBinsM13      = static_cast<int>(floor(thrust::get<1>(t)[5] + 0.5));
		auto binCenterM13  = upperBoundM13 - lowerBoundM13;
		binCenterM13 /= numBinsM13;
		binCenterM13 *= (globalBinNumber + 0.5);
		binCenterM13 += lowerBoundM13;

		auto indices = paramIndices + parameters;
		auto ret = device_DalitzPlot_calcIntegrals(binCenterM12, binCenterM13, resonance_i, resonance_j, cudaArray, indices);

		fptype fakeEvt[10]; // Need room for many observables in case m12 or m13 were assigned a high index in an
		// event-weighted fit.
		fakeEvt[indices[indices[0] + 2 + 0]] = binCenterM12;
		fakeEvt[indices[indices[0] + 2 + 1]] = binCenterM13;
		auto numResonances           = indices[2];
		auto effFunctionIdx                   = parIndexFromResIndex_DP(numResonances);
		auto eff                           = callFunction(fakeEvt, indices[effFunctionIdx], indices[effFunctionIdx + 1]);
		if(m_no_eff) eff = 1.;
		// Multiplication by eff, not sqrt(eff), is correct:
		// These complex numbers will not be squared when they
		// go into the integrals. They've been squared already,
		// as it were.
		ret *= eff;
		// printf("ret %f %f %f \n", ret.real(), ret.imag(), eff );
		return ret;
	}

	SpecialResonanceCalculator::SpecialResonanceCalculator(int pIdx, unsigned int res_idx)
		: resonance_i(res_idx)
		  , parameters(pIdx) {}

	__device__ fpcomplex SpecialResonanceCalculator::operator()(thrust::tuple<int, fptype *, int> t) const {
		// Calculates the BW values for a specific resonance.
		fpcomplex ret;
		auto evtNum  = thrust::get<0>(t);
		auto evt = thrust::get<1>(t) + (evtNum * thrust::get<2>(t));

		auto indices = paramIndices + parameters; // Jump to DALITZPLOT position within parameters array
		auto m12            = evt[indices[2 + indices[0]]];
		auto m13            = evt[indices[3 + indices[0]]];

		auto motherMass = functorConstants[indices[1] + 0];
		auto daug1Mass  = functorConstants[indices[1] + 1];
		auto daug2Mass  = functorConstants[indices[1] + 2];
		auto daug3Mass  = functorConstants[indices[1] + 3];

		if(!inDalitz(m12, m13, motherMass, daug1Mass, daug2Mass, daug3Mass))
			return ret;

		auto m23= motherMass * motherMass + daug1Mass * daug1Mass + daug2Mass * daug2Mass + daug3Mass * daug3Mass - m12 - m13;

		auto parameter_i = parIndexFromResIndex_DP(resonance_i); // Find position of this resonance relative to DALITZPLOT start

		auto functn_i = indices[parameter_i + 2];
		auto params_i = indices[parameter_i + 3];

		ret = getResonanceAmplitude(m12, m13, m23, functn_i, params_i);

		return ret;
	}

} // namespace GooFit
