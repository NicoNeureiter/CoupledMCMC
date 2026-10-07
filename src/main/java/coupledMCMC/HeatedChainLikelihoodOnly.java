package coupledMCMC;

import beast.base.core.BEASTInterface;
import beast.base.core.Description;
import beast.base.inference.CompoundDistribution;
import beast.base.inference.Distribution;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;

import java.util.List;

@Description("Like heated chain, but samples log(prior) + beta * log(likelihood) "
		   + "instead of beta * (log(prior) + log(likelihood))")
public class HeatedChainLikelihoodOnly extends HeatedChain {

	
	Distribution likelihood;
	double oldLogP = Double.NaN, newLogP;
	double oldLogPrior = Double.NaN, newLogPrior;
	// the posterior's components other than the likelihood; null if the likelihood is not one of them
	List<Distribution> priorParts;
	
	@Override
	public void initAndValidate() {
		super.initAndValidate();
		
		BEASTInterface o = findLikelihood(posteriorInput.get());
		if (o == null) {
			throw new IllegalArgumentException("Expected an element with id='likelihood', but could not find one in the posterior");
		}
		if (!(o instanceof Distribution)) {
			throw new IllegalArgumentException("Expected the element with id='likelihood' to be a Distribution");
		}
		likelihood = (Distribution) o;
		if (posteriorInput.get() instanceof CompoundDistribution compound && compound.pDistributions.get().contains(likelihood)) {
			priorParts = compound.pDistributions.get().stream().filter(d -> d != likelihood).toList();
		}
	}

	private BEASTInterface findLikelihood(BEASTInterface o) {
		for (BEASTInterface o2 : o.listActiveBEASTObjects()) {
			if (o2.getID() != null && o2.getID().equals("likelihood")) {
				return o2;
			}
			BEASTInterface likelihood = findLikelihood(o2);
			if (likelihood != null) {
				return likelihood;
			}
		}
		return null;
	}
	
	@Override
	protected double getCurrentLogLikelihood() {
		return tempered(oldLogPrior, oldLogP, this.beta);
	}
	
		
	@Override
	protected double getScaledLogLikelihood(double beta) {
		return tempered(oldLogPrior, oldLogP, beta);
	}

	// at beta = 0 the target is the prior alone, also where the likelihood is -Infinity (0 * -Infinity would be NaN)
	private static double tempered(double logPrior, double logLikelihood, double beta) {
		return beta == 0 ? logPrior : logPrior + beta * logLikelihood;
	}

	// CompoundDistribution stops evaluating its components at the first -Infinity, so recompute every dirty one
	private double calculateLogPrior() {
		double logPrior = 0;
		for (Distribution part : priorParts) {
			logPrior += part.somethingIsDirty() ? part.calculateLogP() : part.getCurrentLogP();
		}
		return logPrior;
	}

	
    /**
     * Perform a single MCMC propose+accept/reject step.
     *
     * @param sampleNr the index of the current MCMC step
     * @return the selected {@link beast.base.inference.Operator}
     */
	@Override
    protected Operator propagateState(final long sampleNr) {
		if (Double.isNaN(oldLogP)) {
			oldLogP = likelihood.getCurrentLogP();
			oldLogPrior = oldLogLikelihood - oldLogP;
		}
		
        state.store(sampleNr);
//            if (m_nStoreEvery > 0 && sample % m_nStoreEvery == 0 && sample > 0) {
//                state.storeToFile(sample);
//            	operatorSchedule.storeToFile();
//            }

        final Operator operator = operatorSchedule.selectOperator();

//        if (printDebugInfo) System.err.print("\n" + sampleNr + " " + operator.getName()+ ":");

        final double logHastingsRatio = operator.proposal();

        if (logHastingsRatio != Double.NEGATIVE_INFINITY) {

            if (operator.requiresStateInitialisation()) {
                state.storeCalculationNodes();
                state.checkCalculationNodesDirtiness();
            }

            newLogLikelihood = posterior.calculateLogP();
    		newLogP = likelihood.getCurrentLogP();

            newLogPrior = beta == 0 && priorParts != null ? calculateLogPrior() : newLogLikelihood - newLogP;

            logAlpha = tempered(newLogPrior, newLogP, beta) - tempered(oldLogPrior, oldLogP, beta) + logHastingsRatio;
            
            if (logAlpha >= 0 || Randomizer.nextDouble() < Math.exp(logAlpha)) {
                // accept
                oldLogLikelihood = newLogLikelihood;
                oldLogP = newLogP;
                oldLogPrior = newLogPrior;
                state.acceptCalculationNodes();

                if (sampleNr >= 0) {
                    operator.accept();
                }
            } else {
                // reject
                if (sampleNr >= 0) {
                    operator.reject(newLogLikelihood == Double.NEGATIVE_INFINITY ? -1 : 0);
                }
                state.restore();
                state.restoreCalculationNodes();
            }
            state.setEverythingDirty(false);
        } else {
            // operation failed
            if (sampleNr >= 0) {
                operator.reject(-2);
            }
            state.restore();
            if (!operator.requiresStateInitialisation()) {
                state.setEverythingDirty(false);
                state.restoreCalculationNodes();
            }
        }
        log(sampleNr);
        return operator;
    }
}
