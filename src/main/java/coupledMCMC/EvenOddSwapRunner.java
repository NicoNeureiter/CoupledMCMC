package coupledMCMC;

import beast.base.util.Randomizer;

/**
 * Non-reversible swap scheme (deterministic even-odd, Syed et al. 2021, JRSS-B 84:321-350):
 * each round attempts every even neighbour pair (0,1),(2,3),... or every odd pair (1,2),(3,4),...,
 * alternating, so states travel across K temperatures in O(K) rounds rather than O(K^2).
 * Temperatures are not adapted.
 */
final class EvenOddSwapRunner {
    private final CoupledMCMC mc3;

    // indexed by the colder slot of a neighbour pair
    long[] pairAttempts, pairAccepts;
    // a replica's trip from the coldest slot to the hottest and back; few round trips mean
    // the hot chains' exploration rarely reaches the cold end
    int roundTrips;
    // per chain: 0 until it has been at the coldest slot, then LAST_AT_COLD, then HOT_SINCE_COLD
    private static final int LAST_AT_COLD = 1, HOT_SINCE_COLD = 2;
    private int[] tripState;

    EvenOddSwapRunner(CoupledMCMC mc3) {
        this.mc3 = mc3;
    }

    static int[][] neighbourPairs(int nrOfChains, long round) {
        int first = (int) (round % 2);
        int[][] pairs = new int[(nrOfChains - first) / 2][];
        for (int p = 0; p < pairs.length; p++) {
            pairs[p] = new int[]{first + 2 * p, first + 2 * p + 1};
        }
        return pairs;
    }

    void run() {
        HeatedChain[] chains = mc3.chains;
        Thread[] threads = new Thread[chains.length];
        mc3.threads = threads;
        long chainLength = mc3.currentChainLength();
        int[] slotToChain = new int[chains.length];
        pairAttempts = new long[chains.length - 1];
        pairAccepts = new long[chains.length - 1];
        tripState = new int[chains.length];
        updateRoundTrips(chains);

        System.out.println("sample\tswapsColdChain\tswapProbability");
        long round = 0;
        for (long until = mc3.resampleEvery; ; until += mc3.resampleEvery, round++) {
            long end = Math.min(until, chainLength);
            // unlike HeatedChainThread, keep failures: a chain that stopped would otherwise still feed the estimate
            Throwable[] failures = new Throwable[chains.length];
            for (int k = 0; k < chains.length; k++) {
                final int chainIndex = k;
                threads[k] = new Thread(() -> {
                    try {
                        mc3.finishTimes[chainIndex] = chains[chainIndex].runTillResample(end);
                    } catch (Throwable t) {
                        failures[chainIndex] = t;
                    }
                }, CoupledMCMC.chainThreadName(chainIndex));
                threads[k].start();
                if (round == 0) {
                    // Randomizer registers a thread's generator in an unsynchronised map on first use
                    join(threads[k]);
                }
            }
            for (Thread thread : threads) {
                join(thread);
            }
            for (int k = 0; k < chains.length; k++) {
                if (failures[k] != null) {
                    throw new RuntimeException("Chain " + k + " failed at sample " + end, failures[k]);
                }
            }
            if (end == chainLength) {
                break;
            }

            for (int k = 0; k < chains.length; k++) {
                slotToChain[chains[k].getChainNr()] = k;
            }
            // pairs within a round are disjoint, so slotToChain stays valid while swapping
            for (int[] pair : neighbourPairs(chains.length, round)) {
                attemptSwap(chains[slotToChain[pair[0]]], chains[slotToChain[pair[1]]]);
            }
            updateRoundTrips(chains);

            System.out.println((end + mc3.sampleOffset) + "\t" + mc3.successfullSwaps0 + "\t"
                    + ((double) mc3.successfullSwaps / mc3.totalSwaps));
            if (end % mc3.storeEveryInput.get() == 0) {
                try {
                    mc3.storeStateToFile(end);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }

        System.err.println("#Swap attempts = " + mc3.totalSwaps);
        System.err.println("#Successful swaps = " + mc3.successfullSwaps);
        System.err.println("#Successful swaps with cold chain = " + mc3.successfullSwaps0);
        printDiagnostics(chains);
    }

    private void updateRoundTrips(HeatedChain[] chains) {
        int hottest = chains.length - 1;
        for (int k = 0; k < chains.length; k++) {
            int slot = chains[k].getChainNr();
            if (slot == 0) {
                if (tripState[k] == HOT_SINCE_COLD) {
                    roundTrips++;
                }
                tripState[k] = LAST_AT_COLD;
            } else if (slot == hottest && tripState[k] == LAST_AT_COLD) {
                tripState[k] = HOT_SINCE_COLD;
            }
        }
    }

    private void printDiagnostics(HeatedChain[] chains) {
        double[] slotBeta = new double[chains.length];
        for (HeatedChain chain : chains) {
            slotBeta[chain.getChainNr()] = chain.getBeta();
        }
        System.out.println("\nSwap acceptance per neighbour pair");
        System.out.println("slots   beta(colder)  beta(hotter)  attempts   acceptance");
        for (int s = 0; s < pairAttempts.length; s++) {
            System.out.println(String.format("%-7s %-13.6g %-13.6g %-10d %.4f", s + "-" + (s + 1), slotBeta[s], slotBeta[s + 1],
                    pairAttempts[s], pairAttempts[s] > 0 ? (double) pairAccepts[s] / pairAttempts[s] : Double.NaN));
        }
        System.out.println("Round trips (coldest -> hottest -> coldest slot): " + roundTrips);
    }

    private static void join(Thread thread) {
        try {
            thread.join();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    // a chain never holds a state with NaN tempered log probability, so NaN means a broken likelihood or posterior calculation
    private static void checkSwapAcceptance(double logAlpha, HeatedChain chain1, HeatedChain chain2) {
        if (Double.isNaN(logAlpha)) {
            throw new IllegalStateException("Swap acceptance probability is NaN for the chains at beta="
                    + chain1.getBeta() + " and beta=" + chain2.getBeta());
        }
    }

    /** colder must hold the lower chain number (higher beta) */
    void attemptSwap(HeatedChain colder, HeatedChain hotter) {
        double logAlpha = colder.getScaledLogLikelihood(hotter.getBeta()) + hotter.getScaledLogLikelihood(colder.getBeta())
                - colder.getCurrentLogLikelihood() - hotter.getCurrentLogLikelihood();
        checkSwapAcceptance(logAlpha, colder, hotter);
        int pair = colder.getChainNr();
        mc3.totalSwaps++;
        pairAttempts[pair]++;
        if (Math.exp(logAlpha) <= Randomizer.nextDouble()) {
            return;
        }
        mc3.successfullSwaps++;
        pairAccepts[pair]++;
        if (colder.getChainNr() == 0) {
            mc3.successfullSwaps0++;
        }
        double beta = colder.getBeta();
        colder.setBeta(hotter.getBeta());
        hotter.setBeta(beta);
        int chainNr = colder.getChainNr();
        colder.setChainNr(hotter.getChainNr());
        hotter.setChainNr(chainNr);
        mc3.swapLoggers(colder, hotter);
        mc3.swapOperatorTuning(colder, hotter);
    }
}
