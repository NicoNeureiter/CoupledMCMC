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
        long chainLength = mc3.chainLengthInput.get();
        int[] slotToChain = new int[chains.length];

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
                });
                threads[k].start();
            }
            for (Thread thread : threads) {
                try {
                    thread.join();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
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
    }

    /** colder must hold the lower chain number (higher beta) */
    private void attemptSwap(HeatedChain colder, HeatedChain hotter) {
        double logAlpha = colder.getScaledLogLikelihood(hotter.getBeta()) + hotter.getScaledLogLikelihood(colder.getBeta())
                - colder.getCurrentLogLikelihood() - hotter.getCurrentLogLikelihood();
        mc3.totalSwaps++;
        if (Math.exp(logAlpha) <= Randomizer.nextDouble()) {
            return;
        }
        mc3.successfullSwaps++;
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
