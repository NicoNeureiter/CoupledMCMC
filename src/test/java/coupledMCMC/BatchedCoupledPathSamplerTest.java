package coupledMCMC;

import beast.base.inference.CompoundDistribution;
import beast.base.spec.domain.PositiveReal;
import beast.base.spec.domain.Real;
import beast.base.spec.inference.distribution.Normal;
import beast.base.spec.inference.parameter.RealScalarParam;
import beast.base.util.Randomizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

public class BatchedCoupledPathSamplerTest {

    // y = 1 observed, y | mu ~ N(mu, 1), mu ~ N(0, 1)  =>  y ~ N(0, 2); the chains start at mu = start
    static BatchedCoupledPathSampler sampler(Path dir, double start, int steps, Integer batchSize, int preBurnin, long chainLength) {
        RealScalarParam<Real> mu = new RealScalarParam<>(start, Real.INSTANCE);
        Normal likelihood = new Normal();
        likelihood.initByName("param", new RealScalarParam<>(1.0, Real.INSTANCE), "mean", mu,
                "sigma", new RealScalarParam<>(1.0, PositiveReal.INSTANCE));
        CompoundDistribution posterior = new CompoundDistribution();
        posterior.initByName("distribution", CoupledPathSamplerTest.standardNormalPrior(mu), "distribution", likelihood);
        BatchedCoupledPathSampler sampler = new BatchedCoupledPathSampler();
        if (batchSize != null) {
            sampler.batchSizeInput.setValue(batchSize, sampler);
        }
        sampler.burnInInput.setValue(preBurnin, sampler);
        return CoupledPathSamplerTest.configure(sampler, dir, dir.toString(), dir.resolve("trace.log").toString(),
                mu, posterior, likelihood, steps, chainLength);
    }

    /** the first logged likelihood of a step, i.e. where its chain was when logging began */
    static double firstLogged(BatchedCoupledPathSampler sampler, int step) throws Exception {
        File log = new File(sampler.stepDir(step), CoupledPathSampler.LIKELIHOOD_LOG_FILE);
        return Files.readAllLines(log.toPath()).stream().filter(l -> l.matches("^\\d.*"))
                .map(l -> Double.parseDouble(l.split("\t")[1])).findFirst().orElseThrow();
    }

    @Test
    void aBatchOfOneChainRunsToCompletion(@TempDir Path dir) {
        // batchSize 1 (or a remainder of 1) gives batches of a single chain, which has nothing to swap with
        BatchedCoupledPathSampler sampler = sampler(dir, 0.0, 3, 1, 0, 2_000);
        assertTimeoutPreemptively(java.time.Duration.ofSeconds(60), () -> CoupledPathSamplerTest.runSampler(sampler));
        assertFalse(Double.isNaN(sampler.logMarginalLikelihood));
    }

    @Test
    void batchRangesSplitTheStepsAndKeepTheRemainder() {
        assertArrayEquals(new int[][]{{0, 8}, {8, 8}, {16, 4}}, BatchedCoupledPathSampler.batchRanges(20, 8));
        assertArrayEquals(new int[][]{{0, 5}}, BatchedCoupledPathSampler.batchRanges(5, 5));
        assertArrayEquals(new int[][]{{0, 2}, {2, 2}, {4, 2}}, BatchedCoupledPathSampler.batchRanges(6, 2));
    }

    @Test
    void rejectsSettingsThatCannotWork(@TempDir Path dir) {
        assertThrows(RuntimeException.class, () -> sampler(dir, 0.0, 6, 0, 0, 1000));
        assertThrows(RuntimeException.class, () -> sampler(dir, 0.0, 6, 3, -1, 1000));
        BatchedCoupledPathSampler sampler = sampler(dir, 0.0, 6, 3, 0, 1000);
        sampler.setStateFile(dir.resolve("cps.state").toString(), true);
        assertThrows(UnsupportedOperationException.class, () -> CoupledPathSamplerTest.runSampler(sampler));
    }

    @Test
    void recoversAnalyticMarginalLikelihoodInBatches(@TempDir Path dir) throws Exception {
        Randomizer.setSeed(127);
        BatchedCoupledPathSampler sampler = sampler(dir, 0.0, 12, 4, 0, 200_000);
        CoupledPathSamplerTest.runSampler(sampler);
        double expected = -0.5 * Math.log(4 * Math.PI) - 0.25;
        assertEquals(expected, sampler.logMarginalLikelihood, 0.07);
        for (int k = 0; k < 12; k++) {
            assertTrue(new File(sampler.stepDir(k), CoupledPathSampler.LIKELIHOOD_LOG_FILE).exists(), "step " + k);
        }
        assertTrue(sampler.successfullSwaps > 0);
    }

    @Test
    void laterBatchesStartFromTheWarmStartedState(@TempDir Path dir) throws Exception {
        // mu starts at 30, where the likelihood is about -420; a chain that begins there is cold-started
        Randomizer.setSeed(127);
        BatchedCoupledPathSampler sampler = sampler(dir, 30.0, 6, 3, 0, 20_000);
        CoupledPathSamplerTest.runSampler(sampler);
        for (int k = 0; k < 3; k++) {
            assertTrue(firstLogged(sampler, k) < -100, "batch 1, step " + k + " starts cold");
        }
        for (int k = 3; k < 6; k++) {
            assertTrue(firstLogged(sampler, k) > -50, "batch 2, step " + k + " is warm-started");
        }
    }

    @Test
    void preBurnInOnlyWarmsUpTheFirstBatch(@TempDir Path dir) throws Exception {
        Randomizer.setSeed(127);
        BatchedCoupledPathSampler sampler = sampler(dir, 30.0, 6, 3, 5_000, 20_000);
        CoupledPathSamplerTest.runSampler(sampler);
        for (int k = 0; k < 6; k++) {
            assertTrue(firstLogged(sampler, k) > -50, "step " + k);
        }
        assertFalse(Files.exists(dir.resolve("cps.state.warmstart")), "the temporary warm-start file is cleaned up");
    }
}
