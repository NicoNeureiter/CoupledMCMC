package coupledMCMC;

import beast.base.core.Description;
import beast.base.core.Input;
import org.apache.commons.statistics.distribution.BetaDistribution;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Description("Path sampling / stepping stone marginal likelihood estimation in which the power posteriors "
        + "run as coupled MCMC chains, so that neighbouring steps can swap states")
public class CoupledPathSampler extends CoupledMCMC {
    public static final String LIKELIHOOD_LOG_FILE = "likelihood.log";
    static final String LIKELIHOOD_LOGGER_ID = "pathSamplingLikelihood";
    static final String LIKELIHOOD_ID = "likelihood";

    public Input<Double> alphaInput = new Input<>("alpha", "alpha parameter of the Beta(alpha,1) distribution used to space out steps. "
            + "If alpha <= 0, uniform intervals are used", 0.3);
    public Input<String> rootDirInput = new Input<>("rootdir", "directory for the per-step likelihood logs", ".");
    public Input<Integer> burnInPercentageInput = new Input<>("burnInPercentage", "burn-in percentage used for analysing the likelihood logs", 50);
    public Input<Integer> likelihoodLogEveryInput = new Input<>("likelihoodLogEvery", "log interval of the per-step likelihood logs (default chainLength/1000)");

    double[] betas;

    public CoupledPathSampler() {
        heatLikelihoodOnlyInput.setValue(true, this);
        optimiseInput.setValue(false, this);
        neighbourSwappingInput.setValue(true, this);
    }

    @Override
    public void initAndValidate() {
        if (!heatLikelihoodOnlyInput.get()) {
            throw new IllegalArgumentException("heatLikelihoodOnly must be true: path sampling needs power posteriors prior * likelihood^beta");
        }
        if (optimiseInput.get()) {
            throw new IllegalArgumentException("optimise must be false: path sampling needs a fixed beta schedule");
        }
        if (nrOfChainsInput.get() < 2) {
            throw new IllegalArgumentException("chains (= number of steps) must be at least 2");
        }
        int burnIn = burnInPercentageInput.get();
        if (burnIn < 0 || burnIn >= 100) {
            throw new IllegalArgumentException("burnInPercentage must be in [0, 100)");
        }
        betas = betaSchedule(nrOfChainsInput.get(), alphaInput.get());
        super.initAndValidate();
    }

    static double[] betaSchedule(int nrOfSteps, double alpha) {
        BetaDistribution dist = alpha > 0 ? BetaDistribution.of(alpha, 1.0) : null;
        double[] b = new double[nrOfSteps];
        for (int k = 0; k < nrOfSteps; k++) {
            double q = (nrOfSteps - 1.0 - k) / (nrOfSteps - 1);
            b[k] = dist != null ? dist.inverseCumulativeProbability(q) : q;
        }
        return b;
    }

    // HeatedChain.setTemperature maps t to 1/(1+t); t = +Infinity gives beta = 0 exactly
    @Override
    protected double getTemperature(int i) {
        return 1.0 / betas[i] - 1.0;
    }

    String stepDir(int k) {
        return new File(rootDirInput.get()).getAbsolutePath() + File.separator + "step" + k;
    }

    static double[] steppingStoneContributions(double[] betasAscending, double[][] logLs) {
        // Xie et al. 2011, Syst Biol 60:150-160, p.153; as in modelselection.inference.PathSampleAnalyser
        double[] contrib = new double[betasAscending.length - 1];
        for (int i = 0; i < contrib.length; i++) {
            double w = betasAscending[i + 1] - betasAscending[i];
            double max = Arrays.stream(logLs[i]).max().getAsDouble();
            double sum = 0;
            for (double l : logLs[i]) {
                sum += Math.exp(w * (l - max));
            }
            contrib[i] = w * max + Math.log(sum / logLs[i].length);
        }
        return contrib;
    }

    static double thermodynamicIntegration(double[] betasAscending, double[] meanLogLs) {
        double logML = 0;
        for (int i = 0; i < betasAscending.length - 1; i++) {
            logML += (betasAscending[i + 1] - betasAscending[i]) * (meanLogLs[i] + meanLogLs[i + 1]) / 2;
        }
        return logML;
    }

    static List<Double> readTrace(File file, String column, int burnInPercentage) throws IOException {
        List<String> lines = Files.readAllLines(file.toPath()).stream()
                .filter(l -> !l.startsWith("#") && !l.isBlank()).toList();
        int col = Arrays.asList(lines.get(0).split("\t")).indexOf(column);
        if (col < 0) {
            throw new IOException("No column '" + column + "' in " + file);
        }
        List<Double> values = new ArrayList<>();
        for (String line : lines.subList(1, lines.size())) {
            values.add(Double.parseDouble(line.split("\t")[col]));
        }
        return new ArrayList<>(values.subList(values.size() * burnInPercentage / 100, values.size()));
    }
}
