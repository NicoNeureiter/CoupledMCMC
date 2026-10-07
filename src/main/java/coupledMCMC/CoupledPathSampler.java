package coupledMCMC;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Log;
import beast.base.spec.inference.util.ESS;
import org.apache.commons.statistics.distribution.BetaDistribution;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@Description("Path sampling / stepping stone marginal likelihood estimation in which the power posteriors "
        + "run as coupled MCMC chains, so that neighbouring steps can swap states")
public class CoupledPathSampler extends CoupledMCMC {
    public static final String LIKELIHOOD_LOG_FILE = "likelihood.log";
    static final String LIKELIHOOD_LOGGER_ID = "pathSamplingLikelihood";
    static final String LIKELIHOOD_ID = "likelihood";
    static final String BETA_SCHEDULE_FILE = "betas.txt";

    public Input<Double> alphaInput = new Input<>("alpha", "alpha parameter of the Beta(alpha,1) distribution used to space out steps. "
            + "If alpha <= 0, uniform intervals are used", 0.3);
    public Input<String> rootDirInput = new Input<>("rootdir", "directory for the per-step likelihood logs", ".");
    public Input<Integer> burnInPercentageInput = new Input<>("burnInPercentage", "burn-in percentage used for analysing the likelihood logs", 50);
    public Input<Integer> likelihoodLogEveryInput = new Input<>("likelihoodLogEvery", "log interval of the per-step likelihood logs (default chainLength/1000)");

    public Input<Boolean> evenOddSwapsInput = new Input<>("evenOddSwaps", "if true, every round attempts all even or all odd "
            + "neighbour pairs (alternating) instead of a single random neighbour pair", true);

    double[] betas;
    EvenOddSwapRunner swapRunner;
    double logMarginalLikelihood = Double.NaN;

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
        if (likelihoodLogEveryInput.get() != null && likelihoodLogEveryInput.get() <= 0) {
            throw new IllegalArgumentException("likelihoodLogEvery must be positive");
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

    // swaps are between neighbours only, so the schedule of random pairs is never used; building it also never
    // finishes for a single chain, which has no partner to pair with
    @Override
    void buildSchedule() {
    }

    @Override
    protected void configureChain(HeatedChain chain, int chainNr) {
        // added after CoupledMCMC suppresses heated-chain loggers, so every temperature slot is logged;
        // swapLoggers swaps it by ID, so each file follows its beta rather than its replica
        File dir = stepDir(chainNr);
        if (!dir.exists() && !dir.mkdirs()) {
            throw new RuntimeException("Could not create directory " + dir);
        }
        int logEvery = likelihoodLogEveryInput.get() != null
                ? likelihoodLogEveryInput.get()
                : (int) Math.max(1, chainLengthInput.get() / 1000);
        CoupledLogger logger = new CoupledLogger();
        logger.initByName("fileName", stepDirName(chainNr) + File.separator + LIKELIHOOD_LOG_FILE,
                "log", ((HeatedChainLikelihoodOnly) chain).likelihood, "logEvery", logEvery);
        logger.setID(LIKELIHOOD_LOGGER_ID);
        chain.coupledLoggersInput.get().add(logger);
    }

    // resumed logs are appended, so a changed schedule would mix samples from different betas in one step
    void checkOrWriteBetaSchedule() throws IOException {
        File file = new File(stepDir(0).getParentFile(), BETA_SCHEDULE_FILE);
        String schedule = Arrays.stream(betas).mapToObj(Double::toString).collect(Collectors.joining("\n"));
        if (restoreFromFile) {
            if (!file.exists() || !Files.readString(file.toPath()).equals(schedule)) {
                throw new IllegalStateException("Cannot resume: the beta schedule differs from the one in " + file
                        + " (chains and alpha must not change between runs)");
            }
        } else {
            if (!file.getParentFile().exists() && !file.getParentFile().mkdirs()) {
                throw new IOException("Could not create directory " + file.getParentFile());
            }
            Files.writeString(file.toPath(), schedule);
        }
    }

    @Override
    public void run() throws IOException {
        checkOrWriteBetaSchedule();
        super.run();
        // CoupledMCMC never closes its loggers; close the likelihood logs before reading them back
        for (HeatedChain chain : chains) {
            for (CoupledLogger logger : chain.coupledLoggersInput.get()) {
                if (LIKELIHOOD_LOGGER_ID.equals(logger.getID())) {
                    logger.close();
                }
            }
        }
        analyse();
    }

    @Override
    protected void runChains() {
        if (evenOddSwapsInput.get()) {
            swapRunner = new EvenOddSwapRunner(this);
            swapRunner.run();
        } else {
            super.runChains();
        }
    }

    void analyse() throws IOException {
        int n = betas.length;
        double[] betasAscending = new double[n];
        double[][] logLs = new double[n][];
        double[] means = new double[n];
        double[] ess = new double[n];
        for (int k = 0; k < n; k++) {
            int i = n - 1 - k;
            List<Double> trace = readTrace(new File(stepDir(k), LIKELIHOOD_LOG_FILE), LIKELIHOOD_ID, burnInPercentageInput.get());
            betasAscending[i] = betas[k];
            logLs[i] = trace.stream().mapToDouble(Double::doubleValue).toArray();
            means[i] = Arrays.stream(logLs[i]).average().orElse(Double.NaN);
            ess[i] = ESS.calcESS(trace);
        }
        double[] contrib = steppingStoneContributions(betasAscending, logLs);
        double ss = Arrays.stream(contrib).sum();
        double ti = thermodynamicIntegration(betasAscending, means);
        logMarginalLikelihood = alphaInput.get() > 0 ? ss : ti;

        Log.info.println("\nStep  beta          mean logL     contribution  ESS");
        for (int k = 0; k < n; k++) {
            int i = n - 1 - k;
            Log.info.println(String.format("%-5d %-13.6g %-13.6g %-13s %.1f", k, betas[k], means[i],
                    i < n - 1 ? String.format("%.6g", contrib[i]) : "-", ess[i]));
        }
        Log.info.println("stepping stone estimate = " + ss);
        Log.info.println("thermodynamic integration estimate = " + ti);
        Log.info.println("marginal L estimate = " + logMarginalLikelihood);
    }

    String stepDirName(int k) {
        return rootDirInput.get() + File.separator + "step" + k;
    }

    // Logger prepends file.name.prefix (BEAST's -prefix / -working) to its file name, so mirror that here
    File stepDir(int k) {
        String prefix = System.getProperty("file.name.prefix");
        return new File((prefix != null ? prefix : "") + stepDirName(k));
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
