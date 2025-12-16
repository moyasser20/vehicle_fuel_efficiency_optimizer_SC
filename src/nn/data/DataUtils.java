package nn.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Utility class for data handling operations.
 */
public class DataUtils {
    
    /**
     * Split dataset into training and testing sets.
     * @param inputs Input samples
     * @param outputs Output samples
     * @param testRatio Ratio of test set (0.0 to 1.0)
     * @return Array containing [trainInputs, trainOutputs, testInputs, testOutputs]
     */
    public static double[][][] trainTestSplit(double[][] inputs, double[][] outputs, double testRatio) {
        if (inputs.length != outputs.length) {
            throw new IllegalArgumentException("Input and output arrays must have the same length");
        }
        if (testRatio < 0.0 || testRatio > 1.0) {
            throw new IllegalArgumentException("Test ratio must be between 0.0 and 1.0");
        }
        
        int totalSamples = inputs.length;
        int testSize = (int) (totalSamples * testRatio);
        int trainSize = totalSamples - testSize;
        
        // Shuffle indices
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalSamples; i++) {
            indices.add(i);
        }
        Random random = new Random();
        for (int i = indices.size() - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = indices.get(i);
            indices.set(i, indices.get(j));
            indices.set(j, temp);
        }
        
        // Split data
        double[][] trainInputs = new double[trainSize][];
        double[][] trainOutputs = new double[trainSize][];
        double[][] testInputs = new double[testSize][];
        double[][] testOutputs = new double[testSize][];
        
        for (int i = 0; i < trainSize; i++) {
            int idx = indices.get(i);
            trainInputs[i] = inputs[idx].clone();
            trainOutputs[i] = outputs[idx].clone();
        }
        
        for (int i = 0; i < testSize; i++) {
            int idx = indices.get(trainSize + i);
            testInputs[i] = inputs[idx].clone();
            testOutputs[i] = outputs[idx].clone();
        }
        
        return new double[][][]{trainInputs, trainOutputs, testInputs, testOutputs};
    }
    
    /**
     * Normalize data using min-max normalization.
     * @param data Data to normalize
     * @return Normalized data and normalization parameters [min, max] for each feature
     */
    public static NormalizationResult normalize(double[][] data) {
        if (data.length == 0) {
            throw new IllegalArgumentException("Data array cannot be empty");
        }
        
        int numFeatures = data[0].length;
        double[] mins = new double[numFeatures];
        double[] maxs = new double[numFeatures];
        
        // Initialize with first sample
        for (int j = 0; j < numFeatures; j++) {
            mins[j] = data[0][j];
            maxs[j] = data[0][j];
        }
        
        // Find min and max for each feature
        for (int i = 1; i < data.length; i++) {
            for (int j = 0; j < numFeatures; j++) {
                if (Double.isNaN(data[i][j]) || Double.isInfinite(data[i][j])) {
                    data[i][j] = 0.0; // Handle invalid values
                }
                mins[j] = Math.min(mins[j], data[i][j]);
                maxs[j] = Math.max(maxs[j], data[i][j]);
            }
        }
        
        // Normalize data
        double[][] normalized = new double[data.length][numFeatures];
        for (int i = 0; i < data.length; i++) {
            normalized[i] = new double[numFeatures];
            for (int j = 0; j < numFeatures; j++) {
                double range = maxs[j] - mins[j];
                if (range == 0.0) {
                    normalized[i][j] = 0.0; // Avoid division by zero
                } else {
                    normalized[i][j] = (data[i][j] - mins[j]) / range;
                }
            }
        }
        
        return new NormalizationResult(normalized, mins, maxs);
    }
    
    /**
     * Normalize data using z-score normalization (standardization).
     * @param data Data to normalize
     * @return Normalized data and normalization parameters [mean, std] for each feature
     */
    public static StandardizationResult standardize(double[][] data) {
        if (data.length == 0) {
            throw new IllegalArgumentException("Data array cannot be empty");
        }
        
        int numFeatures = data[0].length;
        double[] means = new double[numFeatures];
        double[] stds = new double[numFeatures];
        
        // Calculate mean for each feature
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < numFeatures; j++) {
                if (Double.isNaN(data[i][j]) || Double.isInfinite(data[i][j])) {
                    data[i][j] = 0.0; // Handle invalid values
                }
                means[j] += data[i][j];
            }
        }
        for (int j = 0; j < numFeatures; j++) {
            means[j] /= data.length;
        }
        
        // Calculate standard deviation for each feature
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < numFeatures; j++) {
                double diff = data[i][j] - means[j];
                stds[j] += diff * diff;
            }
        }
        for (int j = 0; j < numFeatures; j++) {
            stds[j] = Math.sqrt(stds[j] / data.length);
            if (stds[j] == 0.0) {
                stds[j] = 1.0; // Avoid division by zero
            }
        }
        
        // Standardize data
        double[][] standardized = new double[data.length][numFeatures];
        for (int i = 0; i < data.length; i++) {
            standardized[i] = new double[numFeatures];
            for (int j = 0; j < numFeatures; j++) {
                standardized[i][j] = (data[i][j] - means[j]) / stds[j];
            }
        }
        
        return new StandardizationResult(standardized, means, stds);
    }
    
    /**
     * Result class for normalization.
     */
    public static class NormalizationResult {
        public final double[][] normalizedData;
        public final double[] mins;
        public final double[] maxs;
        
        public NormalizationResult(double[][] normalizedData, double[] mins, double[] maxs) {
            this.normalizedData = normalizedData;
            this.mins = mins;
            this.maxs = maxs;
        }
        
        /**
         * Denormalize a single sample.
         */
        public double[] denormalize(double[] normalized) {
            double[] original = new double[normalized.length];
            for (int i = 0; i < normalized.length; i++) {
                double range = maxs[i] - mins[i];
                original[i] = normalized[i] * range + mins[i];
            }
            return original;
        }
    }
    
    /**
     * Result class for standardization.
     */
    public static class StandardizationResult {
        public final double[][] standardizedData;
        public final double[] means;
        public final double[] stds;
        
        public StandardizationResult(double[][] standardizedData, double[] means, double[] stds) {
            this.standardizedData = standardizedData;
            this.means = means;
            this.stds = stds;
        }
        
        /**
         * Destandardize a single sample.
         */
        public double[] destandardize(double[] standardized) {
            double[] original = new double[standardized.length];
            for (int i = 0; i < standardized.length; i++) {
                original[i] = standardized[i] * stds[i] + means[i];
            }
            return original;
        }
    }
}

