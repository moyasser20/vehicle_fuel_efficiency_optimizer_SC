package nn.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DataUtils {
    
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
    
    public static NormalizationResult normalize(double[][] data) {
        if (data.length == 0) {
            throw new IllegalArgumentException("Data array cannot be empty");
        }
        
        int numFeatures = data[0].length;
        double[] mins = new double[numFeatures];
        double[] maxs = new double[numFeatures];
        
        for (int j = 0; j < numFeatures; j++) {
            mins[j] = data[0][j];
            maxs[j] = data[0][j];
        }
        
        for (int i = 1; i < data.length; i++) {
            for (int j = 0; j < numFeatures; j++) {
                if (Double.isNaN(data[i][j]) || Double.isInfinite(data[i][j])) {
                    data[i][j] = 0.0;
                }
                mins[j] = Math.min(mins[j], data[i][j]);
                maxs[j] = Math.max(maxs[j], data[i][j]);
            }
        }
        
        double[][] normalized = new double[data.length][numFeatures];
        for (int i = 0; i < data.length; i++) {
            normalized[i] = new double[numFeatures];
            for (int j = 0; j < numFeatures; j++) {
                double range = maxs[j] - mins[j];
                if (range == 0.0) {
                    normalized[i][j] = 0.0;
                } else {
                    normalized[i][j] = (data[i][j] - mins[j]) / range;
                }
            }
        }
        
        return new NormalizationResult(normalized, mins, maxs);
    }
    
    public static StandardizationResult standardize(double[][] data) {
        if (data.length == 0) {
            throw new IllegalArgumentException("Data array cannot be empty");
        }
        
        int numFeatures = data[0].length;
        double[] means = new double[numFeatures];
        double[] stds = new double[numFeatures];
        
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < numFeatures; j++) {
                if (Double.isNaN(data[i][j]) || Double.isInfinite(data[i][j])) {
                    data[i][j] = 0.0;
                }
                means[j] += data[i][j];
            }
        }
        for (int j = 0; j < numFeatures; j++) {
            means[j] /= data.length;
        }
        
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < numFeatures; j++) {
                double diff = data[i][j] - means[j];
                stds[j] += diff * diff;
            }
        }
        for (int j = 0; j < numFeatures; j++) {
            stds[j] = Math.sqrt(stds[j] / data.length);
            if (stds[j] == 0.0) {
                stds[j] = 1.0;
            }
        }
        
        double[][] standardized = new double[data.length][numFeatures];
        for (int i = 0; i < data.length; i++) {
            standardized[i] = new double[numFeatures];
            for (int j = 0; j < numFeatures; j++) {
                standardized[i][j] = (data[i][j] - means[j]) / stds[j];
            }
        }
        
        return new StandardizationResult(standardized, means, stds);
    }
    
    public static class NormalizationResult {
        public final double[][] normalizedData;
        public final double[] mins;
        public final double[] maxs;
        
        public NormalizationResult(double[][] normalizedData, double[] mins, double[] maxs) {
            this.normalizedData = normalizedData;
            this.mins = mins;
            this.maxs = maxs;
        }
        
        public double[] denormalize(double[] normalized) {
            double[] original = new double[normalized.length];
            for (int i = 0; i < normalized.length; i++) {
                double range = maxs[i] - mins[i];
                original[i] = normalized[i] * range + mins[i];
            }
            return original;
        }
    }
    
    public static class StandardizationResult {
        public final double[][] standardizedData;
        public final double[] means;
        public final double[] stds;
        
        public StandardizationResult(double[][] standardizedData, double[] means, double[] stds) {
            this.standardizedData = standardizedData;
            this.means = means;
            this.stds = stds;
        }
        
        public double[] destandardize(double[] standardized) {
            double[] original = new double[standardized.length];
            for (int i = 0; i < standardized.length; i++) {
                original[i] = standardized[i] * stds[i] + means[i];
            }
            return original;
        }
    }
    
    public static double[][] loadCSV(String filePath, int[] columnIndices, boolean skipHeader) {
        List<double[]> data = new ArrayList<>();
        
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            boolean isFirstLine = true;
            
            while ((line = br.readLine()) != null) {
                if (skipHeader && isFirstLine) {
                    isFirstLine = false;
                    continue;
                }
                
                String[] values = line.split(",");
                double[] row = new double[columnIndices.length];
                
                for (int i = 0; i < columnIndices.length; i++) {
                    int colIdx = columnIndices[i];
                    if (colIdx < values.length) {
                        try {
                            String value = values[colIdx].trim();
                            if (value.equalsIgnoreCase("True")) {
                                row[i] = 1.0;
                            } else if (value.equalsIgnoreCase("False")) {
                                row[i] = 0.0;
                            } else {
                                row[i] = Double.parseDouble(value);
                            }
                        } catch (NumberFormatException e) {
                            row[i] = 0.0;
                        }
                    } else {
                        row[i] = 0.0;
                    }
                }
                
                data.add(row);
            }
        } catch (IOException e) {
            throw new RuntimeException("Error reading CSV file: " + filePath, e);
        }
        
        return data.toArray(new double[data.size()][]);
    }
}


