package fuzzy.defuzzification;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class MeanOfMaxDefuzzifier implements Defuzzifier {
    @Override
    public double defuzzify(Map<Double, Double> aggregated) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : aggregated.values()) if (v > max) max = v;
        if (max <= 0.0) return 0.0;
        List<Double> xs = new ArrayList<>();
        for (Map.Entry<Double, Double> e : aggregated.entrySet()) {
            if (Math.abs(e.getValue() - max) < 1e-9) xs.add(e.getKey());
        }
        double sum = 0.0;
        for (double x : xs) sum += x;
        return xs.isEmpty() ? 0.0 : sum / xs.size();
    }
}
