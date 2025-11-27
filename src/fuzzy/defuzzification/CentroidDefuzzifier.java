package fuzzy.defuzzification;

import java.util.Map;

public class CentroidDefuzzifier implements Defuzzifier {
    @Override
    public double defuzzify(Map<Double, Double> aggregated) {
        double num = 0.0;
        double den = 0.0;
        for (Map.Entry<Double, Double> e : aggregated.entrySet()) {
            double x = e.getKey();
            double mu = e.getValue();
            num += x * mu;
            den += mu;
        }
        return den == 0.0 ? 0.0 : num / den;
    }
}
