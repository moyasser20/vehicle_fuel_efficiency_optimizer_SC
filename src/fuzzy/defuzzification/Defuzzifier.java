package fuzzy.defuzzification;

import java.util.Map;

public interface Defuzzifier {
    double defuzzify(Map<Double, Double> aggregated);
}
