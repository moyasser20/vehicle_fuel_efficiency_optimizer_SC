package fuzzy.core;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

public class FuzzyRule implements Serializable {
    private static final long serialVersionUID = 1L;

    private final String id = UUID.randomUUID().toString();

    private final Map<String, String> antecedent = new HashMap<>();
    private final Map<String, String> consequent = new HashMap<>();
    private final Map<String, Double> sugenoConsequent = new HashMap<>();

    private boolean enabled = true;
    private double weight = 1.0;

    public FuzzyRule() {}

    public String getId() { return id; }

    public void addAntecedent(String var, String setLabel) { antecedent.put(var, setLabel); }
    public void addConsequent(String var, String setLabel) { consequent.put(var, setLabel); }
    public void addSugenoConsequent(String var, double value) { sugenoConsequent.put(var, value); }

    public Map<String, String> getAntecedent() { return antecedent; }
    public Map<String, String> getConsequent() { return consequent; }
    public Map<String, Double> getSugenoConsequent() { return sugenoConsequent; }

    public boolean isEnabled() { return enabled; }
    public void setEnabled(boolean enabled) { this.enabled = enabled; }

    public double getWeight() { return weight; }
    public void setWeight(double weight) { this.weight = weight; }
}
