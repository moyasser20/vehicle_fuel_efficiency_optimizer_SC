package fuzzy.core;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * Represents a fuzzy IF-THEN rule with editing metadata and optional Sugeno constants.
 */
public class FuzzyRule implements Serializable {
    private static final long serialVersionUID = 1L;

    // Unique identifier for rule management
    private final String id = UUID.randomUUID().toString();

    // antecedent: variable name -> fuzzy set label
    private final Map<String, String> antecedent = new HashMap<>();
    // consequent: variable name -> fuzzy set label
    private final Map<String, String> consequent = new HashMap<>();
    // optional Sugeno-style numeric consequents: variable name -> constant output
    private final Map<String, Double> sugenoConsequent = new HashMap<>();

    // editor metadata
    private boolean enabled = true;
    private double weight = 1.0; // rule weight (0..1 typical)

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
