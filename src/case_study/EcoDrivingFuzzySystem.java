package case_study;

import fuzzy.core.FuzzyRule;
import fuzzy.core.FuzzySet;
import fuzzy.core.FuzzySystem;
import fuzzy.core.LinguisticVariable;
import fuzzy.core.EvaluationResult; 
import fuzzy.defuzzification.CentroidDefuzzifier;
import fuzzy.defuzzification.MeanOfMaxDefuzzifier;
import fuzzy.inference.MamdaniEngine;
import fuzzy.inference.InferenceTrace; 
import fuzzy.norm.MinMaxNorm;
import fuzzy.norm.NormOperator;
import fuzzy.mf.TrapezoidalMF;
import fuzzy.mf.TriangularMF;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;

public class EcoDrivingFuzzySystem {
    
    public static void main(String[] args) {
        // 1. Initialize Scanner for User Input
        Scanner scanner = new Scanner(System.in);
        System.out.println("==========================================");
        System.out.println(" ECO-DRIVING FUZZY LOGIC SYSTEM");
        System.out.println("==========================================");

        // 2. Build the System Structure
        FuzzySystem sys = new FuzzySystem();

        // --- Define Variables ---
        
        // Input 1: Speed (0 - 120 km/h)
        LinguisticVariable speed = new LinguisticVariable("speed");
        speed.addFuzzySet(new FuzzySet("Slow", new TrapezoidalMF(0, 0, 20, 40), 0, 120));
        speed.addFuzzySet(new FuzzySet("Optimal", new TriangularMF(40, 65, 90), 0, 120));
        speed.addFuzzySet(new FuzzySet("Fast", new TrapezoidalMF(90, 100, 120, 120), 0, 120));
        sys.addVariable(speed);

        // Input 2: Acceleration (0 - 100 %)
        LinguisticVariable accel = new LinguisticVariable("accel");
        accel.addFuzzySet(new FuzzySet("Soft", new TrapezoidalMF(0, 0, 20, 40), 0, 100));
        accel.addFuzzySet(new FuzzySet("Moderate", new TriangularMF(30, 50, 70), 0, 100));
        accel.addFuzzySet(new FuzzySet("Hard", new TrapezoidalMF(60, 80, 100, 100), 0, 100));
        sys.addVariable(accel);

        // Output: Efficiency (0 - 100 %)
        LinguisticVariable eco = new LinguisticVariable("efficiency");
        eco.addFuzzySet(new FuzzySet("Poor", new TrapezoidalMF(0, 0, 20, 40), 0, 100));
        eco.addFuzzySet(new FuzzySet("Average", new TriangularMF(30, 50, 70), 0, 100));
        eco.addFuzzySet(new FuzzySet("Excellent", new TrapezoidalMF(60, 80, 100, 100), 0, 100));
        sys.addVariable(eco);

        // --- Define Rules ---
        
        // R1: Optimal Speed + Soft Accel -> Excellent Efficiency
        FuzzyRule r1 = new FuzzyRule();
        r1.addAntecedent("speed", "Optimal");
        r1.addAntecedent("accel", "Soft");
        r1.addConsequent("efficiency", "Excellent");
        sys.addRule(r1);

        // R2: Optimal Speed + Moderate Accel -> Average Efficiency
        FuzzyRule r2 = new FuzzyRule();
        r2.addAntecedent("speed", "Optimal");
        r2.addAntecedent("accel", "Moderate");
        r2.addConsequent("efficiency", "Average");
        sys.addRule(r2);

        // R3: Fast Speed + Hard Accel -> Poor Efficiency
        FuzzyRule r3 = new FuzzyRule();
        r3.addAntecedent("speed", "Fast");
        r3.addAntecedent("accel", "Hard");
        r3.addConsequent("efficiency", "Poor");
        sys.addRule(r3);

        // R4: Slow Speed + Hard Accel -> Poor Efficiency (Wasting gas)
        FuzzyRule r4 = new FuzzyRule();
        r4.addAntecedent("speed", "Slow");
        r4.addAntecedent("accel", "Hard");
        r4.addConsequent("efficiency", "Poor");
        sys.addRule(r4);

        // R5: Slow Speed + Soft Accel -> Average Efficiency (Too slow is not optimal)
        FuzzyRule r5 = new FuzzyRule();
        r5.addAntecedent("speed", "Slow");
        r5.addAntecedent("accel", "Soft");
        r5.addConsequent("efficiency", "Average");
        sys.addRule(r5);

        // --- Configuration Menu ---

        System.out.println("  CONFIGURATION:");
        System.out.println("Select Defuzzification Method:");
        System.out.println("   [1] Centroid (Center of Gravity)");
        System.out.println("   [2] Mean of Max (Highest Plateau Center)");
        System.out.print(" Choice: ");
        
        int defuzzChoice = 1;
        try {
            defuzzChoice = scanner.nextInt();
        } catch(Exception e) {
            scanner.next();
        }

        // Apply Configuration
        NormOperator norm = new MinMaxNorm();
        // Standard Mamdani Inference
        sys.setInferenceEngine(new MamdaniEngine(sys.getVariableMap(), sys.getRulesList(), norm, null));

        if (defuzzChoice == 2) {
            sys.setDefuzzifier(new MeanOfMaxDefuzzifier());
            System.out.println(" Selected: Mean of Max");
        } else {
            sys.setDefuzzifier(new CentroidDefuzzifier());
            System.out.println(" Selected: Centroid (Default)");
        }

        // 3. Interactive Loop
        while (true) {
            System.out.println("\n------------------------------------------------");
            System.out.println("ENTER INPUT VALUES (or type -1 in Speed to Exit):");
            
            System.out.print(" Enter Speed (0 - 120 km/h): ");
            double inSpeed = scanner.nextDouble();
            
            if (inSpeed == -1) {
                System.out.println("Exiting System. Goodbye! ");
                break;
            }

            System.out.print(" Enter Acceleration (0 - 100 %): ");
            double inAccel = scanner.nextDouble();

            Map<String, Double> inputs = new HashMap<>();
            inputs.put("speed", inSpeed);
            inputs.put("accel", inAccel);

            try {
                // 4. Evaluate with Trace
                EvaluationResult fullResult = sys.evaluateWithTrace(inputs);
                Double finalScore = fullResult.getCrispOutputs().get("efficiency");

                // ================= PRINT REPORT =================
                System.out.println("\n RESULTS:");
                
                // A. Fuzzification
                Map<String, Map<String, Double>> fuzzifiedInputs = fullResult.getFuzzifiedInputs();
                System.out.println("   [Input Fuzzification]");
                for (Entry<String, Double> inputEntry : inputs.entrySet()) {
                    String varName = inputEntry.getKey();
                    Map<String, Double> memberships = fuzzifiedInputs.get(varName);
                    
                    if (memberships != null) {
                        for (Entry<String, Double> setVal : memberships.entrySet()) {
                            if (setVal.getValue() > 0.001) {
                                System.out.printf("     - %s is '%s': %.2f%n", varName, setVal.getKey(), setVal.getValue());
                            }
                        }
                    }
                }

                // B. Rules
                System.out.println("\n   [Active Rules]");
                InferenceTrace trace = fullResult.getInferenceTrace();
                Map<String, Double> ruleFirings = trace.getRuleFirings();
                boolean fired = false;
                int rIndex = 1;
                
                for (FuzzyRule rule : sys.getRulesList()) {
                     double strength = ruleFirings.getOrDefault(rule.getId(), 0.0);
                     if (strength > 0.0) {
                         fired = true;
                         // Clean up string for display
                         String ant = rule.getAntecedent().toString().replaceAll("[{}]", "");
                         String con = rule.getConsequent().toString().replaceAll("[{}]", "");
                         System.out.printf("     R%d: IF %s THEN %s (Strength: %.2f)%n", rIndex, ant, con, strength);
                     }
                     rIndex++;
                }
                if (!fired) System.out.println("     (No rules fired)");

                // C. Final Output
                if (finalScore != null) {
                    System.out.printf("\n    FINAL EFFICIENCY: %.2f%%%n", finalScore);
                    
                    String interpretation;
                    if (finalScore < 40) interpretation = "Poor (Wasteful)";
                    else if (finalScore < 70) interpretation = "Average";
                    else interpretation = "Excellent (Eco-Friendly)";
                    
                    System.out.println("      Rating: " + interpretation);
                } else {
                    System.err.println("    Result is NULL (Check input range or rules).");
                }

            } catch (Exception e) {
                System.err.println("Error: " + e.getMessage());
                e.printStackTrace();
            }
        }
        scanner.close();
    }
}