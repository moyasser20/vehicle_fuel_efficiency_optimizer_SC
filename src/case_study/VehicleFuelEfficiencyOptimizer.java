package case_study;

import ga.core.GeneticAlgorithm;
import ga.core.Chromosome;
import ga.operators.selection.TournamentSelection;
import ga.operators.crossover.TwoPointCrossover;
import ga.operators.mutation.FloatingMutation;
import ga.replacement.ElitismReplacement;

public class VehicleFuelEfficiencyOptimizer {
    
    public static void runOptimization() {
        System.out.println("=== Vehicle Fuel Efficiency Optimizer ===");
        System.out.println("Finding optimal driving behavior for maximum fuel efficiency");
        System.out.println();
        
        // Create genetic algorithm
        GeneticAlgorithm ga = new GeneticAlgorithm();
        
        // Configure parameters
        ga.setPopulationSize(50);
        ga.setChromosomeLength(10); // 5 speed-gear pairs
        ga.setGenerations(30);
        ga.setCrossoverRate(0.8);
        ga.setMutationRate(0.1);
        
        // Set fitness function and infeasibility handler
        ga.setFitnessFunction(new FuelEfficiencyFitness());
        ga.setInfeasibilityHandler(new FuelInfeasibilityHandler());
        
        // Set operators
        ga.setSelectionMethod(new TournamentSelection());
        ga.setCrossoverMethod(new TwoPointCrossover());
        ga.setMutationMethod(new FloatingMutation());
        ga.setReplacementStrategy(new ElitismReplacement(2));
        
        System.out.println("Running optimization...");
        ga.run();
        
        // Show results
        Chromosome bestSolution = ga.getBestSolution();
        System.out.println();
        System.out.println("=== RESULTS ===");
        System.out.println("Best Fuel Efficiency: " + String.format("%.2f", bestSolution.getFitness()) + " km/L");
        System.out.println();
        System.out.println("Optimal Driving Pattern:");
        showDrivingPattern(bestSolution);
    }
    
    private static void showDrivingPattern(Chromosome chromosome) {
        System.out.println("Step | Speed (km/h) | Gear");
        System.out.println("-----|--------------|-----");
        
        for (int i = 0; i < chromosome.getLength(); i += 2) {
            if (i + 1 < chromosome.getLength()) {
                double speedNorm = chromosome.getDoubleValue(i);
                double gearNorm = chromosome.getDoubleValue(i + 1);
                
                double speed = speedNorm * 100.0;
                int gear = (int) Math.round(gearNorm * 5 + 1);
                
                System.out.printf("%4d | %12.1f | %4d%n", (i/2 + 1), speed, gear);
            }
        }
    }
}
