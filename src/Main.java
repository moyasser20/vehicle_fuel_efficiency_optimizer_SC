import case_study.VehicleFuelEfficiencyOptimizer;
import ga.core.GeneticAlgorithm;
import ga.core.Chromosome;

public class Main {
    public static void main(String[] args) {
        System.out.println("=== Soft Computing Library - Phase 1: Genetic Algorithm Demo ===");
        System.out.println();
        VehicleFuelEfficiencyOptimizer.runOptimization();
        
        System.out.println();
        System.out.println("=== Simple Function Optimization Demo ===");
        runSimpleDemo();
    }
    
    private static void runSimpleDemo() {
        GeneticAlgorithm.FitnessFunction fitnessFunction = new GeneticAlgorithm.FitnessFunction() {
            @Override
            public double evaluate(Chromosome individual) {
                if (individual.getLength() < 2) return 0.0;
                
                double x = individual.getDoubleValue(0) * 10.0 - 5.0;
                double y = individual.getDoubleValue(1) * 10.0 - 5.0;
                
                return x * x + y * y;
            }
        };
        
        GeneticAlgorithm ga = new GeneticAlgorithm();
        ga.setPopulationSize(30);
        ga.setChromosomeLength(2);
        ga.setGenerations(20);
        ga.setCrossoverRate(0.7);
        ga.setMutationRate(0.02);
        ga.setFitnessFunction(fitnessFunction);
        
        System.out.println("Optimizing function: f(x,y) = x² + y²");
        System.out.println("Target: Find maximum value in range [-5, 5] for both x and y");
        System.out.println();
        
        ga.run();
        
        Chromosome bestSolution = ga.getBestSolution();
        double x = bestSolution.getDoubleValue(0) * 10.0 - 5.0;
        double y = bestSolution.getDoubleValue(1) * 10.0 - 5.0;
        double maxValue = x * x + y * y;
        
        System.out.println();
        System.out.println("Best solution found:");
        System.out.printf("x = %.3f, y = %.3f%n", x, y);
        System.out.printf("Maximum value = %.3f%n", maxValue);
        System.out.println("(Theoretical maximum = 50.0 at x=5, y=5)");
    }
}