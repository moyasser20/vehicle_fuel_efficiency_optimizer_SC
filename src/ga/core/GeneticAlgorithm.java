package ga.core;

import ga.operators.selection.SelectionMethod;
import ga.operators.selection.RouletteSelection;
import ga.operators.crossover.CrossoverMethod;
import ga.operators.crossover.SinglePointCrossover;
import ga.operators.mutation.MutationMethod;
import ga.operators.mutation.FloatingMutation;
import ga.replacement.ReplacementStrategy;
import ga.replacement.GenerationalReplacement;
import java.util.List;

public class GeneticAlgorithm {
    private int populationSize = 50;
    private int generations = 100;
    private double crossoverRate = 0.7;
    private double mutationRate = 0.02;
    private int chromosomeLength = 10;

    private SelectionMethod selectionMethod;
    private CrossoverMethod crossoverMethod;
    private MutationMethod mutationMethod;
    private ReplacementStrategy replacementStrategy;
    private FitnessFunction fitnessFunction;
    private InfeasibilityHandler infeasibilityHandler;

    private Population population;

    public interface FitnessFunction {
        double evaluate(Chromosome c);
    }

    public interface InfeasibilityHandler {
        boolean isInfeasible(Chromosome c);
        Chromosome repair(Chromosome c);
    }

    public GeneticAlgorithm() {
        this.selectionMethod = new RouletteSelection();
        this.crossoverMethod = new SinglePointCrossover();
        this.mutationMethod = new FloatingMutation();
        this.replacementStrategy = new GenerationalReplacement();
    }

    // Setters
    public void setPopulationSize(int size) { this.populationSize = size; }
    public void setGenerations(int generations) { this.generations = generations; }
    public void setCrossoverRate(double rate) { this.crossoverRate = rate; }
    public void setMutationRate(double rate) { this.mutationRate = rate; }
    public void setChromosomeLength(int len) { this.chromosomeLength = len; }
    public void setSelectionMethod(SelectionMethod s) { this.selectionMethod = s; }
    public void setCrossoverMethod(CrossoverMethod c) { this.crossoverMethod = c; }
    public void setMutationMethod(MutationMethod m) { this.mutationMethod = m; }
    public void setReplacementStrategy(ReplacementStrategy r) { this.replacementStrategy = r; }
    public void setFitnessFunction(FitnessFunction f) { this.fitnessFunction = f; }
    public void setInfeasibilityHandler(InfeasibilityHandler h) { this.infeasibilityHandler = h; }

//    // Getters
//    public int getPopulationSize() { return populationSize; }
//    public int getGenerations() { return generations; }
//    public double getCrossoverRate() { return crossoverRate; }
//    public double getMutationRate() { return mutationRate; }
//    public int getChromosomeLength() { return chromosomeLength; }
//    public Population getPopulation() { return population; }
    public Chromosome getBestSolution() { 
        return population != null ? population.getFittest() : null; 
    }

    // Initialize population with random floating-point values
    private void initializePopulation() {
        population = new Population();
        for (int i = 0; i < populationSize; i++) {
            Chromosome c = new Chromosome();
            for (int j = 0; j < chromosomeLength; j++) {
                // Random values between 0 and 1
                Gene g = new Gene(Math.random());
                c.addGene(g);
            }
            
            // Handle infeasible solutions if handler is provided
            if (infeasibilityHandler != null && infeasibilityHandler.isInfeasible(c)) {
                c = infeasibilityHandler.repair(c);
            }
            
            c.setFitness(fitnessFunction.evaluate(c));
            population.addIndividual(c);
        }
    }

    public void run() {
        if (fitnessFunction == null) {
            throw new IllegalStateException("Fitness function must be set before running the algorithm");
        }

        initializePopulation();

        for (int gen = 0; gen < generations; gen++) {
            Population newPopulation = new Population();

            while (newPopulation.size() < populationSize) {
                // Selection
                List<Chromosome> parents = selectionMethod.select(population);

                // Crossover
                Chromosome offspring1 = parents.get(0).copy();
                Chromosome offspring2 = parents.get(1).copy();
                if (Math.random() < crossoverRate) {
                    List<Chromosome> children = crossoverMethod.crossover(offspring1, offspring2);
                    offspring1 = children.get(0);
                    offspring2 = children.get(1);
                }

                // Mutation
                if (Math.random() < mutationRate) mutationMethod.mutate(offspring1);
                if (Math.random() < mutationRate) mutationMethod.mutate(offspring2);

                // Handle infeasible solutions
                if (infeasibilityHandler != null) {
                    if (infeasibilityHandler.isInfeasible(offspring1)) {
                        offspring1 = infeasibilityHandler.repair(offspring1);
                    }
                    if (infeasibilityHandler.isInfeasible(offspring2)) {
                        offspring2 = infeasibilityHandler.repair(offspring2);
                    }
                }

                // Evaluate fitness
                offspring1.setFitness(fitnessFunction.evaluate(offspring1));
                offspring2.setFitness(fitnessFunction.evaluate(offspring2));

                newPopulation.addIndividual(offspring1);
                newPopulation.addIndividual(offspring2);
            }

            // Replacement
            population = replacementStrategy.replace(population, newPopulation);

            System.out.println("Generation " + gen + " Best Fitness: " + population.getFittest().getFitness());
        }

        System.out.println("Best solution found: " + population.getFittest());
    }
}
