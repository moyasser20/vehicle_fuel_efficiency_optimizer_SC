package case_study;

import ga.core.Chromosome;
import ga.core.Gene;
import ga.core.GeneticAlgorithm;

public class FuelInfeasibilityHandler implements GeneticAlgorithm.InfeasibilityHandler {
    
    @Override
    public boolean isInfeasible(Chromosome chromosome) {
        if (chromosome.getLength() % 2 != 0) {
            return true;
        }
        
        for (int i = 0; i < chromosome.getLength(); i++) {
            double value = chromosome.getDoubleValue(i);
            if (value < 0.0 || value > 1.0) {
                return true;
            }
        }
        
        return false;
    }
    
    @Override
    public Chromosome repair(Chromosome chromosome) {
        Chromosome repaired = chromosome.copy();
        
        if (repaired.getLength() % 2 != 0) {
            repaired.addGene(new Gene(0.5));
        }
        
        for (int i = 0; i < repaired.getLength(); i++) {
            double value = repaired.getDoubleValue(i);
            if (value < 0.0) {
                repaired.setGene(i, new Gene(0.0));
            } else if (value > 1.0) {
                repaired.setGene(i, new Gene(1.0));
            }
        }
        
        return repaired;
    }
}
