package ga.replacement;

import ga.core.Chromosome;
import ga.core.Population;
import java.util.ArrayList;
import java.util.List;

public class ElitismReplacement implements ReplacementStrategy {
    private int eliteSize = 2;

    public ElitismReplacement() {}

    public ElitismReplacement(int eliteSize) {
        this.eliteSize = eliteSize;
    }

    @Override
    public Population replace(Population currentPopulation, Population newPopulation) {
        Population result = new Population();
        
        List<Chromosome> sortedCurrent = new ArrayList<>(currentPopulation.getIndividuals());
        sortedCurrent.sort((c1, c2) -> Double.compare(c2.getFitness(), c1.getFitness()));
        
        for (int i = 0; i < Math.min(eliteSize, sortedCurrent.size()); i++) {
            result.addIndividual(sortedCurrent.get(i));
        }
        
        List<Chromosome> newIndividuals = new ArrayList<>(newPopulation.getIndividuals());
        newIndividuals.sort((c1, c2) -> Double.compare(c2.getFitness(), c1.getFitness()));
        
        for (int i = 0; i < newPopulation.size() - eliteSize; i++) {
            result.addIndividual(newIndividuals.get(i));
        }
        
        return result;
    }
}
