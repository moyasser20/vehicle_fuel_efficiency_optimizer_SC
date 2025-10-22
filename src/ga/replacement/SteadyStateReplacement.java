package ga.replacement;

import ga.core.Chromosome;
import ga.core.Population;
import java.util.ArrayList;
import java.util.List;

public class SteadyStateReplacement implements ReplacementStrategy {
    @Override
    public Population replace(Population currentPopulation, Population newPopulation) {
        Population result = new Population();
        
        List<Chromosome> combined = new ArrayList<>();
        combined.addAll(currentPopulation.getIndividuals());
        combined.addAll(newPopulation.getIndividuals());
        
        combined.sort((c1, c2) -> Double.compare(c2.getFitness(), c1.getFitness()));
        
        for (int i = 0; i < currentPopulation.size(); i++) {
            result.addIndividual(combined.get(i));
        }
        
        return result;
    }
}
