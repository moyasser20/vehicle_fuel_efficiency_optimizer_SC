package ga.replacement;

import ga.core.Population;

public class GenerationalReplacement implements ReplacementStrategy {
    @Override
    public Population replace(Population currentPopulation, Population newPopulation) {
        return newPopulation;
    }
}
