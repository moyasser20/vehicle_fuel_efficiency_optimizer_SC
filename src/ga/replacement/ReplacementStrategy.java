package ga.replacement;

import ga.core.Population;

public interface ReplacementStrategy {
    Population replace(Population currentPopulation, Population newPopulation);
}
