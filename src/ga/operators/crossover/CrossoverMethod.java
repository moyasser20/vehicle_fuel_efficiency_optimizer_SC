package ga.operators.crossover;

import ga.core.Chromosome;
import java.util.List;

public interface CrossoverMethod {
    List<Chromosome> crossover(Chromosome parent1, Chromosome parent2);
}
