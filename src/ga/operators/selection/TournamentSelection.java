package ga.operators.selection;

import ga.core.Chromosome;
import ga.core.Population;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class TournamentSelection implements SelectionMethod {
    private final Random random = new Random();
    private final int tournamentSize = 3; // can be adjusted

    @Override
    public List<Chromosome> select(Population population) {
        List<Chromosome> parents = new ArrayList<>();

        for (int i = 0; i < 2; i++) {
            Chromosome best = null;
            for (int j = 0; j < tournamentSize; j++) {
                Chromosome candidate = population.getIndividuals()
                        .get(random.nextInt(population.size()));
                if (best == null || candidate.getFitness() > best.getFitness()) {
                    best = candidate;
                }
            }
            parents.add(best);
        }

        return parents;
    }
}
