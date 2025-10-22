package ga.operators.selection;

import ga.core.Chromosome;
import ga.core.Population;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


// select with Roulette
public class RouletteSelection implements SelectionMethod {
    private final Random random = new Random();

    @Override
    public List<Chromosome> select(Population population) {
        List<Chromosome> parents = new ArrayList<>();

        double totalFitness = 0;
        for (Chromosome c : population.getIndividuals()) {
            totalFitness += c.getFitness();
        }

        for (int i = 0; i < 2; i++) {
            double rand = random.nextDouble() * totalFitness;
            double sum = 0;
            for (Chromosome c : population.getIndividuals()) {
                sum += c.getFitness();
                if (sum >= rand) {
                    parents.add(c);
                    break;
                }
            }
        }

        return parents;
    }
}
