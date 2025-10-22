package ga.core;

import java.util.ArrayList;
import java.util.List;

//da bymsl mgmo3et chromosome
public class Population {
    private List<Chromosome> individuals;

    public Population() {
        this.individuals = new ArrayList<>();
    }

    public Population(int size) {
        this.individuals = new ArrayList<>(size);
    }

    public void addIndividual(Chromosome c) {
        individuals.add(c);
    }

    public List<Chromosome> getIndividuals() {
        return individuals;
    }

    public Chromosome getFittest() {
        Chromosome best = individuals.get(0);
        for (Chromosome c : individuals) {
            if (c.getFitness() > best.getFitness()) {
                best = c;
            }
        }
        return best;
    }

    public int size() {
        return individuals.size();
    }

    @Override
    public String toString() {
        return "Population{" +
                "individuals=" + individuals +
                '}';
    }
}
